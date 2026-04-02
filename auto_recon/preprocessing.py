"""Frame extraction and bottom-mask preprocessing for 360° equirectangular video.

This module implements the first stage of the auto-recon pipeline:
1. Extract frames from MP4 video at a configurable FPS interval
2. Score frame sharpness and discard blurry frames
3. Generate a nadir (bottom) mask for equirectangular images
4. Apply the mask to all selected frames
"""

from __future__ import annotations

import datetime
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# JST timezone & logging helpers
# ---------------------------------------------------------------------------

_JST = datetime.timezone(datetime.timedelta(hours=9))


def _jst_now() -> datetime.datetime:
    return datetime.datetime.now(tz=_JST)


def _setup_logger(output_dir: str | Path) -> logging.Logger:
    """Create a logger that writes to both console and a timestamped log file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("auto_recon.preprocessing")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # File handler with JST timestamp
    ts = _jst_now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(output_dir / f"{ts}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# 1. Frame extraction
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 1.0,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Extract frames from *video_path* at *fps* frames-per-second using ffmpeg.

    Frames are saved as high-quality JPG files (``-qscale:v 2``).

    Returns a sorted list of extracted frame paths.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or logging.getLogger("auto_recon.preprocessing")

    pattern = str(output_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-qscale:v", "2",
        pattern,
    ]

    logger.info("Extracting frames: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ffmpeg stderr:\n%s", result.stderr)
        raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    logger.info("Extracted %d frames to %s", len(frames), output_dir)
    return frames


# ---------------------------------------------------------------------------
# 2. Blur detection / sharp-frame selection
# ---------------------------------------------------------------------------


def _laplacian_variance(image_path: str | Path) -> tuple[str, float]:
    """Return (filename, Laplacian variance) of a grayscale image (higher = sharper).

    Returns a tuple for multiprocessing compatibility (Path objects can't
    be pickled across processes on Windows).
    """
    path = str(image_path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (path, 0.0)
    return (path, float(cv2.Laplacian(img, cv2.CV_64F).var()))


def select_sharp_frames(
    frame_paths: list[Path],
    threshold: float = 100.0,
    min_keep_ratio: float = 0.25,
    logger: logging.Logger | None = None,
    num_workers: int = 0,
) -> list[Path]:
    """Return frames whose Laplacian-variance sharpness exceeds *threshold*.

    If fewer than ``min_keep_ratio * len(frame_paths)`` frames pass, the
    threshold is automatically lowered to retain at least that fraction
    (sorted by sharpness descending).

    Parameters
    ----------
    num_workers:
        Number of parallel workers for sharpness evaluation.
        0 = auto (cpu_count), 1 = sequential (no multiprocessing).
    """
    logger = logger or logging.getLogger("auto_recon.preprocessing")

    if num_workers == 0:
        import os, sys
        cpu = os.cpu_count() or 1
        if sys.platform == "win32":
            cpu = min(cpu, 32)
        num_workers = min(cpu, len(frame_paths))

    str_paths = [str(p) for p in frame_paths]

    if num_workers > 1 and len(frame_paths) > 4:
        from concurrent.futures import ThreadPoolExecutor
        logger.info("Evaluating sharpness with %d threads (%d frames)", num_workers, len(frame_paths))
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(_laplacian_variance, str_paths))
        scores = [(Path(p), s) for p, s in results]
        for p, s in scores:
            logger.debug("Sharpness %s: %.2f", p.name, s)
    else:
        scores = []
        for p in frame_paths:
            _, s = _laplacian_variance(str(p))
            scores.append((p, s))
            logger.debug("Sharpness %s: %.2f", p.name, s)

    # Sort descending by sharpness
    scores.sort(key=lambda x: x[1], reverse=True)

    passing = [(p, s) for p, s in scores if s >= threshold]
    min_keep = max(1, int(len(frame_paths) * min_keep_ratio))

    if len(passing) >= min_keep:
        selected = [p for p, _ in passing]
        logger.info(
            "Sharp-frame selection: %d / %d frames passed (threshold=%.1f)",
            len(selected), len(frame_paths), threshold,
        )
    else:
        # Auto-lower: take the top min_keep frames regardless of threshold
        selected = [p for p, _ in scores[:min_keep]]
        effective_threshold = scores[min_keep - 1][1] if min_keep <= len(scores) else 0.0
        logger.warning(
            "Only %d frames passed threshold %.1f; auto-lowered to %.1f, keeping %d frames",
            len(passing), threshold, effective_threshold, len(selected),
        )

    # Return in original filename order
    selected_set = set(selected)
    return [p for p in frame_paths if p in selected_set]


# ---------------------------------------------------------------------------
# 3. Nadir mask generation
# ---------------------------------------------------------------------------


def generate_nadir_mask(
    width: int,
    height: int,
    mask_ratio: float = 0.18,
    output_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Generate a binary nadir mask for an equirectangular image.

    The mask is white (255) in the region to *keep* and black (0) in the
    nadir region to mask out.  The nadir region is an ellipse centred at
    the bottom-centre of the image covering roughly the bottom
    ``mask_ratio`` of the image height.

    Returns the mask as a single-channel uint8 numpy array.
    """
    logger = logger or logging.getLogger("auto_recon.preprocessing")

    mask = np.full((height, width), 255, dtype=np.uint8)

    # Ellipse centre at the bottom-centre of the image
    cx = width // 2
    cy = height  # centre is at the very bottom edge

    # Semi-axes: wide ellipse spanning most of the width in the nadir zone
    axis_x = int(width * 0.45)  # horizontal reach ~90% of half-width
    axis_y = int(height * mask_ratio)  # vertical reach into the image

    cv2.ellipse(mask, (cx, cy), (axis_x, axis_y), 0, 0, 360, 0, -1)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), mask)
        logger.info("Nadir mask saved to %s (%dx%d, ratio=%.2f)", output_path, width, height, mask_ratio)

    return mask


# ---------------------------------------------------------------------------
# 4. Apply mask
# ---------------------------------------------------------------------------


def apply_mask(
    frame_paths: list[Path],
    mask: np.ndarray,
    output_dir: str | Path,
    inpaint: bool = False,
    inpaint_radius: int = 5,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Apply *mask* to every frame and save the results as JPG.

    Masked pixels are filled with black.  If *inpaint* is ``True``,
    OpenCV's Navier-Stokes inpainting is used instead.

    Returns a list of output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or logging.getLogger("auto_recon.preprocessing")

    # The inpaint mask needs the *inverse* (white = region to fill)
    inpaint_mask = cv2.bitwise_not(mask)

    results: list[Path] = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Could not read %s, skipping", p)
            continue

        # Resize mask if frame dimensions differ
        h, w = img.shape[:2]
        if (mask.shape[1], mask.shape[0]) != (w, h):
            cur_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            cur_inpaint_mask = cv2.bitwise_not(cur_mask)
        else:
            cur_mask = mask
            cur_inpaint_mask = inpaint_mask

        if inpaint:
            out = cv2.inpaint(img, cur_inpaint_mask, inpaint_radius, cv2.INPAINT_NS)
        else:
            out = cv2.bitwise_and(img, img, mask=cur_mask)

        out_path = output_dir / p.name
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        results.append(out_path)

    logger.info("Applied nadir mask to %d frames -> %s", len(results), output_dir)
    return results


# ---------------------------------------------------------------------------
# 5. Main orchestration
# ---------------------------------------------------------------------------


def preprocess_video(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 1.0,
    blur_threshold: float = 100.0,
    mask_ratio: float = 0.18,
    inpaint: bool = False,
    sam3_mode: str | None = None,
    sam3_confidence: float = 0.3,
    sam3_prompt: str = "person",
    sam3_batch_size: int = 4,
    sam3_scale: float = 1.0,
) -> dict[str, str | Path]:
    """Run the full preprocessing pipeline on a 360° video.

    Steps:
        1. Extract frames at *fps*
        2. Select sharp frames (discard blurry ones)
        3. Generate nadir mask
        4. (Optional) SAM3 person detection masks
        5. Apply combined masks to selected frames

    Parameters
    ----------
    sam3_mode:
        SAM3 person masking mode. ``None`` to disable, ``"equirect"`` to run
        SAM3 directly on equirectangular images, ``"pinhole"`` to extract
        perspective views first.
    sam3_confidence:
        SAM3 detection confidence threshold (0-1).
    sam3_prompt:
        Text prompt for SAM3 (default: "person").

    Returns a dict with keys:
        - ``frames_dir``: directory containing the final processed frames
        - ``mask_path``: path to the generated nadir mask image
        - ``frame_count``: number of frames after processing
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    logger = _setup_logger(output_dir)
    logger.info("=== Preprocessing started at %s ===", _jst_now().isoformat())
    logger.info("Video: %s", video_path)

    # Sub-directories
    raw_dir = output_dir / "frames_raw"
    masked_dir = output_dir / "frames_masked"

    # 1. Extract frames
    all_frames = extract_frames(video_path, raw_dir, fps=fps, logger=logger)
    if not all_frames:
        raise RuntimeError("No frames extracted - check video file and ffmpeg installation")

    # 2. Select sharp frames
    sharp_frames = select_sharp_frames(all_frames, threshold=blur_threshold, logger=logger)

    # 3. Generate nadir mask (use dimensions of the first frame)
    sample = cv2.imread(str(sharp_frames[0]))
    h, w = sample.shape[:2]
    mask_path = output_dir / "nadir_mask.jpg"
    mask = generate_nadir_mask(w, h, mask_ratio=mask_ratio, output_path=mask_path, logger=logger)

    # 4. Copy sharp frames to output (no black-fill — masking is done via
    #    mask images passed to 3DGS with --mask-mode ignore).
    masked_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    processed: list[Path] = []
    for p in sharp_frames:
        dst = masked_dir / p.name
        shutil.copy2(p, dst)
        processed.append(dst)
    logger.info("Copied %d sharp frames to %s", len(processed), masked_dir)

    # 5. Build per-frame mask images (255=keep, 0=ignore).
    #    Always includes the nadir mask.  SAM3 person masks are merged when
    #    enabled.  All mask sources are combined with bitwise AND.
    masks_dir = output_dir / "masks_combined"
    masks_dir.mkdir(parents=True, exist_ok=True)

    sam3_mask_dir: Path | None = None
    if sam3_mode is not None:
        from auto_recon.sam3_masking import (
            mask_persons_equirect,
            mask_persons_pinhole,
            mask_persons_equirect_trt,
            mask_persons_pinhole_trt,
        )

        sam3_mask_dir = output_dir / "sam3_masks"
        logger.info("SAM3 person masking: mode=%s, prompt=%r, confidence=%.2f",
                     sam3_mode, sam3_prompt, sam3_confidence)

        if sam3_mode == "equirect":
            mask_persons_equirect(
                sharp_frames, sam3_mask_dir,
                prompt=sam3_prompt, confidence=sam3_confidence,
                scale=sam3_scale,
            )
        elif sam3_mode == "pinhole":
            mask_persons_pinhole(
                sharp_frames, sam3_mask_dir,
                prompt=sam3_prompt, confidence=sam3_confidence,
                batch_size=sam3_batch_size,
                scale=sam3_scale,
            )
        elif sam3_mode == "trt":
            mask_persons_pinhole_trt(
                sharp_frames, sam3_mask_dir,
                confidence=sam3_confidence,
            )
        elif sam3_mode == "trt-equirect":
            mask_persons_equirect_trt(
                sharp_frames, sam3_mask_dir,
                confidence=sam3_confidence,
            )
        else:
            raise ValueError(
                f"Unknown sam3_mode: {sam3_mode!r}. "
                "Use 'pinhole', 'equirect', 'trt', or 'trt-equirect'."
            )

    for frame_path in processed:
        # Start with nadir mask (resize if needed)
        combined = mask.copy()

        # Merge SAM3 mask if available
        if sam3_mask_dir is not None:
            sam3_mask_path = sam3_mask_dir / f"{frame_path.stem}_mask.png"
            if sam3_mask_path.exists():
                sam3_m = cv2.imread(str(sam3_mask_path), cv2.IMREAD_GRAYSCALE)
                if sam3_m.shape[:2] != (h, w):
                    sam3_m = cv2.resize(sam3_m, (w, h), interpolation=cv2.INTER_NEAREST)
                combined = cv2.bitwise_and(combined, sam3_m)

        cv2.imwrite(str(masks_dir / f"{frame_path.stem}.png"), combined)

    logger.info("Generated %d mask images (nadir%s) -> %s",
                len(processed),
                " + SAM3" if sam3_mode else "",
                masks_dir)

    logger.info(
        "=== Preprocessing complete: %d frames ready in %s ===",
        len(processed), masked_dir,
    )

    return {
        "frames_dir": masked_dir,
        "mask_path": mask_path,
        "masks_dir": masks_dir,
        "frame_count": len(processed),
    }


# ---------------------------------------------------------------------------
# Standalone entry-point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess 360° video for Gaussian Splatting")
    parser.add_argument("video", type=str, help="Path to MP4 video file")
    parser.add_argument("-o", "--output", type=str, default="./output/preprocessing", help="Output directory")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate (default: 1.0)")
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="Laplacian variance threshold")
    parser.add_argument("--mask-ratio", type=float, default=0.18, help="Nadir mask height ratio (0-1)")
    parser.add_argument("--inpaint", action="store_true", help="Use inpainting instead of black fill")

    args = parser.parse_args()

    result = preprocess_video(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        blur_threshold=args.blur_threshold,
        mask_ratio=args.mask_ratio,
        inpaint=args.inpaint,
    )

    print(f"\nDone! {result['frame_count']} frames in: {result['frames_dir']}")
    print(f"Nadir mask: {result['mask_path']}")
