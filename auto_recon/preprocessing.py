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


def _laplacian_variance(image_path: str | Path) -> float:
    """Return the Laplacian variance of a grayscale image (higher = sharper)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def select_sharp_frames(
    frame_paths: list[Path],
    threshold: float = 100.0,
    min_keep_ratio: float = 0.25,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Return frames whose Laplacian-variance sharpness exceeds *threshold*.

    If fewer than ``min_keep_ratio * len(frame_paths)`` frames pass, the
    threshold is automatically lowered to retain at least that fraction
    (sorted by sharpness descending).
    """
    logger = logger or logging.getLogger("auto_recon.preprocessing")

    scores: list[tuple[Path, float]] = []
    for p in frame_paths:
        s = _laplacian_variance(p)
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
) -> dict[str, str | Path]:
    """Run the full preprocessing pipeline on a 360° video.

    Steps:
        1. Extract frames at *fps*
        2. Select sharp frames (discard blurry ones)
        3. Generate nadir mask
        4. Apply mask to selected frames

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

    # 4. Apply mask
    processed = apply_mask(sharp_frames, mask, masked_dir, inpaint=inpaint, logger=logger)

    logger.info(
        "=== Preprocessing complete: %d frames ready in %s ===",
        len(processed), masked_dir,
    )

    return {
        "frames_dir": masked_dir,
        "mask_path": mask_path,
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
