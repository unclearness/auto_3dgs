"""SAM3-based person masking for 360° images.

Detects and masks people (and other moving objects) using SAM3's
text-prompted segmentation.  Two modes are supported:

- **equirect mode**: Run SAM3 directly on equirectangular images.
  Faster but may miss small or distorted figures near poles.

- **pinhole mode**: Extract perspective views from equirectangular,
  run SAM3 on each view, then project masks back to equirect space.
  Higher quality but slower (N perspective views per frame).

Both modes produce per-frame binary masks where masked (person) regions
are 0 and background is 255, matching the nadir mask convention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("auto_recon.sam3_masking")

# Default SAM3 assets path
_SAM3_BPE_PATH = (
    Path(__file__).resolve().parent.parent / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
)


# ---------------------------------------------------------------------------
# SAM3 model singleton (lazy-loaded, heavy)
# ---------------------------------------------------------------------------

_processor = None


def _get_processor(confidence: float = 0.3):
    """Lazy-load SAM3 model and processor."""
    global _processor
    if _processor is None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info("Loading SAM3 model...")
        model = build_sam3_image_model(bpe_path=str(_SAM3_BPE_PATH))
        _processor = Sam3Processor(model, confidence_threshold=confidence)
        logger.info("SAM3 model loaded (confidence=%.2f)", confidence)
    else:
        if _processor.confidence_threshold != confidence:
            _processor.set_confidence_threshold(confidence)
    return _processor


# ---------------------------------------------------------------------------
# Core: run SAM3 on a single image
# ---------------------------------------------------------------------------


def _segment_persons(
    image: np.ndarray,
    processor,
    prompt: str = "person",
) -> np.ndarray:
    """Run SAM3 text-prompted segmentation on an image.

    Parameters
    ----------
    image:
        BGR image (H, W, 3).
    processor:
        Sam3Processor instance.
    prompt:
        Text prompt for segmentation.

    Returns
    -------
    np.ndarray
        Binary mask (H, W) uint8 -255 where person detected, 0 elsewhere.
    """
    h, w = image.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(pil_image)
        output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"]  # (N, 1, H, W) bool tensor
    if masks.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Union of all detected person masks
    combined = masks.any(dim=0)[0].cpu().numpy().astype(np.uint8) * 255
    return combined


# ---------------------------------------------------------------------------
# Equirectangular perspective extraction helpers (lightweight, no COLMAP)
# ---------------------------------------------------------------------------


def _equirect_to_persp_map(
    eq_w: int,
    eq_h: int,
    fov_deg: float,
    out_w: int,
    out_h: int,
    yaw_deg: float,
    pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute remap coordinates from equirect to a perspective view.

    Returns (map_x, map_y) for cv2.remap.
    """
    fov = np.radians(fov_deg)
    f = out_w / (2.0 * np.tan(fov / 2.0))
    cx, cy = out_w / 2.0, out_h / 2.0

    u = np.arange(out_w, dtype=np.float64) - cx
    v = np.arange(out_h, dtype=np.float64) - cy
    uu, vv = np.meshgrid(u, v)

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    x, y, z = uu, vv, np.full_like(uu, f)
    y1 = y * cos_p - z * sin_p
    z1 = y * sin_p + z * cos_p
    x2 = x * cos_y + z1 * sin_y
    z2 = -x * sin_y + z1 * cos_y
    y2 = y1

    norm = np.sqrt(x2**2 + y2**2 + z2**2)
    theta = np.arctan2(x2, z2)
    phi = np.arcsin(np.clip(y2 / norm, -1, 1))

    src_x = ((theta / np.pi + 1.0) / 2.0 * eq_w).astype(np.float32) % eq_w
    src_y = ((phi / (np.pi / 2.0) + 1.0) / 2.0 * eq_h).astype(np.float32)
    return src_x, src_y


def _persp_mask_to_equirect(
    persp_mask: np.ndarray,
    eq_w: int,
    eq_h: int,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    """Project a perspective mask back to equirectangular space.

    Returns a (eq_h, eq_w) uint8 mask.
    """
    out_h, out_w = persp_mask.shape[:2]
    fov = np.radians(fov_deg)
    f = out_w / (2.0 * np.tan(fov / 2.0))
    cx, cy = out_w / 2.0, out_h / 2.0

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # For each equirect pixel, compute corresponding perspective pixel
    eq_u = np.arange(eq_w, dtype=np.float64)
    eq_v = np.arange(eq_h, dtype=np.float64)
    eq_uu, eq_vv = np.meshgrid(eq_u, eq_v)

    # Equirect to spherical
    lon = (eq_uu / eq_w - 0.5) * 2 * np.pi   # [-pi, pi]
    lat = -(eq_vv / eq_h - 0.5) * np.pi       # [pi/2, -pi/2]

    # Spherical to 3D
    x3 = np.cos(lat) * np.sin(lon)
    y3 = -np.sin(lat)
    z3 = np.cos(lat) * np.cos(lon)

    # Inverse rotation: world -> camera
    # R = R_yaw @ R_pitch, so R_inv = R_pitch^T @ R_yaw^T
    x_r = x3 * cos_y - z3 * sin_y
    z_r = x3 * sin_y + z3 * cos_y
    y_r = y3

    y_rr = y_r * cos_p + z_r * sin_p
    z_rr = -y_r * sin_p + z_r * cos_p
    x_rr = x_r

    # Project to perspective
    valid = z_rr > 0
    px = np.full_like(x_rr, -1.0)
    py = np.full_like(y_rr, -1.0)
    px[valid] = f * x_rr[valid] / z_rr[valid] + cx
    py[valid] = f * y_rr[valid] / z_rr[valid] + cy

    # Check bounds
    in_bounds = valid & (px >= 0) & (px < out_w) & (py >= 0) & (py < out_h)

    eq_mask = np.zeros((eq_h, eq_w), dtype=np.uint8)
    px_int = px[in_bounds].astype(int)
    py_int = py[in_bounds].astype(int)
    eq_mask[in_bounds] = persp_mask[py_int, px_int]
    return eq_mask


# ---------------------------------------------------------------------------
# Mode 1: Equirectangular direct
# ---------------------------------------------------------------------------


def mask_persons_equirect(
    image_paths: list[Path],
    output_dir: Path,
    *,
    prompt: str = "person",
    confidence: float = 0.3,
    dilate_px: int = 15,
) -> list[Path]:
    """Run SAM3 directly on equirectangular images.

    Parameters
    ----------
    image_paths:
        List of equirectangular image file paths.
    output_dir:
        Directory to write mask images.
    prompt:
        Text prompt (default: "person").
    confidence:
        SAM3 confidence threshold.
    dilate_px:
        Dilate detected masks by this many pixels for safety margin.

    Returns
    -------
    list[Path]
        Paths to output mask images (255=keep, 0=masked).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = _get_processor(confidence)

    mask_paths: list[Path] = []
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read %s, skipping", img_path)
            continue

        person_mask = _segment_persons(img, processor, prompt)

        # Dilate to add safety margin
        if dilate_px > 0 and person_mask.any():
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            person_mask = cv2.dilate(person_mask, kernel)

        # Invert: 255=keep, 0=person (masked)
        keep_mask = cv2.bitwise_not(person_mask)

        out_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), keep_mask)
        mask_paths.append(out_path)

        if i % 5 == 0 or i == len(image_paths) - 1:
            n_person_px = int((person_mask > 0).sum())
            logger.info(
                "[equirect] %d/%d %s -%d person pixels",
                i + 1, len(image_paths), img_path.name, n_person_px,
            )

    return mask_paths


# ---------------------------------------------------------------------------
# Mode 2: Pinhole (perspective) views
# ---------------------------------------------------------------------------


def mask_persons_pinhole(
    image_paths: list[Path],
    output_dir: Path,
    *,
    prompt: str = "person",
    confidence: float = 0.3,
    fov_deg: float = 90.0,
    out_size: tuple[int, int] = (1024, 1024),
    pitch_angles: list[float] | None = None,
    yaw_step_deg: float = 90.0,
    dilate_px: int = 15,
) -> list[Path]:
    """Run SAM3 on perspective views extracted from equirectangular images.

    For each equirect image, extracts a grid of perspective views, runs SAM3
    on each view, then projects the masks back to equirect space and merges.

    Parameters
    ----------
    image_paths:
        List of equirectangular image file paths.
    output_dir:
        Directory to write mask images.
    prompt:
        Text prompt (default: "person").
    confidence:
        SAM3 confidence threshold.
    fov_deg:
        Field of view for perspective views.
    out_size:
        (width, height) of perspective views.
    pitch_angles:
        Pitch angles in degrees. Default: [-30, 0, 30].
    yaw_step_deg:
        Step between yaw angles. Default: 90° (4 views).
    dilate_px:
        Dilate detected masks by this many pixels.

    Returns
    -------
    list[Path]
        Paths to output mask images (255=keep, 0=masked).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = _get_processor(confidence)

    if pitch_angles is None:
        pitch_angles = [-30.0, 0.0, 30.0]
    yaw_angles = list(np.arange(0, 360, yaw_step_deg))
    out_w, out_h = out_size

    n_views = len(yaw_angles) * len(pitch_angles)
    logger.info(
        "Pinhole SAM3: %d images x %d views (%d yaw x %d pitch)",
        len(image_paths), n_views, len(yaw_angles), len(pitch_angles),
    )

    mask_paths: list[Path] = []
    for i, img_path in enumerate(image_paths):
        equirect = cv2.imread(str(img_path))
        if equirect is None:
            logger.warning("Cannot read %s, skipping", img_path)
            continue

        eq_h, eq_w = equirect.shape[:2]
        combined_person_mask = np.zeros((eq_h, eq_w), dtype=np.uint8)

        for pitch in pitch_angles:
            for yaw in yaw_angles:
                # Extract perspective view
                map_x, map_y = _equirect_to_persp_map(
                    eq_w, eq_h, fov_deg, out_w, out_h, yaw, pitch,
                )
                persp = cv2.remap(
                    equirect, map_x, map_y,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
                )

                # Run SAM3 on perspective view
                persp_mask = _segment_persons(persp, processor, prompt)

                if persp_mask.any():
                    # Project mask back to equirect
                    eq_mask = _persp_mask_to_equirect(
                        persp_mask, eq_w, eq_h, fov_deg, yaw, pitch,
                    )
                    combined_person_mask = np.maximum(combined_person_mask, eq_mask)

        # Dilate for safety margin
        if dilate_px > 0 and combined_person_mask.any():
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            combined_person_mask = cv2.dilate(combined_person_mask, kernel)

        # Invert: 255=keep, 0=person
        keep_mask = cv2.bitwise_not(combined_person_mask)

        out_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), keep_mask)
        mask_paths.append(out_path)

        n_person_px = int((combined_person_mask > 0).sum())
        logger.info(
            "[pinhole] %d/%d %s -%d person pixels",
            i + 1, len(image_paths), img_path.name, n_person_px,
        )

    return mask_paths


# ---------------------------------------------------------------------------
# Combined masking (SAM3 + nadir)
# ---------------------------------------------------------------------------


def apply_combined_masks(
    image_paths: list[Path],
    sam3_mask_dir: Path,
    nadir_mask: np.ndarray,
    output_dir: Path,
    *,
    inpaint: bool = False,
    inpaint_radius: int = 5,
) -> list[Path]:
    """Apply both SAM3 person masks and nadir mask to images.

    Parameters
    ----------
    image_paths:
        Original equirectangular images.
    sam3_mask_dir:
        Directory with SAM3 masks ({stem}_mask.png).
    nadir_mask:
        Nadir mask array (255=keep, 0=masked).
    output_dir:
        Output directory for masked images.
    inpaint:
        Use inpainting instead of black fill.
    inpaint_radius:
        Inpainting radius.

    Returns
    -------
    list[Path]
        Paths to output masked images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read %s, skipping", img_path)
            continue

        h, w = img.shape[:2]

        # Load SAM3 mask
        sam3_mask_path = sam3_mask_dir / f"{img_path.stem}_mask.png"
        if sam3_mask_path.exists():
            sam3_mask = cv2.imread(str(sam3_mask_path), cv2.IMREAD_GRAYSCALE)
            if sam3_mask.shape[:2] != (h, w):
                sam3_mask = cv2.resize(sam3_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            sam3_mask = np.full((h, w), 255, dtype=np.uint8)

        # Resize nadir mask if needed
        if nadir_mask.shape[:2] != (h, w):
            cur_nadir = cv2.resize(nadir_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            cur_nadir = nadir_mask

        # Combine: keep only where BOTH masks say keep
        combined = cv2.bitwise_and(sam3_mask, cur_nadir)

        if inpaint:
            inpaint_mask = cv2.bitwise_not(combined)
            out = cv2.inpaint(img, inpaint_mask, inpaint_radius, cv2.INPAINT_NS)
        else:
            out = cv2.bitwise_and(img, img, mask=combined)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        results.append(out_path)

    logger.info("Applied combined masks to %d images -> %s", len(results), output_dir)
    return results
