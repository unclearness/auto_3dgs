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
_trt_predictor = None

# Default TRT engine paths (relative to project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRT_BACKBONE = _PROJECT_ROOT / "hf_backbone_fp16.engine"
_DEFAULT_TRT_ENC_DEC = _PROJECT_ROOT / "enc_dec_fp16.engine"


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


def _get_trt_predictor(
    confidence: float = 0.5,
    trt_backbone: str | Path | None = None,
    trt_enc_dec: str | Path | None = None,
):
    """Lazy-load DART TRT predictor for fast bbox detection."""
    global _trt_predictor
    if _trt_predictor is None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

        bb = str(trt_backbone or _DEFAULT_TRT_BACKBONE)
        ed = str(trt_enc_dec or _DEFAULT_TRT_ENC_DEC)
        if not Path(bb).exists():
            raise FileNotFoundError(f"TRT backbone engine not found: {bb}")
        if not Path(ed).exists():
            raise FileNotFoundError(f"TRT enc-dec engine not found: {ed}")

        logger.info("Loading DART TRT predictor (backbone=%s, enc_dec=%s)", bb, ed)
        model = build_sam3_image_model(bpe_path=str(_SAM3_BPE_PATH))
        _trt_predictor = Sam3MultiClassPredictorFast(
            model, device="cuda", use_fp16=True,
            shared_encoder=True, generic_prompt="person",
            detection_only=True,
            trt_engine_path=bb,
            trt_enc_dec_engine_path=ed,
        )
        _trt_predictor.set_classes(["person"])
        _trt_predictor._confidence = confidence
        logger.info("DART TRT predictor loaded")
    return _trt_predictor


# ---------------------------------------------------------------------------
# Core: run SAM3 on a single image
# ---------------------------------------------------------------------------


def _detect_persons_trt(
    image: np.ndarray,
    predictor,
    confidence: float = 0.5,
    bbox_expand: float = 0.1,
) -> np.ndarray:
    """Detect persons using TRT and return a rectangular bbox mask.

    Returns binary mask (H, W) uint8 -- 255 inside expanded bboxes.
    """
    h, w = image.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    state = predictor.set_image(pil_image)
    results = predictor.predict(state, confidence_threshold=confidence)

    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = results["boxes"]
    if boxes is not None and len(boxes) > 0:
        boxes_np = boxes.cpu().numpy()
        for x1, y1, x2, y2 in boxes_np:
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - bw * bbox_expand))
            y1 = max(0, int(y1 - bh * bbox_expand))
            x2 = min(w, int(x2 + bw * bbox_expand))
            y2 = min(h, int(y2 + bh * bbox_expand))
            mask[y1:y2, x1:x2] = 255
    return mask


def _segment_persons(
    image: np.ndarray,
    processor,
    prompt: str = "person",
    scale: float = 1.0,
) -> np.ndarray:
    """Run SAM3 text-prompted segmentation on a single image.

    Parameters
    ----------
    scale:
        Downscale factor for SAM3 input (0.5 = half resolution).
        The output mask is upscaled back to the original size.

    Returns binary mask (H, W) uint8 -- 255 where person detected.
    """
    h, w = image.shape[:2]
    if scale < 1.0:
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = image

    pil_image = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(pil_image)
        output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"]  # (N, 1, H, W) bool tensor
    if masks.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    combined = masks.any(dim=0)[0].cpu().numpy().astype(np.uint8) * 255

    # Upscale mask back to original size if downscaled
    if scale < 1.0:
        combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
    return combined


def _segment_persons_batch(
    images: list[np.ndarray],
    processor,
    prompt: str = "person",
    scale: float = 1.0,
) -> list[np.ndarray]:
    """Run SAM3 on a batch of same-size images.

    Backbone feature extraction is batched for efficiency.
    Grounding (text prompt -> mask) is run per-image because the SAM3
    grounding head expects single-image state.

    Returns list of binary masks (H, W) uint8.
    """
    if not images:
        return []

    orig_sizes = [(img.shape[0], img.shape[1]) for img in images]
    if scale < 1.0:
        images = [cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA) for img in images]

    pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    sizes = [(img.shape[0], img.shape[1]) for img in images]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # Batch backbone forward pass
        batch_state = processor.set_image_batch(pil_images)
        backbone_out = batch_state["backbone_out"]

    results: list[np.ndarray] = []
    for i, (h, w) in enumerate(sizes):
        # Extract per-image backbone features
        per_image_state = {
            "original_height": h,
            "original_width": w,
            "backbone_out": {},
        }
        for key, val in backbone_out.items():
            if isinstance(val, torch.Tensor) and val.shape[0] == len(images):
                per_image_state["backbone_out"][key] = val[i : i + 1]
            elif isinstance(val, list):
                per_image_state["backbone_out"][key] = [
                    v[i : i + 1] if isinstance(v, torch.Tensor) and v.shape[0] == len(images) else v
                    for v in val
                ]
            elif isinstance(val, dict):
                per_image_state["backbone_out"][key] = {
                    k: v[i : i + 1] if isinstance(v, torch.Tensor) and v.shape[0] == len(images) else
                    [vv[i : i + 1] if isinstance(vv, torch.Tensor) and vv.shape[0] == len(images) else vv for vv in v] if isinstance(v, list) else v
                    for k, v in val.items()
                }
            else:
                per_image_state["backbone_out"][key] = val

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = processor.set_text_prompt(state=per_image_state, prompt=prompt)

        masks = output["masks"]
        oh, ow = orig_sizes[i]
        if masks.shape[0] == 0:
            results.append(np.zeros((oh, ow), dtype=np.uint8))
        else:
            combined = masks.any(dim=0)[0].cpu().numpy().astype(np.uint8) * 255
            if scale < 1.0:
                combined = cv2.resize(combined, (ow, oh), interpolation=cv2.INTER_NEAREST)
            results.append(combined)

    return results


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
    scale: float = 1.0,
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

        person_mask = _segment_persons(img, processor, prompt, scale=scale)

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
# Mode 1b: Equirectangular direct (TRT bbox)
# ---------------------------------------------------------------------------


def mask_persons_equirect_trt(
    image_paths: list[Path],
    output_dir: Path,
    *,
    confidence: float = 0.5,
    bbox_expand: float = 0.1,
    dilate_px: int = 15,
    trt_backbone: str | Path | None = None,
    trt_enc_dec: str | Path | None = None,
) -> list[Path]:
    """Detect persons using TRT and produce rectangular bbox masks.

    ~2.4x faster than SAM3 PyTorch with Recall ~0.98 against pixel masks.
    Requires pre-built TRT engines (see scripts/build_trt_engines.py).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = _get_trt_predictor(confidence, trt_backbone, trt_enc_dec)

    mask_paths: list[Path] = []
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read %s, skipping", img_path)
            continue

        person_mask = _detect_persons_trt(img, predictor, confidence, bbox_expand)

        if dilate_px > 0 and person_mask.any():
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            person_mask = cv2.dilate(person_mask, kernel)

        keep_mask = cv2.bitwise_not(person_mask)
        out_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), keep_mask)
        mask_paths.append(out_path)

        if i % 10 == 0 or i == len(image_paths) - 1:
            n_person_px = int((person_mask > 0).sum())
            logger.info(
                "[trt-equirect] %d/%d %s - %d person pixels",
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
    batch_size: int = 4,
    scale: float = 1.0,
) -> list[Path]:
    """Run SAM3 on perspective views extracted from equirectangular images.

    Performance notes
    -----------------
    - **Backbone batching** (controlled by ``batch_size``): batches the ViT
      backbone forward pass across multiple perspective views.  batch_size=12
      (all views at once) gives ~12% speedup over batch_size=1.  The gain is
      modest because the grounding head (text prompt -> mask decoding) runs
      per-image regardless of batch size and dominates total inference time.
    - **GPU multiprocessing** is not recommended: SAM3 loads ~2 GB of VRAM,
      and concurrent GPU processes would cause VRAM contention or OOM.
    - **CPU parallelism** for perspective extraction (cv2.remap) and mask
      back-projection (_persp_mask_to_equirect) could help but is not yet
      implemented.  These are numpy/cv2 operations that release the GIL.

    For each equirect image, extracts a grid of perspective views, runs SAM3
    on batches of views, then projects the masks back to equirect space.

    Parameters
    ----------
    batch_size:
        Number of perspective views to process in a single SAM3 batch.
        Higher values use more VRAM but are faster.  Default: 4.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = _get_processor(confidence)

    if pitch_angles is None:
        pitch_angles = [-30.0, 0.0, 30.0]
    yaw_angles = list(np.arange(0, 360, yaw_step_deg))
    out_w, out_h = out_size

    # Pre-compute view grid
    view_grid = [(pitch, yaw) for pitch in pitch_angles for yaw in yaw_angles]
    n_views = len(view_grid)
    logger.info(
        "Pinhole SAM3: %d images x %d views, batch_size=%d",
        len(image_paths), n_views, batch_size,
    )

    mask_paths: list[Path] = []
    for i, img_path in enumerate(image_paths):
        equirect = cv2.imread(str(img_path))
        if equirect is None:
            logger.warning("Cannot read %s, skipping", img_path)
            continue

        eq_h, eq_w = equirect.shape[:2]
        combined_person_mask = np.zeros((eq_h, eq_w), dtype=np.uint8)

        # Extract all perspective views
        persp_images: list[np.ndarray] = []
        for pitch, yaw in view_grid:
            map_x, map_y = _equirect_to_persp_map(
                eq_w, eq_h, fov_deg, out_w, out_h, yaw, pitch,
            )
            persp = cv2.remap(
                equirect, map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
            )
            persp_images.append(persp)

        # Process in batches
        persp_masks: list[np.ndarray] = []
        for b_start in range(0, n_views, batch_size):
            batch = persp_images[b_start : b_start + batch_size]
            batch_masks = _segment_persons_batch(batch, processor, prompt, scale=scale)
            persp_masks.extend(batch_masks)

        # Project masks back to equirect
        for idx, (pitch, yaw) in enumerate(view_grid):
            persp_mask = persp_masks[idx]
            if persp_mask.any():
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

        keep_mask = cv2.bitwise_not(combined_person_mask)

        out_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), keep_mask)
        mask_paths.append(out_path)

        n_person_px = int((combined_person_mask > 0).sum())
        logger.info(
            "[pinhole] %d/%d %s - %d person pixels",
            i + 1, len(image_paths), img_path.name, n_person_px,
        )

    return mask_paths


# ---------------------------------------------------------------------------
# Mode 2b: Pinhole TRT (perspective views + TRT bbox detection)
# ---------------------------------------------------------------------------


def mask_persons_pinhole_trt(
    image_paths: list[Path],
    output_dir: Path,
    *,
    confidence: float = 0.5,
    fov_deg: float = 90.0,
    out_size: tuple[int, int] = (1024, 1024),
    pitch_angles: list[float] | None = None,
    yaw_step_deg: float = 90.0,
    bbox_expand: float = 0.1,
    dilate_px: int = 15,
    trt_backbone: str | Path | None = None,
    trt_enc_dec: str | Path | None = None,
) -> list[Path]:
    """Detect persons in perspective views using TRT bbox detection.

    Like mask_persons_pinhole but uses TRT for ~2x faster inference.
    Produces rectangular bbox masks instead of pixel-precise masks.
    Recall vs SAM3 pixel masks is ~0.98.

    Requires pre-built TRT engines (see scripts/build_trt_engines.py).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = _get_trt_predictor(confidence, trt_backbone, trt_enc_dec)

    if pitch_angles is None:
        pitch_angles = [-30.0, 0.0, 30.0]
    yaw_angles = list(np.arange(0, 360, yaw_step_deg))
    out_w, out_h = out_size

    view_grid = [(pitch, yaw) for pitch in pitch_angles for yaw in yaw_angles]
    n_views = len(view_grid)
    logger.info(
        "Pinhole TRT: %d images x %d views",
        len(image_paths), n_views,
    )

    mask_paths: list[Path] = []
    for i, img_path in enumerate(image_paths):
        equirect = cv2.imread(str(img_path))
        if equirect is None:
            logger.warning("Cannot read %s, skipping", img_path)
            continue

        eq_h, eq_w = equirect.shape[:2]
        combined_person_mask = np.zeros((eq_h, eq_w), dtype=np.uint8)

        for pitch, yaw in view_grid:
            map_x, map_y = _equirect_to_persp_map(
                eq_w, eq_h, fov_deg, out_w, out_h, yaw, pitch,
            )
            persp = cv2.remap(
                equirect, map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
            )

            persp_mask = _detect_persons_trt(persp, predictor, confidence, bbox_expand)

            if persp_mask.any():
                eq_mask = _persp_mask_to_equirect(
                    persp_mask, eq_w, eq_h, fov_deg, yaw, pitch,
                )
                combined_person_mask = np.maximum(combined_person_mask, eq_mask)

        if dilate_px > 0 and combined_person_mask.any():
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            combined_person_mask = cv2.dilate(combined_person_mask, kernel)

        keep_mask = cv2.bitwise_not(combined_person_mask)
        out_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), keep_mask)
        mask_paths.append(out_path)

        n_person_px = int((combined_person_mask > 0).sum())
        logger.info(
            "[trt-pinhole] %d/%d %s - %d person pixels",
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
