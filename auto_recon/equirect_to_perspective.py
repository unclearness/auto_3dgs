"""Convert equirectangular images to perspective (pinhole) projections.

Given equirectangular images and their Metashape camera transforms, this
module generates a set of perspective views (cubemap-like or configurable
pitch/yaw grid) for each camera position with corresponding COLMAP-format
camera parameters using the PINHOLE model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("auto_recon.equirect_to_perspective")

# ---------------------------------------------------------------------------
# Core: equirectangular to perspective reprojection
# ---------------------------------------------------------------------------


def _equirect_to_perspective(
    equirect: np.ndarray,
    fov_deg: float,
    out_size: tuple[int, int],
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    """Extract a perspective view from an equirectangular image.

    Parameters
    ----------
    equirect:
        Input equirectangular image (H, W, 3).
    fov_deg:
        Horizontal field of view in degrees.
    out_size:
        (width, height) of the output perspective image.
    yaw_deg:
        Horizontal rotation in degrees (0 = front, 90 = right).
    pitch_deg:
        Vertical rotation in degrees (0 = horizon, +90 = up).

    Returns
    -------
    np.ndarray
        Perspective image (out_h, out_w, 3).
    """
    out_w, out_h = out_size
    eq_h, eq_w = equirect.shape[:2]

    fov = np.radians(fov_deg)
    f = out_w / (2.0 * np.tan(fov / 2.0))

    cx = out_w / 2.0
    cy = out_h / 2.0

    # Pixel grid in perspective image
    u = np.arange(out_w, dtype=np.float64) - cx
    v = np.arange(out_h, dtype=np.float64) - cy
    uu, vv = np.meshgrid(u, v)

    # Direction vectors in camera frame (x-right, y-down, z-forward)
    x = uu
    y = vv
    z = np.full_like(uu, f)

    # Rotation matrices for yaw and pitch
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Rotation: first pitch (around x-axis), then yaw (around y-axis)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # Combined rotation: R_yaw * R_pitch
    # R_pitch rotates around x: y' = y*cos - z*sin, z' = y*sin + z*cos
    y1 = y * cos_p - z * sin_p
    z1 = y * sin_p + z * cos_p
    x1 = x

    # R_yaw rotates around y: x' = x*cos + z*sin, z' = -x*sin + z*cos
    x2 = x1 * cos_y + z1 * sin_y
    z2 = -x1 * sin_y + z1 * cos_y
    y2 = y1

    # Convert to spherical coordinates
    norm = np.sqrt(x2**2 + y2**2 + z2**2)
    theta = np.arctan2(x2, z2)  # longitude [-pi, pi]
    phi = np.arcsin(np.clip(y2 / norm, -1, 1))  # latitude [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    src_x = ((theta / np.pi + 1.0) / 2.0 * eq_w).astype(np.float32)
    src_y = ((phi / (np.pi / 2.0) + 1.0) / 2.0 * eq_h).astype(np.float32)

    # Wrap x
    src_x = src_x % eq_w

    # Remap
    perspective = cv2.remap(
        equirect, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return perspective


# ---------------------------------------------------------------------------
# Generate perspective views for all cameras
# ---------------------------------------------------------------------------


def convert_equirect_to_perspectives(
    images_dir: str | Path,
    colmap_sparse_dir: str | Path | None,
    output_dir: str | Path,
    fov_deg: float = 90.0,
    out_size: tuple[int, int] = (1024, 1024),
    pitch_angles: list[float] | None = None,
    yaw_step_deg: float = 90.0,
    masks_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Convert equirectangular images + COLMAP Spherical data to perspective views.

    For each equirectangular input image, generates multiple perspective views
    at different yaw/pitch angles. Outputs a new COLMAP dataset with PINHOLE
    camera model.

    Parameters
    ----------
    images_dir:
        Directory containing equirectangular images.
    colmap_sparse_dir:
        Directory with cameras.txt, images.txt, points3D.txt from Metashape.
    output_dir:
        Output directory for the perspective dataset.
    fov_deg:
        Field of view for each perspective view (degrees).
    out_size:
        (width, height) of output perspective images.
    pitch_angles:
        List of pitch angles in degrees. Default: [-30, 0, 30].
    yaw_step_deg:
        Step between yaw angles. Default: 90° (4 views per pitch level).

    Returns
    -------
    dict[str, Path]
        Keys: ``"sparse_dir"``, ``"images_dir"``, ``"point_cloud"``.
    """
    images_dir = Path(images_dir).resolve()
    output_dir = Path(output_dir).resolve()
    has_sparse = colmap_sparse_dir is not None
    if has_sparse:
        colmap_sparse_dir = Path(colmap_sparse_dir).resolve()
    has_masks = masks_dir is not None
    if has_masks:
        masks_dir = Path(masks_dir).resolve()

    if pitch_angles is None:
        pitch_angles = [-30.0, 0.0, 30.0]

    yaw_angles = list(np.arange(0, 360, yaw_step_deg))

    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_sparse_dir = output_dir / "sparse" / "0"
    out_sparse_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir: Path | None = None
    if has_masks:
        out_masks_dir = output_dir / "masks"
        out_masks_dir.mkdir(parents=True, exist_ok=True)

    out_w, out_h = out_size
    f = out_w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    cx = out_w / 2.0
    cy = out_h / 2.0

    # Parse original images.txt if available (Metashape path).
    # When colmap_sparse_dir is None (COLMAP/RealityScan path), we generate
    # perspective images without camera poses — SfM will be run afterward.
    if has_sparse:
        orig_images = _parse_images_txt(colmap_sparse_dir / "images.txt")
    else:
        # Build a fake image list from the directory (no poses)
        _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        orig_images = [
            {"name": p.name, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
             "tx": 0.0, "ty": 0.0, "tz": 0.0}
            for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in _IMAGE_EXTS
        ]

    logger.info(
        "Converting %d equirect images x %d views (yaw) x %d pitches = %d perspective images",
        len(orig_images), len(yaw_angles), len(pitch_angles),
        len(orig_images) * len(yaw_angles) * len(pitch_angles),
    )

    # --- cameras.txt (single PINHOLE camera) ---
    # Only write COLMAP sparse files when we have camera poses.
    cam_lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"1 PINHOLE {out_w} {out_h} {f:.6f} {f:.6f} {cx:.1f} {cy:.1f}",
    ]
    if has_sparse:
        (out_sparse_dir / "cameras.txt").write_text(
            "\n".join(cam_lines) + "\n", encoding="utf-8"
        )

    # --- Generate perspective images and images.txt ---
    img_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]

    new_img_id = 1
    for orig_img in orig_images:
        name = orig_img["name"]
        qw, qx, qy, qz = orig_img["qw"], orig_img["qx"], orig_img["qy"], orig_img["qz"]
        tx, ty, tz = orig_img["tx"], orig_img["ty"], orig_img["tz"]

        # Load equirectangular image
        eq_path = images_dir / name
        if not eq_path.exists():
            logger.warning("Image not found: %s, skipping", eq_path)
            continue
        equirect = cv2.imread(str(eq_path))
        if equirect is None:
            logger.warning("Cannot read: %s, skipping", eq_path)
            continue

        # Load equirectangular mask if available
        eq_mask = None
        if has_masks:
            mask_stem = Path(name).stem
            mask_path = masks_dir / f"{mask_stem}.png"
            if mask_path.exists():
                eq_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                logger.debug("No mask for %s", name)

        # Original camera rotation (world-to-camera)
        R_orig = _quat_to_rotation_matrix(qw, qx, qy, qz)
        t_orig = np.array([tx, ty, tz])

        stem = Path(name).stem

        for pitch in pitch_angles:
            for yaw in yaw_angles:
                # Generate perspective view
                persp = _equirect_to_perspective(
                    equirect, fov_deg, out_size, yaw, pitch
                )

                # Output filename
                out_name = f"{stem}_p{pitch:+.0f}_y{yaw:.0f}.jpg"
                cv2.imwrite(
                    str(out_images_dir / out_name), persp,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

                # Convert mask to perspective if available
                if eq_mask is not None and out_masks_dir is not None:
                    persp_mask = _equirect_to_perspective(
                        cv2.cvtColor(eq_mask, cv2.COLOR_GRAY2BGR),
                        fov_deg, out_size, yaw, pitch,
                    )
                    # Use nearest-neighbor threshold to keep mask binary
                    persp_mask_gray = cv2.cvtColor(persp_mask, cv2.COLOR_BGR2GRAY)
                    _, persp_mask_bin = cv2.threshold(persp_mask_gray, 127, 255, cv2.THRESH_BINARY)
                    mask_out_name = f"{stem}_p{pitch:+.0f}_y{yaw:.0f}.png"
                    cv2.imwrite(str(out_masks_dir / mask_out_name), persp_mask_bin)

                if has_sparse:
                    # Compute combined world-to-camera rotation for this sub-view.
                    R_extract = _yaw_pitch_to_extract_matrix(yaw, pitch)
                    R_sub = R_extract.T
                    R_combined = R_sub @ R_orig
                    t_combined = R_sub @ t_orig

                    qw_c, qx_c, qy_c, qz_c = _rotation_matrix_to_quaternion(R_combined)

                    img_lines.append(
                        f"{new_img_id} {qw_c:.10f} {qx_c:.10f} {qy_c:.10f} {qz_c:.10f} "
                        f"{t_combined[0]:.10f} {t_combined[1]:.10f} {t_combined[2]:.10f} "
                        f"1 {out_name}"
                    )
                    img_lines.append("")  # empty POINTS2D

                new_img_id += 1

    # Write COLMAP files only when we have camera poses (Metashape path)
    if has_sparse:
        (out_sparse_dir / "images.txt").write_text(
            "\n".join(img_lines) + "\n", encoding="utf-8"
        )

        orig_p3d = colmap_sparse_dir / "points3D.txt"
        if orig_p3d.exists():
            import shutil
            shutil.copy2(orig_p3d, out_sparse_dir / "points3D.txt")
        else:
            (out_sparse_dir / "points3D.txt").write_text(
                "# 3D point list (empty)\n", encoding="utf-8"
            )

    # Copy point cloud if it exists
    point_cloud_dst = output_dir / "point_cloud.ply"
    if has_sparse:
        point_cloud_src = colmap_sparse_dir.parent / "point_cloud.ply"
        if point_cloud_src.exists():
            import shutil
            shutil.copy2(point_cloud_src, point_cloud_dst)

    logger.info(
        "Generated %d perspective images in %s", new_img_id - 1, output_dir,
    )

    return {
        "sparse_dir": out_sparse_dir,
        "images_dir": out_images_dir,
        "point_cloud": point_cloud_dst if point_cloud_dst.exists() else None,
        "masks_dir": out_masks_dir,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_images_txt(path: Path) -> list[dict]:
    """Parse COLMAP images.txt, returning list of image entries."""
    images = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10:
                images.append({
                    "id": int(parts[0]),
                    "qw": float(parts[1]),
                    "qx": float(parts[2]),
                    "qy": float(parts[3]),
                    "qz": float(parts[4]),
                    "tx": float(parts[5]),
                    "ty": float(parts[6]),
                    "tz": float(parts[7]),
                    "camera_id": int(parts[8]),
                    "name": parts[9],
                })
                # Skip the POINTS2D line
                next(f, None)
    return images


def _quat_to_rotation_matrix(
    qw: float, qx: float, qy: float, qz: float,
) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _yaw_pitch_to_extract_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation that maps sub-view directions to equirect directions.

    Matches the order used in ``_equirect_to_perspective``:
    first pitch (around X), then yaw (around Y).
    Result: R_yaw @ R_pitch.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)

    R_yaw = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y],
    ])

    R_pitch = np.array([
        [1, 0, 0],
        [0, cos_p, -sin_p],
        [0, sin_p, cos_p],
    ])

    return R_yaw @ R_pitch


def _rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return (float(w), float(x), float(y), float(z))
