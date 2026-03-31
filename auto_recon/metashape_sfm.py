"""Metashape SfM integration module.

Uses the Metashape Python module (``import Metashape``) directly to perform
Structure from Motion on 360° equirectangular images.  Outputs a
COLMAP-compatible sparse reconstruction consumed by LichtFeld Studio.

Exports are done by reading Metashape's in-memory data structures directly
(camera transforms, sensor calibration, tie points) rather than using
``chunk.exportCameras()`` / ``chunk.exportPointCloud()`` which require a
license.
"""

from __future__ import annotations

import datetime
import logging
import shutil
import struct
from pathlib import Path

import Metashape
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_JST = datetime.timezone(datetime.timedelta(hours=9))

logger = logging.getLogger("auto_recon.metashape_sfm")

# ---------------------------------------------------------------------------
# Image extensions
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS: set[str] = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".exr",
}

# ---------------------------------------------------------------------------
# Rotation matrix -> quaternion
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Export COLMAP text from chunk (no license needed)
# ---------------------------------------------------------------------------


def _export_colmap_text(chunk: Metashape.Chunk, sparse_dir: Path) -> None:
    """Write cameras.txt, images.txt, points3D.txt from chunk data."""
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # --- cameras.txt ---
    cam_lines: list[str] = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    sensor_to_id: dict[int, int] = {}
    for idx, sensor in enumerate(chunk.sensors, start=1):
        sensor_to_id[sensor.key] = idx
        w = sensor.width
        h = sensor.height
        calib = sensor.calibration
        if sensor.type == Metashape.Sensor.Type.Spherical:
            f = calib.f if calib else w / (2.0 * np.pi)
            cx = (calib.cx if calib else 0.0) + w / 2.0
            cy = (calib.cy if calib else 0.0) + h / 2.0
            cam_lines.append(
                f"{idx} SIMPLE_RADIAL {w} {h} {f:.6f} {cx:.1f} {cy:.1f} 0.0"
            )
        else:
            f = calib.f if calib else float(max(w, h))
            cx = (calib.cx if calib else 0.0) + w / 2.0
            cy = (calib.cy if calib else 0.0) + h / 2.0
            cam_lines.append(
                f"{idx} SIMPLE_RADIAL {w} {h} {f:.6f} {cx:.1f} {cy:.1f} 0.0"
            )

    (sparse_dir / "cameras.txt").write_text(
        "\n".join(cam_lines) + "\n", encoding="utf-8"
    )

    # --- images.txt ---
    img_lines: list[str] = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]

    # Build point track index: track_key -> point3D_id (1-based)
    tie_points = chunk.tie_points
    point3d_id_map: dict[int, int] = {}  # track index -> point3D_id
    if tie_points:
        p3d_id = 1
        for i in range(len(tie_points.points)):
            pt = tie_points.points[i]
            if pt.valid:
                point3d_id_map[i] = p3d_id
                p3d_id += 1

    img_id = 1
    for camera in chunk.cameras:
        if camera.transform is None:
            continue

        cam_id = sensor_to_id.get(camera.sensor.key, 1)

        # camera.transform is camera-to-chunk; invert for world-to-camera
        T_inv = camera.transform.inv()
        R = np.array([
            [T_inv[0, 0], T_inv[0, 1], T_inv[0, 2]],
            [T_inv[1, 0], T_inv[1, 1], T_inv[1, 2]],
            [T_inv[2, 0], T_inv[2, 1], T_inv[2, 2]],
        ])
        t = np.array([T_inv[0, 3], T_inv[1, 3], T_inv[2, 3]])
        qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)

        # Use photo filename (with extension) rather than label (which may lack it)
        if camera.photo and camera.photo.path:
            img_name = Path(camera.photo.path).name
        else:
            img_name = camera.label

        img_lines.append(
            f"{img_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
            f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {cam_id} {img_name}"
        )

        # POINTS2D line: projections for this camera
        points2d_parts: list[str] = []
        if tie_points:
            projections = tie_points.projections
            projs = projections[camera]
            tracks = tie_points.tracks
            for proj_idx in range(len(projs)):
                proj = projs[proj_idx]
                track = tracks[proj.track_id] if hasattr(proj, 'track_id') else None
                # Map track index to point3D id
                track_idx = proj.track_id if hasattr(proj, 'track_id') else proj_idx
                p3d_id = point3d_id_map.get(track_idx, -1)
                points2d_parts.append(
                    f"{proj.coord.x:.2f} {proj.coord.y:.2f} {p3d_id}"
                )
        img_lines.append(" ".join(points2d_parts))
        img_id += 1

    (sparse_dir / "images.txt").write_text(
        "\n".join(img_lines) + "\n", encoding="utf-8"
    )

    # --- points3D.txt ---
    p3d_lines: list[str] = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    if tie_points:
        for i in range(len(tie_points.points)):
            pt = tie_points.points[i]
            if not pt.valid:
                continue
            p3d_id = point3d_id_map[i]
            coord = pt.coord
            # Default color (white) and error
            p3d_lines.append(
                f"{p3d_id} {coord.x:.6f} {coord.y:.6f} {coord.z:.6f} "
                f"200 200 200 0.0"
            )

    (sparse_dir / "points3D.txt").write_text(
        "\n".join(p3d_lines) + "\n", encoding="utf-8"
    )

    logger.info(
        "Exported COLMAP text: %d cameras, %d images, %d points3D to %s",
        len(chunk.sensors), img_id - 1, len(point3d_id_map), sparse_dir,
    )


# ---------------------------------------------------------------------------
# Export sparse point cloud as PLY (no license needed)
# ---------------------------------------------------------------------------


def _export_point_cloud_ply(chunk: Metashape.Chunk, ply_path: Path) -> None:
    """Write tie points as a PLY file."""
    tie_points = chunk.tie_points
    if not tie_points:
        logger.warning("No tie points to export")
        return

    valid_points: list[tuple[float, float, float]] = []
    for i in range(len(tie_points.points)):
        pt = tie_points.points[i]
        if pt.valid:
            valid_points.append((pt.coord.x, pt.coord.y, pt.coord.z))

    if not valid_points:
        logger.warning("No valid tie points to export")
        return

    ply_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(valid_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in valid_points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    logger.info("Exported %d points to %s", len(valid_points), ply_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_metashape_sfm(
    image_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Run Metashape SfM on equirectangular images using the Python API.

    Performs matching and alignment, then exports camera parameters and
    tie points directly from memory (no license required for export).

    Parameters
    ----------
    image_dir:
        Directory containing equirectangular images.
    output_dir:
        Directory where COLMAP sparse reconstruction will be written.

    Returns
    -------
    dict[str, Path]
        Keys: ``"sparse_dir"``, ``"images_dir"``, ``"point_cloud"``.
    """
    image_dir = Path(image_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    photos = sorted(
        str(p) for p in image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not photos:
        raise FileNotFoundError(f"No images found in {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Symlink/copy input images
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        dst = images_out / img_path.name
        if not dst.exists():
            try:
                dst.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst)

    now = datetime.datetime.now(tz=_JST).strftime("%Y%m%d_%H%M%S")
    logger.info("[%s] Starting Metashape SfM (%d images)", now, len(photos))

    # Create document and chunk (no save needed)
    doc = Metashape.Document()
    chunk = doc.addChunk()
    chunk.addPhotos(photos)
    logger.info("Added %d cameras", len(chunk.cameras))

    # Set sensor type to Spherical
    for sensor in chunk.sensors:
        sensor.type = Metashape.Sensor.Type.Spherical
    logger.info("Sensor type set to Spherical")

    # Match and align
    chunk.matchPhotos(
        downscale=1,
        generic_preselection=True,
        reference_preselection=False,
    )
    logger.info("Photo matching complete")

    chunk.alignCameras()
    aligned = sum(1 for c in chunk.cameras if c.transform is not None)
    logger.info("Aligned: %d / %d cameras", aligned, len(chunk.cameras))

    if aligned == 0:
        raise RuntimeError(
            "No cameras aligned. Images may lack sufficient overlap."
        )

    # Export COLMAP text (direct memory access, no license)
    _export_colmap_text(chunk, sparse_dir)

    # Export point cloud PLY (direct memory access, no license)
    point_cloud_path = output_dir / "point_cloud.ply"
    _export_point_cloud_ply(chunk, point_cloud_path)

    now = datetime.datetime.now(tz=_JST).strftime("%Y%m%d_%H%M%S")
    logger.info("[%s] Metashape SfM complete", now)

    return {
        "sparse_dir": sparse_dir,
        "images_dir": images_out,
        "point_cloud": point_cloud_path,
    }
