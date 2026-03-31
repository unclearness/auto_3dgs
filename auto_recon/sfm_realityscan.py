"""RealityScan SfM backend with panorama rig support.

Converts equirectangular images to virtual pinhole views (same as the COLMAP
backend), flattens them into a single directory with unique names, then runs
RealityScan CLI.  Exports COLMAP-format registration for downstream 3DGS.

RealityScan strips subdirectory prefixes from image names in its COLMAP
export, so images are flattened before import (unlike the COLMAP backend
which uses per-camera subdirectories natively).

References:
    https://qiita.com/Tks_Yoshinaga/items/896713e9c637cbfe35d1#34-realityscan
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from auto_recon.pano_render import (
    PanoRenderOptions,
    DEFAULT_RENDER_OPTIONS,
    render_perspective_images,
)
from auto_recon.sfm_backend import SfMBackend, SfMResult

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Default install location on Windows.
_DEFAULT_RS_EXE = r"C:\Program Files\Epic Games\RealityScan_2.1\RealityScan.exe"

# Resolve the config directory shipped alongside the reference pipeline.
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_DIR = _SCRIPT_DIR / "realityscan_pipeline" / "config"


def _to_rs_path(p: str | Path) -> str:
    """Convert *p* to a path string that RealityScan understands."""
    if sys.platform == "win32":
        return str(Path(p))
    cp = subprocess.run(
        ["winepath", "-w", str(Path(p).resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip()


# ---------------------------------------------------------------------------
# Image flattening
# ---------------------------------------------------------------------------


def _flatten_images(src_dir: Path, dst_dir: Path) -> int:
    """Flatten images from per-camera subdirectories into a single directory.

    Renames ``pano_camera0/frame_000001.jpg`` to ``cam00_frame_000001.jpg``
    etc. to avoid name collisions.  Returns the number of images copied.

    If *src_dir* contains no subdirectories (already flat), copies as-is.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for p in sorted(src_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTS:
            continue
        rel = p.relative_to(src_dir)
        if len(rel.parts) > 1:
            # Subdirectory structure: build a flat unique name.
            # e.g. "pano_camera0/frame_000001.jpg" -> "cam00_frame_000001.jpg"
            subdir = rel.parts[0]
            # Extract numeric camera index from "pano_cameraNN" prefix.
            cam_idx = subdir.replace("pano_camera", "")
            try:
                cam_idx_int = int(cam_idx)
                flat_name = f"cam{cam_idx_int:02d}_{rel.parts[-1]}"
            except ValueError:
                flat_name = f"{subdir}_{rel.parts[-1]}"
        else:
            flat_name = p.name

        shutil.copy2(str(p), str(dst_dir / flat_name))
        count += 1

    return count


# ---------------------------------------------------------------------------
# Output reorganization
# ---------------------------------------------------------------------------


def _reorganize_output(output_dir: Path, image_dir: Path) -> dict[str, Path]:
    """Reorganize RealityScan exports into the standard COLMAP layout.

    Expected final structure::

        output_dir/
          sparse/0/
            cameras.txt
            images.txt
            points3D.txt
          images/   (flat directory with unique image names)
          point_cloud.ply
    """
    output_dir = Path(output_dir).resolve()
    image_dir = Path(image_dir).resolve()

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # --- Move COLMAP text files ---
    colmap_src = output_dir / "colmap_undistorted"

    for fname in ("cameras.txt", "images.txt", "points3D.txt"):
        src = colmap_src / fname
        dst = sparse_dir / fname
        if src.is_file():
            shutil.move(str(src), str(dst))
            logger.debug("Moved %s -> %s", src, dst)
        else:
            logger.warning("Expected COLMAP file not found: %s", src)

    # Create empty points3D.txt if missing
    points3d = sparse_dir / "points3D.txt"
    if not points3d.exists():
        points3d.write_text(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0\n"
        )

    # --- Point cloud ---
    sparse_ply = output_dir / "sparse.ply"
    point_cloud = output_dir / "point_cloud.ply"
    if sparse_ply.is_file() and not point_cloud.exists():
        shutil.move(str(sparse_ply), str(point_cloud))

    # --- Images directory ---
    # image_dir is already output_dir / "images" (created by the backend
    # before running RealityScan), so a symlink/junction is only needed
    # when it lives elsewhere.
    images_link = output_dir / "images"
    if not images_link.exists():
        try:
            images_link.symlink_to(image_dir, target_is_directory=True)
        except OSError:
            if sys.platform == "win32":
                try:
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J",
                         str(images_link), str(image_dir)],
                        check=True, capture_output=True,
                    )
                except subprocess.CalledProcessError:
                    shutil.copytree(str(image_dir), str(images_link))
            else:
                shutil.copytree(str(image_dir), str(images_link))

    # Cleanup
    if colmap_src.is_dir() and not any(colmap_src.iterdir()):
        colmap_src.rmdir()

    return {
        "sparse_dir": sparse_dir,
        "images_dir": images_link,
        "point_cloud": point_cloud,
    }


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class RealityScanSfMBackend(SfMBackend):
    """SfM backend using RealityScan CLI with panorama rig support.

    Converts equirectangular images to virtual pinhole views, flattens
    them into a single directory with unique names (to work around
    RealityScan stripping subdirectory prefixes in COLMAP exports), then
    runs SfM and exports COLMAP registration.

    Parameters
    ----------
    realityscan_path:
        Path to ``RealityScan.exe``.
    headless:
        Run RealityScan in headless mode.
    config_dir:
        Directory containing the XML parameter files for export.
    render_options:
        Panorama render options. Default: 4 yaw steps x 3 pitches.
    """

    def __init__(
        self,
        realityscan_path: str | Path | None = None,
        headless: bool = True,
        config_dir: str | Path | None = None,
        render_options: PanoRenderOptions | None = None,
    ) -> None:
        self._rs_exe = str(realityscan_path or _DEFAULT_RS_EXE)
        self._headless = headless
        self._config_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR
        self._render_options = render_options or DEFAULT_RENDER_OPTIONS

    @property
    def supports_equirectangular(self) -> bool:
        return True

    def run(
        self, image_dir: str | Path, output_dir: str | Path, **kwargs
    ) -> SfMResult:
        """Run RealityScan SfM on equirectangular images.

        Pipeline:
        1. Render equirect -> virtual pinhole views (per-camera subdirectories)
        2. Flatten into a single directory with unique names
        3. RealityScan: import, align, export COLMAP registration + sparse PLY
        4. Reorganize to standard COLMAP layout

        Returns SfMResult with is_pinhole=True.
        """
        image_dir = Path(image_dir).resolve()
        output_dir = Path(output_dir).resolve()

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        rs_exe = Path(self._rs_exe)
        if not rs_exe.is_file():
            raise FileNotFoundError(f"RealityScan executable not found: {rs_exe}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Temporary directory for per-camera subdirectory images
        persp_subdir = output_dir / "_persp_raw"
        persp_subdir.mkdir(exist_ok=True, parents=True)

        # Final flat images directory
        persp_image_dir = output_dir / "images"
        persp_image_dir.mkdir(exist_ok=True, parents=True)

        # Step 1: Render perspective images into per-camera subdirectories
        logger.info("Step 1: Rendering perspective images from panoramas")
        _rig_cameras, _virtual_cam = render_perspective_images(
            image_dir, persp_subdir, self._render_options, mask_dir=None,
        )

        # Step 2: Flatten into a single directory with unique names
        logger.info("Step 2: Flattening perspective images")
        n_flat = _flatten_images(persp_subdir, persp_image_dir)
        logger.info("Flattened %d images into %s", n_flat, persp_image_dir)

        # Clean up the subdirectory version
        shutil.rmtree(str(persp_subdir), ignore_errors=True)

        # Step 3: Build and run RealityScan command
        logger.info("Step 3: Running RealityScan SfM")
        cmd = self._build_command(persp_image_dir, output_dir)
        logger.info("Command: %s", " ".join(shlex.quote(c) for c in cmd))

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"RealityScan exited with code {result.returncode}."
            )

        # Step 4: Reorganize output
        logger.info("Step 4: Reorganizing output to COLMAP format")
        paths = _reorganize_output(output_dir, persp_image_dir)

        logger.info("RealityScan panorama SfM complete: %s", output_dir)

        return SfMResult(
            sparse_dir=paths["sparse_dir"],
            images_dir=paths["images_dir"],
            point_cloud=paths["point_cloud"],
            is_pinhole=True,
        )

    def _build_command(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> list[str]:
        """Build the RealityScan CLI command.

        Images are already flattened into a single directory, so we do NOT
        need ``-set appIncSubdirs=true`` or ``-setConstantCalibrationGroups``.
        """
        config_dir = self._config_dir.resolve()
        sparse_params = config_dir / "sparse_point_cloud.xml"
        registration_params = config_dir / "colmap_undistorted.xml"

        if not sparse_params.is_file():
            raise FileNotFoundError(f"Sparse point cloud params XML not found: {sparse_params}")
        if not registration_params.is_file():
            raise FileNotFoundError(f"COLMAP registration params XML not found: {registration_params}")

        sparse_ply = output_dir / "sparse.ply"
        colmap_dir = output_dir / "colmap_undistorted"
        colmap_txt = colmap_dir / "colmap.txt"
        project_path = output_dir / "project.rsproj"

        colmap_dir.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [self._rs_exe, "-stdConsole"]
        if self._headless:
            cmd += ["-headless"]

        # Scene setup — flat directory, no subdirectory handling needed
        cmd += ["-newScene"]
        cmd += ["-addFolder", _to_rs_path(image_dir)]
        cmd += ["-selectAllImages"]

        # Camera model: NONE (no distortion — already undistorted pinhole)
        cmd += ["-editInputSelection", "inpDistortionModel=0"]

        # SfM alignment
        cmd += ["-align"]
        cmd += ["-selectMaximalComponent"]

        # Export sparse point cloud
        cmd += [
            "-exportSparsePointCloud",
            _to_rs_path(sparse_ply),
            _to_rs_path(sparse_params),
        ]

        # Export COLMAP registration (undistorted)
        cmd += [
            "-exportRegistration",
            _to_rs_path(colmap_txt),
            _to_rs_path(registration_params),
        ]

        # Save and quit
        cmd += ["-save", _to_rs_path(project_path)]
        cmd += ["-quit"]

        return cmd


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run RealityScan SfM on equirectangular images",
    )
    parser.add_argument("image_dir", type=Path, help="Input panorama directory")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--rc", default=_DEFAULT_RS_EXE,
        help=f"Path to RealityScan.exe (default: {_DEFAULT_RS_EXE})",
    )
    parser.add_argument(
        "--config-dir", default=str(_DEFAULT_CONFIG_DIR),
        help=f"XML config directory (default: {_DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

    backend = RealityScanSfMBackend(
        realityscan_path=args.rc,
        headless=not args.no_headless,
        config_dir=args.config_dir,
    )
    result = backend.run(args.image_dir, args.output_dir)
    print(f"Sparse dir : {result.sparse_dir}")
    print(f"Images dir : {result.images_dir}")
    print(f"Point cloud: {result.point_cloud}")
