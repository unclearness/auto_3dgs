"""RealityScan SfM backend.

Wraps the RealityScan CLI behind the
:class:`~auto_recon.sfm_backend.SfMBackend` interface so it can be used
interchangeably with other SfM engines (COLMAP, Metashape, etc.).

RealityScan does **not** support equirectangular images.  The pipeline must
convert equirectangular frames to pinhole perspectives *before* invoking
this backend.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from auto_recon.sfm_backend import SfMBackend, SfMResult

logger = logging.getLogger(__name__)

# Distortion model codes used by RealityScan CLI.
CAMERA_MODEL_MAP: dict[str, int] = {
    "NONE": 0,
    "DIVISION": 1,
    "BROWN3": 2,
    "BROWN4": 3,
    "BROWN3_TAN2": 4,
    "BROWN4_TAN2": 5,
}

# Default install location on Windows.
_DEFAULT_RS_EXE = r"C:\Program Files\Epic Games\RealityScan_2.1\RealityScan.exe"

# Resolve the config directory shipped alongside the reference pipeline.
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_DIR = _SCRIPT_DIR / "realityscan_pipeline" / "config"


def _to_rs_path(p: str | Path) -> str:
    """Convert *p* to a path string that RealityScan understands.

    On Windows the native path is used directly; on Linux a Wine path
    conversion is applied.
    """
    if sys.platform == "win32":
        return str(Path(p))
    cp = subprocess.run(
        ["winepath", "-w", str(Path(p).resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip()


def _build_command(
    image_dir: Path,
    output_dir: Path,
    *,
    realityscan_path: str | Path = _DEFAULT_RS_EXE,
    headless: bool = True,
    config_dir: str | Path = _DEFAULT_CONFIG_DIR,
    camera_model: str = "NONE",
) -> list[str]:
    """Build the single RealityScan CLI command that performs SfM.

    The command chain:
      1. ``-newScene``
      2. ``-addFolder <image_dir>``
      3. ``-selectAllImages``
      4. Camera model (NONE -- undistorted pinhole)
      5. ``-align`` (SfM)
      6. ``-selectMaximalComponent``
      7. ``-exportSparsePointCloud``
      8. ``-exportRegistration`` (COLMAP undistorted)
      9. ``-save <project.rsproj>``
     10. ``-quit``
    """
    config_dir = Path(config_dir).resolve()
    image_dir = Path(image_dir).resolve()
    output_dir = Path(output_dir).resolve()

    sparse_params = config_dir / "sparse_point_cloud.xml"
    registration_params = config_dir / "colmap_undistorted.xml"

    if not sparse_params.is_file():
        raise FileNotFoundError(
            f"Sparse point cloud params XML not found: {sparse_params}"
        )
    if not registration_params.is_file():
        raise FileNotFoundError(
            f"COLMAP registration params XML not found: {registration_params}"
        )

    # Output paths
    sparse_ply = output_dir / "sparse.ply"
    colmap_dir = output_dir / "colmap_undistorted"
    colmap_txt = colmap_dir / "colmap.txt"
    project_path = output_dir / "project.rsproj"

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [str(Path(realityscan_path))]

    # Console / headless flags
    cmd += ["-stdConsole"]
    if headless:
        cmd += ["-headless"]

    # Scene setup
    cmd += ["-newScene"]
    cmd += ["-addFolder", _to_rs_path(image_dir)]
    cmd += ["-selectAllImages"]

    # Camera model -- NONE (0) means no distortion (already undistorted pinhole)
    model_val = CAMERA_MODEL_MAP[camera_model.upper()]
    cmd += ["-editInputSelection", f"inpDistortionModel={model_val}"]

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

    # Save project and quit
    cmd += ["-save", _to_rs_path(project_path)]
    cmd += ["-quit"]

    return cmd


def _reorganize_output(output_dir: Path, image_dir: Path) -> dict[str, Path]:
    """Reorganize RealityScan exports into the standard COLMAP layout.

    Expected final structure::

        output_dir/
          sparse/0/
            cameras.txt
            images.txt
            points3D.txt
          images/   (symlink or copy from input)
          point_cloud.ply

    Returns a dict with keys ``sparse_dir``, ``images_dir``, ``point_cloud``.
    """
    output_dir = Path(output_dir).resolve()
    image_dir = Path(image_dir).resolve()

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # --- Move COLMAP text files -----------------------------------------------
    colmap_src = output_dir / "colmap_undistorted"

    # RealityScan exports cameras.txt / images.txt / points3D.txt into the
    # directory containing the target path given to -exportRegistration.
    for fname in ("cameras.txt", "images.txt", "points3D.txt"):
        src = colmap_src / fname
        dst = sparse_dir / fname
        if src.is_file():
            shutil.move(str(src), str(dst))
            logger.debug("Moved %s -> %s", src, dst)
        else:
            logger.warning("Expected COLMAP file not found: %s", src)

    # If points3D.txt was not exported by RealityScan registration, create an
    # empty one so downstream tools don't crash.
    points3d = sparse_dir / "points3D.txt"
    if not points3d.exists():
        points3d.write_text(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0\n"
        )
        logger.info("Created empty points3D.txt placeholder.")

    # --- Point cloud ----------------------------------------------------------
    sparse_ply = output_dir / "sparse.ply"
    point_cloud = output_dir / "point_cloud.ply"
    if sparse_ply.is_file() and not point_cloud.exists():
        shutil.move(str(sparse_ply), str(point_cloud))
        logger.debug("Moved %s -> %s", sparse_ply, point_cloud)

    # --- Images directory (symlink to originals) ------------------------------
    images_link = output_dir / "images"
    if not images_link.exists():
        try:
            images_link.symlink_to(image_dir, target_is_directory=True)
            logger.debug("Symlinked images -> %s", image_dir)
        except OSError:
            # Symlink creation may fail on Windows without developer mode.
            # Fall back to a directory junction or plain copy.
            if sys.platform == "win32":
                try:
                    # Try a directory junction (no special privileges needed).
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J",
                         str(images_link), str(image_dir)],
                        check=True,
                        capture_output=True,
                    )
                    logger.debug("Created junction images -> %s", image_dir)
                except subprocess.CalledProcessError:
                    shutil.copytree(str(image_dir), str(images_link))
                    logger.info(
                        "Copied images to %s (symlink/junction failed)",
                        images_link,
                    )
            else:
                shutil.copytree(str(image_dir), str(images_link))
                logger.info("Copied images to %s (symlink failed)", images_link)

    # --- Cleanup leftover directories ----------------------------------------
    if colmap_src.is_dir() and not any(colmap_src.iterdir()):
        colmap_src.rmdir()

    return {
        "sparse_dir": sparse_dir,
        "images_dir": images_link,
        "point_cloud": point_cloud,
    }


class RealityScanSfMBackend(SfMBackend):
    """SfM backend using Epic Games RealityScan CLI.

    RealityScan operates on standard pinhole (perspective) images only.
    The pipeline must convert equirectangular images to perspectives
    *before* invoking this backend.

    Parameters
    ----------
    realityscan_path : str | Path | None
        Path to ``RealityScan.exe``.  Defaults to the standard install
        location on Windows.
    headless : bool
        Run RealityScan in headless mode (no GUI).
    config_dir : str | Path | None
        Directory containing the XML parameter files for export.
        Defaults to ``realityscan_pipeline/config/`` relative to the
        project root.
    """

    def __init__(
        self,
        realityscan_path: str | Path | None = None,
        headless: bool = True,
        config_dir: str | Path | None = None,
    ) -> None:
        self._rs_exe = realityscan_path or _DEFAULT_RS_EXE
        self._headless = headless
        self._config_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR

    @property
    def supports_equirectangular(self) -> bool:  # noqa: D102
        return False

    def run(
        self, image_dir: str | Path, output_dir: str | Path, **kwargs
    ) -> SfMResult:
        """Run RealityScan SfM on pinhole images.

        Parameters
        ----------
        image_dir:
            Directory containing pinhole (perspective) images produced by
            the equirect-to-perspective conversion stage.
        output_dir:
            Directory where the COLMAP sparse reconstruction and point
            cloud will be written.
        **kwargs:
            Additional options forwarded to the command builder.  Supported
            keys: ``camera_model`` (str, default ``"NONE"``).

        Returns
        -------
        SfMResult
            ``is_pinhole`` is always ``True``.

        Raises
        ------
        RuntimeError
            If the RealityScan process exits with a non-zero return code.
        FileNotFoundError
            If the RealityScan executable or config XML files are missing.
        """
        image_dir = Path(image_dir).resolve()
        output_dir = Path(output_dir).resolve()

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: {image_dir}"
            )

        rs_exe = Path(self._rs_exe)
        if not rs_exe.is_file():
            raise FileNotFoundError(
                f"RealityScan executable not found: {rs_exe}"
            )

        camera_model: str = kwargs.get("camera_model", "NONE")

        cmd = _build_command(
            image_dir,
            output_dir,
            realityscan_path=self._rs_exe,
            headless=self._headless,
            config_dir=self._config_dir,
            camera_model=camera_model,
        )

        logger.info("Running RealityScan SfM")
        logger.info("Command: %s", " ".join(shlex.quote(c) for c in cmd))

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"RealityScan exited with code {result.returncode}. "
                f"Check the console output for details."
            )

        logger.info(
            "RealityScan finished successfully, reorganizing output..."
        )

        paths = _reorganize_output(output_dir, image_dir)

        return SfMResult(
            sparse_dir=paths["sparse_dir"],
            images_dir=paths["images_dir"],
            point_cloud=paths["point_cloud"],
            is_pinhole=True,
        )


# ---------------------------------------------------------------------------
# Standalone entry point for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run RealityScan SfM backend standalone",
    )
    parser.add_argument("--images", required=True, help="Input image directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--rc",
        default=_DEFAULT_RS_EXE,
        help=f"Path to RealityScan.exe (default: {_DEFAULT_RS_EXE})",
    )
    parser.add_argument(
        "--config-dir",
        default=str(_DEFAULT_CONFIG_DIR),
        help=f"XML config directory (default: {_DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Disable headless mode (show GUI)",
    )
    parser.add_argument(
        "--camera-model",
        default="NONE",
        choices=list(CAMERA_MODEL_MAP.keys()),
        help="Distortion model (default: NONE)",
    )
    args = parser.parse_args()

    backend = RealityScanSfMBackend(
        realityscan_path=args.rc,
        headless=not args.no_headless,
        config_dir=args.config_dir,
    )
    sfm_result = backend.run(
        args.images,
        args.output,
        camera_model=args.camera_model,
    )
    print(f"Sparse dir : {sfm_result.sparse_dir}")
    print(f"Images dir : {sfm_result.images_dir}")
    print(f"Point cloud: {sfm_result.point_cloud}")
