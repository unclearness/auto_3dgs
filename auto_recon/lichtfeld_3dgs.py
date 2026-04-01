"""LichtFeld Studio 3D Gaussian Splatting integration module.

Provides functions to prepare COLMAP-format data and run 3DGS training
via LichtFeld-Studio.exe CLI.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# Default executable path relative to the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if platform.system() == "Windows":
    DEFAULT_LICHTFELD_EXE = (
        _PROJECT_ROOT
        / "LichtFeld-Studio-windows-v0.5.1"
        / "bin"
        / "LichtFeld-Studio.exe"
    )
else:
    DEFAULT_LICHTFELD_EXE = (
        _PROJECT_ROOT
        / "LichtFeld-Studio"
        / "build"
        / "LichtFeld-Studio"
    )

# ---------------------------------------------------------------------------
# COLMAP text-to-binary helpers
# ---------------------------------------------------------------------------

def _need_txt_to_bin_conversion(sparse_dir: str | Path) -> bool:
    """Return True if the sparse directory contains .txt but not .bin files."""
    sparse_dir = Path(sparse_dir)
    has_txt = (sparse_dir / "cameras.txt").exists()
    has_bin = (sparse_dir / "cameras.bin").exists()
    return has_txt and not has_bin


def _convert_colmap_txt_to_bin(sparse_dir: str | Path) -> None:
    """Convert COLMAP text model files to binary format in-place.

    Uses the ``colmap model_converter`` CLI if available.  When COLMAP is not
    installed the function logs a warning and leaves the text files as-is
    (LichtFeld Studio may accept them directly).
    """
    sparse_dir = Path(sparse_dir)
    colmap_exe = shutil.which("colmap")
    if colmap_exe is None:
        logger.warning(
            "colmap executable not found on PATH; skipping txt-to-bin "
            "conversion. LichtFeld Studio may still accept .txt format."
        )
        return

    logger.info("Converting COLMAP text model to binary: %s", sparse_dir)
    cmd = [
        colmap_exe,
        "model_converter",
        "--input_path", str(sparse_dir),
        "--output_path", str(sparse_dir),
        "--output_type", "BIN",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("colmap model_converter failed:\n%s", result.stderr)
        raise RuntimeError(f"COLMAP model conversion failed: {result.stderr}")
    logger.info("COLMAP model conversion complete.")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data_directory(
    colmap_sparse_dir: str | Path,
    images_dir: str | Path,
    output_data_dir: str | Path,
    masks_dir: str | Path | None = None,
) -> Path:
    """Create the directory layout expected by LichtFeld Studio.

    The resulting structure under *output_data_dir*::

        output_data_dir/
            images/          <- symlinked / copied from *images_dir*
            masks/           <- optional, from *masks_dir*
            sparse/
                0/           <- COLMAP model files

    Parameters
    ----------
    colmap_sparse_dir:
        Directory containing COLMAP model files (cameras.txt/.bin, etc.).
    images_dir:
        Directory with the training images.
    output_data_dir:
        Destination root that will be passed to ``LichtFeld-Studio -d``.
    masks_dir:
        Optional directory with mask images (same filenames as images).

    Returns
    -------
    Path
        The *output_data_dir* as an absolute ``Path``.
    """
    colmap_sparse_dir = Path(colmap_sparse_dir).resolve()
    images_dir = Path(images_dir).resolve()
    output_data_dir = Path(output_data_dir).resolve()

    now = datetime.now(tz=JST).strftime("%Y%m%d_%H%M%S")
    logger.info("[%s] Preparing LichtFeld data directory: %s", now, output_data_dir)

    # --- images ---
    dst_images = output_data_dir / "images"
    _link_or_copy_dir(images_dir, dst_images)

    # --- sparse/0 ---
    dst_sparse = output_data_dir / "sparse" / "0"
    _link_or_copy_dir(colmap_sparse_dir, dst_sparse)

    # Convert .txt -> .bin if needed.
    if _need_txt_to_bin_conversion(dst_sparse):
        _convert_colmap_txt_to_bin(dst_sparse)

    # --- masks (optional) ---
    if masks_dir is not None:
        masks_dir = Path(masks_dir).resolve()
        dst_masks = output_data_dir / "masks"
        _link_or_copy_dir(masks_dir, dst_masks)

    logger.info("Data directory ready: %s", output_data_dir)
    return output_data_dir


def _link_or_copy_dir(src: Path, dst: Path) -> None:
    """Create *dst* as a directory junction / symlink to *src*, falling back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        logger.debug("Destination already exists, removing: %s", dst)
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    # On Windows, directory symlinks often require elevated privileges.
    # Try symlink first, fall back to copy.
    try:
        dst.symlink_to(src, target_is_directory=True)
        logger.debug("Created symlink %s -> %s", dst, src)
    except OSError:
        logger.debug("Symlink failed; copying %s -> %s", src, dst)
        shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _count_images(data_dir: Path) -> int:
    """Count image files in *data_dir*/images/ (including subdirectories)."""
    images_path = data_dir / "images"
    if not images_path.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sum(1 for f in images_path.rglob("*") if f.is_file() and f.suffix.lower() in exts)


def run_training(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    lichtfeld_exe: str | Path | None = None,
    iterations: int = 30_000,
    strategy: str = "mrnf",
    sh_degree: int = 3,
    max_cap: int | None = None,
    steps_scaler: float | None = None,
    tile_mode: int | None = None,
    mask_mode: str | None = None,
    invert_masks: bool = False,
    resize_factor: str | None = None,
    undistort: bool = False,
    init_path: str | Path | None = None,
    ppisp: bool = True,
    extra_args: list[str] | None = None,
) -> Path:
    """Run LichtFeld Studio 3DGS training.

    Parameters
    ----------
    data_dir:
        Root directory prepared by :func:`prepare_data_directory`.
    output_dir:
        Where LichtFeld Studio writes its output (splat file, etc.).
    lichtfeld_exe:
        Path to ``LichtFeld-Studio.exe``.  Defaults to the bundled copy.
    iterations:
        Number of training iterations.
    strategy:
        Training strategy: ``"mrnf"``, ``"mcmc"``, ``"adc"``, or ``"igs+"``.
    sh_degree:
        Max spherical-harmonics degree (0-3).
    max_cap:
        Maximum number of Gaussians (for MCMC / MRNF / igs+).
    steps_scaler:
        Scale factor for training steps.  If *None*, auto-computed as
        ``image_count / 300``.
    tile_mode:
        Tile mode for memory efficiency (1, 2, or 4).
    mask_mode:
        Mask mode: ``"none"``, ``"segment"``, ``"ignore"``, ``"alpha_consistent"``.
    invert_masks:
        Invert mask values.
    resize_factor:
        Resize factor for images: ``"auto"``, ``"1"``, ``"2"``, ``"4"``, ``"8"``.
    undistort:
        Undistort images on-the-fly during training.
    init_path:
        Path to a splat file (.ply, .sog, .spz, .resume) for initialisation.
    ppisp:
        Enable PPISP (Per-Pixel Image-Space Prediction) for per-camera
        appearance modeling.  Recommended for outdoor/varying lighting.
    extra_args:
        Additional CLI arguments passed verbatim.

    Returns
    -------
    Path
        Path to the output directory (containing the splat file).
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if lichtfeld_exe is None:
        lichtfeld_exe = DEFAULT_LICHTFELD_EXE
    lichtfeld_exe = Path(lichtfeld_exe).resolve()
    if not lichtfeld_exe.is_file():
        raise FileNotFoundError(f"LichtFeld-Studio executable not found: {lichtfeld_exe}")

    # Auto-compute steps_scaler from image count.
    if steps_scaler is None:
        n_images = _count_images(data_dir)
        if n_images > 0:
            steps_scaler = n_images / 300.0
            logger.info("Auto steps_scaler=%.3f (from %d images)", steps_scaler, n_images)
        else:
            steps_scaler = 1.0
            logger.warning("No images found in %s/images; defaulting steps_scaler=1.0", data_dir)

    # Build command.
    cmd: list[str] = [
        str(lichtfeld_exe),
        "--headless",
        "--train",
        "-d", str(data_dir),
        "-o", str(output_dir),
        "-i", str(iterations),
        "--strategy", strategy,
        "--sh-degree", str(sh_degree),
        "--steps-scaler", f"{steps_scaler:.4f}",
    ]

    if max_cap is not None:
        cmd += ["--max-cap", str(max_cap)]
    if tile_mode is not None:
        cmd += ["--tile-mode", str(tile_mode)]
    if mask_mode is not None:
        cmd += ["--mask-mode", mask_mode]
    if invert_masks:
        cmd.append("--invert-masks")
    if resize_factor is not None:
        cmd += ["--resize_factor", resize_factor]
    if undistort:
        cmd.append("--undistort")
    if ppisp:
        cmd.append("--ppisp")
    if init_path is not None:
        cmd += ["--init", str(Path(init_path).resolve())]
    if extra_args:
        cmd.extend(extra_args)

    now = datetime.now(tz=JST).strftime("%Y%m%d_%H%M%S")
    logger.info("[%s] Starting LichtFeld training", now)
    logger.info("Command: %s", " ".join(cmd))

    # Run the process, streaming stdout/stderr.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            # Attempt to detect iteration progress lines.
            logger.info("[LichtFeld] %s", line)

    retcode = process.wait()
    now = datetime.now(tz=JST).strftime("%Y%m%d_%H%M%S")
    if retcode != 0:
        logger.error("[%s] LichtFeld training failed with exit code %d", now, retcode)
        raise RuntimeError(f"LichtFeld-Studio exited with code {retcode}")

    logger.info("[%s] LichtFeld training completed successfully.", now)
    logger.info("Output directory: %s", output_dir)

    return output_dir


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_lichtfeld_pipeline(
    colmap_sparse_dir: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    masks_dir: str | Path | None = None,
    *,
    lichtfeld_exe: str | Path | None = None,
    iterations: int = 30_000,
    strategy: str = "mrnf",
    sh_degree: int = 3,
    max_cap: int | None = None,
    steps_scaler: float | None = None,
    tile_mode: int | None = None,
    mask_mode: str | None = None,
    invert_masks: bool = False,
    resize_factor: str | None = None,
    undistort: bool = False,
    init_path: str | Path | None = None,
    ppisp: bool = True,
    extra_args: list[str] | None = None,
) -> Path:
    """Run the full LichtFeld 3DGS pipeline: prepare data then train.

    This is the main entry point for downstream callers.  It creates a
    ``lichtfeld_data/`` subdirectory inside *output_dir* for the prepared
    dataset, then runs training with output written to *output_dir*.

    Returns
    -------
    Path
        Path to the training output directory.
    """
    output_dir = Path(output_dir).resolve()
    data_dir = output_dir / "lichtfeld_data"

    now = datetime.now(tz=JST).strftime("%Y%m%d_%H%M%S")
    logger.info("[%s] === LichtFeld 3DGS Pipeline Start ===", now)

    # Step 1: Prepare data directory.
    prepare_data_directory(
        colmap_sparse_dir=colmap_sparse_dir,
        images_dir=images_dir,
        output_data_dir=data_dir,
        masks_dir=masks_dir,
    )

    # Step 2: Run training.
    result_dir = run_training(
        data_dir=data_dir,
        output_dir=output_dir,
        lichtfeld_exe=lichtfeld_exe,
        iterations=iterations,
        strategy=strategy,
        sh_degree=sh_degree,
        max_cap=max_cap,
        steps_scaler=steps_scaler,
        tile_mode=tile_mode,
        mask_mode=mask_mode,
        invert_masks=invert_masks,
        resize_factor=resize_factor,
        undistort=undistort,
        init_path=init_path,
        ppisp=ppisp,
        extra_args=extra_args,
    )

    now = datetime.now(tz=JST).strftime("%Y%m%d_%H%M%S")
    logger.info("[%s] === LichtFeld 3DGS Pipeline Complete ===", now)
    return result_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) < 4:
        print(
            "Usage: python -m auto_recon.lichtfeld_3dgs "
            "<colmap_sparse_dir> <images_dir> <output_dir> [masks_dir]"
        )
        sys.exit(1)

    colmap_sparse = sys.argv[1]
    images = sys.argv[2]
    output = sys.argv[3]
    masks = sys.argv[4] if len(sys.argv) > 4 else None

    run_lichtfeld_pipeline(
        colmap_sparse_dir=colmap_sparse,
        images_dir=images,
        output_dir=output,
        masks_dir=masks,
    )
