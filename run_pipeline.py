#!/usr/bin/env python3
"""run_pipeline.py - Fully automated 360° video to Gaussian Splatting pipeline.

Usage:
    python run_pipeline.py "./data/20260330/0330 (1).mp4" -o ./output/20260330

Pipeline stages:
    1. Preprocessing  - Frame extraction (ffmpeg) + nadir mask removal
    2. SfM            - Metashape (Spherical camera, COLMAP export)
    3. 3DGS           - LichtFeld Studio (Gaussian Splatting training)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

from auto_recon.preprocessing import preprocess_video
from auto_recon.metashape_sfm import run_metashape_sfm
from auto_recon.equirect_to_perspective import convert_equirect_to_perspectives
from auto_recon.lichtfeld_3dgs import run_lichtfeld_pipeline

_JST = datetime.timezone(datetime.timedelta(hours=9))


def _setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("auto_recon")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        ts = datetime.datetime.now(tz=_JST).strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(output_dir / f"pipeline_{ts}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def run_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 1.0,
    blur_threshold: float = 100.0,
    mask_ratio: float = 0.18,
    inpaint: bool = False,
    metashape_path: str | Path | None = None,
    lichtfeld_exe: str | Path | None = None,
    iterations: int = 30_000,
    strategy: str = "mcmc",
) -> dict[str, Path]:
    """Run the full 360° video to Gaussian Splatting pipeline.

    Parameters
    ----------
    video_path:
        Path to input 360° MP4 video.
    output_dir:
        Root output directory. Subdirectories are created for each stage.
    fps:
        Frame extraction rate (frames per second).
    blur_threshold:
        Laplacian variance threshold for sharp frame selection.
    mask_ratio:
        Nadir mask height ratio (0-1). Default 0.18 covers ~18% of bottom.
    inpaint:
        Use OpenCV inpainting instead of black fill for masked regions.
    metashape_path:
        Path to metashape.exe. Defaults to standard install location.
    lichtfeld_exe:
        Path to LichtFeld-Studio.exe. Defaults to bundled copy.
    iterations:
        Number of 3DGS training iterations.
    strategy:
        3DGS optimization strategy (mcmc, adc, igs+).

    Returns
    -------
    dict[str, Path]
        Keys: ``preprocessing_dir``, ``sfm_dir``, ``splat_dir``
    """
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()

    logger = _setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("360° Video to Gaussian Splatting Pipeline")
    logger.info("=" * 60)
    logger.info("Input:  %s", video_path)
    logger.info("Output: %s", output_dir)

    # -- Stage 1: Preprocessing ------------------------------------------------
    logger.info("-" * 40)
    logger.info("Stage 1: Preprocessing (frame extraction + nadir mask)")
    logger.info("-" * 40)

    preprocess_dir = output_dir / "01_preprocessing"
    preprocess_result = preprocess_video(
        video_path=video_path,
        output_dir=preprocess_dir,
        fps=fps,
        blur_threshold=blur_threshold,
        mask_ratio=mask_ratio,
        inpaint=inpaint,
    )

    frames_dir = Path(preprocess_result["frames_dir"])
    frame_count = preprocess_result["frame_count"]
    logger.info("Stage 1 complete: %d frames extracted", frame_count)

    if frame_count == 0:
        raise RuntimeError("No frames survived preprocessing. "
                           "Try lowering --blur-threshold or --fps.")

    # -- Stage 2: Metashape SfM ------------------------------------------------
    logger.info("-" * 40)
    logger.info("Stage 2: Metashape SfM (Spherical camera)")
    logger.info("-" * 40)

    sfm_dir = output_dir / "02_sfm"
    sfm_result = run_metashape_sfm(
        image_dir=frames_dir,
        output_dir=sfm_dir,
    )

    sfm_sparse_dir = sfm_result["sparse_dir"]
    sfm_images_dir = sfm_result["images_dir"]
    logger.info("Stage 2 complete: COLMAP sparse in %s", sfm_sparse_dir)

    # -- Stage 2.5: Equirect to Perspective conversion -------------------------
    logger.info("-" * 40)
    logger.info("Stage 2.5: Equirectangular to Perspective conversion")
    logger.info("-" * 40)

    persp_dir = output_dir / "02b_perspective"
    persp_result = convert_equirect_to_perspectives(
        images_dir=sfm_images_dir,
        colmap_sparse_dir=sfm_sparse_dir,
        output_dir=persp_dir,
        fov_deg=90.0,
        out_size=(1024, 1024),
        pitch_angles=[-30.0, 0.0, 30.0],
        yaw_step_deg=90.0,
    )

    sparse_dir = persp_result["sparse_dir"]
    images_dir = persp_result["images_dir"]
    # Use SfM point cloud for initialization (lives in sfm output dir)
    init_ply = sfm_result.get("point_cloud")
    logger.info("Stage 2.5 complete: %s", persp_dir)

    # -- Stage 3: LichtFeld Studio 3DGS ---------------------------------------
    logger.info("-" * 40)
    logger.info("Stage 3: LichtFeld Studio 3DGS training")
    logger.info("-" * 40)

    splat_dir = output_dir / "03_3dgs"

    run_lichtfeld_pipeline(
        colmap_sparse_dir=sparse_dir,
        images_dir=images_dir,
        output_dir=splat_dir,
        lichtfeld_exe=lichtfeld_exe,
        iterations=iterations,
        strategy=strategy,
        init_path=init_ply if init_ply and Path(init_ply).exists() else None,
    )

    logger.info("Stage 3 complete: output in %s", splat_dir)

    # -- Summary ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Preprocessed frames: %s", preprocess_dir)
    logger.info("  SfM output:          %s", sfm_dir)
    logger.info("  3DGS output:         %s", splat_dir)
    logger.info("=" * 60)

    return {
        "preprocessing_dir": preprocess_dir,
        "sfm_dir": sfm_dir,
        "splat_dir": splat_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated 360° video to Gaussian Splatting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", type=str, help="Path to 360° MP4 video")
    parser.add_argument("-o", "--output", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frame extraction rate in fps (default: 1.0)")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                        help="Laplacian variance threshold for blur detection")
    parser.add_argument("--mask-ratio", type=float, default=0.18,
                        help="Nadir mask height ratio 0-1 (default: 0.18)")
    parser.add_argument("--inpaint", action="store_true",
                        help="Inpaint masked region instead of black fill")
    parser.add_argument("--metashape", type=str, default=None,
                        help="Path to metashape.exe")
    parser.add_argument("--lichtfeld", type=str, default=None,
                        help="Path to LichtFeld-Studio.exe")
    parser.add_argument("--iterations", type=int, default=30_000,
                        help="3DGS training iterations (default: 30000)")
    parser.add_argument("--strategy", type=str, default="mcmc",
                        choices=["mcmc", "adc", "igs+"],
                        help="3DGS optimization strategy (default: mcmc)")

    args = parser.parse_args()

    result = run_pipeline(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        blur_threshold=args.blur_threshold,
        mask_ratio=args.mask_ratio,
        inpaint=args.inpaint,
        metashape_path=args.metashape,
        lichtfeld_exe=args.lichtfeld,
        iterations=args.iterations,
        strategy=args.strategy,
    )

    print(f"\nDone! Output in: {result['splat_dir']}")


if __name__ == "__main__":
    main()
