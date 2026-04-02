#!/usr/bin/env python3
"""run_pipeline.py - Fully automated 360° video to Gaussian Splatting pipeline.

Usage:
    python run_pipeline.py "./data/20260330/0330 (1).mp4" -o ./output/20260330

    # Run only Stage 3 (reuse previous stages' output):
    python run_pipeline.py "./data/20260330/0330 (1).mp4" -o ./output/20260330 --from-stage 3 --iterations 30000

Pipeline stages:
    1. Preprocessing  - Frame extraction (ffmpeg) + nadir mask removal
    2. SfM            - Structure from Motion (Metashape/COLMAP/RealityScan)
       - If backend supports equirect: SfM on equirect, then convert to pinhole
       - If not: convert equirect to pinhole first, then run SfM
    3. 3DGS           - LichtFeld Studio (Gaussian Splatting training)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import shutil
import sys
from pathlib import Path

from auto_recon.preprocessing import preprocess_video
from auto_recon.equirect_to_perspective import convert_equirect_to_perspectives
from auto_recon.lichtfeld_3dgs import run_lichtfeld_pipeline
from auto_recon.sfm_backend import SfMBackend, SfMResult
from auto_recon.sfm_metashape import MetashapeSfMBackend
from auto_recon.sfm_colmap import ColmapSfMBackend
from auto_recon.sfm_realityscan import RealityScanSfMBackend

_SFM_BACKENDS: dict[str, type[SfMBackend]] = {
    "metashape": MetashapeSfMBackend,
    "colmap": ColmapSfMBackend,
    "realityscan": RealityScanSfMBackend,
}

_JST = datetime.timezone(datetime.timedelta(hours=9))

# Stage directory names (fixed convention)
_STAGE_DIRS = {
    1: "01_preprocessing",
    2: "02_sfm",
    "2b": "02b_perspective",
    3: "03_3dgs",
}


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
    lichtfeld_exe: str | Path | None = None,
    iterations: int = 30_000,
    strategy: str = "mrnf",
    ppisp: bool = True,
    from_stage: int = 1,
    sfm_backend: str = "metashape",
    sam3_mode: str | None = None,
    sam3_confidence: float = 0.3,
    sam3_prompt: str = "person",
    sam3_batch_size: int = 4,
    sam3_scale: float = 1.0,
    render_eval: bool = False,
) -> dict[str, Path]:
    """Run the 360° video to Gaussian Splatting pipeline.

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
        Nadir mask height ratio (0-1).
    inpaint:
        Use OpenCV inpainting instead of black fill for masked regions.
    lichtfeld_exe:
        Path to LichtFeld-Studio.exe. Defaults to bundled copy.
    iterations:
        Number of 3DGS training iterations.
    strategy:
        3DGS optimization strategy (mcmc, adc, igs+).
    from_stage:
        Start from this stage (1, 2, or 3). Earlier stages' output is reused.
    sfm_backend:
        SfM backend to use: ``"metashape"``, ``"colmap"``, or
        ``"realityscan"`` (default: ``"metashape"``).
    """
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Resolve SfM backend
    if sfm_backend not in _SFM_BACKENDS:
        raise ValueError(
            f"Unknown SfM backend: {sfm_backend!r}. "
            f"Choose from: {', '.join(_SFM_BACKENDS)}"
        )
    backend: SfMBackend = _SFM_BACKENDS[sfm_backend]()

    logger = _setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("360° Video to Gaussian Splatting Pipeline")
    logger.info("=" * 60)
    logger.info("Input:  %s", video_path)
    logger.info("Output: %s", output_dir)
    logger.info("SfM backend: %s (equirect support: %s)",
                sfm_backend, backend.supports_equirectangular)
    if from_stage > 1:
        logger.info("Resuming from stage %d (reusing earlier outputs)", from_stage)

    preprocess_dir = output_dir / _STAGE_DIRS[1]
    sfm_dir = output_dir / _STAGE_DIRS[2]
    persp_dir = output_dir / _STAGE_DIRS["2b"]
    splat_dir = output_dir / _STAGE_DIRS[3]

    # -- Stage 1: Preprocessing ------------------------------------------------
    if from_stage <= 1:
        logger.info("-" * 40)
        logger.info("Stage 1: Preprocessing (frame extraction + nadir mask)")
        logger.info("-" * 40)

        preprocess_result = preprocess_video(
            video_path=video_path,
            output_dir=preprocess_dir,
            fps=fps,
            blur_threshold=blur_threshold,
            mask_ratio=mask_ratio,
            inpaint=inpaint,
            sam3_mode=sam3_mode,
            sam3_confidence=sam3_confidence,
            sam3_prompt=sam3_prompt,
            sam3_batch_size=sam3_batch_size,
            sam3_scale=sam3_scale,
        )

        frames_dir = Path(preprocess_result["frames_dir"])
        equirect_masks_dir = Path(preprocess_result["masks_dir"]) if preprocess_result.get("masks_dir") else None
        frame_count = preprocess_result["frame_count"]
        logger.info("Stage 1 complete: %d frames extracted", frame_count)
        if equirect_masks_dir:
            logger.info("Mask images: %s", equirect_masks_dir)

        if frame_count == 0:
            raise RuntimeError("No frames survived preprocessing. "
                               "Try lowering --blur-threshold or --fps.")
    else:
        frames_dir = preprocess_dir / "frames_masked"
        if not frames_dir.is_dir():
            raise FileNotFoundError(
                f"Stage 1 output not found: {frames_dir}. "
                "Run from stage 1 first."
            )
        # Check if mask images exist from a previous run
        equirect_masks_dir = preprocess_dir / "masks_combined"
        if not equirect_masks_dir.is_dir():
            equirect_masks_dir = None
        logger.info("Stage 1: Reusing %s", frames_dir)

    # -- Stage 2: SfM (+ equirect conversion where needed) ----------------------
    if from_stage <= 2:
        if backend.supports_equirectangular:
            # Backend handles equirect natively: run SfM first, then convert.
            logger.info("-" * 40)
            logger.info("Stage 2: %s SfM (equirectangular)", sfm_backend)
            logger.info("-" * 40)

            sfm_result = backend.run(image_dir=frames_dir, output_dir=sfm_dir)
            logger.info("Stage 2 complete: COLMAP sparse in %s", sfm_result.sparse_dir)

            if sfm_result.is_pinhole:
                # Backend already produced pinhole images (e.g. COLMAP with
                # internal equirect-to-perspective conversion).  Skip Stage 2.5.
                logger.info("Backend produced pinhole images; skipping Stage 2.5")
                sparse_dir = sfm_result.sparse_dir
                images_dir = sfm_result.images_dir
                init_ply = sfm_result.point_cloud
                masks_dir = None  # TODO: convert equirect masks to rig perspective masks
            else:
                # -- Stage 2.5: Equirect to Perspective conversion -----------------
                logger.info("-" * 40)
                logger.info("Stage 2.5: Equirectangular to Perspective conversion")
                logger.info("-" * 40)

                persp_result = convert_equirect_to_perspectives(
                    images_dir=sfm_result.images_dir,
                    colmap_sparse_dir=sfm_result.sparse_dir,
                    output_dir=persp_dir,
                    fov_deg=90.0,
                    out_size=(1024, 1024),
                    pitch_angles=[-30.0, 0.0, 30.0],
                    yaw_step_deg=90.0,
                    masks_dir=equirect_masks_dir,
                )

                sparse_dir = persp_result["sparse_dir"]
                images_dir = persp_result["images_dir"]
                init_ply = sfm_result.point_cloud
                masks_dir = persp_result.get("masks_dir")
                logger.info("Stage 2.5 complete: %s", persp_dir)
        else:
            # Backend needs pinhole images: convert first, then run SfM.
            logger.info("-" * 40)
            logger.info("Stage 2a: Equirectangular to Perspective conversion (pre-SfM)")
            logger.info("-" * 40)

            # For non-equirect backends we don't have a COLMAP sparse dir yet.
            # convert_equirect_to_perspectives needs colmap_sparse_dir; run a
            # lightweight conversion that only produces perspective images
            # without requiring camera poses.  However, the current
            # convert_equirect_to_perspectives requires a sparse dir.
            #
            # For backends that do NOT support equirect, we generate
            # perspective images from the equirectangular frames using a
            # uniform yaw/pitch grid (no existing camera poses needed).
            persp_result = convert_equirect_to_perspectives(
                images_dir=frames_dir,
                colmap_sparse_dir=None,
                output_dir=persp_dir,
                fov_deg=90.0,
                out_size=(1024, 1024),
                pitch_angles=[-30.0, 0.0, 30.0],
                yaw_step_deg=90.0,
            )

            persp_images_dir = persp_result["images_dir"]
            logger.info("Stage 2a complete: perspective images in %s", persp_images_dir)

            logger.info("-" * 40)
            logger.info("Stage 2b: %s SfM (pinhole)", sfm_backend)
            logger.info("-" * 40)

            sfm_result = backend.run(image_dir=persp_images_dir, output_dir=sfm_dir)
            logger.info("Stage 2b complete: COLMAP sparse in %s", sfm_result.sparse_dir)

            sparse_dir = sfm_result.sparse_dir
            images_dir = sfm_result.images_dir
            init_ply = sfm_result.point_cloud
            masks_dir = None  # TODO: masks for non-equirect path
    else:
        sparse_dir = persp_dir / "sparse" / "0"
        images_dir = persp_dir / "images"
        init_ply = sfm_dir / "point_cloud.ply"
        masks_dir = persp_dir / "masks"
        if not masks_dir.is_dir():
            masks_dir = None
        if not sparse_dir.is_dir():
            # Fall back: check sfm_dir directly (non-equirect backends
            # store the sparse dir inside sfm_dir, not persp_dir).
            alt_sparse = sfm_dir / "sparse" / "0"
            if alt_sparse.is_dir():
                sparse_dir = alt_sparse
                images_dir = sfm_dir / "images"
            else:
                raise FileNotFoundError(
                    f"Stage 2 output not found: {sparse_dir}. "
                    "Run from stage 2 first."
                )
        logger.info("Stage 2/2.5: Reusing %s", sparse_dir.parent.parent)

    # -- Stage 3: LichtFeld Studio 3DGS ---------------------------------------
    logger.info("-" * 40)
    logger.info("Stage 3: LichtFeld Studio 3DGS training (%d iterations)", iterations)
    logger.info("-" * 40)

    # Clean previous 3DGS output if re-running
    if splat_dir.exists():
        shutil.rmtree(splat_dir)

    run_lichtfeld_pipeline(
        colmap_sparse_dir=sparse_dir,
        images_dir=images_dir,
        output_dir=splat_dir,
        masks_dir=masks_dir,
        lichtfeld_exe=lichtfeld_exe,
        iterations=iterations,
        strategy=strategy,
        ppisp=ppisp,
        mask_mode="ignore" if masks_dir else None,
        init_path=init_ply if init_ply and Path(init_ply).exists() else None,
        eval_images=render_eval,
        test_every=8,
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
    parser.add_argument("--lichtfeld", type=str, default=None,
                        help="Path to LichtFeld-Studio.exe")
    parser.add_argument("--iterations", type=int, default=30_000,
                        help="3DGS training iterations (default: 30000)")
    parser.add_argument("--strategy", type=str, default="mrnf",
                        choices=["mrnf", "mcmc", "adc", "igs+"],
                        help="3DGS optimization strategy (default: mrnf)")
    parser.add_argument("--no-ppisp", action="store_true",
                        help="Disable PPISP per-camera appearance modeling")
    parser.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3],
                        help="Start from this stage, reusing earlier outputs (default: 1)")
    parser.add_argument("--sfm-backend", type=str, default="metashape",
                        choices=["metashape", "colmap", "realityscan"],
                        help="SfM backend to use (default: metashape)")
    parser.add_argument("--sam3", type=str, default="pinhole",
                        choices=["pinhole", "equirect", "trt", "trt-equirect", "off"],
                        help="SAM3 person masking mode (default: pinhole; trt for 2x faster bbox masks)")
    parser.add_argument("--sam3-confidence", type=float, default=0.3,
                        help="SAM3 confidence threshold (default: 0.3)")
    parser.add_argument("--sam3-prompt", type=str, default="person",
                        help="SAM3 text prompt (default: person)")
    parser.add_argument("--sam3-batch", type=int, default=4,
                        help="SAM3 batch size for pinhole mode (default: 4)")
    parser.add_argument("--sam3-scale", type=float, default=1.0,
                        help="SAM3 input scale factor, e.g. 0.5 = half resolution (default: 1.0)")
    parser.add_argument("--render", action="store_true",
                        help="Render camera viewpoints after training (Stage 4)")

    args = parser.parse_args()

    result = run_pipeline(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        blur_threshold=args.blur_threshold,
        mask_ratio=args.mask_ratio,
        inpaint=args.inpaint,
        lichtfeld_exe=args.lichtfeld,
        iterations=args.iterations,
        strategy=args.strategy,
        ppisp=not args.no_ppisp,
        from_stage=args.from_stage,
        sfm_backend=args.sfm_backend,
        sam3_mode=args.sam3 if args.sam3 != "off" else None,
        sam3_confidence=args.sam3_confidence,
        sam3_prompt=args.sam3_prompt,
        sam3_batch_size=args.sam3_batch,
        sam3_scale=args.sam3_scale,
        render_eval=args.render,
    )

    print(f"\nDone! Output in: {result['splat_dir']}")


if __name__ == "__main__":
    main()
