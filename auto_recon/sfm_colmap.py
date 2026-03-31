"""COLMAP SfM backend using CLI subprocess calls.

Supports 360° equirectangular images by converting them to virtual pinhole
camera rig views and running incremental SfM with rig constraints via the
COLMAP CLI (``colmap.exe``).

No ``pycolmap`` dependency — all COLMAP operations are executed through
subprocess calls to avoid hanging issues observed with the Python bindings.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from auto_recon.pano_render import (
    PanoRenderOptions,
    DEFAULT_RENDER_OPTIONS,
    RigCameraInfo,
    VirtualCamera,
    render_perspective_images,
    rotation_matrix_to_quaternion,
)
from auto_recon.sfm_backend import SfMBackend, SfMResult

logger = logging.getLogger("auto_recon.sfm_colmap")

# Default COLMAP executable — resolved relative to the project root.
_DEFAULT_COLMAP_EXE = (
    Path(__file__).resolve().parent.parent / "colmap-x64-windows-cuda" / "bin" / "colmap.exe"
)


# ---------------------------------------------------------------------------
# Rig config JSON generation (for COLMAP CLI rig_configurator)
# ---------------------------------------------------------------------------


def _write_rig_config_json(
    rig_cameras: list[RigCameraInfo],
    output_path: Path,
) -> None:
    """Write a COLMAP rig configuration JSON file."""
    cameras_json = []
    for cam in rig_cameras:
        entry: dict = {"image_prefix": cam.image_prefix}
        if cam.ref_sensor:
            entry["ref_sensor"] = True
        if cam.cam_from_rig_rotation is not None:
            qw, qx, qy, qz = rotation_matrix_to_quaternion(cam.cam_from_rig_rotation)
            entry["cam_from_rig_rotation"] = [qw, qx, qy, qz]
            entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
        cameras_json.append(entry)

    rig_json = [{"cameras": cameras_json}]
    output_path.write_text(json.dumps(rig_json, indent=2), encoding="utf-8")
    logger.info("Wrote rig config to %s", output_path)


# ---------------------------------------------------------------------------
# COLMAP CLI runner
# ---------------------------------------------------------------------------


def _run_colmap(
    colmap_exe: Path,
    command: str,
    args: dict[str, str],
    *,
    timeout: int | None = None,
) -> None:
    """Run a COLMAP CLI command via subprocess."""
    cmd = [str(colmap_exe), command]
    for key, val in args.items():
        cmd.extend([f"--{key}", str(val)])

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )

    if result.stdout:
        for line in result.stdout.splitlines():
            logger.debug("[colmap %s] %s", command, line)

    if result.returncode != 0:
        tail = "\n".join((result.stdout or "").splitlines()[-20:])
        logger.error("colmap %s failed (rc=%d):\n%s", command, result.returncode, tail)
        raise RuntimeError(
            f"colmap {command} failed with return code {result.returncode}"
        )


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class ColmapSfMBackend(SfMBackend):
    """SfM backend using COLMAP CLI with native 360° panorama rig support.

    Parameters
    ----------
    colmap_exe:
        Path to the COLMAP executable. Defaults to the bundled binary.
    matcher:
        Matching strategy: ``"sequential"`` or ``"exhaustive"``.
    render_options:
        Panorama render options. Default: 4 yaw steps × 3 pitches (12 views).
    """

    def __init__(
        self,
        colmap_exe: str | Path | None = None,
        matcher: str = "sequential",
        render_options: PanoRenderOptions | None = None,
    ) -> None:
        self._colmap_exe = Path(colmap_exe) if colmap_exe else _DEFAULT_COLMAP_EXE
        if not self._colmap_exe.exists():
            raise FileNotFoundError(
                f"COLMAP executable not found: {self._colmap_exe}"
            )
        self._matcher = matcher
        self._render_options = render_options or DEFAULT_RENDER_OPTIONS

    @property
    def supports_equirectangular(self) -> bool:
        return True

    def run(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        **kwargs,
    ) -> SfMResult:
        """Run the COLMAP panorama SfM pipeline via CLI.

        1. Render equirectangular images to virtual pinhole views with masks
        2. Extract features (masked)
        3. Apply rig configuration
        4. Match features (with rig verification)
        5. Run incremental mapping (with fixed intrinsics/rig)
        6. Export model to TXT + PLY
        """
        image_dir = Path(image_dir).resolve()
        output_dir = Path(output_dir).resolve()

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        persp_image_dir = output_dir / "images"
        mask_dir = output_dir / "masks"
        persp_image_dir.mkdir(exist_ok=True, parents=True)
        mask_dir.mkdir(exist_ok=True, parents=True)

        database_path = output_dir / "database.db"
        if database_path.exists():
            database_path.unlink()

        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True, parents=True)

        # Step 1: Render perspective images + masks
        logger.info("Step 1: Rendering perspective images from panoramas")
        rig_cameras, virtual_cam = render_perspective_images(
            image_dir, persp_image_dir, self._render_options, mask_dir=mask_dir,
        )

        # Step 2: Feature extraction (with masks)
        logger.info("Step 2: Extracting features")
        _run_colmap(self._colmap_exe, "feature_extractor", {
            "database_path": database_path,
            "image_path": persp_image_dir,
            "ImageReader.mask_path": mask_dir,
            "ImageReader.camera_model": "SIMPLE_PINHOLE",
            "ImageReader.camera_params": f"{virtual_cam.focal:.6f},{virtual_cam.cx:.6f},{virtual_cam.cy:.6f}",
            "ImageReader.single_camera_per_folder": "1",
        })

        # Step 3: Write and apply rig configuration
        logger.info("Step 3: Applying rig configuration")
        rig_config_path = output_dir / "rig_config.json"
        _write_rig_config_json(rig_cameras, rig_config_path)
        _run_colmap(self._colmap_exe, "rig_configurator", {
            "database_path": database_path,
            "rig_config_path": rig_config_path,
        })

        # Step 4: Feature matching
        logger.info("Step 4: Matching features (%s)", self._matcher)
        matcher_cmd = (
            "sequential_matcher" if self._matcher == "sequential"
            else "exhaustive_matcher"
        )
        _run_colmap(self._colmap_exe, matcher_cmd, {
            "database_path": database_path,
            "FeatureMatching.rig_verification": "1",
            "FeatureMatching.skip_image_pairs_in_same_frame": "1",
        })

        # Step 5: Incremental mapping (fixed intrinsics and rig)
        logger.info("Step 5: Running incremental mapping")
        _run_colmap(self._colmap_exe, "mapper", {
            "database_path": database_path,
            "image_path": persp_image_dir,
            "output_path": sparse_dir,
            "Mapper.ba_refine_sensor_from_rig": "0",
            "Mapper.ba_refine_focal_length": "0",
            "Mapper.ba_refine_principal_point": "0",
            "Mapper.ba_refine_extra_params": "0",
        })

        # Find the best (largest) reconstruction
        model_dirs = sorted(sparse_dir.iterdir())
        if not model_dirs:
            raise RuntimeError(
                "COLMAP produced no reconstructions. "
                "Check image overlap and quality."
            )

        best_model_dir = model_dirs[0]
        if len(model_dirs) > 1:
            logger.info("Found %d reconstructions, picking largest", len(model_dirs))
            best_count = 0
            for md in model_dirs:
                txt_tmp = md / "_tmp_txt"
                txt_tmp.mkdir(exist_ok=True)
                try:
                    _run_colmap(self._colmap_exe, "model_converter", {
                        "input_path": md,
                        "output_path": txt_tmp,
                        "output_type": "TXT",
                    })
                    images_txt = txt_tmp / "images.txt"
                    if images_txt.exists():
                        lines = [l for l in images_txt.read_text().splitlines()
                                 if l and not l.startswith("#")]
                        count = len(lines) // 2
                        if count > best_count:
                            best_count = count
                            best_model_dir = md
                finally:
                    shutil.rmtree(txt_tmp, ignore_errors=True)

        logger.info("Best reconstruction: %s", best_model_dir)

        # Step 6: Export to TXT and PLY
        logger.info("Step 6: Exporting model to TXT")
        _run_colmap(self._colmap_exe, "model_converter", {
            "input_path": best_model_dir,
            "output_path": best_model_dir,
            "output_type": "TXT",
        })

        point_cloud_path = output_dir / "point_cloud.ply"
        logger.info("Step 6: Exporting point cloud to PLY")
        _run_colmap(self._colmap_exe, "model_converter", {
            "input_path": best_model_dir,
            "output_path": point_cloud_path,
            "output_type": "PLY",
        })

        logger.info("COLMAP panorama SfM complete: %s", output_dir)

        return SfMResult(
            sparse_dir=best_model_dir,
            images_dir=persp_image_dir,
            point_cloud=point_cloud_path,
            is_pinhole=True,
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Run COLMAP panorama SfM on equirectangular images.",
    )
    parser.add_argument("image_dir", type=Path, help="Input panorama directory")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--colmap-exe", type=Path, default=None,
        help="Path to colmap executable",
    )
    parser.add_argument(
        "--matcher", default="sequential",
        choices=["sequential", "exhaustive"],
    )
    args = parser.parse_args()

    backend = ColmapSfMBackend(colmap_exe=args.colmap_exe, matcher=args.matcher)
    result = backend.run(args.image_dir, args.output_dir)
    print(f"\nDone! sparse: {result.sparse_dir}, images: {result.images_dir}")
