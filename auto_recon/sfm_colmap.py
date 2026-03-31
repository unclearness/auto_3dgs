"""COLMAP SfM backend using pycolmap's panorama rig pipeline.

Supports 360° equirectangular images natively by converting them to virtual
pinhole camera rig views (similar to a cubemap) and running incremental SfM
with rig constraints.  Based on COLMAP 4.0.2's ``panorama_sfm.py`` example.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

import pycolmap

from auto_recon.sfm_backend import SfMBackend, SfMResult

logger = logging.getLogger("auto_recon.sfm_colmap")

# ---------------------------------------------------------------------------
# Panorama render configuration
# ---------------------------------------------------------------------------


@dataclass
class PanoRenderOptions:
    num_steps_yaw: int
    pitches_deg: Sequence[float]
    hfov_deg: float
    vfov_deg: float


DEFAULT_RENDER_OPTIONS = PanoRenderOptions(
    num_steps_yaw=4,
    pitches_deg=(-35.0, 0.0, 35.0),
    hfov_deg=90.0,
    vfov_deg=90.0,
)

# ---------------------------------------------------------------------------
# Virtual camera helpers (from COLMAP panorama_sfm.py)
# ---------------------------------------------------------------------------


def _create_virtual_camera(
    pano_width: int,
    pano_height: int,
    hfov_deg: float,
    vfov_deg: float,
) -> pycolmap.Camera:
    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    return pycolmap.Camera.create_from_model_id(
        camera_id=0,
        model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
        focal_length=focal,
        width=image_width,
        height=image_height,
    )


def _get_virtual_camera_rays(camera: pycolmap.Camera) -> npt.NDArray:
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    xy += 0.5
    xy_norm = camera.cam_from_img(image_points=xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def _spherical_img_from_cam(
    image_size: tuple[int, int],
    rays_in_cam: npt.NDArray,
) -> npt.NDArray:
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def _get_virtual_rotations(
    opts: PanoRenderOptions,
) -> list[npt.NDArray]:
    cams_from_pano_r: list[npt.NDArray] = []
    yaws = np.linspace(0, 360, opts.num_steps_yaw, endpoint=False)
    for pitch_deg in opts.pitches_deg:
        yaw_offset = (360 / opts.num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


def _create_pano_rig_config(
    cams_from_pano_rotation: Sequence[npt.NDArray],
    ref_idx: int = 0,
) -> pycolmap.RigConfig:
    rig_cameras = []
    zero_t = np.zeros((3, 1), dtype=np.float64)
    for idx, cam_from_pano_r in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_r = cam_from_pano_r @ cams_from_pano_rotation[ref_idx].T
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_r), zero_t,
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


# ---------------------------------------------------------------------------
# Panorama rendering
# ---------------------------------------------------------------------------


class _PanoProcessor:
    """Render virtual pinhole views from equirectangular panoramas."""

    def __init__(
        self,
        pano_image_dir: Path,
        output_image_dir: Path,
        mask_dir: Path,
        opts: PanoRenderOptions,
    ) -> None:
        self.opts = opts
        self.pano_image_dir = pano_image_dir
        self.output_image_dir = output_image_dir
        self.mask_dir = mask_dir

        self.cams_from_pano_r = _get_virtual_rotations(opts)
        self.rig_config = _create_pano_rig_config(self.cams_from_pano_r)

        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_r, [0, 0, 1]
        )
        self._lock = Lock()
        self._camera: pycolmap.Camera | None = None
        self._pano_size: tuple[int, int] | None = None
        self._rays_in_cam: npt.NDArray | None = None

    def process(self, pano_name: str) -> None:
        pano_path = self.pano_image_dir / pano_name
        pano_image = cv2.imread(str(pano_path))
        if pano_image is None:
            logger.warning("Cannot read %s, skipping", pano_path)
            return

        pano_h, pano_w = pano_image.shape[:2]

        with self._lock:
            if self._camera is None:
                self._camera = _create_virtual_camera(
                    pano_w, pano_h, self.opts.hfov_deg, self.opts.vfov_deg,
                )
                for rig_cam in self.rig_config.cameras:
                    rig_cam.camera = self._camera
                self._pano_size = (pano_w, pano_h)
                self._rays_in_cam = _get_virtual_camera_rays(self._camera)

        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_r):
            rays_in_pano = self._rays_in_cam @ cam_from_pano_r
            xy_in_pano = _spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                self._camera.width, self._camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5
            x_coords, y_coords = np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0])
            image = cv2.remap(
                pano_image, x_coords, y_coords,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
            )

            # Mask: each pixel belongs to the nearest virtual camera
            closest_camera = np.argmax(
                rays_in_pano @ self.cam_centers_in_pano.T, -1
            )
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(self._camera.width, self._camera.height)
                .transpose()
            )

            image_name = self.rig_config.cameras[cam_idx].image_prefix + pano_name
            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

            mask_name = f"{image_name}.png"
            mask_path = self.mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(mask_path), mask)


def _render_perspective_images(
    pano_image_dir: Path,
    output_image_dir: Path,
    mask_dir: Path,
    opts: PanoRenderOptions,
) -> pycolmap.RigConfig:
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    pano_names = sorted(
        p.name for p in pano_image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTS
    )
    logger.info("Rendering %d panoramas to perspective views", len(pano_names))

    processor = _PanoProcessor(pano_image_dir, output_image_dir, mask_dir, opts)

    # Process sequentially — pycolmap Camera objects are not thread-safe and
    # concurrent cv2.remap on large panoramas can cause segfaults on Windows.
    for name in pano_names:
        processor.process(name)

    logger.info(
        "Rendered %d perspective images in %s",
        len(pano_names) * len(processor.cams_from_pano_r),
        output_image_dir,
    )
    return processor.rig_config


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class ColmapSfMBackend(SfMBackend):
    """SfM backend using pycolmap with native 360° panorama rig support.

    Converts equirectangular images to virtual pinhole views with rig
    constraints, then runs incremental SfM via pycolmap.

    Parameters
    ----------
    matcher:
        Matching strategy: ``"sequential"`` or ``"exhaustive"``.
    render_options:
        Panorama render options. Default: 4 yaw steps × 3 pitches (12 views).
    """

    def __init__(
        self,
        matcher: str = "sequential",
        render_options: PanoRenderOptions | None = None,
    ) -> None:
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
        """Run the COLMAP panorama SfM pipeline.

        1. Render equirectangular images to virtual pinhole views with masks
        2. Extract features (masked)
        3. Apply rig configuration
        4. Match features (with rig verification)
        5. Run incremental mapping (with fixed intrinsics/rig)
        6. Export model to TXT + PLY

        Returns SfMResult with is_pinhole=True (output images are pinhole).
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

        pycolmap.set_random_seed(0)

        # Step 1: Render perspective images + masks
        logger.info("Step 1: Rendering perspective images from panoramas")
        rig_config = _render_perspective_images(
            image_dir, persp_image_dir, mask_dir, self._render_options,
        )

        # Step 2: Feature extraction (with masks)
        logger.info("Step 2: Extracting features")
        pycolmap.extract_features(
            database_path,
            persp_image_dir,
            reader_options=pycolmap.ImageReaderOptions(mask_path=mask_dir),
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
        )

        # Step 3: Apply rig configuration
        logger.info("Step 3: Applying rig configuration")
        with pycolmap.Database.open(database_path) as db:
            pycolmap.apply_rig_config([rig_config], db)

        # Step 4: Feature matching
        logger.info("Step 4: Matching features (%s)", self._matcher)
        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.rig_verification = True
        matching_options.skip_image_pairs_in_same_frame = True

        if self._matcher == "sequential":
            pycolmap.match_sequential(
                database_path,
                pairing_options=pycolmap.SequentialPairingOptions(
                    loop_detection=False,
                ),
                matching_options=matching_options,
            )
        elif self._matcher == "exhaustive":
            pycolmap.match_exhaustive(
                database_path,
                matching_options=matching_options,
            )
        else:
            raise ValueError(f"Unknown matcher: {self._matcher}")

        # Step 5: Incremental mapping (fixed intrinsics and rig)
        logger.info("Step 5: Running incremental mapping")
        pipeline_opts = pycolmap.IncrementalPipelineOptions(
            ba_refine_sensor_from_rig=False,
            ba_refine_focal_length=False,
            ba_refine_principal_point=False,
            ba_refine_extra_params=False,
        )
        recs = pycolmap.incremental_mapping(
            database_path, persp_image_dir, sparse_dir, pipeline_opts,
        )

        if not recs:
            raise RuntimeError(
                "COLMAP produced no reconstructions. "
                "Check image overlap and quality."
            )

        # Pick the largest reconstruction
        best_idx = max(recs, key=lambda k: recs[k].num_reg_images())
        best_rec = recs[best_idx]
        logger.info(
            "Best reconstruction #%d: %d images, %d points",
            best_idx, best_rec.num_reg_images(), best_rec.num_points3D(),
        )

        model_dir = sparse_dir / str(best_idx)

        # Step 6: Export to TXT and PLY
        logger.info("Step 6: Exporting model")
        best_rec.write_text(str(model_dir))
        point_cloud_path = output_dir / "point_cloud.ply"
        best_rec.export_PLY(str(point_cloud_path))

        logger.info("COLMAP panorama SfM complete: %s", output_dir)

        return SfMResult(
            sparse_dir=model_dir,
            images_dir=persp_image_dir,
            point_cloud=point_cloud_path,
            is_pinhole=True,
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

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
        "--matcher", default="sequential",
        choices=["sequential", "exhaustive"],
    )
    args = parser.parse_args()

    backend = ColmapSfMBackend(matcher=args.matcher)
    result = backend.run(args.image_dir, args.output_dir)
    print(f"\nDone! sparse: {result.sparse_dir}, images: {result.images_dir}")
