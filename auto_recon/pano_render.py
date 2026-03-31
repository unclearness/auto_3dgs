"""Shared panorama rendering utilities.

Converts equirectangular (360°) images to virtual pinhole camera views
organized in per-camera subdirectories.  Used by both the COLMAP and
RealityScan SfM backends.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger("auto_recon.pano_render")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ---------------------------------------------------------------------------
# Render configuration
# ---------------------------------------------------------------------------


@dataclass
class PanoRenderOptions:
    num_steps_yaw: int = 4
    pitches_deg: Sequence[float] = (-35.0, 0.0, 35.0)
    hfov_deg: float = 90.0
    vfov_deg: float = 90.0


DEFAULT_RENDER_OPTIONS = PanoRenderOptions()


# ---------------------------------------------------------------------------
# Pure-numpy rotation helpers
# ---------------------------------------------------------------------------


def _rot_x(angle_rad: float) -> npt.NDArray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> npt.NDArray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotation_matrix_to_quaternion(
    R: npt.NDArray,
) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to (qw, qx, qy, qz) quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


# ---------------------------------------------------------------------------
# Virtual camera
# ---------------------------------------------------------------------------


@dataclass
class VirtualCamera:
    """Simple pinhole camera parameters."""
    width: int
    height: int
    focal: float

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0


def create_virtual_camera(
    pano_width: int,
    pano_height: int,
    hfov_deg: float,
    vfov_deg: float,
) -> VirtualCamera:
    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    return VirtualCamera(width=image_width, height=image_height, focal=focal)


def _get_virtual_camera_rays(cam: VirtualCamera) -> npt.NDArray:
    x_coords = np.arange(cam.width, dtype=np.float64) + 0.5
    y_coords = np.arange(cam.height, dtype=np.float64) + 0.5
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    rays_x = (xx - cam.cx) / cam.focal
    rays_y = (yy - cam.cy) / cam.focal
    rays_z = np.ones_like(rays_x)
    rays = np.stack([rays_x, rays_y, rays_z], axis=-1).reshape(-1, 3)
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


def get_virtual_rotations(
    opts: PanoRenderOptions,
) -> list[npt.NDArray]:
    """Compute cam_from_pano rotation matrices for all virtual views."""
    cams_from_pano_r: list[npt.NDArray] = []
    yaws = np.linspace(0, 360, opts.num_steps_yaw, endpoint=False)
    for pitch_deg in opts.pitches_deg:
        yaw_offset = (360 / opts.num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = _rot_y(np.deg2rad(-yaw_deg)) @ _rot_x(np.deg2rad(-pitch_deg))
            cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


# ---------------------------------------------------------------------------
# Rig camera info
# ---------------------------------------------------------------------------


@dataclass
class RigCameraInfo:
    """Info for one virtual camera in the rig."""
    image_prefix: str
    cam_from_rig_rotation: npt.NDArray | None  # None for the reference camera
    ref_sensor: bool = False


def build_rig_cameras(
    cams_from_pano_rotation: Sequence[npt.NDArray],
    ref_idx: int = 0,
) -> list[RigCameraInfo]:
    cameras = []
    for idx, cam_from_pano_r in enumerate(cams_from_pano_rotation):
        is_ref = idx == ref_idx
        if is_ref:
            cam_from_rig_r = None
        else:
            cam_from_rig_r = cam_from_pano_r @ cams_from_pano_rotation[ref_idx].T
        cameras.append(RigCameraInfo(
            image_prefix=f"pano_camera{idx}/",
            cam_from_rig_rotation=cam_from_rig_r,
            ref_sensor=is_ref,
        ))
    return cameras


# ---------------------------------------------------------------------------
# Panorama rendering
# ---------------------------------------------------------------------------


class PanoProcessor:
    """Render virtual pinhole views from equirectangular panoramas."""

    def __init__(
        self,
        pano_image_dir: Path,
        output_image_dir: Path,
        mask_dir: Path | None,
        opts: PanoRenderOptions,
    ) -> None:
        self.opts = opts
        self.pano_image_dir = pano_image_dir
        self.output_image_dir = output_image_dir
        self.mask_dir = mask_dir

        self.cams_from_pano_r = get_virtual_rotations(opts)
        self.rig_cameras = build_rig_cameras(self.cams_from_pano_r)

        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_r, [0, 0, 1]
        )
        self.camera: VirtualCamera | None = None
        self._pano_size: tuple[int, int] | None = None
        self._rays_in_cam: npt.NDArray | None = None

    def process(self, pano_name: str) -> None:
        pano_path = self.pano_image_dir / pano_name
        pano_image = cv2.imread(str(pano_path))
        if pano_image is None:
            logger.warning("Cannot read %s, skipping", pano_path)
            return

        pano_h, pano_w = pano_image.shape[:2]

        if self.camera is None:
            self.camera = create_virtual_camera(
                pano_w, pano_h, self.opts.hfov_deg, self.opts.vfov_deg,
            )
            self._pano_size = (pano_w, pano_h)
            self._rays_in_cam = _get_virtual_camera_rays(self.camera)

        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_r):
            rays_in_pano = self._rays_in_cam @ cam_from_pano_r
            xy_in_pano = _spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                self.camera.height, self.camera.width, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5
            x_coords = xy_in_pano[:, :, 0]
            y_coords = xy_in_pano[:, :, 1]
            image = cv2.remap(
                pano_image, x_coords, y_coords,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
            )

            image_name = self.rig_cameras[cam_idx].image_prefix + pano_name
            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Masks are optional (COLMAP needs them, RealityScan does not)
            if self.mask_dir is not None:
                closest_camera = np.argmax(
                    self._rays_in_cam @ cam_from_pano_r @ self.cam_centers_in_pano.T, -1
                )
                mask = (
                    ((closest_camera == cam_idx) * 255)
                    .astype(np.uint8)
                    .reshape(self.camera.height, self.camera.width)
                )
                mask_name = f"{image_name}.png"
                mask_path = self.mask_dir / mask_name
                mask_path.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(mask_path), mask)


def render_perspective_images(
    pano_image_dir: Path,
    output_image_dir: Path,
    opts: PanoRenderOptions,
    mask_dir: Path | None = None,
) -> tuple[list[RigCameraInfo], VirtualCamera]:
    """Render perspective images and return rig cameras + virtual camera info."""
    pano_names = sorted(
        p.name for p in pano_image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTS
    )
    logger.info("Rendering %d panoramas to perspective views", len(pano_names))

    processor = PanoProcessor(pano_image_dir, output_image_dir, mask_dir, opts)

    for name in pano_names:
        processor.process(name)

    n_views = len(pano_names) * len(processor.cams_from_pano_r)
    logger.info("Rendered %d perspective images in %s", n_views, output_image_dir)
    return processor.rig_cameras, processor.camera
