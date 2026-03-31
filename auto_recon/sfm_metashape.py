"""Metashape SfM backend.

Wraps :func:`auto_recon.metashape_sfm.run_metashape_sfm` behind the
:class:`~auto_recon.sfm_backend.SfMBackend` interface so it can be used
interchangeably with other SfM engines.
"""

from __future__ import annotations

from pathlib import Path

from auto_recon.metashape_sfm import run_metashape_sfm
from auto_recon.sfm_backend import SfMBackend, SfMResult


class MetashapeSfMBackend(SfMBackend):
    """SfM backend using Agisoft Metashape (Spherical camera mode).

    Metashape natively supports equirectangular (360 degree) images via its
    Spherical sensor type, so no pre-conversion to pinhole images is needed.
    The equirectangular-to-perspective step happens *after* SfM in the
    pipeline.
    """

    @property
    def supports_equirectangular(self) -> bool:  # noqa: D102
        return True

    def run(self, image_dir: str | Path, output_dir: str | Path, **kwargs) -> SfMResult:
        """Run Metashape SfM on equirectangular images.

        Parameters
        ----------
        image_dir:
            Directory containing equirectangular images.
        output_dir:
            Directory where COLMAP sparse reconstruction will be written.

        Returns
        -------
        SfMResult
            ``is_pinhole`` is ``False`` because Metashape processes the
            raw equirectangular images.
        """
        result = run_metashape_sfm(image_dir=image_dir, output_dir=output_dir)
        return SfMResult(
            sparse_dir=result["sparse_dir"],
            images_dir=result["images_dir"],
            point_cloud=result["point_cloud"],
            is_pinhole=False,
        )
