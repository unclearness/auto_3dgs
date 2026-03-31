"""Abstract SfM backend interface.

Defines the :class:`SfMResult` data class and the :class:`SfMBackend` abstract
base class so that multiple SfM engines (Metashape, COLMAP, RealityScan, etc.)
can be used interchangeably through a single pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SfMResult:
    """Result of a Structure-from-Motion run.

    Attributes
    ----------
    sparse_dir:
        Path to a COLMAP-format ``sparse/0/`` directory containing
        ``cameras.txt``, ``images.txt``, and ``points3D.txt``.
    images_dir:
        Directory with the images used during reconstruction (may be
        symlinks to the originals).
    point_cloud:
        Path to a PLY point cloud exported from the reconstruction.
    is_pinhole:
        ``True`` if the images are already pinhole projections and do
        **not** require equirectangular-to-perspective conversion.
        ``False`` when the reconstruction was done on equirectangular
        images (e.g. Metashape Spherical mode) and a conversion step
        is still needed downstream.
    """

    sparse_dir: Path
    images_dir: Path
    point_cloud: Path
    is_pinhole: bool

    def as_dict(self) -> dict[str, Path]:
        """Return a dict compatible with the legacy return format.

        Keys: ``"sparse_dir"``, ``"images_dir"``, ``"point_cloud"``.
        """
        return {
            "sparse_dir": self.sparse_dir,
            "images_dir": self.images_dir,
            "point_cloud": self.point_cloud,
        }


class SfMBackend(ABC):
    """Abstract base class for SfM backends.

    Subclasses must implement :meth:`run` and the
    :attr:`supports_equirectangular` property.
    """

    @abstractmethod
    def run(self, image_dir: str | Path, output_dir: str | Path, **kwargs) -> SfMResult:
        """Run Structure-from-Motion on the given images.

        Parameters
        ----------
        image_dir:
            Directory containing input images.
        output_dir:
            Directory where the COLMAP sparse reconstruction (and
            supporting files) will be written.
        **kwargs:
            Backend-specific options.

        Returns
        -------
        SfMResult
            The reconstruction result.
        """
        ...

    @property
    @abstractmethod
    def supports_equirectangular(self) -> bool:
        """Whether this backend can process equirectangular (360°) images.

        If ``True`` the pipeline will feed equirectangular images directly
        into SfM and convert to perspective afterwards.

        If ``False`` the pipeline must convert equirectangular images to
        perspective *before* running SfM.
        """
        ...
