"""auto_recon - Automated 360° video to Gaussian Splatting pipeline."""

from auto_recon.preprocessing import preprocess_video
from auto_recon.metashape_sfm import run_metashape_sfm
from auto_recon.lichtfeld_3dgs import run_lichtfeld_pipeline
from auto_recon.sfm_backend import SfMBackend, SfMResult

# Backend classes are imported lazily by run_pipeline.py to avoid pulling in
# heavy dependencies (Metashape, COLMAP, etc.) when they are not needed.
# They can still be imported explicitly:
#   from auto_recon.sfm_metashape import MetashapeSfMBackend
#   from auto_recon.sfm_colmap import ColmapSfMBackend
#   from auto_recon.sfm_realityscan import RealityScanSfMBackend

__all__ = [
    "preprocess_video",
    "run_metashape_sfm",
    "run_lichtfeld_pipeline",
    "SfMBackend",
    "SfMResult",
]
