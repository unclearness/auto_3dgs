# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fully automated pipeline from 360° video/images to 3D Gaussian Splatting output.

### Input
- 360° images (equirectangular format)
- For video: extract frames → process as images
- Sample data: mp4 files under `./data/20260330/`

### Pipeline Flow
```
360° video → Frame extraction (1s interval) → Preprocessing (nadir mask removal) → Metashape SfM → LichtFeld-Studio 3DGS → Final output
```

## Required Preprocessing

The camera is mounted on a selfie stick, so the photographer's head/body appears at the nadir (bottom) of equirectangular images. This must be removed with a fixed-angle mask.

### Optional Preprocessing
- Moving object detection/masking: SAM3
- Mask region inpainting: OpenCV
- General image processing: OpenCV

## SfM Backend

### Currently Used: Metashape
- Natively supports 360° equirectangular input (camera type: Spherical)
- Standard or Professional Edition (either works)
- Windows path: `C:/Program Files/Agisoft/Metashape/metashape.exe`
- Python module: `import Metashape` (via Metashape built-in Python or external Python)
- Exports camera parameters in XML/PLY format

### Future Options (not currently used)
- COLMAP / RealityScan: No direct 360° input. Requires cubemap conversion + rig configuration.
- `colmap_openmvs_pipeline/`, `realityscan_pipeline/` are previously created photogrammetry automation pipelines. For reference when switching SfM backends.

## 3DGS Backend: LichtFeld Studio

- Windows: pre-built binary at `LichtFeld-Studio_Windows_vXXX/`
- Linux: built from source at `LichtFeld-Studio/build/LichtFeld-Studio`
- CUDA-based 3D Gaussian Splatting. Supports MCMC/ADC/IGS+ strategies.

### CLI Examples
```bash
# Training
LichtFeld-Studio -d <data_path> -o <output_path> -i <iterations> --strategy mcmc

# Camera import (COLMAP-format sparse folder)
LichtFeld-Studio --import-cameras <sparse_folder> -d <data_path> -o <output_path>

# Viewer
LichtFeld-Studio -v <splat_file.ply>

# Resume from checkpoint
LichtFeld-Studio --resume <checkpoint_file>
```

### Key Options
- `--strategy`: mrnf (default, v0.5.1+) / mcmc / adc / igs+
- `--ppisp`: Enable per-camera appearance modeling (default: on, v0.5.1+)
- `--mask-mode`: none / segment / ignore / alpha_consistent
- `--steps-scaler`: Training steps scale factor (num_images / 300 as guideline)
- `--max-cap`: Maximum number of Gaussians for MRNF / MCMC / IGS+
- `--tile-mode`: Tile mode for memory efficiency (1/2/4)
- `--init`: Initialize from SfM point cloud (.ply)
- `--undistort`: Undistort images before training

## Development Conventions

- Python 3.12+ (`str | Path` union syntax)
- Timestamps in JST (UTC+9)
- Logs written to timestamped files in `YYYYMMDD_HHMMSS.log` format
- **All code, comments, documentation, and commit messages must be written in English**
