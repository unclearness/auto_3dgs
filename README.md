# auto_3dgs

Fully automated pipeline for generating 3D Gaussian Splatting from 360° video.

## Pipeline Overview

```
360° Video (.mp4)
  │
  ├─ Stage 1: Preprocessing
  │    ├─ Frame extraction (ffmpeg, configurable FPS)
  │    ├─ Blur detection & rejection (Laplacian variance)
  │    ├─ Nadir mask (remove photographer from bottom)
  │    └─ SAM3 person masking (optional)
  │
  ├─ Stage 2: SfM (Structure from Motion)
  │    ├─ Metashape (recommended / native Spherical camera support)
  │    ├─ COLMAP
  │    └─ RealityScan
  │
  ├─ Stage 2.5: Equirectangular → Perspective conversion
  │
  └─ Stage 3: 3D Gaussian Splatting
       └─ LichtFeld Studio v0.5.1 (MRNF + PPISP / MCMC / IGS+ / ADC)
```

## Requirements

| | Windows | Linux (Ubuntu 24.04) |
|------|---------|---------------------|
| Python | 3.12+ | 3.12+ |
| CUDA Toolkit | 13.0+ | 13.0+ |
| NVIDIA GPU | Required | Required |
| Metashape | Standard Edition | Python wheel (auto-installed) |
| COLMAP | [Pre-built binary](https://colmap.github.io/install.html#pre-built-binaries) | [Build from source](https://colmap.github.io/install.html#build-from-source) |
| RealityScan | [Official installer](https://www.capturingreality.com/realityscan) | [Official installer](https://www.capturingreality.com/realityscan) |
| LichtFeld Studio | Pre-built binary (v0.5.1+) | Build from source (v0.5.1+) |

> **SfM backends**: Only the backend you choose (`--sfm-backend`) needs to be installed. Metashape is recommended; COLMAP is the fully open-source alternative.

## Setup

### Common

```bash
git clone --recursive https://github.com/<your-repo>/auto_3dgs.git
cd auto_3dgs
```

> If you forgot `--recursive`: `git submodule update --init --recursive`

### Windows

1. Install [Agisoft Metashape](https://www.agisoft.com/) Standard Edition
2. Place [LichtFeld Studio](https://lichtfeld.io/) pre-built binary. For example, `LichtFeld-Studio-windows-v0.5.1/`
3. *(If using COLMAP)* Download [COLMAP pre-built binary](https://colmap.github.io/install.html#pre-built-binaries) and add to PATH
4. *(If using RealityScan)* Install via [official installer](https://www.capturingreality.com/realityscan)
5. Install Python dependencies:

```bash
pip install uv
uv sync
```

### Linux (Ubuntu 24.04)

#### 1. Install Python dependencies

```bash
pip install uv
uv sync
```

#### 2. Install SfM backend (if not using Metashape)

- **COLMAP**: [Build from source](https://colmap.github.io/install.html#build-from-source) following the official instructions
- **RealityScan**: Install via [official installer](https://www.capturingreality.com/realityscan)

#### 3. Build LichtFeld Studio

The build script handles everything:

```bash
chmod +x scripts/build_lichtfeld_linux.sh
./scripts/build_lichtfeld_linux.sh
```

What the script does:

1. Install **GCC 14** and configure alternatives (`sudo` required)
2. Install **CMake 4.x** (3.30+ required)
3. Clone and bootstrap **vcpkg**
4. Auto-detect **CUDA Toolkit** (12.8+ required)
5. CMake configure & build **LichtFeld Studio**

To specify a custom CUDA path:

```bash
CUDA_ROOT=/usr/local/cuda-13.0 ./scripts/build_lichtfeld_linux.sh
```

After a successful build, the binary is located at `LichtFeld-Studio/build/LichtFeld-Studio`. The pipeline automatically uses this path on Linux.

> **Note**: The first build takes 20–30 minutes due to vcpkg dependency compilation.

## Usage

### Basic

```bash
uv run python run_pipeline.py "data/20260330/0330 (1).mp4" -o ./output/20260330
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--fps` | `1.0` | Frame extraction rate (fps) |
| `--sfm-backend` | `metashape` | SfM backend (`metashape` / `colmap` / `realityscan`) |
| `--iterations` | `30000` | 3DGS training iterations (scaled by steps_scaler) |
| `--strategy` | `mrnf` | 3DGS strategy (`mrnf` / `mcmc` / `igs+` / `adc`) |
| `--no-ppisp` | off | Disable PPISP per-camera appearance modeling |
| `--sam3` | `pinhole` | Person masking mode (see [SAM3 Masking Modes](#sam3-masking-modes)) |
| `--sam3-batch` | `4` | SAM3 batch size for pinhole mode |
| `--sam3-scale` | `1.0` | SAM3 input scale (e.g. `0.5` for half resolution) |
| `--from-stage` | `1` | Resume from stage (`1` / `2` / `3`) |
| `--mask-ratio` | `0.18` | Nadir mask height ratio (0-1) |
| `--blur-threshold` | `100.0` | Blur detection threshold (Laplacian variance) |
| `--lichtfeld` | Auto-detect | Path to LichtFeld Studio binary |

### Recommended Setup (Metashape)

Metashape natively supports equirectangular images as Spherical cameras, producing the most accurate SfM results. Requires a Metashape license (Standard or Professional, either works).

```bash
uv run python run_pipeline.py "video.mp4" -o ./output \
    --sfm-backend metashape
```

### Fully Open-Source Setup (COLMAP)

No commercial license required. COLMAP does not support equirectangular input directly, so the pipeline internally converts to perspective views before running SfM.

```bash
uv run python run_pipeline.py "video.mp4" -o ./output \
    --sfm-backend colmap
```

### More Examples

```bash
# Resume from Stage 3 (reuse previous Stage 1-2 output)
uv run python run_pipeline.py "video.mp4" -o ./output/existing --from-stage 3

# 2FPS frame extraction, MCMC strategy with 50000 iterations
uv run python run_pipeline.py "video.mp4" -o ./output \
    --fps 2.0 --iterations 50000 --strategy mcmc --no-ppisp

# Disable SAM3, nadir mask only
uv run python run_pipeline.py "video.mp4" -o ./output --sam3 off

# TRT mode (2x faster, requires engine build)
uv run python run_pipeline.py "video.mp4" -o ./output --sam3 trt
```

## SAM3 Masking Modes

The pipeline detects and masks people (moving objects that degrade SfM/3DGS) using [SAM3](https://github.com/facebookresearch/sam3).  Masks are passed to LichtFeld Studio via `--mask-mode ignore` so masked regions are excluded from training without black-filling the images.

| Mode | Flag | Speed (per equirect frame) | Mask Type | Quality |
|------|------|---------------------------|-----------|---------|
| **Pinhole** (default) | `--sam3 pinhole` | ~1.2 s | Pixel-precise | Best |
| Equirect | `--sam3 equirect` | ~0.5 s | Pixel-precise | Good (some distortion artifacts) |
| **TRT Pinhole** | `--sam3 trt` | ~0.5 s | Bounding boxes | Recall ~0.98 vs pixel masks |
| TRT Equirect | `--sam3 trt-equirect` | ~0.1 s | Bounding boxes | Recall ~0.98 vs pixel masks |
| Off | `--sam3 off` | — | — | — |

- **Pinhole modes** extract 12 perspective views per equirectangular frame, run detection on each view, then project masks back to equirectangular space.  More accurate because SAM3 works better on undistorted images.
- **TRT modes** use TensorRT-accelerated inference with bounding-box masks instead of pixel-precise segmentation.  ~2x faster with ~98% recall.  Slightly over-masks (bboxes include background) but this is acceptable for 3DGS training exclusion.
- **Equirect modes** run detection directly on the equirectangular image.  Faster but may miss small figures distorted near the poles.

### Building TRT Engines (one-time setup)

TRT modes require pre-built TensorRT engines specific to your GPU.  Build them once:

```bash
uv run python scripts/build_trt_engines.py
```

This produces two engine files in the project root:

| File | Size | Description |
|------|------|-------------|
| `hf_backbone_fp16.engine` | ~874 MB | HuggingFace ViT-H backbone (FP16) |
| `enc_dec_fp16.engine` | ~46 MB | Encoder-decoder head (FP16) |

Requirements: `tensorrt`, `onnx`, `onnxscript`, `transformers` (install with `uv add tensorrt onnx onnxscript onnxslim transformers`).  First build takes 5-10 minutes.  Engines must be rebuilt when switching GPUs.

## Output Directory Structure

```
output/
├── 01_preprocessing/
│   ├── frames_raw/          # Extracted frames (all)
│   ├── frames_masked/       # Sharp frames (blur-filtered)
│   ├── sam3_masks/          # Per-frame SAM3 person masks
│   └── masks_combined/      # Combined nadir + SAM3 masks (for 3DGS)
├── 02_sfm/
│   ├── sparse/0/            # COLMAP-format camera parameters
│   └── point_cloud.ply      # SfM point cloud
├── 02b_perspective/
│   ├── images/              # Perspective-projected images
│   ├── masks/               # Perspective-projected masks
│   └── sparse/0/            # Perspective camera parameters
├── 03_3dgs/
│   ├── lichtfeld_data/      # LichtFeld input data (symlinks)
│   ├── checkpoints/         # Training checkpoints
│   └── ...                  # 3DGS output files
└── pipeline_YYYYMMDD_HHMMSS.log
```

## License

- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) — GPL-3.0
- [Agisoft Metashape](https://www.agisoft.com/) — Commercial license (Standard or Professional, either works)
