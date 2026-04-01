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
| `--sam3` | `pinhole` | SAM3 person masking (`pinhole` / `equirect` / `off`) |
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
```

## Output Directory Structure

```
output/
├── 01_preprocessing/
│   ├── frames/              # Extracted frames
│   ├── frames_masked/       # Frames with masks applied
│   └── masks/               # SAM3 mask images
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
