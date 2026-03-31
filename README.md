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
       └─ LichtFeld Studio (IGS+ / MCMC / ADC)
```

## Requirements

| | Windows | Linux (Ubuntu 24.04) |
|------|---------|---------------------|
| Python | 3.12+ | 3.12+ |
| CUDA Toolkit | 12.8+ | 12.8+ |
| NVIDIA GPU | Required | Required |
| Metashape | Standard Edition | Python wheel (auto-installed) |
| LichtFeld Studio | Pre-built binary | Build from source |

## Setup

### Common

```bash
git clone --recursive https://github.com/<your-repo>/auto_3dgs.git
cd auto_3dgs
```

> If you forgot `--recursive`: `git submodule update --init --recursive`

### Windows

1. Install [Agisoft Metashape](https://www.agisoft.com/) Standard Edition
2. Place [LichtFeld Studio v0.5.0](https://lichtfeld-studio.com/) pre-built binary in `LichtFeld-Studio_Windows_v0.5.0/`
3. Install Python dependencies:

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

#### 2. Build LichtFeld Studio

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
| `--iterations` | `30000` | 3DGS training iterations |
| `--strategy` | `igs+` | 3DGS optimization strategy (`igs+` / `mcmc` / `adc`) |
| `--sam3` | `pinhole` | SAM3 person masking (`pinhole` / `equirect` / `off`) |
| `--from-stage` | `1` | Resume from stage (`1` / `2` / `3`) |
| `--mask-ratio` | `0.18` | Nadir mask height ratio (0–1) |
| `--blur-threshold` | `100.0` | Blur detection threshold (Laplacian variance) |
| `--lichtfeld` | Auto-detect | Path to LichtFeld Studio binary |

### Recommended Setup (Metashape)

Metashape natively supports equirectangular images as Spherical cameras, producing the most accurate SfM results. Requires a Metashape license (Standard or Professional, either works).

```bash
uv run python run_pipeline.py "video.mp4" -o ./output \
    --sfm-backend metashape --sam3 pinhole --strategy igs+
```

### Fully Open-Source Setup (COLMAP)

No commercial license required. COLMAP does not support equirectangular input directly, so the pipeline internally converts to perspective views before running SfM.

```bash
uv run python run_pipeline.py "video.mp4" -o ./output \
    --sfm-backend colmap --sam3 pinhole --strategy igs+
```

### More Examples

```bash
# Resume from Stage 3 (reuse previous Stage 1-2 output)
uv run python run_pipeline.py "video.mp4" -o ./output/existing --from-stage 3

# 2FPS frame extraction, MCMC strategy with 50000 iterations
uv run python run_pipeline.py "video.mp4" -o ./output \
    --fps 2.0 --iterations 50000 --strategy mcmc

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
