# auto_3dgs

360° 動画から 3D Gaussian Splatting を全自動で生成するパイプライン。

## パイプライン概要

```
360° 動画 (.mp4)
  │
  ├─ Stage 1: 前処理
  │    ├─ フレーム切り出し (ffmpeg, 指定FPS)
  │    ├─ ブレ検出・除去 (Laplacian variance)
  │    ├─ 底面マスク (撮影者除去)
  │    └─ SAM3 人物マスキング (オプション)
  │
  ├─ Stage 2: SfM (Structure from Motion)
  │    ├─ Metashape (推奨 / Spherical カメラ対応)
  │    ├─ COLMAP
  │    └─ RealityScan
  │
  ├─ Stage 2.5: Equirectangular → Perspective 変換
  │
  └─ Stage 3: 3D Gaussian Splatting
       └─ LichtFeld Studio (IGS+ / MCMC / ADC)
```

## 動作環境

| 要件 | Windows | Linux (Ubuntu 24.04) |
|------|---------|---------------------|
| Python | 3.12+ | 3.12+ |
| CUDA Toolkit | 12.8+ | 12.8+ |
| NVIDIA GPU | 必須 | 必須 |
| Metashape | Standard Edition | Python wheel (自動インストール) |
| LichtFeld Studio | プリビルドバイナリ | ソースからビルド |

## セットアップ

### 共通手順

```bash
git clone --recursive https://github.com/<your-repo>/auto_3dgs.git
cd auto_3dgs
```

> `--recursive` を忘れた場合: `git submodule update --init --recursive`

### Windows

1. [Agisoft Metashape](https://www.agisoft.com/) Standard Edition をインストール
2. [LichtFeld Studio v0.5.0](https://lichtfeld-studio.com/) プリビルドバイナリを `LichtFeld-Studio_Windows_v0.5.0/` に配置
3. Python 依存のインストール:

```bash
pip install uv
uv sync
```

### Linux (Ubuntu 24.04)

#### 1. Python 依存のインストール

```bash
pip install uv
uv sync
```

#### 2. LichtFeld Studio のビルド

ビルドスクリプトが一括で実行します:

```bash
chmod +x scripts/build_lichtfeld_linux.sh
./scripts/build_lichtfeld_linux.sh
```

スクリプトが行う処理:

1. **GCC 14** のインストールと設定 (`sudo` 必要)
2. **CMake 4.x** のインストール (3.30+ 必須)
3. **vcpkg** のクローンとブートストラップ
4. **CUDA Toolkit** の自動検出 (12.8+ 必須)
5. **LichtFeld Studio** の CMake configure & ビルド

CUDA パスを明示指定する場合:

```bash
CUDA_ROOT=/usr/local/cuda-13.0 ./scripts/build_lichtfeld_linux.sh
```

ビルド成功後、バイナリは `LichtFeld-Studio/build/LichtFeld-Studio` に生成されます。パイプラインは Linux 上では自動的にこのパスを参照します。

> **注意**: 初回ビルドでは vcpkg による依存ライブラリのビルドに 20〜30 分かかります。

## 使い方

### 基本

```bash
uv run python run_pipeline.py "data/20260330/0330 (1).mp4" -o ./output/20260330
```

### 主要オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--fps` | `1.0` | フレーム抽出レート (fps) |
| `--sfm-backend` | `metashape` | SfM バックエンド (`metashape` / `colmap` / `realityscan`) |
| `--iterations` | `30000` | 3DGS トレーニングイテレーション数 |
| `--strategy` | `igs+` | 3DGS 最適化戦略 (`igs+` / `mcmc` / `adc`) |
| `--sam3` | `pinhole` | SAM3 人物マスキング (`pinhole` / `equirect` / `off`) |
| `--from-stage` | `1` | 指定ステージから再開 (`1` / `2` / `3`) |
| `--mask-ratio` | `0.18` | 底面マスク高さ比率 (0〜1) |
| `--blur-threshold` | `100.0` | ブレ検出閾値 (Laplacian variance) |
| `--lichtfeld` | 自動検出 | LichtFeld Studio バイナリのパス |

### 実行例

```bash
# 2FPS で COLMAP バックエンド、SAM3 無効
uv run python run_pipeline.py "video.mp4" -o ./output --fps 2.0 --sfm-backend colmap --sam3 off

# Stage 3 から再開 (前回の Stage 1-2 出力を再利用)
uv run python run_pipeline.py "video.mp4" -o ./output/existing --from-stage 3

# イテレーション数とストラテジーを指定
uv run python run_pipeline.py "video.mp4" -o ./output --iterations 50000 --strategy mcmc
```

## 出力ディレクトリ構造

```
output/
├── 01_preprocessing/
│   ├── frames/              # 抽出フレーム
│   ├── frames_masked/       # マスク適用済みフレーム
│   └── masks/               # SAM3 マスク画像
├── 02_sfm/
│   ├── sparse/0/            # COLMAP 形式カメラパラメータ
│   └── point_cloud.ply      # SfM ポイントクラウド
├── 02b_perspective/
│   ├── images/              # Perspective 変換画像
│   ├── masks/               # Perspective 変換マスク
│   └── sparse/0/            # Perspective カメラパラメータ
├── 03_3dgs/
│   ├── lichtfeld_data/      # LichtFeld 入力データ (symlink)
│   ├── checkpoints/         # トレーニングチェックポイント
│   └── ...                  # 3DGS 出力ファイル
└── pipeline_YYYYMMDD_HHMMSS.log
```

## ライセンス

- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) — GPL-3.0
- [Agisoft Metashape](https://www.agisoft.com/) — 商用ライセンス (別途必要)
