# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

360°動画/画像から3D Gaussian Splatting出力を完全自動で得るパイプライン。

### 入力
- 360°画像（equirectangularフォーマット）
- 動画の場合：フレーム切り出し→画像として処理
- サンプルデータ: `./data/20260330/` 以下のmp4

### パイプライン全体像
```
360°動画 → フレーム切り出し(1秒間隔) → 前処理(直下マスク除去) → Metashape SfM → LichtFeld-Studio 3DGS → 最終出力
```

## 必須の前処理

棒を持って撮影しているため、equirectangular画像の直下（底面）に撮影者の頭・体が写る。これを固定角度マスクで除去する必要がある。

### オプションの前処理
- 動く物体の検出・マスキング: SAM3
- マスク領域のインペインティング: OpenCV
- 一般的な画像処理: OpenCV

## SfMバックエンド

### 現在使用: Metashape
- 360° equirectangular入力を直接サポート（カメラタイプ: Spherical）
- Windows版 Standard Edition がインストール済み
- パス: `C:/Program Files/Agisoft/Metashape/metashape.exe`
- Pythonモジュール: `import Metashape` (Metashape内蔵Pythonまたは外部Pythonから利用可能)
- カメラパラメータをXML/PLY形式で出力可能

### 将来的な選択肢（現段階では使用しない）
- COLMAP / RealityScan: 360°直接入力不可。キューブマップ変換+リグ構成が必要
- `colmap_openmvs_pipeline/`, `realityscan_pipeline/` は過去に作成したフォトグラメトリ自動化パイプライン。SfMバックエンド切替時の参考用。

## 3DGSバックエンド: LichtFeld Studio

- ローカルの `LichtFeld-Studio_Windows_v0.5.0/` を使用
- 実行ファイル: `LichtFeld-Studio_Windows_v0.5.0/bin/LichtFeld-Studio.exe`
- CUDA実装の3D Gaussian Splatting。MCMC/ADC/IGS+ストラテジー対応

### CLI使用例
```bash
# トレーニング
LichtFeld-Studio.exe -d <data_path> -o <output_path> -i <iterations> --strategy mcmc

# カメラインポート (COLMAPフォーマットのsparseフォルダ)
LichtFeld-Studio.exe --import-cameras <sparse_folder> -d <data_path> -o <output_path>

# ビューア
LichtFeld-Studio.exe -v <splat_file.ply>

# チェックポイントからの再開
LichtFeld-Studio.exe --resume <checkpoint_file>
```

### 主要オプション
- `--mask-mode`: none / segment / ignore / alpha_consistent
- `--steps-scaler`: トレーニングステップのスケール係数（画像数÷300が目安）
- `--max-cap`: MCMCやIGS+のガウシアン最大数
- `--tile-mode`: メモリ効率のためのタイルモード (1/2/4)
- `--init`: SfMポイントクラウドからの初期化 (.ply)
- `--undistort`: トレーニング前に画像のアンディストーション

## 実装すべきもの

「360° Gaussian v1.3.0」相当のPythonベースの自動化パイプライン:
1. 360°動画からのフレーム切り出し（ブレの少ないフレームを選択）
2. equirectangular画像の底面マスク生成・適用
3. MetashapeでのSfM実行（Sphericalカメラ、アライメント、カメラパラメータ・ポイントクラウド出力）
4. Metashape出力をLichtFeld Studio入力形式に変換
5. LichtFeld StudioでのGaussian Splattingトレーニング実行
6. 最終出力の取得

## 参考情報
- 手順参考（ただし完全自動ではなく外部ツール使用）: https://qiita.com/Tks_Yoshinaga/items/896713e9c637cbfe35d1

## 開発規約

- Python 3.11+（`str | Path` 型ユニオン構文使用）
- タイムスタンプはJST (UTC+9)
- ログは `YYYYMMDD_HHMMSS.log` 形式でタイムスタンプ付きファイルに出力
