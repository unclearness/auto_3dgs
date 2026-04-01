#!/usr/bin/env python3
"""Build TensorRT engines for DART fast person detection.

Exports the HuggingFace SAM3 backbone and encoder-decoder as TRT FP16
engines.  These are used by the ``--sam3 trt`` pipeline mode for ~2x
faster person masking (bbox-based) compared to SAM3 PyTorch.

Requirements:
    - tensorrt >= 10.9.0  (``uv add tensorrt``)
    - onnx, onnxscript     (``uv add onnx onnxscript onnxslim``)
    - transformers         (``uv add transformers``)
    - A sample image for backbone tracing

Usage:
    uv run python scripts/build_trt_engines.py

    # Custom output directory:
    uv run python scripts/build_trt_engines.py --output-dir ./engines

    # Custom image size (must match pipeline usage):
    uv run python scripts/build_trt_engines.py --imgsz 1008

The script produces two engine files:
    hf_backbone_fp16.engine  (~874 MB) - ViT-H backbone
    enc_dec_fp16.engine      (~46 MB)  - Encoder-decoder head

First build takes 5-10 minutes.  Engines are hardware-specific and must
be rebuilt when switching GPUs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TRT engines for DART fast person detection",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Directory to write engine files (default: project root)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Input resolution for backbone (default: 1008)",
    )
    parser.add_argument(
        "--max-classes", type=int, default=1,
        help="Max classes for enc-dec engine (default: 1, person only)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    # Download checkpoint if needed
    print("=== Locating SAM3 checkpoint ===")
    from huggingface_hub import hf_hub_download
    checkpoint = hf_hub_download("facebook/sam3", "sam3.pt")
    print(f"Checkpoint: {checkpoint}")

    # Create a temporary sample image for backbone export
    sample_image = output_dir / "_trt_sample.jpg"
    if not sample_image.exists():
        import numpy as np
        import cv2
        img = np.random.randint(0, 255, (1008, 1008, 3), dtype=np.uint8)
        cv2.imwrite(str(sample_image), img)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Export encoder-decoder ONNX
        enc_dec_onnx = tmpdir / "enc_dec.onnx"
        enc_dec_engine = output_dir / "enc_dec_fp16.engine"
        print(f"\n=== Step 1: Export encoder-decoder ONNX ===")
        _run([
            python, "-m", "sam3.trt.export_enc_dec",
            "--checkpoint", checkpoint,
            "--output", str(enc_dec_onnx),
            "--max-classes", str(args.max_classes),
            "--imgsz", str(args.imgsz),
            "--no-validate",
        ])

        # Step 2: Build encoder-decoder TRT engine
        print(f"\n=== Step 2: Build encoder-decoder TRT engine ===")
        _run([
            python, "-m", "sam3.trt.build_engine",
            "--onnx", str(enc_dec_onnx),
            "--output", str(enc_dec_engine),
            "--fp16", "--mixed-precision", "none",
        ])
        print(f"Encoder-decoder engine: {enc_dec_engine} ({enc_dec_engine.stat().st_size / 1e6:.1f} MB)")

        # Step 3: Export HuggingFace backbone ONNX + TRT
        # export_hf_backbone.py handles both ONNX export and TRT build
        print(f"\n=== Step 3: Export HuggingFace backbone ===")
        hf_script = Path(__file__).parent / "export_hf_backbone.py"
        if not hf_script.exists():
            print(f"ERROR: {hf_script} not found.")
            print("Download from: https://github.com/mkturkcan/DART/blob/main/scripts/export_hf_backbone.py")
            sys.exit(1)

        _run([
            python, str(hf_script),
            "--image", str(sample_image),
            "--imgsz", str(args.imgsz),
        ], env_extra={"PYTHONIOENCODING": "utf-8"})

        # Move backbone engine to output directory
        bb_engine_src = Path("hf_backbone_fp16.engine")
        bb_engine_dst = output_dir / "hf_backbone_fp16.engine"
        if bb_engine_src.exists() and bb_engine_src != bb_engine_dst:
            shutil.move(str(bb_engine_src), str(bb_engine_dst))
        print(f"Backbone engine: {bb_engine_dst} ({bb_engine_dst.stat().st_size / 1e6:.1f} MB)")

    # Cleanup
    sample_image.unlink(missing_ok=True)

    print(f"\n=== Done ===")
    print(f"Engines in: {output_dir}")
    print(f"  {enc_dec_engine.name}")
    print(f"  {bb_engine_dst.name}")
    print(f"\nUsage: uv run python run_pipeline.py video.mp4 -o output --sam3 trt")


def _run(cmd: list, env_extra: dict | None = None) -> None:
    """Run a subprocess, raising on failure."""
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
