#!/usr/bin/env python3
"""Build TensorRT engines for DART fast person detection.

Temporarily clones DART (https://github.com/mkturkcan/DART) to run the
ONNX export scripts, builds TRT FP16 engines, then removes the clone.
The upstream sam3 submodule is never modified.

Requirements:
    - tensorrt >= 10.9.0  (``uv add tensorrt``)
    - onnx, onnxscript     (``uv add onnx onnxscript onnxslim``)
    - transformers         (``uv add transformers``)

Usage:
    uv run python scripts/build_trt_engines.py

    # Custom output directory:
    uv run python scripts/build_trt_engines.py --output-dir ./engines

The script produces two engine files:
    hf_backbone_fp16.engine  (~874 MB) - ViT-H backbone (HuggingFace export)
    enc_dec_fp16.engine      (~46 MB)  - Encoder-decoder head

First build takes 5-10 minutes.  Engines are GPU-specific and must be
rebuilt when switching hardware.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DART_REPO = "https://github.com/mkturkcan/DART.git"


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
    parser.add_argument(
        "--keep-dart", action="store_true",
        help="Keep the DART clone after build (for debugging)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    # Locate SAM3 checkpoint (auto-downloads from HuggingFace)
    print("=== Locating SAM3 checkpoint ===")
    from huggingface_hub import hf_hub_download
    checkpoint = hf_hub_download("facebook/sam3", "sam3.pt")
    print(f"Checkpoint: {checkpoint}")

    # Create a temporary sample image for backbone tracing
    sample_image = output_dir / "_trt_sample.jpg"
    import numpy as np
    import cv2
    img = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    cv2.imwrite(str(sample_image), img)

    # Clone DART to a temporary directory
    dart_dir = Path(tempfile.mkdtemp(prefix="dart_"))
    print(f"\n=== Cloning DART to {dart_dir} ===")
    _run(["git", "clone", "--depth", "1", DART_REPO, str(dart_dir)])

    try:
        # DART's sam3 package must be on PYTHONPATH for the export scripts.
        # We prepend it so DART's sam3 (with trt module) takes priority
        # over the upstream sam3 during export only.
        dart_env = {
            "PYTHONPATH": str(dart_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "PYTHONIOENCODING": "utf-8",
        }

        with tempfile.TemporaryDirectory() as onnx_dir:
            onnx_dir = Path(onnx_dir)

            # Step 1: Export encoder-decoder ONNX
            enc_dec_onnx = onnx_dir / "enc_dec.onnx"
            enc_dec_engine = output_dir / "enc_dec_fp16.engine"
            print(f"\n=== Step 1: Export encoder-decoder ONNX ===")
            _run([
                python, "-m", "sam3.trt.export_enc_dec",
                "--checkpoint", checkpoint,
                "--output", str(enc_dec_onnx),
                "--max-classes", str(args.max_classes),
                "--imgsz", str(args.imgsz),
                "--no-validate",
            ], env_extra=dart_env)

            # Step 2: Build encoder-decoder TRT engine
            print(f"\n=== Step 2: Build encoder-decoder TRT engine ===")
            _run([
                python, "-m", "sam3.trt.build_engine",
                "--onnx", str(enc_dec_onnx),
                "--output", str(enc_dec_engine),
                "--fp16", "--mixed-precision", "none",
            ], env_extra=dart_env)
            print(f"  -> {enc_dec_engine} ({enc_dec_engine.stat().st_size / 1e6:.1f} MB)")

            # Step 3: Export HuggingFace backbone (ONNX + TRT in one step)
            print(f"\n=== Step 3: Export HuggingFace backbone ===")
            hf_script = dart_dir / "scripts" / "export_hf_backbone.py"
            _run([
                python, str(hf_script),
                "--image", str(sample_image),
                "--imgsz", str(args.imgsz),
            ], env_extra=dart_env, cwd=str(output_dir))

            # The HF script writes hf_backbone_fp16.engine in cwd
            bb_engine = output_dir / "hf_backbone_fp16.engine"
            if not bb_engine.exists():
                # Check if it was written in the original cwd
                alt = Path("hf_backbone_fp16.engine")
                if alt.exists():
                    shutil.move(str(alt), str(bb_engine))
            print(f"  -> {bb_engine} ({bb_engine.stat().st_size / 1e6:.1f} MB)")

    finally:
        # Cleanup
        sample_image.unlink(missing_ok=True)
        if not args.keep_dart:
            shutil.rmtree(dart_dir, ignore_errors=True)
            print(f"\nCleaned up DART clone")
        else:
            print(f"\nDART clone kept at: {dart_dir}")

    # Clean up ONNX artifacts that may have been written to output_dir
    for pattern in ["*.onnx", "*.onnx.data", "onnx_hf_backbone"]:
        for f in output_dir.glob(pattern):
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)

    print(f"\n=== Done ===")
    print(f"Engines in: {output_dir}")
    print(f"  {enc_dec_engine.name}")
    print(f"  {bb_engine.name}")
    print(f"\nUsage: uv run python run_pipeline.py video.mp4 -o output --sam3 trt")


def _run(
    cmd: list,
    env_extra: dict | None = None,
    cwd: str | None = None,
) -> None:
    """Run a subprocess, raising on failure."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
