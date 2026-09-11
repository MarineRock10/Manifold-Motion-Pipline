#!/usr/bin/env python3
"""Fetch the frozen-SONIC ONNX assets from HuggingFace (nvidia/GEAR-SONIC).

The models are not stored in git; run from the repository root:

    python3 download_from_hf.py            # download missing files
    python3 download_from_hf.py --force    # re-download everything

Behind a proxy export ``https_proxy``/``http_proxy``; for a mirror set
``HF_ENDPOINT`` (huggingface_hub honors all three).
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "nvidia/GEAR-SONIC"
OUTPUT_DIR = Path(__file__).resolve().parent / "gear_sonic_deploy"

# (filename in the HF repo, destination relative to gear_sonic_deploy/)
FILES = [
    ("model_encoder.onnx", "policy/release/model_encoder.onnx"),
    ("model_decoder.onnx", "policy/release/model_decoder.onnx"),
    ("observation_config.yaml", "policy/release/observation_config.yaml"),
    ("planner_sonic.onnx", "planner/target_vel/V2/planner_sonic.onnx"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the SONIC ONNX assets")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required: python3 -m pip install huggingface_hub")
        return 1

    for remote, local in FILES:
        target = OUTPUT_DIR / local
        if target.exists() and not args.force:
            print(f"[skip] {local} ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[fetch] {remote}")
        downloaded = Path(hf_hub_download(repo_id=REPO_ID, filename=remote, local_dir=OUTPUT_DIR))
        target.parent.mkdir(parents=True, exist_ok=True)
        if downloaded.resolve() != target.resolve():
            downloaded.replace(target)
        print(f"  -> {local} ({target.stat().st_size / 1e6:.1f} MB)")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
