#!/usr/bin/env python3
"""Prepare the pinned O4 CodeSearchNet/WikiText corpus; CPU container only."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.lm_data import prepare_lm_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-token-budget", type=int, default=25_000_000)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    manifest = prepare_lm_data(**vars(args))
    print({"manifest": str(args.output_dir / "manifest.json"), "splits": manifest["splits"]}, flush=True)


if __name__ == "__main__":
    main()
