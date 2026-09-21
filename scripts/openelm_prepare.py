"""Prepare pinned OpenELM-1.1B Stage A artifacts; no training or GPU execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdrm.pretrained.artifacts import (
    ArtifactSpec, CHECKPOINT_FILENAME, CHECKPOINT_SHA256, CHECKPOINT_SIZE,
    CHECKPOINT_URL, CORENET_REVISION, HF_REVISION, load_native_state_dict,
    prepare_checkpoint, prepare_sources, prepare_tokenizer, refresh_manifest,
    state_dict_manifest, validate_artifact, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--only", choices=("all", "sources", "checkpoint", "tokenizer", "inspect"), default="all")
    parser.add_argument("--tokenizer-model", type=Path, help="Existing official tokenizer.model; must match the pinned SHA256")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = {"schema_version": 1, "artifact_root": str(root), "corenet_revision": CORENET_REVISION, "hf_revision": HF_REVISION}
    if args.only in ("all", "sources"):
        print("Preparing pinned source snapshots", flush=True)
        output["sources"] = prepare_sources(root)
    if args.only in ("all", "tokenizer"):
        print("Preparing official Llama tokenizer", flush=True)
        output["tokenizer"] = prepare_tokenizer(root, args.tokenizer_model)
    if args.only in ("all", "checkpoint"):
        print("Preparing native 1.1B 300k model-only checkpoint", flush=True)
        output["checkpoint"] = prepare_checkpoint(root)
    if args.only == "inspect":
        from cdrm.pretrained.openelm import OpenELMConfig, OpenELMModel

        path = root / "checkpoint" / CHECKPOINT_FILENAME
        validate_artifact(path, ArtifactSpec(CHECKPOINT_URL, CHECKPOINT_SIZE, CHECKPOINT_SHA256))
        meta_model = OpenELMModel(OpenELMConfig.native_1_1b(), device="meta")
        shapes = {key: tuple(value.shape) for key, value in meta_model.state_dict().items()}
        state = load_native_state_dict(path, expected_shapes=shapes)
        output["checkpoint_inspection"] = state_dict_manifest(state)
        write_json(root / "checkpoint_inspection.json", output["checkpoint_inspection"])
    write_json(root / f"prepare_{args.only}.json", output)
    refresh_manifest(root)
    print(json.dumps({"status": "prepared", "mode": args.only, "artifact_root": str(root)}, sort_keys=True))


if __name__ == "__main__":
    main()
