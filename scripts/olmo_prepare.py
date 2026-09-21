"""Prepare immutable original OLMo-1B step-60k weights/tokenizer; no GPU execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdrm.pretrained.artifacts import state_dict_manifest, write_json
from cdrm.pretrained.olmo_artifacts import prepare_artifacts, load_native_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true", help="Also strictly load and inspect all native FP32 tensors on CPU")
    args = parser.parse_args()
    manifest = prepare_artifacts(args.artifact_root)
    result = {"status": "prepared", "revision": manifest["revision"], "artifact_root": str(args.artifact_root.resolve())}
    if args.inspect:
        state = load_native_state_dict(args.artifact_root)
        inspection = state_dict_manifest(state)
        write_json(args.artifact_root / "checkpoint-inspection.json", inspection)
        result["inspection"] = {"tensors": len(state), "parameters": sum(x.numel() for x in state.values())}
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
