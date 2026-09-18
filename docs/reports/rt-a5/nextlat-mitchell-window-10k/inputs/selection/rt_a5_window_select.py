"""Select positions for the authorized window2 pilot from fixed 10k evidence.

Standard-library, read-only artifact analysis except its new selection record.
No training, model evaluation or checkpoint modification occurs here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / ".runtime/rt-a5/20260911T191702Z-nextlat/train-rt-nextlat"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def local(path):
    return Path(str(path).replace("/workspace/cdrm-w-latent", str(ROOT)))


def read_arm(directory, expected_schema):
    directory = Path(directory).resolve()
    report = json.loads((directory / "report.json").read_text())
    if (report["schema"] != expected_schema or report["status"] != "complete"
            or report["completed_updates"] != 10000 or report["start_update"] != 0
            or report["endpoint"] != 10000 or report["parent_checkpoint"] is not None
            or report["confirmation_evaluated"] or report["latent_rollout_evaluated"]
            or report["wandb"]["status"] != "synced"):
        raise ValueError("Selection requires complete fresh matched 10k development pilots")
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    if [r["update"] for r in history] != list(range(1, 10001)):
        raise ValueError("Incomplete training history")
    if any(r["examples_seen"] != r["update"] * 1024 for r in history):
        raise ValueError("Different training exposure")
    if history[-1]["order_chain"] != report["order_chain"]:
        raise ValueError("History/report order chain differs")
    metrics = {}
    for role, length in (("dev", 12), ("ood_dev", 36)):
        candidates = [m for m in report["evaluations"] if m["role"] == role and m["update"] == 10000]
        if len(candidates) != 1 or candidates[0]["rows"] != 102400 or candidates[0]["length"] != length:
            raise ValueError("Selection needs the full fixed-endpoint evaluation")
        m = candidates[0]
        if m.get("route") != "backbone_only":
            raise ValueError("Selection requires backbone-only inference")
        e = m["cumulative_prefix_exactness"]
        if (len(e) != length or any(not math.isfinite(v) or not 0 <= v <= 1 for v in e)
                or any(a < b for a, b in zip(e, e[1:]))):
            raise ValueError("Invalid exact-prefix curve")
        counts = [round(v * 102400) for v in e]
        if any(abs(v * 102400 - c) > 1e-6 for v, c in zip(e, counts)):
            raise ValueError("Exactness cannot be recovered as integer word counts")
        metrics[role] = {"metric": m, "counts": counts}
    checkpoints = [c for c in report["checkpoints"] if c["completed_updates"] == 10000]
    if len(checkpoints) != 1:
        raise ValueError("Expected exactly one fixed endpoint checkpoint")
    checkpoint = checkpoints[0]
    if sha(local(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("Selected checkpoint differs from its training record")
    for path, digest in report["source_files"].items():
        if sha(directory / "source" / path) != digest:
            raise ValueError("Training source snapshot changed")
    return {"directory": str(directory), "report": report, "history": history,
            "metrics": metrics, "checkpoint": checkpoint,
            "report_sha256": sha(directory / "report.json")}


def select(baseline_dir, sinusoidal_dir, protocol_path):
    baseline = read_arm(baseline_dir, "rt-a5-nextlat-training-v1")
    sinusoidal = read_arm(sinusoidal_dir, "rt-a5-window-training-v1")
    left, right = baseline["report"], sinusoidal["report"]
    if right["contract"]["experiment_config"]["position_encoding"] != "sinusoidal":
        raise ValueError("The new candidate must use sinusoids")
    if right["contract"]["experiment_config"]["second_layer_window"] is not None:
        raise ValueError("Choose positions before changing the attention window")
    if right["initialization"]["baseline_initialization"] != left["initialization"]:
        raise ValueError("Both arms must use the original paired initialization")
    if right["initialization"]["changed_parameter_slices"] != []:
        raise ValueError("Mitchell rollback must change no learned initial tensors")
    a, b = copy.deepcopy(left["contract"]), copy.deepcopy(right["contract"])
    for value in (a, b):
        value.pop("schema")
        value.pop("source_sha256")
    for key in ("experiment_config", "training_step", "evaluation", "one_step_diagnostics"):
        b.pop(key)
    if a["model_config"]["alibi"] is not True or b["model_config"]["alibi"] is not False:
        raise ValueError("Expected only the ALiBi-to-sinusoidal position difference")
    b["model_config"]["alibi"] = True
    if a != b:
        raise ValueError("Unplanned data/model/optimizer/loss/runtime contract differences")
    if any(right["source_files"].get(k) != v for k, v in left["source_files"].items()):
        raise ValueError("Historical execution source changed")
    if [r["order_chain"] for r in baseline["history"]] != [r["order_chain"] for r in sinusoidal["history"]]:
        raise ValueError("Pilot minibatch orders differ")
    protocol = json.loads(Path(protocol_path).read_text())
    selection = protocol["position_selection"]
    if (selection["checkpoint"] != 10000 or selection["role"] != "ood_dev"
            or selection["rows"] != 102400 or selection["exact_tie"] != "alibi"
            or selection["primary"] != "mean cumulative-prefix exactness E(t) over t=13..36 inclusive"):
        raise ValueError("Prospective selection criterion differs")
    arms = {}
    for position, arm in (("alibi", baseline), ("sinusoidal", sinusoidal)):
        count_sum = sum(arm["metrics"]["ood_dev"]["counts"][12:36])
        arms[position] = {"directory": arm["directory"], "report_sha256": arm["report_sha256"],
                          "checkpoint": arm["checkpoint"], "exact_prefix_count_sum_13_36": count_sum,
                          "denominator": 102400 * 24, "score": count_sum / (102400 * 24),
                          "endpoint_metrics": {role: value["metric"] for role, value in arm["metrics"].items()}}
    winner = "sinusoidal" if arms["sinusoidal"]["exact_prefix_count_sum_13_36"] > arms["alibi"]["exact_prefix_count_sum_13_36"] else "alibi"
    return {"schema": "rt-a5-window-position-selection-v1", "status": "selected",
            "criterion": selection, "arms": arms, "selected_position": winner,
            "selected_full_attention_directory": arms[winner]["directory"],
            "protocol_sha256": sha(protocol_path), "selector_sha256": sha(__file__),
            "matched_order_chain": left["order_chain"], "confirmation_evaluated": False,
            "next_run": {"position_encoding": winner, "second_layer_window": 2,
                         "initialization": "fresh original Mitchell", "updates": 10000}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path, default=BASELINE)
    p.add_argument("--sinusoidal-dir", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the original selection record")
    result = select(args.baseline_dir, args.sinusoidal_dir, args.protocol)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected_position": result["selected_position"],
                      "scores": {k: v["score"] for k, v in result["arms"].items()}}))


if __name__ == "__main__":
    main()
