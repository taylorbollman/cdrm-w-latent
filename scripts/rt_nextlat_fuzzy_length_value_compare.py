#!/usr/bin/env python3
"""Evaluate the completed value-route model against the retained baseline length probe."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import torch

from cdrm import rt_nextlat_task_embeddings as adapter
from scripts import rt_nextlat_a5_fuzzy_embedding_compare as paired
from scripts import rt_nextlat_a5_fuzzy_report as saved
from scripts import rt_nextlat_fuzzy_length_baseline as baseline_probe
from scripts import rt_nextlat_fuzzy_length_eval as common
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.rt_a5_common import configure_fp32_runtime
from scripts.rt_a5_train import atomic_json
from scripts.rt_nextlat_fuzzy_length_prepare import verify_manifest
from scripts.stage_a_common import require_cuda_container


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-fuzzy-value-baseline-length-comparison-v1"
QUALIFICATION = (
    "Two fixed 15,000-update checkpoints: uninjected baseline and upper-layer permanent-value addition "
    "v_t = W_V h_t + 0.01 P_e e_t. Both trained on mixed A5 T12/Fuzzy T400 with NextLat, paired initial "
    "shared tensors, identical ordered data and optimization. The value route adds 16,384 parameters. "
    "T400 metrics come from the saved training endpoints. T512/T1024 use the same held-out pools of "
    "1,280 examples; baseline measurements are reused from the verified earlier probe. "
    "This tests length generalization, not training at longer lengths. More tokens can add repeated "
    "mappings as well as longer retrieval distances. Single-seed development results do not establish "
    "a replicated architectural effect or reveal how the baseline represents embeddings. "
    "No training, final confirmation, A5 re-evaluation or autonomous latent rollout is performed."
)


def load_cached_baseline(directory, data_manifest_sha256):
    evidence = saved.read_json(directory / "evidence.json")
    published = saved.read_json(directory / "report.json")
    common.require(evidence.get("schema") == baseline_probe.SCHEMA and evidence.get("status") == "complete"
                   and all(published.get(key) == value for key, value in evidence.items()),
                   "Require the unchanged completed baseline length probe")
    common.require(evidence.get("data_manifest_sha256") == data_manifest_sha256,
                   "Cached baseline used different length-evaluation data")
    common.require(evidence.get("optimizer_updates") == 0 and evidence.get("confirmation_evaluated") is False
                   and evidence.get("latent_rollout_evaluated") is False,
                   "Cached baseline scope differs")
    arm = evidence["arms"]["baseline"]
    common.require(arm.get("model_unchanged") is True
                   and arm["model_sha256_before"] == arm["model_sha256_after"], "Cached baseline model-state audit differs")
    common.require(set(arm["metrics"]) == {"400", "512", "1024"}, "Cached baseline lengths differ")
    for metric in arm["metrics"].values():
        common.check_metric(metric)
    for name, digest in evidence["source_manifest"].items():
        common.require(saved.sha(directory / "source" / name) == digest
                       and saved.sha(ROOT / name) == digest, "Cached baseline source implementation changed")
    return copy.deepcopy(arm), {"directory": str(directory.resolve()),
        "evidence_sha256": saved.sha(directory / "evidence.json"), "report_sha256": saved.sha(directory / "report.json"),
        "report_wandb": published.get("report_wandb")}


def source_manifest(runs):
    sources = common.verify_live_sources(runs)
    sources.update(adapter.source_manifest())
    for name in ("scripts/rt_nextlat_fuzzy_length_baseline.py", "scripts/rt_nextlat_fuzzy_length_value_compare.py"):
        sources[name] = saved.sha(ROOT / name)
    return dict(sorted(sources.items()))


def markdown(summary):
    fields = ("answer_accuracy", "first_value_token_accuracy", "terminal_probe_accuracy",
              "terminal_first_value_token_accuracy", "answer_motif_exact_match", "sequence_exact_match")
    lines = ["# Value-route versus baseline: Fuzzy length generalization", "", QUALIFICATION, "",
             "| Arm | Length | Answer | First value | Terminal | Terminal first | Motif exact | Sequence exact |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    percent = lambda value: "n/a" if value is None else f"{100*value:.4f}%"
    for label, arm in summary["arms"].items():
        for length in (400, 512, 1024):
            metric = arm["metrics"][str(length)]
            lines.append(f"| {label} | {length} | " + " | ".join(percent(metric.get(key)) for key in fields) + " |")
    lines += ["", "## Value minus baseline", "", "Differences are percentage points at the fixed endpoints, not best-checkpoint selections.", "",
              "| Length | Answer | First value | Terminal | Terminal first | Motif exact | Sequence exact |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for length in (400, 512, 1024):
        a = summary["arms"]["baseline"]["metrics"][str(length)]
        b = summary["arms"]["value"]["metrics"][str(length)]
        lines.append(f"| {length} | " + " | ".join(f"{100*(b[key]-a[key]):+.4f}" for key in fields) + " |")
    lines += ["", "Near-zero sequence exactness is a floor and near-perfect answer accuracy is a ceiling. "
              "Interpret first-value, terminal, known-history and distance-bin results alongside them. "
              "A change in this aggregate outcome cannot identify the baseline's internal mechanism or rule out "
              "other embedding routes, gains or initialization seeds.", "",
              f"Evaluation microbatch {summary['eval_microbatch']}, FP32 eager, no TF32/autocast/compile/CUDA graphs. "
              "All longer-length scoring populations are identical between arms. The baseline cache is bound to its "
              "original report, checkpoint, data and frozen source hashes; its reconstructed checkpoint tensor digest "
              "matches the earlier probe. The value model and all executed source/data hashes are checked before and "
              "after evaluation. Native teacher-conditioned NextLat diagnostics are retained; no latent rollout occurs.", ""]
    for label, arm in summary["arms"].items():
        lines.append(f"- {label}: checkpoint `{arm['checkpoint']['sha256']}`; {arm['parameters']:,} parameters.")
    lines += ["", f"Data manifest SHA256: `{summary['data_manifest_sha256']}`.", ""]
    for name in summary["figures"]:
        lines += [f"![{name}]({name}.png)", f"[PDF]({name}.pdf)", ""]
    return "\n".join(lines)


def run(args):
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    common.require(type(args.eval_microbatch) is int and args.eval_microbatch > 0, "Positive evaluation microbatch required")
    common.require(not args.output.exists(), "Use a new paired length-evaluation output")
    manifest, datasets = verify_manifest(args.data, verify_sources=True)
    data_sha = saved.sha(args.data / "manifest.json")
    baseline, cache = load_cached_baseline(args.baseline_report, data_sha)
    base_run = baseline_probe.load_baseline(saved.local_path(baseline["directory"]))
    common.require(baseline["checkpoint"]["sha256"] == base_run["checkpoints"][15000]["sha256"],
                   "Baseline cache references a different endpoint checkpoint")
    cached_t400 = baseline["metrics"]["400"]
    common.require(all(cached_t400.get(key) == value for key, value in saved.endpoint_metrics(base_run)["fuzzy/dev"].items()),
                   "Cached T400 baseline metrics differ from the training endpoint")
    base_model, base_digest = common.restore_model(base_run)
    common.require(base_digest == baseline["model_sha256_before"], "Baseline checkpoint differs from cached evaluated tensors")
    del base_model
    value_run = saved.load_run(args.train)
    common.require(value_run["endpoint"] == 15000 and value_run["report"]["status"] == "complete",
                   "Require the completed value-route 15k checkpoint")
    audit = paired.compatibility(base_run, value_run)
    common.require(audit["variant"]["variant"] == "value", "Only the approved permanent-value variant is allowed")
    runs = {"baseline": base_run, "value": value_run}
    common.require(all(current["report"]["contract"]["runtime"] == runtime for current in runs.values()),
                   "Saved evaluation precision/runtime differs")
    sources = source_manifest(runs)
    args.output.mkdir(parents=True)
    for name, digest in sources.items():
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        common.require(saved.sha(destination) == digest, "Source changed during setup")
    atomic_json(args.output / "source-manifest.json", sources)
    summary = {"schema": SCHEMA, "status": "running", "qualification": QUALIFICATION,
        "training_update": 15000, "training_fuzzy_length": 400, "new_evaluation_lengths": [512, 1024],
        "eval_microbatch": args.eval_microbatch, "runtime": runtime, "hardware": hardware,
        "data_manifest": manifest, "data_manifest_sha256": data_sha, "baseline_cache": cache,
        "source_manifest": sources, "compatibility": audit, "optimizer_updates": 0,
        "confirmation_evaluated": False, "latent_rollout_evaluated": False, "arms": {"baseline": baseline}}
    atomic_json(args.output / "report.json", summary)
    tracker = (OnlineTracker(project="rt-nextlat-fuzzy-a5", entity="taylorbollman", output_dir=args.output,
                             name="d128-b2560-value-baseline-length-15k") if args.wandb else None)
    try:
        if tracker:
            tracker.start({"scope": QUALIFICATION, "checkpoint": value_run["checkpoints"][15000]["sha256"],
                           "baseline_cache": cache, "lengths": [512, 1024]})
        model, before = common.restore_model(value_run)
        model.to(device="cuda", dtype=torch.float32)
        metrics = {"400": copy.deepcopy(saved.endpoint_metrics(value_run)["fuzzy/dev"])}
        common.check_metric(metrics["400"])
        metrics["400"].update(sequence_length=400, source="saved training-endpoint development evaluation")
        metrics.update(common.evaluate_model(model, datasets, microbatch=args.eval_microbatch))
        checkpoint = value_run["checkpoints"][15000]
        for length in (512, 1024):
            metrics[str(length)]["checkpoint"] = checkpoint
            common.require(all(metrics[str(length)].get(f"{name}_tokens") == count for name, count in
                               manifest["lengths"][str(length)]["evaluation_regions"].items()),
                           "Evaluation denominators differ from frozen dataset regions")
        after = common.model_digest(model)
        common.require(before == after, "Evaluation changed model tensors")
        summary["arms"]["value"] = {"directory": value_run["directory"], "checkpoint": checkpoint,
            "parameters": value_run["report"]["parameter_count"], "mechanism": audit["variant"], "metrics": metrics,
            "model_sha256_before": before, "model_sha256_after": after, "model_unchanged": True,
            "lineage": value_run["lineage"], "training_wandb": value_run["report"].get("wandb")}
        common.validate_shared_counts(summary["arms"])
        common.require(saved.sha(args.data / "manifest.json") == data_sha and source_manifest(runs) == sources,
                       "Source or dataset manifest changed during evaluation")
        common.require(saved.sha(args.baseline_report / "evidence.json") == cache["evidence_sha256"]
                       and saved.sha(args.baseline_report / "report.json") == cache["report_sha256"],
                       "Baseline cache changed during evaluation")
        summary["figures"] = common.make_plots(summary, args.output)
        summary["figure_sha256"] = {f"{name}.{suffix}": saved.sha(args.output / f"{name}.{suffix}")
                                   for name in summary["figures"] for suffix in ("png", "pdf")}
        summary["status"] = "complete"
        (args.output / "report.md").write_text(markdown(summary))
        atomic_json(args.output / "evidence.json", summary)
        if tracker:
            import wandb
            for label, arm in summary["arms"].items():
                for length, metric in arm["metrics"].items():
                    tracker.log(scalar_metrics(metric, f"length_eval/{label}/length_{length}"))
            tracker.log({f"report/{name}": wandb.Image(str(args.output / f"{name}.png")) for name in summary["figures"]})
            tracker.summary({"results": {label: arm["metrics"] for label, arm in summary["arms"].items()},
                             "qualification": QUALIFICATION, "optimizer_updates": 0})
            artifact = wandb.Artifact(f"value-baseline-length-{tracker.record['run_id']}", type="development-report")
            for path in sorted(args.output.iterdir()):
                if path.is_file() and path.suffix in (".json", ".md", ".png", ".pdf"):
                    artifact.add_file(str(path), name=path.name)
            tracker._call("artifact logging", lambda: tracker._run.log_artifact(artifact))
            tracker.finish(succeeded=True)
            summary["report_wandb"] = tracker.record
        atomic_json(args.output / "report.json", summary)
        return summary
    except BaseException as error:
        summary.update(status="failed", error_type=type(error).__name__)
        atomic_json(args.output / "report.json", summary)
        if tracker:
            tracker.finish(succeeded=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-microbatch", type=int, default=64)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output": str(args.output), "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
