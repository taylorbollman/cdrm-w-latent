#!/usr/bin/env python3
"""Immediate baseline-only Fuzzy length probe, without training or new variants."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import torch

from cdrm import rt_nextlat_task_embeddings as adapter
from scripts import rt_nextlat_fuzzy_length_eval as length_eval
from scripts import rt_nextlat_a5_fuzzy_report as saved
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.rt_a5_common import configure_fp32_runtime
from scripts.rt_a5_train import atomic_json
from scripts.rt_nextlat_fuzzy_length_prepare import verify_manifest
from scripts.stage_a_common import require_cuda_container


SCHEMA = "rt-nextlat-fuzzy-baseline-length-probe-v1"
ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = (
    "Baseline-only out-of-distribution length-generalization probe at the fixed 15,000-update checkpoint. "
    "This uninjected model trained on mixed A5 T12 and Fuzzy T400 with NextLat. T400 is its retained endpoint "
    "development evaluation; T512/T1024 are newly evaluated held-out development pools of 1,280 examples each. "
    "The longer pools are shared with the planned later architecture comparison, but no embedding variant "
    "is evaluated in this report. Longer sequences are not necessarily harder: repeated mappings may reduce "
    "effective retrieval difficulty. Native answer masks, history coverage, retrieval distances and previous "
    "occurrence counts are retained. No optimizer update, final confirmation, A5 re-evaluation or autonomous "
    "latent rollout is performed. These single-seed observations screen length difficulty; they do not compare architectures."
)


def load_baseline(directory):
    current = saved.load_run(directory)
    contract = current["report"]["contract"]
    config = contract["model_config"]
    length_eval.require(current["endpoint"] == 15000 and current["report"]["status"] == "complete",
                        "Require the completed baseline 15k checkpoint")
    length_eval.require("embedding_injection" not in config, "Baseline probe cannot evaluate an embedding variant")
    length_eval.require(contract["mode"] == "mixed" and contract["batch_per_task"] == 2560
                        and contract["streams"]["fuzzy"]["length"] == 400
                        and config["backbone"]["d_model"] == 128
                        and config["backbone"]["n_layers"] == 2
                        and config["backbone"]["max_sequence_length"] >= 1024,
                        "Unexpected baseline training architecture or data contract")
    metric = saved.endpoint_metrics(current)["fuzzy/dev"]
    length_eval.require(metric["checkpoint"]["sha256"] == current["checkpoints"][15000]["sha256"],
                        "Saved T400 evaluation belongs to another checkpoint")
    length_eval.check_metric(metric)
    return current


def source_manifest(current):
    sources = length_eval.verify_live_sources({"baseline": current})
    for name, digest in adapter.source_manifest().items():
        length_eval.require(name not in sources or sources[name] == digest, "Inconsistent adapter source closure")
        sources[name] = digest
    sources["scripts/rt_nextlat_fuzzy_length_baseline.py"] = saved.sha(Path(__file__))
    return dict(sorted(sources.items()))


def markdown(summary):
    arm = summary["arms"]["baseline"]
    lines = ["# Baseline Fuzzy Recall length probe", "", QUALIFICATION, "",
             f"The same saved baseline checkpoint is used at every length: **15,000 optimizer updates**, "
             f"{arm['parameters']:,} parameters. This probe performs **zero training updates**.", "",
             "| Length | Answer | Motif exact | Sequence exact | First value | Terminal | Known history |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    percent = lambda value: "n/a" if value is None else f"{100*value:.4f}%"
    for length in (400, 512, 1024):
        metric = arm["metrics"][str(length)]
        fields = ("answer_accuracy", "answer_motif_exact_match", "sequence_exact_match",
                  "first_value_token_accuracy", "terminal_probe_accuracy", "known_history_accuracy")
        lines.append(f"| {length} | " + " | ".join(percent(metric.get(key)) for key in fields) + " |")
    lines += ["", "## Scoring populations", "",
              "The points at different lengths use different development examples. Native teacher-forced "
              "answer accuracy and whole-sequence exactness must not be interpreted as interchangeable. "
              "A zero-count distance bin has undefined accuracy, not 0%.", "",
              "| Length | Scored answers | First values | Known history | Unavailable history | History coverage |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for length in (400, 512, 1024):
        metric = arm["metrics"][str(length)]
        lines.append(f"| {length} | {metric['answer_tokens']:,} | {metric['first_value_token_tokens']:,} | "
                     f"{metric['known_history_tokens']:,} | {metric['unavailable_history_tokens']:,} | "
                     f"{percent(metric['oracle_coverage'])} |")
    lines += ["", "## Provenance", "",
              f"Checkpoint SHA256: `{arm['checkpoint']['sha256']}`. "
              f"Data manifest SHA256: `{summary['data_manifest_sha256']}`.", "",
              f"Evaluation uses FP32 eager execution and microbatch {summary['eval_microbatch']}; "
              "TF32, autocast, compilation and CUDA graphs remain off. The original evaluator supplies native "
              "Fuzzy metrics plus a teacher-conditioned NextLat loss diagnostic. Canonical initialization, "
              "strict checkpoint restoration, finite FP32 state and frozen source hashes were checked. "
              "Model tensors, source hashes and the data manifest were checked again after evaluation. "
              "Complete scoring counts and retrieval-distance/previous-occurrence bins are in evidence.json.", "",
              "This probe does not establish a winning embedding design. If longer lengths remain near ceiling, "
              "they may offer limited discrimination for the planned comparison.", ""]
    for name in summary["figures"]:
        lines += [f"![{name}]({name}.png)", f"[PDF]({name}.pdf)", ""]
    return "\n".join(lines)


def run(args):
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    length_eval.require(type(args.eval_microbatch) is int and args.eval_microbatch > 0,
                        "Positive evaluation microbatch required")
    length_eval.require(not args.output.exists(), "Use a new baseline-probe output directory")
    current = load_baseline(args.train)
    length_eval.require(current["report"]["contract"]["runtime"] == runtime, "Saved FP32 runtime differs")
    sources = source_manifest(current)
    manifest, datasets = verify_manifest(args.data, verify_sources=True)
    length_eval.require(set(datasets) == {512, 1024}, "Require the frozen T512/T1024 development pools")
    args.output.mkdir(parents=True)
    for name, digest in sources.items():
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        length_eval.require(saved.sha(destination) == digest, "Source changed during setup")
    atomic_json(args.output / "source-manifest.json", sources)
    summary = {"schema": SCHEMA, "status": "running", "qualification": QUALIFICATION,
               "training_update": 15000, "training_fuzzy_length": 400, "new_evaluation_lengths": [512, 1024],
               "eval_microbatch": args.eval_microbatch, "data_directory": str(args.data.resolve()),
               "data_manifest": manifest, "data_manifest_sha256": saved.sha(args.data / "manifest.json"),
               "runtime": runtime, "hardware": hardware, "source_manifest": sources,
               "optimizer_updates": 0, "confirmation_evaluated": False, "latent_rollout_evaluated": False,
               "arms": {}}
    atomic_json(args.output / "report.json", summary)
    tracker = (OnlineTracker(project="rt-nextlat-fuzzy-a5", entity="taylorbollman", output_dir=args.output,
                             name="d128-b2560-baseline-length-probe-15k") if args.wandb else None)
    try:
        if tracker:
            tracker.start({"scope": QUALIFICATION, "lengths": [512, 1024], "training_update": 15000,
                           "checkpoint_sha256": current["checkpoints"][15000]["sha256"]})
        model, before = length_eval.restore_model(current)
        model.to(device="cuda", dtype=torch.float32)
        metrics = {"400": copy.deepcopy(saved.endpoint_metrics(current)["fuzzy/dev"])}
        metrics["400"].update(sequence_length=400, source="saved training-endpoint development evaluation")
        metrics.update(length_eval.evaluate_model(model, datasets, microbatch=args.eval_microbatch))
        checkpoint = current["checkpoints"][15000]
        for length in (512, 1024):
            metrics[str(length)]["checkpoint"] = checkpoint
            regions = manifest["lengths"][str(length)]["evaluation_regions"]
            length_eval.require(all(metrics[str(length)].get(f"{name}_tokens") == count for name, count in regions.items()),
                                "Evaluation denominators differ from frozen data regions")
        after = length_eval.model_digest(model)
        length_eval.require(before == after, "Evaluation changed baseline model tensors")
        summary["arms"]["baseline"] = {"directory": current["directory"], "checkpoint": checkpoint,
            "parameters": current["report"]["parameter_count"], "metrics": metrics,
            "mechanism": {"variant": "baseline", "description": "No embedding injection"},
            "model_sha256_before": before, "model_sha256_after": after, "model_unchanged": True,
            "lineage": current["lineage"], "training_wandb": current["report"].get("wandb")}
        length_eval.require(saved.sha(args.data / "manifest.json") == summary["data_manifest_sha256"],
                            "Data manifest changed during evaluation")
        length_eval.require(source_manifest(current) == sources, "Executed source changed during evaluation")
        summary["figures"] = length_eval.make_plots(summary, args.output)
        summary["figure_sha256"] = {f"{name}.{suffix}": saved.sha(args.output / f"{name}.{suffix}")
                                   for name in summary["figures"] for suffix in ("png", "pdf")}
        summary["status"] = "complete"
        (args.output / "report.md").write_text(markdown(summary))
        atomic_json(args.output / "evidence.json", summary)
        if tracker:
            import wandb
            for length, metric in metrics.items():
                tracker.log(scalar_metrics(metric, f"length_eval/baseline/length_{length}"))
            tracker.log({f"report/{name}": wandb.Image(str(args.output / f"{name}.png")) for name in summary["figures"]})
            tracker.summary({"baseline_metrics": metrics, "qualification": QUALIFICATION, "optimizer_updates": 0})
            artifact = wandb.Artifact(f"baseline-length-probe-{tracker.record['run_id']}", type="development-report")
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-microbatch", type=int, default=64)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output": str(args.output), "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
