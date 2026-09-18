#!/usr/bin/env python3
"""Evaluate fixed mixed-task checkpoints on shared longer Fuzzy development sets.

Training remains at T400. The T400 point is the saved endpoint evaluation;
T512/T1024 are new length-generalization measurements, not harder-training runs.
This command requires the project CUDA container and never falls back to CPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch

from cdrm.rt_nextlat_task_embeddings import build_model
from scripts import rt_nextlat_a5_fuzzy_embedding_compare as paired
from scripts import rt_nextlat_a5_fuzzy_report as saved
from scripts import rt_nextlat_fuzzy_train as fuzzy
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.rt_a5_common import configure_fp32_runtime
from scripts.rt_a5_train import atomic_json, json_value
from scripts.stage_a_common import require_cuda_container


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-fuzzy-length-evaluation-v1"
LENGTHS = (512, 1024)
EXPECTED_UPDATE = 15000
EXPECTED_ROWS = 1280
METRICS = (
    ("answer_accuracy", "Answer tokens"),
    ("answer_motif_exact_match", "Answer motifs"),
    ("sequence_exact_match", "All answers in a sequence"),
    ("first_value_token_accuracy", "First value token"),
    ("terminal_probe_accuracy", "Terminal probe"),
    ("terminal_first_value_token_accuracy", "Terminal first value"),
    ("known_history_accuracy", "Known-history answers"),
    ("continuation_value_token_accuracy", "Value continuation"),
    ("oracle_coverage", "History available (data coverage)"),
)
QUALIFICATION = (
    "Shared held-out development data for out-of-distribution length generalization, not a new training condition. "
    "All four checkpoints were trained on mixed A5 T12 and Fuzzy T400 with NextLat. "
    "Longer sequences are not necessarily harder: the small vocabulary can yield repeated mappings and shorter "
    "effective retrieval distances. Native teacher-forced answer tokens, first-value/terminal metrics, "
    "available-history coverage, prior occurrence counts and distance bins are reported separately. "
    "T400 is the retained training-endpoint development evaluation; T512/T1024 use separately generated pools. "
    "Length points are not extensions of identical examples. One initialization and reused development "
    "selection do not establish a replicated effect. No final confirmation, optimizer update, A5 re-evaluation "
    "or autonomous latent rollout is performed."
)


def require(condition, message):
    saved.require(condition, message)


def finite_tree(value, *, path="checkpoint"):
    """Inspect saved model/Adam tensors without constructing an optimizer."""
    if isinstance(value, torch.Tensor):
        require(torch.isfinite(value).all().item(), f"Nonfinite saved tensor: {path}")
        if value.is_floating_point():
            require(value.dtype == torch.float32, f"Non-FP32 saved tensor: {path}")
    elif isinstance(value, dict):
        for name, item in value.items():
            finite_tree(item, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, path=f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite saved scalar: {path}")


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def check_metric(metric, *, expected_rows=EXPECTED_ROWS):
    require(metric.get("evaluated_rows", metric.get("examples")) == expected_rows
            and metric.get("examples") == expected_rows, "Incomplete Fuzzy evaluation pool")
    for key, value in metric.items():
        if isinstance(value, float):
            require(math.isfinite(value), f"Nonfinite evaluation metric: {key}")
        if key.endswith("_tokens") and not key.endswith("_ce_tokens"):
            region = key[:-len("_tokens")]
            count, correct, accuracy = value, metric.get(f"{region}_correct"), metric.get(f"{region}_accuracy")
            require(type(count) is int and count >= 0 and type(correct) is int and 0 <= correct <= count,
                    f"Invalid metric counts: {region}")
            require(accuracy is None if count == 0 else isinstance(accuracy, (int, float))
                    and math.isclose(accuracy, correct/count, abs_tol=1e-12),
                    f"Accuracy/count mismatch: {region}")
    require(metric.get("answer_tokens", 0) > 0, "Empty answer evaluation")
    for count_key, total_key, ratio in (("exact_sequences", "examples", "sequence_exact_match"),
                                         ("exact_answer_motifs", "answer_motifs", "answer_motif_exact_match")):
        count, total = metric.get(count_key), metric.get(total_key)
        require(type(count) is int and type(total) is int and total > 0 and 0 <= count <= total
                and math.isclose(metric[ratio], count/total, abs_tol=1e-12), "Exactness count mismatch")
    require(math.isclose(metric["oracle_coverage"], metric["known_history_tokens"] / metric["answer_tokens"],
                         abs_tol=1e-12), "History coverage/count mismatch")
    return metric


def verify_live_sources(runs):
    sources = {}
    for run in runs.values():
        for name, expected in run["sources"].items():
            path = ROOT / name
            require(path.is_file() and saved.sha(path) == expected,
                    f"Executed source differs from the frozen checkpoint implementation: {name}")
            require(name not in sources or sources[name] == expected, "Arms have inconsistent source closures")
            sources[name] = expected
    additions = (
        "scripts/rt_nextlat_fuzzy_length_eval.py", "scripts/rt_nextlat_fuzzy_length_prepare.py",
        "scripts/rt_nextlat_a5_fuzzy_embedding_compare.py", "scripts/rt_nextlat_a5_fuzzy_report.py",
        "scripts/rt_a5_nextlat_report.py", "cdrm/rt_nextlat_task_embeddings.py",
    )
    for name in additions:
        sources[name] = saved.sha(ROOT / name)
    return dict(sorted(sources.items()))


def load_inputs(specifications):
    require(len(specifications) == 4 and len({label for label, _ in specifications}) == 4,
            "Require four uniquely labeled completed training arms")
    runs = {label: saved.load_run(directory) for label, directory in specifications}
    variants = {}
    for label, run in runs.items():
        require(run["endpoint"] == EXPECTED_UPDATE and run["report"]["status"] == "complete",
                "Length evaluation requires each completed 15k endpoint")
        config = run["report"]["contract"]["model_config"]
        contract = run["report"]["contract"]
        kind = paired.mechanism(config)["variant"]
        require(kind not in variants, "Duplicate embedding mechanism")
        variants[kind] = label
        require(contract["mode"] == "mixed" and contract["batch_per_task"] == 2560
                and config["backbone"]["d_model"] == 128
                and contract["streams"]["fuzzy"]["length"] == 400
                and config["backbone"]["max_sequence_length"] >= max(LENGTHS),
                "Unexpected mixed-task architecture, batch or trained length")
        final = saved.endpoint_metrics(run)["fuzzy/dev"]
        require(final["checkpoint"]["sha256"] == run["checkpoints"][EXPECTED_UPDATE]["sha256"],
                "T400 metrics differ from the selected checkpoint")
        check_metric(final)
    require(set(variants) == {"baseline", "input", "value", "head"}, "Missing baseline or embedding mechanism")
    base = runs[variants["baseline"]]
    audits = {label: paired.compatibility(base, run) for label, run in runs.items() if run is not base}
    return runs, audits, verify_live_sources(runs)


def restore_model(run, *, factory=build_model):
    """Strict CPU restoration, followed by a device move only in the GPU runner."""
    packet = paired._packet(run, EXPECTED_UPDATE)
    finite_tree(packet["model"], path="model")
    finite_tree(packet["optimizer"], path="optimizer")
    finite = packet.get("finite_state", {})
    require(all(finite.get(key) is True for key in
                ("parameters_finite_fp32", "gradients_finite_fp32", "adam_finite_fp32")),
            "Saved checkpoint lacks a finite FP32 training-state audit")
    initialization = packet["initialization"]
    model = factory(run["report"]["contract"]["model_config"], seed=initialization["seed"],
                    predictor_seed=initialization["predictor_seed"], fuzzy_seed=initialization["fuzzy_seed"],
                    device="cpu", backend="tiled")
    # Training serializes conversion tuples as JSON lists in checkpoint metadata.
    # Apply that same canonicalization; hashes and all field values remain exact.
    require(json_value(model.initialization) == initialization,
            "Reconstructed canonical initialization differs")
    require(sum(parameter.numel() for parameter in model.parameters()) == run["report"]["parameter_count"],
            "Reconstructed parameter count differs")
    model.load_state_dict(packet["model"], strict=True)
    require(all(parameter.dtype == torch.float32 and parameter.grad is None for parameter in model.parameters()),
            "Restored evaluation model must be FP32 with no gradients")
    model.requires_grad_(False)
    model.eval()
    return model, model_digest(model)


def evaluate_model(model, datasets, *, microbatch, device="cuda", evaluator=fuzzy.evaluate):
    require(type(microbatch) is int and microbatch > 0, "Positive evaluation microbatch required")
    require(set(datasets) == set(LENGTHS), "Require exactly the T512 and T1024 development pools")
    observations = {}
    before = model_digest(model)
    for length in LENGTHS:
        dataset = datasets[length]
        require(len(dataset) == EXPECTED_ROWS and dataset.input_ids.shape == (EXPECTED_ROWS, length),
                "Evaluation data length or pool size differs")
        result = evaluator(model, dataset, microbatch=microbatch, device=device,
                           latent_weight=1.0, limit=EXPECTED_ROWS)
        check_metric(result)
        result.update(sequence_length=length, role="length_generalization_dev", task="fuzzy",
                      training_update=EXPECTED_UPDATE, training_sequence_length=400)
        observations[str(length)] = result
    require(model_digest(model) == before and all(parameter.grad is None for parameter in model.parameters()),
            "Evaluation changed model state or accumulated gradients")
    return observations


def make_plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = []
    colors = dict(zip(summary["arms"], ("#202020", "#d95f02", "#1b9e77", "#7570b3")))
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=160)
        plt.close(fig)
        figures.append(name)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")
    for axis, (key, title) in zip(axes.flat, METRICS):
        for label, arm in summary["arms"].items():
            points = [(length, arm["metrics"][str(length)].get(key)) for length in (400, *LENGTHS)]
            points = [(length, value) for length, value in points if value is not None]
            axis.plot([point[0] for point in points], [100*point[1] for point in points], marker="o",
                      color=colors[label], label=label)
        axis.set(title=title, xlabel="Evaluation sequence length", ylabel="Accuracy / coverage (%)",
                 xticks=(400, *LENGTHS), ylim=(-2, 102))
        axis.axvline(400, color=".6", ls=":")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("Fuzzy length generalization at fixed 15k checkpoints — trained T400, no new training\n"
                 "T400 saved endpoint; T512/T1024 new shared 1,280-example development pools")
    save(fig, "fuzzy-length-generalization")

    from cdrm.rt_nextlat_fuzzy_metrics import DISTANCE_BINS
    names = [f"distance_{lo}_{hi}" if hi is not None else f"distance_{lo}_plus" for lo, hi in DISTANCE_BINS]
    labels = [f"{lo}–{hi}" if hi is not None else f"{lo}+" for lo, hi in DISTANCE_BINS]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    for column, length in enumerate((400, *LENGTHS)):
        for label, arm in summary["arms"].items():
            result = arm["metrics"][str(length)]
            points = [(i, result.get(f"{name}_accuracy")) for i, name in enumerate(names)]
            points = [(i, value) for i, value in points if value is not None]
            axes[0, column].plot([p[0] for p in points], [100*p[1] for p in points], marker="o",
                                 label=label, color=colors[label])
        representative = next(iter(summary["arms"].values()))["metrics"][str(length)]
        axes[1, column].bar(range(len(names)), [representative[f"{name}_tokens"] for name in names], color=".5")
        axes[0, column].set(title=f"T{length}: accuracy by retrieval distance", ylabel="Token accuracy (%)", ylim=(-2, 102))
        axes[0, column].legend(fontsize=7)
        axes[1, column].set(title="Native scored answer tokens in each distance bin", ylabel="Token count")
        for row in (0, 1):
            axes[row, column].set(xticks=range(len(names)), xticklabels=labels, xlabel="Distance to most recent matching key")
            axes[row, column].grid(axis="y", alpha=.2)
    fig.suptitle("Distance is between current and most recent historical matching key starts\n"
                 "Unavailable mappings have no distance bin; zero-count bins are not assigned 0% accuracy")
    save(fig, "fuzzy-retrieval-distance")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    regions = ("prior_occurrences_1", "prior_occurrences_2", "prior_occurrences_3_plus")
    for axis, length in zip(axes, (400, *LENGTHS)):
        for label, arm in summary["arms"].items():
            result = arm["metrics"][str(length)]
            points = [(i, result.get(f"{name}_accuracy")) for i, name in enumerate(regions)]
            points = [(i, value) for i, value in points if value is not None]
            axis.plot([p[0] for p in points], [100*p[1] for p in points], marker="o", color=colors[label], label=label)
        representative = next(iter(summary["arms"].values()))["metrics"][str(length)]
        counts = ", ".join(f"{representative[f'{name}_tokens']:,}" for name in regions)
        axis.set(title=f"T{length}; region token counts: {counts}", xticks=(0, 1, 2), xticklabels=("1", "2", "3+"),
                 xlabel="Prior occurrences of the same key", ylabel="Token accuracy (%)", ylim=(-2, 102))
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("Longer sequences may add repetition rather than longer retrieval distance")
    save(fig, "fuzzy-prior-occurrences")
    return figures


def markdown(summary):
    lines = ["# Fuzzy Recall length generalization", "", QUALIFICATION, "",
             "Every row uses the same selected 15,000-update checkpoint for that arm. "
             "Both new pools contain 1,280 examples and are shared by all four arms; no weight updates occur.", "",
             "| Arm | Length | Answer | Motif exact | Sequence exact | First value | Terminal | Known history |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    percent = lambda value: "n/a" if value is None else f"{100*value:.4f}%"
    for label, arm in summary["arms"].items():
        for length in (400, *LENGTHS):
            result = arm["metrics"][str(length)]
            fields = ("answer_accuracy", "answer_motif_exact_match", "sequence_exact_match",
                      "first_value_token_accuracy", "terminal_probe_accuracy", "known_history_accuracy")
            lines.append(f"| {label} | {length} | " + " | ".join(percent(result.get(key)) for key in fields) + " |")
    lines += ["", "## Dataset coverage", "",
              "Counts are shared across arms at each length. Accuracy conditional on a retrieval distance or prior "
              "occurrence count changes its population across lengths. No zero-denominator region is scored as 0%.", "",
              "| Length | Answer tokens | First value tokens | Known history | Unavailable history | History coverage |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    first = next(iter(summary["arms"].values()))
    for length in (400, *LENGTHS):
        result = first["metrics"][str(length)]
        lines.append(f"| {length} | {result['answer_tokens']:,} | {result['first_value_token_tokens']:,} | "
                     f"{result['known_history_tokens']:,} | {result['unavailable_history_tokens']:,} | {percent(result['oracle_coverage'])} |")
    lines += ["", "## Checkpoint and execution evidence", "",
              "FP32 eager execution, TF32/autocast/compile/CUDA graphs off; evaluation microbatch "
              f"{summary['eval_microbatch']}. Checkpoints, frozen sources, configuration, finite state and shared "
              "initialization/data order were verified before execution. Restored model tensors are hashed before "
              "and after evaluation to verify that no state changed. The NextLat loss is a teacher-conditioned "
              "diagnostic from the original evaluator, not an autonomous latent recurrence.", ""]
    for label, arm in summary["arms"].items():
        lines.append(f"- {label}: {arm['mechanism']['description']}; {arm['parameters']:,} parameters; "
                     f"checkpoint SHA256 `{arm['checkpoint']['sha256']}`.")
    lines += ["", f"Data manifest SHA256: `{summary['data_manifest_sha256']}`. "
              "Exact data provenance, source snapshots, counts and checkpoint identities are recorded in evidence.json.", "",
              "No automatic winner or causal explanation is selected from this single-seed length screen. "
              "If answer accuracy remains saturated, these measurements alone do not establish that one embedding route "
              "has better retrieval capacity.", ""]
    for name in summary["figures"]:
        lines += [f"![{name}]({name}.png)", f"[PDF]({name}.pdf)", ""]
    return "\n".join(lines)


def validate_shared_counts(arms):
    reference = next(iter(arms.values()))["metrics"]
    for arm in arms.values():
        for length, result in arm["metrics"].items():
            count_keys = [key for key in reference[length] if key.endswith("_tokens")
                          and not key.endswith("_ce_tokens")]
            count_keys += ["examples", "answer_motifs"]
            require(all(result.get(key) == reference[length][key] for key in count_keys),
                    f"Arms were evaluated on different native scoring populations at T{length}")


def run(args):
    # This public execution path always requires CUDA and the project container.
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    require(type(args.eval_microbatch) is int and args.eval_microbatch > 0, "Positive evaluation microbatch required")
    require(not args.output.exists(), "Use a new length-evaluation output directory")
    runs, compatibility, sources = load_inputs(args.run)
    from scripts.rt_nextlat_fuzzy_length_prepare import verify_manifest
    manifest, datasets = verify_manifest(args.data, verify_sources=True)
    require(set(datasets) == set(LENGTHS), "Prepared lengths differ from the approved screen")
    for current in runs.values():
        require(current["report"]["contract"]["runtime"] == runtime,
                "Evaluation runtime differs from the saved FP32 training runtime")
    args.output.mkdir(parents=True)
    for name, expected in sources.items():
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        require(saved.sha(destination) == expected, "Source changed during evaluation setup")
    atomic_json(args.output / "source-manifest.json", sources)
    summary = {"schema": SCHEMA, "status": "running", "qualification": QUALIFICATION,
               "training_update": EXPECTED_UPDATE, "training_fuzzy_length": 400,
               "new_evaluation_lengths": list(LENGTHS), "eval_microbatch": args.eval_microbatch,
               "data_directory": str(args.data.resolve()), "data_manifest": manifest,
               "data_manifest_sha256": saved.sha(args.data / "manifest.json"),
               "runtime": runtime, "hardware": hardware, "compatibility": compatibility,
               "source_manifest": sources, "arms": {}, "optimizer_updates": 0,
               "confirmation_evaluated": False, "latent_rollout_evaluated": False}
    atomic_json(args.output / "report.json", summary)
    tracker = (OnlineTracker(project="rt-nextlat-fuzzy-a5", entity="taylorbollman", output_dir=args.output,
                            name="d128-b2560-embedding-length-generalization-15k") if args.wandb else None)
    try:
        if tracker:
            tracker.start({"scope": QUALIFICATION, "lengths": list(LENGTHS), "training_update": EXPECTED_UPDATE,
                           "eval_microbatch": args.eval_microbatch,
                           "checkpoints": {label: current["checkpoints"][EXPECTED_UPDATE]["sha256"] for label, current in runs.items()}})
        for label, current in runs.items():
            model, before = restore_model(current)
            model.to(device="cuda", dtype=torch.float32)
            metrics = {"400": copy.deepcopy(saved.endpoint_metrics(current)["fuzzy/dev"])}
            metrics["400"].update(sequence_length=400, source="saved training-endpoint development evaluation")
            metrics.update(evaluate_model(model, datasets, microbatch=args.eval_microbatch))
            for length in LENGTHS:
                regions = manifest["lengths"][str(length)]["evaluation_regions"]
                require(all(metrics[str(length)].get(f"{name}_tokens") == count for name, count in regions.items()),
                        "Evaluation denominators differ from the frozen data-region counts")
            after = model_digest(model)
            require(before == after, "Checkpoint model changed during evaluation")
            checkpoint = current["checkpoints"][EXPECTED_UPDATE]
            for length in LENGTHS:
                metrics[str(length)]["checkpoint"] = checkpoint
            arm = {"directory": current["directory"], "checkpoint": checkpoint,
                   "mechanism": paired.mechanism(current["report"]["contract"]["model_config"]),
                   "parameters": current["report"]["parameter_count"], "metrics": metrics,
                   "model_sha256_before": before, "model_sha256_after": after,
                   "model_unchanged": True, "lineage": current["lineage"],
                   "training_wandb": current["report"].get("wandb")}
            summary["arms"][label] = arm
            atomic_json(args.output / "report.json", summary)
            if tracker:
                for length, result in metrics.items():
                    tracker.log(scalar_metrics(result, f"length_eval/{label}/length_{length}"))
            del model
            torch.cuda.empty_cache()
        validate_shared_counts(summary["arms"])
        require(saved.sha(args.data / "manifest.json") == summary["data_manifest_sha256"],
                "Data manifest changed during evaluation")
        require(verify_live_sources(runs) == sources, "Executed source closure changed during evaluation")
        summary["figures"] = make_plots(summary, args.output)
        summary["figure_sha256"] = {f"{name}.{suffix}": saved.sha(args.output / f"{name}.{suffix}")
                                   for name in summary["figures"] for suffix in ("png", "pdf")}
        summary["status"] = "complete"
        (args.output / "report.md").write_text(markdown(summary))
        atomic_json(args.output / "evidence.json", summary)
        if tracker:
            import wandb
            tracker.log({f"report/{name}": wandb.Image(str(args.output / f"{name}.png")) for name in summary["figures"]})
            tracker.summary({"results": {label: arm["metrics"] for label, arm in summary["arms"].items()},
                             "qualification": QUALIFICATION, "optimizer_updates": 0})
            artifact = wandb.Artifact(f"mixed-embedding-length-eval-{tracker.record['run_id']}", type="development-report")
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
    parser.add_argument("--run", type=paired.parse_run, action="append", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-microbatch", type=int, default=64)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output": str(args.output), "arms": list(summary["arms"])}))


if __name__ == "__main__":
    main()
