#!/usr/bin/env python3
"""Summarize retained tiled CDRM NUM/OPS evidence and publish one W&B overview.

Explicitly CPU-only: reads JSON/checksums, makes standalone plots and tables,
and never loads a checkpoint or executes a model. Failed machine screens are
preserved independently of the caller's reviewed engineering disposition.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

from experiment_tracking import OnlineTracker, add_wandb_arguments, scalar_metrics

PRIMARY = "tiled_bf16_vs_tiled_fp32"
REFERENCE = "tiled_fp32_vs_naive_fp32"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@contextmanager
def preserve_cpu_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])


def pick(value, names):
    return {name: value[name] for name in names if name in value}


def compact_metric(row):
    return pick(row, ("finite", "rel_l2", "max_abs_error", "maxerr_over_ref_rms",
                      "maximum_over_reference_maximum", "elementwise_pass", "elementwise_failure_count",
                      "scale_aware_pass", "prospective_l2_pass", "prospective_maximum_pass",
                      "global_logit_l2_pass", "ref_numel"))


def compact_gradients(value):
    summary = pick(value, ("pass", "global_parameter_relative_l2", "global_parameter_reference_l2",
                           "global_parameter_error_l2", "global_parameter_fp32_floor_l2",
                           "global_parameter_l2_pass", "parameter_tensor_count", "total_tensor_count",
                           "per_tensor_l2_failures", "per_tensor_maximum_failures", "fp32_elementwise_failures",
                           "fp32_scale_aware_diagnostic_failures", "invalid_tensors"))
    rows = value.get("rows", {})
    valid = {name: row for name, row in rows.items() if row.get("finite")}
    summary["all_present_finite"] = bool(rows) and len(valid) == len(rows)
    summary["elementwise_coordinate_failures"] = sum(row.get("elementwise_failure_count", 0) for row in rows.values())
    if valid:
        for label, key in (("worst_tensor_relative_l2", "rel_l2"),
                           ("worst_tensor_maximum_over_reference_maximum", "maximum_over_reference_maximum"),
                           ("worst_tensor_absolute_error", "max_abs_error")):
            name = max(valid, key=lambda item: valid[item][key])
            summary[label] = {"tensor": name, "value": valid[name][key],
                              "metrics": compact_metric(valid[name])}
    return summary


def compact_numerical(report, reference, role):
    result = {"artifact": reference, "role": role,
              **pick(report, ("status", "machine_screens_pass", "numerical_clearance", "disposition",
                              "criteria", "checkpoint", "scaling_requested", "scaling_pass")),
              "fixture": pick(report.get("fixture", {}), ("sha256", "dataset_sha256", "shape", "example_offset",
                                                            "split", "native_targets", "answer_targets", "loss_semantics")),
              "comparisons": {}, "wandb": report.get("wandb")}
    for label, pair in report.get("comparisons", {}).items():
        result["comparisons"][label] = {
            "machine_screens_pass": pair["machine_screens_pass"],
            "actual_ce_gradients": compact_gradients(pair["actual_ce_gradients"]),
            "independent_side_gradients": compact_gradients(pair["independent_side_gradients"]),
            "logits": compact_metric(pair["logits"]), "absolute_ce_difference": pair["absolute_ce_difference"],
            "fp32_ce_elementwise_pass": pair.get("fp32_ce_elementwise_pass"),
            "unscaled_side_states": {name: compact_metric(row) for name, row in pair["unscaled_side_states"].items()},
            "actual_forward_unscaled_side_states": {name: compact_metric(row) for name, row in
                                                     pair["actual_forward_unscaled_side_states"].items()},
            "adam": pick(pair["adam"], ("global_delta_relative_l2", "global_delta_cosine", "maximum_error_over_lr",
                                        "gradient_near_zero_buckets", "guardrail_pass", "guardrail", "clipping"))}
        if label == PRIMARY:
            for scope in ("actual_ce_gradients", "independent_side_gradients"):
                gradient = result["comparisons"][label][scope]
                reference_norm = gradient["global_parameter_reference_l2"]
                gradient["global_relative_l2_limit_including_fp32_floor"] = (
                    2 * 2**-7 + gradient["global_parameter_fp32_floor_l2"] / reference_norm
                    if reference_norm else None)
    result["scaling_failures"] = {arm: {scale: [name for name, row in rows.items()
                                              if not row["bitwise_normalized_equal"]]
                                        for scale, rows in scales.items()}
                                  for arm, scales in report.get("scaling", {}).items()}
    return result


def operational_identity(report):
    identity = report["identity"]
    if report.get("identity_sha256") != json_digest(identity):
        raise ValueError("Operational identity digest mismatch")
    cfg = identity["model_config"]
    expected = {"n_layers": 5, "d_model": 128, "n_heads": 16, "mlp_hidden_size": 512,
                "cdrm_early_layer": 1, "cdrm_late_layer": 3, "cdrm_backend": "tiled"}
    if any(cfg.get(name) != value for name, value in expected.items()):
        raise ValueError("Operational report lies outside the five-block tiled CDRM scope")
    return identity


def comparable_identity(report):
    identity = copy.deepcopy(operational_identity(report))
    identity.pop("precision")
    identity["model_config"].pop("cdrm_precision_policy")
    return identity


def compact_operational(report, reference):
    identity = operational_identity(report)
    result = {"artifact": reference, "status": report["status"], "identity_sha256": report["identity_sha256"],
            "precision": identity["precision"], "initial_checkpoint_sha256": identity["initial_checkpoint_sha256"],
            "data": identity["data"], "physical_batch": identity["physical_batch"],
            "optimizer": identity["optimizer"], "schedule": identity["schedule"],
            "execution_contract": identity["execution_contract"], "wandb": report.get("wandb"),
            **pick(report, ("completed_updates", "completed_epochs", "training_seconds", "clipping_count",
                            "final_precision", "learning_check", "repeat_batch_sha256", "recovery", "checkpoints")),
            "development": {step: {kind: row[kind] for kind in ("native", "answer")}
                            for step, row in report.get("development", {}).items()}}
    history = report.get("history", [])
    if history:
        signals = {name for row in history for name in row.get("parameter_gradient_signals", {})}
        result["trajectory_audit"] = {
            "monitored_updates": len(history),
            "all_updates_have_finite_fp32_parameters_gradients_moments": all(
                not row.get("precision", {}).get("missing_gradients", ["unrecorded"])
                and all(row.get("precision", {}).get(kind, {}).get("finite") is True
                        and row["precision"][kind].get("dtypes") == ["torch.float32"]
                        for kind in ("parameters", "gradients", "moments")) for row in history),
            "logits_dtypes": sorted({row.get("logits_dtype", "unrecorded") for row in history}),
            "ce_dtypes": sorted({row.get("ce_dtype", "unrecorded") for row in history}),
            "monitored_owner_adapter_parameter_count": len(signals),
            "parameters_without_any_nonzero_observed_gradient": sorted(name for name in signals if not any(
                row.get("parameter_gradient_signals", {}).get(name, {}).get("nonzero") for row in history)),
            "correction_ratios": {name: {"first": values[0], "last": values[-1], "minimum": min(values), "maximum": max(values)}
                for name in ("candidate_correction_over_p3_rms", "bridge_correction_over_p8_rms")
                if (values := [row["state_summaries"][name] for row in history if name in row.get("state_summaries", {})])}}
    return result


def paired_training(fp32, bf16):
    a, b = [comparable_identity(report) for report in (fp32, bf16)]
    if a != b:
        raise ValueError("Training comparison differs beyond its explicit precision policy")
    left, right = fp32.get("history", []), bf16.get("history", [])
    if len(left) != 100 or len(right) != 100:
        raise ValueError("The intended paired OPS comparison requires both complete 100-update histories")
    rows = []
    for index, (aa, bb) in enumerate(zip(left, right), 1):
        for key in ("update", "epoch", "batch_in_epoch", "batch_sha256", "indices_sha256", "learning_rate"):
            if aa[key] != bb[key]:
                raise ValueError(f"Paired update {index} differs in {key}")
        if aa["update"] != index:
            raise ValueError("Noncontiguous training history")
        rows.append({"update": index, "fp32_native_ce": aa["native_loss"], "bf16_native_ce": bb["native_loss"],
                     "bf16_minus_fp32_ce": bb["native_loss"] - aa["native_loss"]})
    gaps = [row["bf16_minus_fp32_ce"] for row in rows]
    development = []
    if fp32["development"].keys() != bf16["development"].keys():
        raise ValueError("Paired development endpoints differ")
    for step in sorted(fp32["development"], key=int):
        for profile, report in (("fp32", fp32), ("bf16", bf16)):
            values = report["development"][step]
            development.append({"update": int(step), "precision": profile,
                                "native_ce": values["native"]["ce"], "answer_ce": values["answer"]["ce"],
                                "answer_token_accuracy": values["answer"]["token_accuracy"],
                                "answer_sequence_exact_match": values["answer"]["sequence_exact_match"]})
    final_gap = bf16["development"]["100"]["native"]["ce"] - fp32["development"]["100"]["native"]["ce"]
    last_mean = statistics.mean(gaps[-50:])
    return {"same_initialization_data_order_schedule": True, "updates": len(rows),
            "mean_bf16_minus_fp32_ce": statistics.mean(gaps), "last50_mean_bf16_minus_fp32_ce": last_mean,
            "maximum_absolute_update_ce_gap": max(map(abs, gaps)), "final_development_ce_gap": final_gap,
            "prospective_loss_guardrail_nats": .02,
            "last_window_and_development_guardrail_pass": last_mean <= .02 and final_gap <= .02,
            "scope": "Bounded operational stability; no long-run learning-equivalence claim"}, rows, development


def compact_benchmark(report, reference):
    result = compact_operational(report, reference)
    result.update(timing=report["timing"], peak_allocated_bytes=report["peak_allocated_bytes"],
                  peak_reserved_bytes=report["peak_reserved_bytes"], warmup_steps=len(report["warmup"]),
                  measured_steps=len(report["measured"]), measurement_compilation=report["measurement_compilation"],
                  benchmark_data_sha256=report["benchmark_data_sha256"])
    measured_mean = statistics.mean(row["seconds"] for row in report["measured"])
    if not math.isclose(measured_mean, result["timing"]["mean_seconds"], rel_tol=1e-12):
        raise ValueError("Recorded benchmark mean differs from its measured updates")
    if not result["measurement_compilation"]["steady_compilation_free"]:
        raise ValueError("Benchmark includes additional compilation during measurement")
    if len(report["measured"]) != 20:
        raise ValueError("The intended warmed benchmark requires 20 measured complete updates")
    return result


def compact_support(value, depth=0):
    """Keep support verdicts and aggregates; link full tensor/coordinate tables."""
    if isinstance(value, dict):
        if depth > 4:
            return {"omitted_nested_details": True}
        return {key: compact_support(item, depth + 1) for key, item in value.items()
                if key not in {"rows", "source_sha256", "provenance", "tensors", "state_dict", "hardware"}}
    if isinstance(value, list):
        return value if len(value) <= 32 and all(isinstance(item, (str, int, float, bool)) for item in value) else {"entries": len(value)}
    return value


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(output, numerical, training_rows, development, benchmarks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"fp32": "#24608a", "bf16": "#b95827"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
    for profile in ("fp32", "bf16"):
        axes[0, 0].plot([row["update"] for row in training_rows],
                        [row[f"{profile}_native_ce"] for row in training_rows],
                        color=colors[profile], label=profile.upper(), linewidth=1.5)
        dev = [row for row in development if row["precision"] == profile]
        axes[0, 1].plot([row["update"] for row in dev], [100 * row["answer_token_accuracy"] for row in dev],
                        "o-", color=colors[profile], label=profile.upper())
        axes[0, 2].plot([row["update"] for row in dev], [100 * row["answer_sequence_exact_match"] for row in dev],
                        "o-", color=colors[profile], label=profile.upper())
    for ax, title, ylabel in zip(axes[0], ("Native training loss", "Development answer-token accuracy", "Development sequence exact match"),
                                ("Cross entropy (nats)", "Accuracy (%)", "Exact match (%)")):
        ax.set(title=title, xlabel="Optimizer update", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    if development and not any(row["answer_sequence_exact_match"] for row in development):
        axes[0, 2].set_ylim(-.05, 1.)
        axes[0, 2].text(.5, .72, "No exact sequences in this bounded run", transform=axes[0, 2].transAxes,
                        ha="center", fontsize=9)
    cases = [case for case in numerical if case["role"] == "confirmation" and PRIMARY in case["comparisons"]]
    positions = np.arange(len(cases))
    for offset, scope, label, color in ((-.18, "actual_ce_gradients", "Full actual CE", "#417c90"),
                                       (.18, "independent_side_gradients", "Unscaled side", "#9871aa")):
        values = [case["comparisons"][PRIMARY][scope] for case in cases]
        bars = axes[1, 0].bar(positions + offset, [100 * row["global_parameter_relative_l2"] for row in values],
                              width=.36, label=label, color=color)
        for bar, row in zip(bars, values):
            if not row["global_parameter_l2_pass"]:
                bar.set_hatch("///")
                bar.set_edgecolor("#8b2020")
        if values:
            axes[1, 0].scatter(positions + offset,
                               [100 * row["global_relative_l2_limit_including_fp32_floor"] for row in values],
                               marker="_", s=140, color="#222222",
                               label="Actual limit with FP32 floor" if scope == "actual_ce_gradients" else None, zorder=3)
    axes[1, 0].axhline(1.5625, color="#6d6d6d", linestyle="--", linewidth=1, label="Nominal global limit (before floors)")
    axes[1, 0].set_xticks(positions, [Path(case["artifact"]["path"]).parent.name for case in cases], rotation=10)
    axes[1, 0].set(title="BF16 versus tiled FP32 gradients", ylabel="Global parameter relative L2 (%)")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].text(.01, -.25, "All frozen failures remain in summary.json; bars do not imply clearance.",
                    transform=axes[1, 0].transAxes, fontsize=8)
    for axis, field, factor, title, ylabel in (
            (axes[1, 1], "mean_seconds", 1000, "Complete warmed-up update", "Time (ms)"),
            (axes[1, 2], "peak_allocated_bytes", 1 / 2**20, "Peak allocated GPU memory", "Memory (MiB)")):
        values = [(benchmarks[p]["timing"][field] if field == "mean_seconds" else benchmarks[p][field]) * factor
                  for p in ("fp32", "bf16")]
        bars = axis.bar(["FP32", "BF16"], values, color=[colors[p] for p in ("fp32", "bf16")])
        axis.bar_label(bars, fmt="%.1f", padding=3)
        axis.set(title=title, ylabel=ylabel, ylim=(0, max(values) * 1.18))
    fig.suptitle("Five-block tiled CDRM · MAD selective copying V16/T256/K96 · bounded validation", fontsize=14)
    for suffix in ("svg", "png"):
        fig.savefig(output / f"summary.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, action="append", default=[])
    parser.add_argument("--diagnostic", type=Path, action="append", default=[])
    parser.add_argument("--train-fp32", type=Path, required=True)
    parser.add_argument("--train-bf16", type=Path, required=True)
    parser.add_argument("--benchmark-fp32", type=Path, required=True)
    parser.add_argument("--benchmark-bf16", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, action="append", default=[])
    parser.add_argument("--repeat", type=Path, action="append", default=[])
    parser.add_argument("--support-report", type=Path, action="append", default=[])
    parser.add_argument("--reviewed-disposition", default="Pending engineering review; machine failures remain unchanged")
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project="cdrm-tiled-bf16-validation", wandb_run_name="milestone-summary")
    args = parser.parse_args()
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Enter the project CPU container with CDRM_DOCKER_GPUS=none")
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("Retained reporting must not initialize or use CUDA")
    if not args.wandb_project:
        parser.error("The graphable milestone summary requires online W&B")
    root, output = args.run_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True)
    inputs = []

    def read(path):
        path = path if path.is_absolute() or path.exists() else root / path
        path = path.resolve()
        if path.is_dir():
            path = path / "report.json"
        if output in path.parents:
            raise ValueError("Retained reports must be outside the new output directory")
        report = json.loads(path.read_text())
        reference = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
        inputs.append(reference)
        if report.get("tensor_artifact"):
            tensor = Path(report["tensor_artifact"]["path"])
            if not tensor.exists():
                tensor = path.parent / tensor.name
            if digest(tensor) != report["tensor_artifact"]["sha256"]:
                raise ValueError(f"Retained NUM tensor checksum differs: {tensor}")
        return report, reference

    summary = {"schema": "cdrm-tiled-milestone-summary-v1", "status": "summarizing", "run_root": str(root),
               "generator_sha256": digest(Path(__file__)), "tracking_helper_sha256": digest(Path(__file__).with_name("experiment_tracking.py")),
               "automatic_numerical_clearance": False, "reviewed_disposition": args.reviewed_disposition,
               "scope": "Retained NUM/OPS reductions; no model execution, new acceptance threshold or long-run learning claim",
               "inputs": inputs}
    tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group or root.name,
                            name=args.wandb_run_name, output_dir=output, preserve_state=preserve_cpu_rng)
    summary["wandb"] = tracker.record
    started = time.monotonic()
    try:
        numerical = []
        for role, paths in (("confirmation", args.confirmation), ("diagnostic", args.diagnostic)):
            for path in paths:
                report, reference = read(path)
                if report.get("schema") != "cdrm-tiled-numerical-v1":
                    raise ValueError("Numerical input is not a CDRM tiled validation report")
                numerical.append(compact_numerical(report, reference, role))
        summary["numerical"] = numerical
        confirmations = [row for row in numerical if row["role"] == "confirmation"]
        summary["all_confirmation_machine_screens_pass"] = (all(row.get("machine_screens_pass") is True for row in confirmations)
                                                            if confirmations else None)
        summary["confirmation_machine_failure_count"] = sum(row.get("machine_screens_pass") is not True for row in confirmations)
        training, raw_training, benchmarks, raw_benchmarks = {}, {}, {}, {}
        for profile in ("fp32", "bf16"):
            policy = "fp32" if profile == "fp32" else "bf16_fp32_state"
            report, reference = read(getattr(args, f"train_{profile}"))
            if operational_identity(report)["precision"] != policy or report["identity"]["purpose"] != "paired_training":
                raise ValueError(f"Training {profile} input has an unexpected precision or purpose")
            training[profile], raw_training[profile] = compact_operational(report, reference), report
            report, reference = read(getattr(args, f"benchmark_{profile}"))
            if operational_identity(report)["precision"] != policy or report["identity"]["purpose"] != "benchmark":
                raise ValueError(f"Benchmark {profile} input has an unexpected precision or purpose")
            benchmarks[profile] = compact_benchmark(report, reference)
            raw_benchmarks[profile] = report
        summary["training"], summary["benchmarks"] = training, benchmarks
        summary["paired_training"], train_rows, dev_rows = paired_training(raw_training["fp32"], raw_training["bf16"])
        if (comparable_identity(raw_benchmarks["fp32"]) != comparable_identity(raw_benchmarks["bf16"])
                or benchmarks["fp32"]["initial_checkpoint_sha256"] != training["fp32"]["initial_checkpoint_sha256"]
                or benchmarks["fp32"]["benchmark_data_sha256"] != benchmarks["bf16"]["benchmark_data_sha256"]):
            raise ValueError("Benchmark profiles differ beyond precision or do not share the training initialization")
        summary["benchmark_comparison"] = {
            "bf16_time_change_fraction": benchmarks["bf16"]["timing"]["mean_seconds"] / benchmarks["fp32"]["timing"]["mean_seconds"] - 1,
            "bf16_allocated_memory_change_fraction": benchmarks["bf16"]["peak_allocated_bytes"] / benchmarks["fp32"]["peak_allocated_bytes"] - 1}
        summary["recovery"] = []
        for path in args.recovery:
            report, reference = read(path)
            result = compact_operational(report, reference)
            comparison = report.get("recovery_comparison", {})
            result["comparison"] = {"bitwise_state_and_nontiming_metrics_equal": comparison.get("bitwise_state_and_nontiming_metrics_equal"),
                                    "difference_count": len(comparison.get("differences", {})),
                                    "difference_paths": list(comparison.get("differences", {})),
                                    "excluded": comparison.get("excluded")}
            result["compiler"] = report.get("compiler")
            summary["recovery"].append(result)
        summary["repeat"] = []
        for path in args.repeat:
            report, reference = read(path)
            summary["repeat"].append(compact_operational(report, reference))
        summary["all_operational_reports_complete"] = all(row["status"] == "complete" for row in (
            *training.values(), *benchmarks.values(), *summary["recovery"], *summary["repeat"]))
        summary["all_supplied_recoveries_exact"] = (all(
            row["comparison"]["bitwise_state_and_nontiming_metrics_equal"] is True for row in summary["recovery"])
            if summary["recovery"] else None)
        summary["support_reports"] = []
        for path in args.support_report:
            report, reference = read(path)
            summary["support_reports"].append({"artifact": reference, "recorded": compact_support(report)})
        gradient_rows = []
        for case in numerical:
            for pair, comparisons in case["comparisons"].items():
                for scope in ("actual_ce_gradients", "independent_side_gradients"):
                    values = comparisons[scope]
                    gradient_rows.append({"case": Path(case["artifact"]["path"]).parent.name, "role": case["role"],
                                          "comparison": pair, "scope": scope,
                                          "global_parameter_relative_l2": values.get("global_parameter_relative_l2"),
                                          "global_parameter_l2_pass": values.get("global_parameter_l2_pass"),
                                          "global_relative_l2_limit_including_fp32_floor": values.get("global_relative_l2_limit_including_fp32_floor"),
                                          "worst_tensor_relative_l2": values.get("worst_tensor_relative_l2", {}).get("value"),
                                          "fp32_elementwise_failed_tensors": len(values.get("fp32_elementwise_failures", [])),
                                          "per_tensor_l2_failures": len(values.get("per_tensor_l2_failures", [])),
                                          "per_tensor_maximum_failures": len(values.get("per_tensor_maximum_failures", [])),
                                          "pair_machine_screens_pass": comparisons["machine_screens_pass"]})
        write_csv(output / "training.csv", train_rows)
        write_csv(output / "development.csv", dev_rows)
        write_csv(output / "gradient-summary.csv", gradient_rows)
        make_plot(output, numerical, train_rows, dev_rows, benchmarks)
        tracker.start({"evidence_class": "NUM-OPS-summary", "source_reports": inputs,
                       "reviewed_disposition": args.reviewed_disposition, "scope": summary["scope"]})
        print(json.dumps({"wandb_run_url": tracker.record["run_url"]}), flush=True)
        with preserve_cpu_rng():
            import wandb
        by_update = {row["update"]: {"update": row["update"], "train/fp32_native_ce": row["fp32_native_ce"],
                                     "train/bf16_native_ce": row["bf16_native_ce"], "train/bf16_minus_fp32_ce": row["bf16_minus_fp32_ce"]}
                     for row in train_rows}
        for row in dev_rows:
            values = by_update.setdefault(row["update"], {"update": row["update"]})
            values.update({f"dev/{row['precision']}/{name}": row[name]
                           for name in ("native_ce", "answer_ce", "answer_token_accuracy", "answer_sequence_exact_match")})
        for _, values in sorted(by_update.items()):
            tracker.log(values)
        columns = list(gradient_rows[0]) if gradient_rows else []
        with preserve_cpu_rng():
            media = {"milestone/summary_plot": wandb.Image(str(output / "summary.png")),
                     "milestone/gradient_summary": wandb.Table(columns=columns, data=[[row[name] for name in columns] for row in gradient_rows])}
        tracker.log(media)
        tracker.summary({"automatic_numerical_clearance": False,
                         "all_confirmation_machine_screens_pass": summary["all_confirmation_machine_screens_pass"],
                         "confirmation_machine_failure_count": summary["confirmation_machine_failure_count"],
                         "all_operational_reports_complete": summary["all_operational_reports_complete"],
                         "all_supplied_recoveries_exact": summary["all_supplied_recoveries_exact"],
                         "reviewed_disposition": args.reviewed_disposition,
                         **scalar_metrics(summary["paired_training"], "paired_training"),
                         **scalar_metrics(summary["benchmark_comparison"], "benchmark")})
        if any(digest(row["path"]) != row["sha256"] for row in inputs):
            raise RuntimeError("A retained source report changed during summarization")
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            tracker.finish(succeeded=summary["status"] == "complete")
        except Exception as error:
            summary.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            summary["elapsed_seconds"] = time.monotonic() - started
            with (output / "summary.json").open("x") as stream:
                json.dump(summary, stream, indent=2, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()
