#!/usr/bin/env python3
"""CPU-only paired A5 pilot report from completed training records.

No model or checkpoint is loaded. Final confirmation remains unevaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.experiment_tracking import OnlineTracker


SCHEMA = "rt-a5-paired-report-v1"
TRAIN_SCHEMA = "rt-a5-training-v1"
ARMS = ("seq", "rt")
ROLES = ("dev", "ood_dev")
SCALARS = ("ce", "token_accuracy", "whole_word_exact_match", "final_state_accuracy")
COLORS = {"seq": "#3574B2", "rt": "#D65B35"}
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_input(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return content, {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def hash_file(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def checkpoint_path(value):
    path = Path(value)
    for prefix in (Path("/workspace/cdrm-w-latent"), Path("/home/taylorbollman/cdrm-w-latent")):
        if path.is_relative_to(prefix):
            return ROOT / path.relative_to(prefix)
    return path


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_arm(directory, architecture):
    directory = Path(directory)
    raw, report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    if report.get("schema") != TRAIN_SCHEMA or report.get("status") != "complete":
        raise ValueError(f"{architecture}: only completed A5 training reports can be compared")
    if report.get("confirmation_evaluated") is not False:
        raise ValueError(f"{architecture}: confirmation must remain unevaluated")
    if report["contract"]["architecture"] != architecture:
        raise ValueError(f"Expected {architecture} architecture")
    for key, expected in {"precision": "fp32", "tf32": False, "compile": False, "cuda_graphs": False}.items():
        if report["contract"].get(key) != expected:
            raise ValueError(f"{architecture}: this report requires {key}={expected}")
    config = report["contract"]["model_config"]
    if config.get("block_type") != ("sequential" if architecture == "seq" else "recurrent"):
        raise ValueError(f"{architecture}: unexpected block implementation")
    for key, expected in {"n_layers": 2, "activation_type": "gelu", "alibi": True,
                          "cdrm_enabled": False, "recurrent_layers": None}.items():
        if config.get(key) != expected:
            raise ValueError(f"{architecture}: unexpected model setting {key}")
    endpoint, start = report["completed_updates"], report["start_update"]
    if type(endpoint) is not int or type(start) is not int or not 0 <= start < endpoint or endpoint != report["endpoint"]:
        raise ValueError(f"{architecture}: incomplete endpoint or invalid update window")
    raw, history_file = read_input(directory / "history.jsonl")
    history = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if [row["update"] for row in history] != list(range(start + 1, endpoint + 1)):
        raise ValueError(f"{architecture}: training history has missing, duplicate or out-of-order updates")
    batch = report["contract"]["batch_size"]
    for row in history:
        if row["examples_seen"] != row["update"] * batch:
            raise ValueError(f"{architecture}: inconsistent training word count")
        for key in ("seconds", "loss", "token_accuracy", "whole_word_exact", "grad_norm"):
            if not finite_number(row[key]) or row[key] < 0:
                raise ValueError(f"{architecture}: nonfinite/negative history value {key}")
        if row["token_accuracy"] > 1 or row["whole_word_exact"] > 1:
            raise ValueError(f"{architecture}: invalid accuracy")
    if history[-1]["order_chain"] != report["order_chain"]:
        raise ValueError(f"{architecture}: final history order chain differs from report")
    checkpoints = [row for row in report["checkpoints"] if row["completed_updates"] == endpoint]
    if len(checkpoints) != 1:
        raise ValueError(f"{architecture}: need one retained endpoint checkpoint")
    checkpoint = hash_file(checkpoint_path(checkpoints[0]["path"]))
    if any(checkpoint[key] != checkpoints[0][key] for key in ("sha256", "bytes")):
        raise ValueError(f"{architecture}: endpoint checkpoint SHA256/size differs")
    source_files = report["source_files"]
    source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if source_digest != report["contract"]["source_sha256"]:
        raise ValueError(f"{architecture}: source file list differs from contract")
    for relative, expected in source_files.items():
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source snapshot paths must stay within the snapshot directory")
        if hash_file(directory / "source" / relative)["sha256"] != expected:
            raise ValueError(f"{architecture}: actual source snapshot SHA256 differs: {relative}")
    endpoint_metrics = {}
    for role in ROLES:
        selected = [row for row in report["evaluations"] if row["role"] == role and row["update"] == endpoint]
        if len(selected) != 1:
            raise ValueError(f"{architecture}: need exactly one endpoint evaluation for {role}")
        row = selected[0]
        if type(row["rows"]) is not int or row["rows"] < 1 or row["tokens"] != row["rows"] * row["length"]:
            raise ValueError(f"{architecture}: invalid {role} evaluation sizes")
        for key in SCALARS:
            if not finite_number(row[key]) or row[key] < 0 or (key != "ce" and row[key] > 1):
                raise ValueError(f"{architecture}: invalid endpoint {key}")
        for key in ("isolated_state_accuracy", "cumulative_prefix_exactness", "per_position_ce"):
            if len(row[key]) != row["length"] or any(not finite_number(v) or v < 0 for v in row[key]):
                raise ValueError(f"{architecture}: invalid {role} {key} curve")
        accuracy, exact = row["isolated_state_accuracy"], row["cumulative_prefix_exactness"]
        if any(a > 1 or e > a + 1e-12 for a, e in zip(accuracy, exact)) or any(
                later > earlier + 1e-12 for earlier, later in zip(exact, exact[1:])):
            raise ValueError(f"{architecture}: invalid cumulative exactness semantics")
        if not (math.isclose(exact[-1], row["whole_word_exact_match"], abs_tol=1e-12)
                and math.isclose(accuracy[-1], row["final_state_accuracy"], abs_tol=1e-12)
                and math.isclose(sum(accuracy) / len(accuracy), row["token_accuracy"], abs_tol=1e-12)):
            raise ValueError(f"{architecture}: endpoint scalars disagree with curves")
        endpoint_metrics[role] = row
    return {"report": report, "history": history, "endpoint_metrics": endpoint_metrics,
            "input_files": {"report": report_file, "history": history_file, "endpoint_checkpoint": checkpoint}}


def training_bins(history, window=100):
    """Nonoverlapping means of up to 100 updates; timing is never smoothed away."""
    if type(window) is not int or window < 1:
        raise ValueError("Training window must be a positive integer")
    result, elapsed = [], 0.0
    for begin in range(0, len(history), window):
        rows = history[begin:begin + window]
        elapsed += sum(row["seconds"] for row in rows)
        result.append({"first_update": rows[0]["update"], "update": rows[-1]["update"],
                       "updates_in_bin": len(rows), "training_seconds": elapsed,
                       **{key: sum(row[key] for row in rows) / len(rows)
                          for key in ("loss", "token_accuracy", "whole_word_exact", "grad_norm")}})
    return result


def compare_runs(seq_dir, rt_dir, expected_eval_rows=102400):
    arms = {"seq": read_arm(seq_dir, "seq"), "rt": read_arm(rt_dir, "rt")}
    seq, rt = (arms[name]["report"] for name in ARMS)
    for key in ("completed_updates", "endpoint", "start_update", "order_chain"):
        if seq[key] != rt[key]:
            raise ValueError(f"Paired {key} differs")
    left, right = dict(seq["contract"]), dict(rt["contract"])
    for contract in (left, right):
        contract.pop("architecture")
        config = dict(contract.pop("model_config"))
        config.pop("block_type")
        contract["shared_model_config"] = config
    if left != right:
        raise ValueError("Paired source/data/model/optimization/runtime contracts differ")
    if seq["initialization"]["canonical_sha256"] != rt["initialization"]["canonical_sha256"]:
        raise ValueError("Paired canonical initialization differs")
    if seq["initialization"]["parameter_count"] != rt["initialization"]["parameter_count"]:
        raise ValueError("Paired parameter counts differ")
    if [row["order_chain"] for row in arms["seq"]["history"]] != [row["order_chain"] for row in arms["rt"]["history"]]:
        raise ValueError("Paired per-update data order differs")
    for role in ROLES:
        if arms["seq"]["endpoint_metrics"][role]["rows"] != expected_eval_rows:
            raise ValueError(f"Expected {expected_eval_rows} endpoint evaluation rows for {role}")
        for key in ("rows", "length", "tokens"):
            if arms["seq"]["endpoint_metrics"][role][key] != arms["rt"]["endpoint_metrics"][role][key]:
                raise ValueError(f"Paired {role} evaluation {key} differs")
    return {
        "schema": SCHEMA, "completed_updates": seq["completed_updates"],
        "start_update": seq["start_update"], "width": left["width"], "seed": left["seed"],
        "training_length": left["length"], "batch_size": left["batch_size"],
        "parameter_count": seq["initialization"]["parameter_count"],
        "source_sha256": left["source_sha256"], "data_manifest_sha256": left["data_manifest_sha256"],
        "source_files": seq["source_files"],
        "canonical_initialization_sha256": seq["initialization"]["canonical_sha256"],
        "order_chain": seq["order_chain"], "confirmation_evaluated": False,
        "scope": "One paired development seed at a fixed update budget; no final-test or convergence claim",
        "timing": "Sum of training-loop update durations since start_update; excludes evaluation, checkpointing and W&B logging",
        "training_bin_updates": 100,
        "arms": {name: {"inputs": record["input_files"], "contract": record["report"]["contract"],
                        "wandb": record["report"].get("wandb"), "endpoint_metrics": record["endpoint_metrics"],
                        "training_curve": training_bins(record["history"])} for name, record in arms.items()},
    }


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = []
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for axis, key, title in zip(axes, ("cumulative_prefix_exactness", "isolated_state_accuracy"),
                                ("All states through position t correct: E(t)", "State at position t correct: A(t)")):
        for arm in ARMS:
            metrics = summary["arms"][arm]["endpoint_metrics"]["ood_dev"]
            axis.plot(range(1, metrics["length"] + 1), metrics[key], label=arm.upper(), color=COLORS[arm], linewidth=2)
        axis.axvline(summary["training_length"], color="#777777", linestyle="--", linewidth=1, label="Training boundary")
        axis.set(xlabel="Operation position / prefix length", ylabel="Fraction correct", ylim=(-0.02, 1.02), title=title)
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle(f"A5 OOD development · {summary['completed_updates']:,} updates · one paired seed")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"length-generalization.{suffix}", dpi=180)
    plt.close(figure)
    figures.append("length-generalization")
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    for col, xkey in enumerate(("update", "training_seconds")):
        for row, (ykey, ylabel) in enumerate((("loss", "Training CE"), ("token_accuracy", "Training token accuracy"))):
            axis = axes[row, col]
            for arm in ARMS:
                curve = summary["arms"][arm]["training_curve"]
                scale = 60 if xkey == "training_seconds" else 1
                axis.plot([point[xkey] / scale for point in curve], [point[ykey] for point in curve],
                          label=arm.upper(), color=COLORS[arm], linewidth=1.7)
            axis.set(xlabel="Optimizer updates" if col == 0 else "Accumulated training-loop time (minutes)", ylabel=ylabel)
            axis.grid(alpha=.2)
            axis.legend()
            if ykey == "token_accuracy":
                axis.set_ylim(-.02, 1.02)
    figure.suptitle("A5 training · nonoverlapping 100-update means\nTime excludes evaluation, checkpointing and W&B logging")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"training-curves.{suffix}", dpi=180)
    plt.close(figure)
    figures.append("training-curves")
    return {name: {suffix: f"{name}.{suffix}" for suffix in ("png", "pdf")} for name in figures}


def markdown_report(summary):
    lines = ["# A5 paired development pilot", "",
             f"Two blocks, width {summary['width']}, {summary['parameter_count']:,} parameters per arm; "
             f"{summary['completed_updates']:,} updates, batch {summary['batch_size']:,}, seed {summary['seed']}.", "",
             "Both arms use full FP32, causal ALiBi, GELU and the same canonical initialization and data order. "
             "Both RT blocks are recurrent.", "",
             "| Development set | Model | Rows | CE | Token accuracy | Whole-word exact match | Final-state accuracy |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for role in ROLES:
        for arm in ARMS:
            row = summary["arms"][arm]["endpoint_metrics"][role]
            lines.append(f"| {role} (L{row['length']}) | {arm.upper()} | {row['rows']:,} | {row['ce']:.8f} | "
                         f"{row['token_accuracy']:.8f} | {row['whole_word_exact_match']:.8f} | {row['final_state_accuracy']:.8f} |")
    lines += ["", "Accuracy columns are fractions. E(t) measures correctness of **every** state through t; "
              "A(t) measures only the state at t. The training boundary is marked on the length curve.", "",
              "![Length generalization](length-generalization.png)", "",
              "![Training curves](training-curves.png)", "",
              "Training curves show nonoverlapping means of up to 100 updates. Their time axis sums training-loop "
              "durations only, excluding evaluation, checkpointing and W&B logging. It is not total wall-clock cost.", "",
              "This is one paired development seed at a fixed update budget. It provides no multi-seed uncertainty "
              "estimate and does not establish convergence or performance at the 400,000-update reference budget. "
              "Final confirmation remains unevaluated.", "",
              "[Plot data and input SHA256 records](plot-data.json) · [Machine-readable report](report.json)", ""]
    return "\n".join(lines)


def run(args):
    summary = compare_runs(args.seq_dir, args.rt_dir, args.expected_eval_rows)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh paired report output directory")
    output.mkdir(parents=True)
    write_json(output / "plot-data.json", summary)
    figures = plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    tracker = OnlineTracker(project=args.wandb_project, entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name=f"paired-report-d{summary['width']}-step{summary['completed_updates']}")
    report = {**summary, "status": "running", "figures": figures,
              "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        tracker.start({key: value for key, value in summary.items() if key != "arms"})
        scalars = {"update": summary["completed_updates"]}
        for arm in ARMS:
            for role in ROLES:
                metrics = summary["arms"][arm]["endpoint_metrics"][role]
                scalars.update({f"dev/{arm}/{role}/{key}": metrics[key] for key in SCALARS})
        tracker.log(scalars)
        import wandb
        tracker.log({f"report/{name}": wandb.Image(str(output / files["png"])) for name, files in figures.items()})
        tracker.summary({"completed_updates": summary["completed_updates"], "confirmation_evaluated": False,
                         "scope": summary["scope"]})
        tracker.finish(succeeded=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        try:
            tracker.finish(succeeded=False)
        except Exception:
            pass
        raise
    finally:
        report["wandb"] = tracker.record
        write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "output_dir": str(output), "wandb": tracker.record["run_url"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-dir", required=True)
    parser.add_argument("--rt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-project", default="rt-a5-state-tracking")
    parser.add_argument("--wandb-group")
    parser.add_argument("--expected-eval-rows", type=int, default=102400,
                        help="Require this endpoint row count in each development role")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
