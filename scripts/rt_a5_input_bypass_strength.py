"""Watch saved INPUT-injection weights and plot their strength on CPU only.

Lambda is fixed at0.02. This measures the learned projection and its added
signal over all60 current raw token embeddings, not over contextual block inputs.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from scripts.experiment_tracking import OnlineTracker

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/rt-a5/20260915T150000Z-embedding-input10k"
CHILD = ROOT / ".runtime/rt-a5/20260915T153500Z-embedding-input20k"
RUN = ROOT / ".runtime/rt-a5/20260915T153500Z-input-bypass-strength"
OUTPUT = ROOT / "docs/reports/rt-a5/input-bypass-strength"
COEFFICIENT = .02
STEPS = (0, 1000, 5000, 10000, 15000, 20000)
COLUMNS = ("update", "fixed_lambda", "projection_weight_rms", "raw_embedding_rms",
           "added_signal_rms", "added_to_raw_rms_ratio", "checkpoint_sha256")


def local(value):
    path = Path(value)
    for prefix in (Path("/workspace/cdrm-w-latent"), Path("/home/taylorbollman/cdrm-w-latent")):
        if path.is_relative_to(prefix):
            path = ROOT / path.relative_to(prefix)
            break
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Evidence must remain in the project")
    return path


def record(path):
    path = local(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 << 20):
            digest.update(block)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def write(path, value, *, immutable=False):
    if immutable and path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_file(saved):
    actual = record(saved["path"])
    if any(actual[key] != saved[key] for key in ("sha256", "bytes") if key in saved):
        raise ValueError("Checkpoint or bound input changed")
    return actual


def check_protocol(lineage, parent=None):
    protocol = json.loads((lineage / "protocol.json").read_text())
    expected_schema = ("rt-a5-embedding-injection-protocol-v1" if parent is None
                       else "rt-a5-embedding-injection-extension-protocol-v1")
    start, end = (0, 10000) if parent is None else (10000, 20000)
    if protocol["schema"] != expected_schema or protocol["start_update"] != start or protocol["endpoint"] != end:
        raise ValueError("Unexpected input-injection continuation scope")
    contract = protocol["strict_contract"]
    if (contract["schema"] != "rt-a5-embedding-injection-training-v1" or contract["variant"] != "input"
            or contract["n_layers"] != 4 or contract["width"] != 512
            or contract["projection_seed"] != 1236 or contract["injection_coefficient"] != COEFFICIENT
            or contract["experiment_config"]["injection_layer"] != 1
            or contract["experiment_config"]["coefficient_learned"] is not False
            or contract["experiment_config"]["embedding_source"] !=
               "attached original raw token embedding lookup, without scaling, normalization, dropout or added positions"):
        raise ValueError("Require the fixed0.02 input route at the second block of four layers")
    sources = protocol["source_files"]
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if len(sources) != 62 or protocol["source_sha256"] != digest or contract["source_sha256"] != digest:
        raise ValueError("Source62 binding differs")
    for name, expected in sources.items():
        if record(ROOT / name)["sha256"] != expected:
            raise ValueError("Frozen training source changed")
    if parent is not None:
        if any(protocol[key] != parent[key] for key in ("strict_contract", "initialization", "source_files")):
            raise ValueError("Child changed the input model, initialization, or training contract")
        checked_file(protocol["parent_checkpoint"])
        checked_file(protocol["parent_protocol"])
        if protocol["parent_checkpoint"]["completed_updates"] != 10000:
            raise ValueError("Child must resume the10k parent")
    return protocol


def inspect_saved(protocol, report, step, torch):
    if (report["schema"] != "rt-a5-embedding-injection-training-v1"
            or report["status"] not in ("running", "complete")
            or report["start_update"] != protocol["start_update"]
            or report["endpoint"] != protocol["endpoint"]
            or report["contract"] != protocol["strict_contract"]
            or report["initialization"] != protocol["initialization"]
            or report["source_files"] != protocol["source_files"]):
        raise ValueError("Training report differs from its approved input protocol")
    directory = local(protocol["training_directory"])
    items = [item for item in report["checkpoints"] if item["completed_updates"] == step]
    if len(items) != 1 or local(items[0]["path"]) != directory / f"checkpoints/step-{step:06d}.pt":
        raise ValueError("Need exactly the selected saved checkpoint in its training directory")
    checkpoint = checked_file(items[0])  # Hash before CPU deserialization.
    packet = torch.load(local(checkpoint["path"]), map_location="cpu", weights_only=False)
    if (packet["schema"] != report["schema"] or packet["contract"] != protocol["strict_contract"]
            or packet["initialization"] != protocol["initialization"]
            or packet["completed_updates"] != step or packet["examples_seen"] != step * 1024):
        raise ValueError("Saved tensor packet has a different model or update contract")
    projection = packet["model"]["backbone.embedding_projection.weight"]
    embeddings = packet["model"]["backbone.transformer.wte.weight"]
    if (projection.shape != (512, 512) or embeddings.shape != (60, 512)
            or any(value.device.type != "cpu" or value.dtype != torch.float32
                   or not torch.isfinite(value).all() for value in (projection, embeddings))):
        raise ValueError("Expected finite CPU FP32 projection and all60 raw embedding rows")
    values = magnitudes(projection, embeddings, torch)
    del packet
    return {"update": step, "fixed_lambda": COEFFICIENT, **values,
            "checkpoint_sha256": checkpoint["sha256"]}, checkpoint


def magnitudes(projection, embeddings, torch):
    # Apply the actual FP32 linear map; use float64 only for scalar RMS reduction.
    added = COEFFICIENT * torch.nn.functional.linear(embeddings, projection)
    rms = lambda tensor: tensor.double().square().mean().sqrt().item()
    raw_rms = rms(embeddings)
    if raw_rms <= 0:
        raise ValueError("Raw embeddings have zero RMS; ratio is undefined")
    return {"projection_weight_rms": rms(projection), "raw_embedding_rms": raw_rms,
            "added_signal_rms": rms(added), "added_to_raw_rms_ratio": rms(added) / raw_rms}


def snapshot(rows, inputs, tracker):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    endpoint = rows[-1]["update"]
    directory = OUTPUT / f"through-{endpoint:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "bypass-strength.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.78, wspace=.37)
    fields = (("projection_weight_rms", "Learned projection weight RMS", "Weight RMS"),
              ("added_signal_rms", "Added signal: 0.02 Pₑ e", "Added signal RMS"),
              ("added_to_raw_rms_ratio", "Added / raw embedding RMS", "RMS ratio"))
    for axis, (key, title, ylabel) in zip(axes, fields):
        axis.plot([row["update"] for row in rows], [row[key] for row in rows], marker="o", color="#236B9E")
        axis.set(title=title, ylabel=ylabel, xlabel="Optimizer updates")
        axis.set_xticks([row["update"] for row in rows],
                        labels=["0" if row["update"] == 0 else f"{row['update']//1000}k" for row in rows])
        axis.grid(alpha=.25)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Input injection · λ = 0.02 is fixed; Pₑ is learned\nAll 60 current raw token embeddings, equally weighted", y=.97)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"bypass-strength.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# Input bypass strength", "", "**λ is fixed at 0.02; it is not a learned gate.** The learned object is the projection Pₑ.", "",
             "These measurements use the projection and raw token-embedding weights from each saved checkpoint. Every one of the60 A5 tokens receives equal weight.", "",
             "| Updates | Projection weight RMS | Raw embedding RMS | Added signal RMS | Added / raw RMS |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['update']:,} | {row['projection_weight_rms']:.6g} | {row['raw_embedding_rms']:.6g} | {row['added_signal_rms']:.6g} | {row['added_to_raw_rms_ratio']:.3%} |")
    lines += ["", "Added signal = 0.02 × Pₑ e. RMS means the square root of the mean squared entries; the ratio divides added-signal RMS by raw-embedding RMS.", "",
              "This ratio is **not the contribution relative to the contextual input of the second block**. It measures only the saved raw-embedding route. Both the projection and embedding table can change during training; the trend reflects both. No model forward pass or task evaluation was run.", "",
              f"[W&B run]({tracker.record['run_url']}) · [CSV](bypass-strength.csv) · [PDF](bypass-strength.pdf)", "", "![Bypass strength](bypass-strength.png)"]
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    summary = {"schema": "rt-a5-input-bypass-strength-v1", "through_update": endpoint, "fixed_lambda": COEFFICIENT,
               "lambda_learned": False, "vocabulary_rows": 60, "token_weighting": "uniform",
               "contextual_hidden_input_ratio_measured": False, "rows": rows, "inputs": inputs,
               "analysis_source": record(Path(__file__)), "wandb": tracker.record,
               "scope": "CPU saved projection/embedding arithmetic only; current weights at each checkpoint; no model/inference."}
    write(directory / "summary.json", summary, immutable=True)
    artifacts = {path.name: record(path) for path in directory.iterdir() if path.is_file()}
    write(directory / "artifacts.json", artifacts, immutable=True)
    import wandb
    tracker.log({"update": endpoint, "figures/input_bypass_strength": wandb.Image(str(directory / "bypass-strength.png"))})
    return directory


def main():
    if not Path("/.dockerenv").is_file() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Run inside the GPU-disabled project container")
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("Saved-weight analysis must never initialize CUDA")
    # A bounded arithmetic check, not a model/numerics campaign.
    fixture = magnitudes(torch.eye(2), torch.tensor([[3., 4.], [0., 0.]]), torch)
    assert math.isclose(fixture["projection_weight_rms"], math.sqrt(.5), rel_tol=1e-7)
    assert math.isclose(fixture["added_signal_rms"], .05, rel_tol=1e-6)
    assert math.isclose(fixture["added_to_raw_rms_ratio"], .02, rel_tol=1e-6)
    if OUTPUT.exists():
        raise FileExistsError("Preserve previous immutable graph snapshots")
    OUTPUT.mkdir(parents=True)
    parent = check_protocol(PARENT)
    parent_report = json.loads((local(parent["training_directory"]) / "report.json").read_text())
    if parent_report["status"] != "complete" or parent_report["wandb"]["status"] != "synced":
        raise ValueError("Initial four points require the completed10k parent")
    rows, inputs = [], {"parent_protocol": record(PARENT / "protocol.json"), "checkpoints": {}}
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=RUN,
                            group="20260915T153500Z-input-bypass-strength", name="input-bypass-strength")
    def status(value, **extra):
        write(RUN / "status.json", {"status": value, "through_update": rows[-1]["update"] if rows else None,
              "updated_utc": datetime.now(timezone.utc).isoformat(), "wandb": tracker.record, **extra})
    def add(protocol, report, step):
        row, saved = inspect_saved(protocol, report, step, torch)
        rows.append(row); inputs["checkpoints"][str(step)] = saved
        tracker.log({"update": step, **{f"bypass/{key}": value for key, value in row.items() if key not in ("update", "checkpoint_sha256")}})
    try:
        tracker.start({"fixed_lambda": COEFFICIENT, "lambda_learned": False, "variant": "input",
                       "vocabulary_rows": 60, "token_weighting": "uniform", "planned_endpoint": 20000})
        for step in STEPS[:4]:
            add(parent, parent_report, step)
        directory = snapshot(rows, inputs, tracker)
        status("waiting_for_continuation", latest_snapshot=str(directory.relative_to(ROOT)))
        print(json.dumps({"event": "snapshot", "through_update": 10000, "path": str(directory), "wandb": tracker.record['run_url']}), flush=True)
        child = None
        while rows[-1]["update"] < 20000:
            launch_path = CHILD / "launch-status.json"
            if launch_path.exists() and json.loads(launch_path.read_text()).get("status") == "failed":
                raise RuntimeError("Input continuation failed; preserve the available graph snapshots")
            if not (CHILD / "protocol.json").exists():
                time.sleep(30); continue
            if child is None:
                child = check_protocol(CHILD, parent)
                inputs["child_protocol"] = record(CHILD / "protocol.json")
                if child["parent_checkpoint"]["sha256"] != inputs["checkpoints"]["10000"]["sha256"]:
                    raise ValueError("Continuation resumed a different10k checkpoint")
            if record(CHILD / "protocol.json")["sha256"] != inputs["child_protocol"]["sha256"]:
                raise ValueError("Continuation protocol changed during analysis")
            report_path = local(child["training_directory"]) / "report.json"
            if not report_path.exists():
                time.sleep(30); continue
            report = json.loads(report_path.read_text())
            if report.get("status") == "failed":
                raise RuntimeError("Continuation report failed; graph remains partial")
            available = {item["completed_updates"] for item in report["checkpoints"]}
            for step in (15000, 20000):
                if step > rows[-1]["update"] and step in available:
                    add(child, report, step)
                    directory = snapshot(rows, inputs, tracker)
                    status("watching", latest_snapshot=str(directory.relative_to(ROOT)))
                    print(json.dumps({"event": "snapshot", "through_update": step, "path": str(directory)}), flush=True)
            if rows[-1]["update"] < 20000:
                time.sleep(30)
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected CUDA initialization")
        tracker.summary({"through_update": 20000, "lambda_learned": False, "fixed_lambda": COEFFICIENT,
                         "final_added_to_raw_rms_ratio": rows[-1]["added_to_raw_rms_ratio"]})
        tracker.finish(succeeded=True)
        write(OUTPUT / "final-summary.json", {"through_update": 20000, "rows": rows, "inputs": inputs,
              "wandb": tracker.record, "cpu_only": True, "final_snapshot": str(directory.relative_to(ROOT)),
              "lambda_learned": False, "fixed_lambda": COEFFICIENT}, immutable=True)
        status("complete", latest_snapshot=str(directory.relative_to(ROOT)))
    except BaseException as error:
        try:
            tracker.finish(succeeded=False)
        finally:
            status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
