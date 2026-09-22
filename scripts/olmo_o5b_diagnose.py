#!/usr/bin/env python3
"""Evaluation-only post-hoc beta/pass diagnostic on the completed FBT endpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.fbt_evaluation import evaluate_fbt_batches
from cdrm.pretrained.lm_data import load_lm_data
from cdrm.pretrained.lm_training import CHECKPOINT_SCHEMA, parameter_layout
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, validate_prepared_manifest
from cdrm.pretrained.olmo_fbt import FBTMode, FBTOnlineMode
from cdrm.pretrained.recurrent import RTMode
from scripts.olmo_lm_common import state_digests
from scripts.olmo_o5b_common import build_model, source_hashes, slice_batch, PilotTracker, preserve_rng
from scripts.olmo_o5b_report import validate_runs
from scripts.olmo_validation import require_container_gpu

SPLITS = ("dev", "retention_dev")
QUALIFICATION = (
    "Exploratory post-hoc development evaluation of the same trained FBT endpoint; "
    "the grid was specified after observing preliminary learning curves and evaluated "
    "only after completion. Any best setting is "
    "development-set model selection, not a confirmatory result. No weights are trained "
    "or changed, no reserved test split is accessed, and RT/NextLat remain off. Beta zero "
    "ablates feedback in the FBT-trained backbone; it is not the separately trained "
    "ordinary control. Full-window and short-prefix results use different contexts and "
    "must not be compared as matched samples. Online evaluation is teacher-forced, "
    "not free-running generation. One seed; no efficacy or generalization claim."
)


@dataclass(frozen=True)
class DiagnosticCase:
    section: str
    beta: float
    passes: int | None
    rows: int
    max_length: int

    @property
    def label(self):
        return f"{self.section}-beta{self.beta:g}-" + ("online" if self.passes is None else f"K{self.passes}")

    def mode(self):
        if self.passes is None:
            return FBTOnlineMode(beta=self.beta, rt_mode=RTMode(()))
        return FBTMode(num_passes=self.passes, beta=self.beta, rt_mode=RTMode(()))


def diagnostic_cases():
    """Fixed before evaluation; no adaptive grid or data selection."""
    full = [DiagnosticCase("full_beta", beta, 2, 128, 512) for beta in (0.0, .25, .5, .75, 1.0)]
    short = [DiagnosticCase("short_passes", beta, passes, 32, 64)
             for beta in (.5, 1.0) for passes in (2, 3, 4, None)]
    return tuple(full + short)


def selected_batches(corpus, split, case, *, batch_size, device):
    if split not in SPLITS:
        raise ValueError("Diagnostic permits development and retention development only")
    if corpus.split_sizes[split] < case.rows:
        raise ValueError("Prepared split has fewer rows than the fixed diagnostic selection")
    for start in range(0, case.rows, batch_size):
        batch = corpus.batch(split, range(start, min(start + batch_size, case.rows)), device=device)
        yield slice_batch(batch, length=case.max_length)


def diagnostic_source_hashes():
    paths = ("scripts/olmo_o5b_diagnose.py", "scripts/olmo_o5b_report.py", "scripts/olmo_o4_report.py",
             "docs/reports/olmo1b-o5b/diagnostic-protocol.md")
    return {path: sha256_file(ROOT/path) for path in paths}


def read_completed_inputs(preflight, runs):
    """Require both original arms, then bind diagnostic to current frozen code."""
    comparison = validate_runs(preflight, runs)
    config = comparison["configuration"]
    if source_hashes() != config["source_hashes"]:
        raise ValueError("Current model/training/evaluation sources differ from frozen O5b sources")
    preflight_path = Path(preflight)/"report.json"
    paths = {"preflight": preflight_path, "configuration": Path(preflight)/"configuration.json",
             **{arm: Path(runs)/arm/"report.json" for arm in ("ordinary", "fbt")}}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    fbt = reports["fbt"]
    total = config["schedule"]["total_updates"]
    records = [record for record in fbt["checkpoints"] if record["optimizer_updates"] == total]
    if len(records) != 1:
        raise ValueError("Need exactly one final FBT checkpoint")
    record = records[0]
    filename = f"update-{total:06d}.pt"
    if Path(record["path"]).name != filename:
        raise ValueError("Final checkpoint filename differs from its completed update")
    checkpoint = Path(runs)/"fbt"/filename
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError("Need the retained final FBT checkpoint as a regular local file")
    if checkpoint.stat().st_size != record["size_bytes"] or sha256_file(checkpoint) != record["sha256"]:
        raise ValueError("Final FBT checkpoint bytes differ from the verified receipt")
    return config, reports, checkpoint, record, {name: sha256_file(path) for name, path in paths.items()}


def validate_checkpoint_payload(payload, model, config, report):
    """Validate identities before strict model-only load; optimizer is never built."""
    expected_config = {**config, "arm": "fbt", "storage_prefix": report["storage_prefix"]}
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "model_type": type(model).__module__ + "." + type(model).__qualname__,
        "configuration": expected_config,
        "source_fingerprint": report["source_fingerprint"],
        "parameter_layout": parameter_layout(model),
        "counters": report["counters"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"Endpoint checkpoint {name} differs from the completed FBT arm")
    if payload.get("data_cursor", {}).get("next_window") != report["data_cursor"]:
        raise ValueError("Endpoint checkpoint data cursor differs")
    state, target = payload.get("model", {}), model.state_dict()
    if set(state) != set(target):
        raise ValueError("Endpoint model tensor keys differ")
    for name, value in target.items():
        saved = state[name]
        if not isinstance(saved, torch.Tensor) or saved.shape != value.shape or saved.dtype != value.dtype:
            raise ValueError(f"Endpoint model tensor shape/dtype differs: {name}")
        if not bool(torch.isfinite(saved).all()):
            raise ValueError(f"Endpoint model tensor is nonfinite: {name}")
    if set(payload.get("module_training", {})) != set(dict(model.named_modules())):
        raise ValueError("Endpoint module ownership differs")
    return state


def load_endpoint_payload(path):
    """Keep weights-only loading; permit only PyTorch's version-string metadata.

    The retained runtime fingerprint contains torch.__version__, a TorchVersion
    string subclass. No arbitrary user classes or unrestricted pickle loader
    are enabled; full checkpoint bytes were already verified before this call.
    """
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def evaluate_case(model, corpus, config, case):
    result = {"case": asdict(case), "label": case.label, "metrics": {}}
    for split in SPLITS:
        batches = selected_batches(corpus, split, case, batch_size=config["eval_batch_size"],
                                   device=model.backbone.readout_weight.device)
        result["metrics"][split] = evaluate_fbt_batches(
            model, batches, mode=case.mode(), precision=config["precision"], include_document_records=True)
    return result


def _signature(metric):
    return [(row["batch_index"], row["row_index"], row["document_id"], row["ce_count"])
            for row in metric["document_records"]]


def tracking_metrics(row, index):
    """Shared beta/pass series plus case-specific complete pass observations."""
    case = row["case"]
    values = {"diagnostic/case_index": index, "diagnostic/beta": case["beta"]}
    if case["section"] == "short_passes":
        values["diagnostic/short_execution_index"] = (2, 3, 4, None).index(case["passes"])
    for split in SPLITS:
        series = f"diagnostic/{case['section']}"
        if case["section"] == "short_passes":
            series += f"/beta{case['beta']:g}"
        final = row["metrics"][split]["passes"][-1]
        values[f"{series}/{split}/final_nll"] = final["mean_nll"]
        values[f"{series}/{split}/final_accuracy"] = final["next_token_accuracy"]
        for p, item in enumerate(row["metrics"][split]["passes"]):
            prefix = f"diagnostic/cases/{row['label']}/{split}/pass{p}"
            values[prefix+"/nll"] = item["mean_nll"]
            values[prefix+"/accuracy"] = item["next_token_accuracy"]
    return values


def summarize_cases(results):
    """Paired point differences only; the grid does not yield test-set inference."""
    if [row["case"] for row in results] != [asdict(case) for case in diagnostic_cases()]:
        raise ValueError("Diagnostic results do not cover the fixed ordered grid")
    for section in ("full_beta", "short_passes"):
        rows = [row for row in results if row["case"]["section"] == section]
        for split in SPLITS:
            signature = _signature(rows[0]["metrics"][split]["passes"][0])
            for row in rows:
                for metric in row["metrics"][split]["passes"]:
                    if _signature(metric) != signature:
                        raise ValueError("Diagnostic settings do not use identical document/target selections")
    summary = {"full_beta": {}, "short_passes": {}}
    for split in SPLITS:
        full = [row for row in results if row["case"]["section"] == "full_beta"]
        best = min(full, key=lambda row: row["metrics"][split]["passes"][-1]["mean_nll"])
        baseline = full[0]["metrics"][split]["passes"][-1]["mean_nll"]
        beta1 = full[-1]["metrics"][split]["passes"][-1]["mean_nll"]
        nll = best["metrics"][split]["passes"][-1]["mean_nll"]
        summary["full_beta"][split] = {"lowest_observed_beta": best["case"]["beta"],
            "lowest_observed_nll": nll, "difference_from_beta0": nll-baseline,
            "difference_from_beta1": nll-beta1, "selection_scope": "post-hoc development grid only"}
        summary["short_passes"][split] = []
        for beta in (.5, 1.0):
            rows = [row for row in results if row["case"]["section"] == "short_passes" and row["case"]["beta"] == beta]
            k2 = rows[0]["metrics"][split]["passes"][-1]["mean_nll"]
            summary["short_passes"][split].append({"beta": beta, "differences_from_finite_K2": {
                "online" if row["case"]["passes"] is None else f"K{row['case']['passes']}":
                row["metrics"][split]["passes"][-1]["mean_nll"]-k2 for row in rows}})
    return summary


def markdown(report):
    lines = ["# O5b endpoint: post-hoc feedback diagnostic", "", QUALIFICATION, "",
             f"Endpoint checkpoint SHA256: `{report['checkpoint']['sha256']}`.", "",
             "The table scores the final finite pass or exact online state separately; it does not report the summed training CE objective.", "",
             "| Selection | Beta | Execution | Code NLL | Retention NLL |",
             "| --- | ---: | --- | ---: | ---: |"]
    for row in report["cases"]:
        case = row["case"]
        label = "128 windows, max512" if case["section"] == "full_beta" else "32 prefixes, max64"
        execution = "Online" if case["passes"] is None else f"K={case['passes']}"
        nll = [row["metrics"][split]["passes"][-1]["mean_nll"] for split in SPLITS]
        lines.append(f"| {label} | {case['beta']:g} | {execution} | {nll[0]:.6f} | {nll[1]:.6f} |")
    lines += ["", f"All model and buffer hashes unchanged: **{report['weights_unchanged']}**. No optimizer or checkpoint was created.", "",
              "Each setting's full per-pass metrics and document records are in [report.json](report.json).",
              "[Standalone figure](diagnostic.pdf) · [PNG](diagnostic.png)", ""]
    return "\n".join(lines)


def write_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for column, split in enumerate(SPLITS):
        full = [row for row in report["cases"] if row["case"]["section"] == "full_beta"]
        axes[0, column].plot([row["case"]["beta"] for row in full],
            [row["metrics"][split]["passes"][-1]["mean_nll"] for row in full], "o-")
        axes[0, column].set(xlabel="Inference feedback beta", ylabel="Token NLL (nats)",
            title=f"{'Code' if column == 0 else 'Retention'} · K2 · 128 windows · max512")
        for beta in (.5, 1.0):
            short = [row for row in report["cases"] if row["case"]["section"] == "short_passes" and row["case"]["beta"] == beta]
            axes[1, column].plot(range(4), [row["metrics"][split]["passes"][-1]["mean_nll"] for row in short],
                                  "o-", label=f"beta={beta:g}")
        axes[1, column].set(xticks=range(4), xticklabels=("K2", "K3", "K4", "Online"),
            ylabel="Token NLL (nats)", title=f"{'Code' if column == 0 else 'Retention'} · 32 prefixes · max64")
        axes[1, column].legend()
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.suptitle("Same FBT endpoint · post-hoc development exploration · no weight updates", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(Path(output)/f"diagnostic.{suffix}", dpi=180)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preflight", "runs", "data", "artifacts", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    for name in ("preflight", "runs", "data", "artifacts"):
        if not getattr(args, name).is_dir():
            parser.error(f"--{name} must be an existing directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    runtime = require_container_gpu()
    config, inputs, checkpoint, checkpoint_record, input_hashes = read_completed_inputs(args.preflight, args.runs)
    if runtime != inputs["preflight"]["runtime"]:
        raise ValueError("Diagnostic runtime differs from the frozen pilot runtime")
    corpus = load_lm_data(args.data)
    if corpus.manifest_sha256 != config["data_manifest_sha256"]:
        raise ValueError("Diagnostic data differs from the frozen prepared corpus")
    native = validate_prepared_manifest(args.artifacts)
    if native["checkpoint"]["sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Native architecture artifact differs from the frozen checkpoint lineage")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "olmo-o5b-posthoc-diagnostic-v1", "status": "running", "qualification": QUALIFICATION,
              "started_utc": datetime.now(timezone.utc).isoformat(), "configuration": config,
              "checkpoint": checkpoint_record, "checkpoint_path": str(checkpoint), "input_file_hashes": input_hashes,
              "source_hashes": source_hashes(), "diagnostic_source_hashes": diagnostic_source_hashes(),
              "data_manifest_sha256": corpus.manifest_sha256, "runtime": runtime,
              "grid": [asdict(case) for case in diagnostic_cases()], "cases": []}
    tracker = PilotTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
            name="olmo-1b-o5b-posthoc-diagnostic", group="olmo1b-o5b-code-pilot", preserve_state=preserve_rng)
    model = None
    before = None
    try:
        tracker.start({"scope": "evaluation only; post-hoc development diagnostic", "checkpoint_sha256": checkpoint_record["sha256"],
                       "data_manifest_sha256": corpus.manifest_sha256, "grid": report["grid"]})
        def configure_axes():
            tracker._run.define_metric("diagnostic/full_beta/*", step_metric="diagnostic/beta")
            tracker._run.define_metric("diagnostic/short_passes/*", step_metric="diagnostic/short_execution_index")
        tracker._call("diagnostic chart axes", configure_axes)
        native_state = load_native_state_dict(args.artifacts)
        model = build_model(native_state, backend=config["attention_backend"])
        del native_state
        payload = load_endpoint_payload(checkpoint)
        trained_state = validate_checkpoint_payload(payload, model, config, inputs["fbt"])
        model.load_state_dict(trained_state, strict=True, assign=False)
        del trained_state, payload
        gc.collect()
        model.eval()
        before = state_digests(model)
        for index, case in enumerate(diagnostic_cases()):
            row = evaluate_case(model, corpus, config, case)
            report["cases"].append(row)
            tracker.log(tracking_metrics(row, index), step=index)
            report["wandb"] = tracker.record
            write_json(args.output_dir/"report.json", report)
            print({"case": case.label, "nll": {split: row["metrics"][split]["passes"][-1]["mean_nll"] for split in SPLITS}}, flush=True)
        report["summary"] = summarize_cases(report["cases"])
        report["state_before"] = before
        report["state_after"] = state_digests(model)
        report["weights_unchanged"] = before == report["state_after"]
        if not report["weights_unchanged"]:
            raise AssertionError("Evaluation changed model or buffer bytes")
        if report["source_hashes"] != source_hashes() or report["diagnostic_source_hashes"] != diagnostic_source_hashes():
            raise AssertionError("Diagnostic sources changed during evaluation")
        write_figures(report, args.output_dir)
        (args.output_dir/"results.md").write_text(markdown(report))
        report["status"] = "passed"
        tracker.summary({"diagnostic/status": "passed", "diagnostic/weights_unchanged": True,
                         "diagnostic/cases": len(report["cases"])})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir/"report.json", report)


if __name__ == "__main__":
    main()
