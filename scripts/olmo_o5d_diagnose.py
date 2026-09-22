#!/usr/bin/env python3
"""Fixed-weight finite-pass versus exact-online evaluation after fusion repair."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.fbt_evaluation import evaluate_fbt_batches
from cdrm.pretrained.lm_data import load_lm_data
from cdrm.pretrained.olmo_fbt import FBTMode, FBTOnlineMode
from cdrm.pretrained.recurrent import RTMode
from scripts.olmo_lm_common import state_digests
from scripts.olmo_o5b_common import PilotTracker, preserve_rng, slice_batch
from scripts.olmo_o5c_common import (endpoint_metadata as source_metadata,
    load_frozen_endpoint, source_hashes, plain_metadata)
from scripts.olmo_o5d_common import endpoint_metadata, load_endpoint
from scripts.olmo_validation import require_container_gpu

SPLITS = ("dev", "retention_dev")
ENDPOINTS = ("source", "mixed")


@dataclass(frozen=True)
class DiagnosticCase:
    section: str
    beta: float
    passes: int | None
    rows: int
    max_length: int

    @property
    def label(self):
        return self.section + "-" + ("online" if self.passes is None else f"K{self.passes}")

    def mode(self):
        if self.passes is None:
            return FBTOnlineMode(beta=self.beta, rt_mode=RTMode(()))
        return FBTMode(num_passes=self.passes, beta=self.beta, rt_mode=RTMode(()))


def diagnostic_cases():
    return tuple(DiagnosticCase(section, 1.0, passes, rows, length)
                 for section, rows, length in (("short_prefix", 32, 64), ("full_context", 512, 512))
                 for passes in (2, 3, 4, None))


def diagnostic_source_hashes():
    names = ("scripts/olmo_o5d_common.py", "scripts/olmo_o5d_diagnose.py",
             "docs/reports/olmo1b-o5d/protocol.md")
    return {name: sha256_file(ROOT/name) for name in names}


def selected_batches(corpus, split, case, *, batch_size=8, device="cuda"):
    if split not in SPLITS:
        raise ValueError("Only development splits may be evaluated")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive integer batch size required")
    if corpus.split_sizes[split] < case.rows:
        raise ValueError("Fewer prepared windows than the fixed selection")
    for start in range(0, case.rows, batch_size):
        yield slice_batch(corpus.batch(split, range(start, min(start+batch_size, case.rows)),
                                       device=device), length=case.max_length)


def fixed_configuration(mixed):
    config = mixed["configuration"]
    return {"precision": config["precision"], "batch_size": config["eval_batch_size"],
        "attention_backend": config["attention_backend"], "beta": 1.0,
        "rt_layers": [], "nextlat_enabled": False, "compile": False,
        "cuda_graphs": False, "tf32": False, "seed": 20260922,
        "base_data_manifest_sha256": config["base_data_manifest_sha256"],
        "bootstrap_repetitions": 1000, "bootstrap_seed": 20260922,
        "teacher_forced": True, "weight_updates": 0}


def validate_resume(report, identity):
    if report.get("status") not in ("running", "paused", "failed"):
        raise ValueError("Resume only an incomplete diagnostic")
    for key, value in identity.items():
        if report.get(key) != value:
            raise ValueError(f"Resume diagnostic identity changed: {key}")
    expected = [(endpoint, asdict(case), endpoint+"-"+case.label)
                for endpoint in ENDPOINTS for case in diagnostic_cases()]
    rows = report.get("cases", [])
    if len(rows) > len(expected):
        raise ValueError("Too many completed cases")
    for row, (endpoint, case, label) in zip(rows, expected):
        if (row.get("endpoint") != endpoint or row.get("case") != case or row.get("label") != label
                or row.get("weights_unchanged") is not True or set(row.get("metrics", {})) != set(SPLITS)):
            raise ValueError("Completed cases must be an unchanged ordered prefix of the fixed grid")


def tracking_metrics(row, index):
    values = {"diagnostic/case_index": index,
              "diagnostic/execution_index": (2, 3, 4, None).index(row["case"]["passes"])}
    prefix = f"diagnostic/{row['endpoint']}/{row['case']['section']}"
    for split in SPLITS:
        metric = row["metrics"][split]["passes"][-1]
        for key in ("mean_nll", "next_token_accuracy", "perplexity"):
            values[f"{prefix}/{split}/{key}"] = metric[key]
        if row["case"]["passes"] is not None:
            values[f"{prefix}/{split}/ordinary_nll"] = row["metrics"][split]["passes"][0]["mean_nll"]
    return values


class RequestedStop(Exception):
    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists() != args.resume:
        parser.error("Use a new output directory, or explicitly --resume an incomplete one")
    runtime = plain_metadata(require_container_gpu())
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mixed, source = endpoint_metadata(), source_metadata()
    if runtime != mixed["source_fingerprint"]["runtime"]:
        raise ValueError("Runtime differs from the completed O5c run")
    config = fixed_configuration(mixed)
    if config["batch_size"] != 8 or config["precision"] != "bf16_mixed":
        raise ValueError("Diagnostic must preserve actual O5c precision and evaluation batch size")
    corpus = load_lm_data(args.data)
    if corpus.manifest_sha256 != config["base_data_manifest_sha256"]:
        raise ValueError("Evaluation data manifest changed")
    identity = {"schema": "olmo-o5d-online-diagnostic-v1", "configuration": config,
        "runtime": runtime, "source_hashes": source_hashes(),
        "diagnostic_source_hashes": diagnostic_source_hashes(),
        "grid": [{"endpoint": endpoint, "case": asdict(case), "label": endpoint+"-"+case.label}
                 for endpoint in ENDPOINTS for case in diagnostic_cases()],
        "input_file_hashes": {"source_report": source["report_sha256"], "mixed_report": mixed["report_sha256"],
                              "data_manifest": corpus.manifest_sha256},
        "checkpoint_identities": {name: item["checkpoint"] for name, item in (("source", source), ("mixed", mixed))}}
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    report_path = args.output_dir/"report.json"
    if args.resume:
        report = json.loads(report_path.read_text())
        validate_resume(report, identity)
    else:
        report = {**identity, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "running", "cases": [], "endpoints": {}, "attempts": []}
    report["attempts"].append({"utc": datetime.now(timezone.utc).isoformat(), "resume": args.resume,
                               "completed_cases": len(report["cases"])})
    report["status"] = "running"
    tracker = PilotTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        name="olmo-1b-o5d-finite-vs-online", group="olmo1b-o5d-online-diagnostic", preserve_state=preserve_rng)
    stop_requested = False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def persist():
        report["wandb"] = tracker.record
        write_json(report_path, report)
    try:
        tracker.start(config, run_id=report.get("wandb", {}).get("run_id") if args.resume else None)
        def axes():
            for endpoint in ENDPOINTS:
                for section in ("short_prefix", "full_context"):
                    tracker._run.define_metric(f"diagnostic/{endpoint}/{section}/*", step_metric="diagnostic/execution_index")
        tracker._call("diagnostic axes", axes)
        persist()
        for name, metadata in (("source", source), ("mixed", mixed)):
            if all(any(row["label"] == name+"-"+case.label for row in report["cases"]) for case in diagnostic_cases()):
                continue
            if name == "source":
                model, provenance = load_frozen_endpoint(metadata["checkpoint_path"], metadata["checkpoint"]["sha256"],
                    expected_configuration=metadata["configuration"], expected_source_fingerprint=metadata["source_fingerprint"])
                model.requires_grad_(False)
            else:
                model, provenance = load_endpoint(metadata["checkpoint_path"], metadata["checkpoint"]["sha256"],
                    expected_size_bytes=metadata["checkpoint"]["size_bytes"],
                    expected_configuration=metadata["configuration"], expected_model_configuration=metadata["model_configuration"],
                    expected_source_fingerprint=metadata["source_fingerprint"], expected_counters=metadata["counters"],
                    expected_data_cursor=metadata["data_cursor"], expected_frozen_state_digests=metadata["frozen_state_digests"])
            before = state_digests(model)
            existing = report["endpoints"].get(name)
            if existing and existing["state_before"] != before:
                raise ValueError("Reloaded checkpoint state differs from saved diagnostic state")
            report["endpoints"][name] = {"checkpoint": metadata["checkpoint"], "provenance": provenance,
                "state_before": before, "state_after": before, "weights_unchanged": True}
            persist()
            for case in diagnostic_cases():
                label = name+"-"+case.label
                if any(row["label"] == label for row in report["cases"]):
                    continue
                began = time.monotonic()
                row = {"endpoint": name, "case": asdict(case), "label": label,
                       "metrics": {}, "split_elapsed_seconds": {}}
                report["in_progress_case"] = row
                for split in SPLITS:
                    start = time.monotonic()
                    def batches():
                        for index, batch in enumerate(selected_batches(corpus, split, case, batch_size=config["batch_size"])):
                            if stop_requested or (args.output_dir/"STOP").exists():
                                raise RequestedStop()
                            yield batch
                            report["progress"] = {"case": label, "split": split, "completed_batches": index+1,
                                "total_batches": (case.rows+config["batch_size"]-1)//config["batch_size"],
                                "seconds": time.monotonic()-start}
                            if (index+1) % 8 == 0:
                                persist(); print(report["progress"], flush=True)
                    row["metrics"][split] = evaluate_fbt_batches(model, batches(), mode=case.mode(),
                        precision=config["precision"], include_document_records=True)
                    torch.cuda.synchronize()
                    row["split_elapsed_seconds"][split] = time.monotonic()-start
                    persist()
                row["elapsed_seconds"] = time.monotonic()-began
                after = state_digests(model)
                if after != before:
                    raise AssertionError("Evaluation changed model or buffer bytes")
                row["weights_unchanged"] = True
                report["endpoints"][name].update(state_after=after, weights_unchanged=True)
                report["cases"].append(row)
                report.pop("in_progress_case", None)
                tracker.log(tracking_metrics(row, len(report["cases"])-1), step=len(report["cases"])-1)
                persist()
                print({"case": label, "seconds": row["elapsed_seconds"],
                    "nll": {s: row["metrics"][s]["passes"][-1]["mean_nll"] for s in SPLITS}}, flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()
        if report["source_hashes"] != source_hashes() or report["diagnostic_source_hashes"] != diagnostic_source_hashes():
            raise AssertionError("Frozen evaluation sources changed")
        report.update(status="passed", weights_unchanged=True)
        tracker.summary({"diagnostic/status": "passed", "diagnostic/cases": len(report["cases"]),
                         "diagnostic/weights_unchanged": True})
    except RequestedStop:
        report["status"] = "paused"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            persist()


if __name__ == "__main__":
    main()
