#!/usr/bin/env python3
"""CPU-only phase-one aggregation of supplied fuzzy-recall experiment artifacts.

This reads reports, arrays and hashes. It never imports a model, generates data,
loads Torch checkpoints, or reinterprets development calibration as a final test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SCHEMA = "cdrm-fuzzy-phase-one-summary-v1"
NUM_SCHEMA = "cdrm-fuzzy-numerical-v1"
PROFILE_SCHEMA = "cdrm-fuzzy-profile-v1"
TRAIN_SCHEMA = "cdrm-fuzzy-calibration-v1"
INIT_SCHEMA = "cdrm-fuzzy-paired-initialization-v1"
PREP_SCHEMA = "cdrm-fuzzy-preparation-v1"
CRITERIA_SHA = "26b1756dd958e0ab1c916cc51e46b691393eabb5598c645cb0d63b5bfdd0ea20"
EXCLUDED = {"source", "wandb", "inductor-cache", "__pycache__", "verification", "staged-retention"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def resolve_record_path(value, project=ROOT):
    path = Path(value)
    for prefix in (Path("/workspace/cdrm-w-latent"), Path("/home/taylorbollman/cdrm-w-latent")):
        if path.is_relative_to(prefix):
            return (project / path.relative_to(prefix)).resolve()
    return (path if path.is_absolute() else project / path).resolve()


def scan_reports(lineage):
    """Walk only this supplied lineage, excluding snapshots and generated reports."""
    result = []
    for directory, dirs, names in os.walk(lineage):
        dirs[:] = [name for name in dirs if name not in EXCLUDED and not (Path(directory) / name).is_symlink()]
        names = set(names)
        for name in ("report.json", "manifest.json"):
            if name in names:
                path = Path(directory) / name
                value = json.loads(path.read_text())
                if value.get("schema") in (NUM_SCHEMA, PROFILE_SCHEMA, TRAIN_SCHEMA, INIT_SCHEMA, PREP_SCHEMA):
                    result.append({"path": str(path), "sha256": digest(path), "report": value,
                                   "observation": "closed_report"})
        if "progress.json" in names and "report.json" not in names:
            path = Path(directory) / "progress.json"
            value = json.loads(path.read_text())
            if value.get("schema") in (NUM_SCHEMA, PROFILE_SCHEMA, TRAIN_SCHEMA):
                result.append({"path": str(path), "sha256": digest(path), "report": value,
                               "observation": "orphan_progress_incomplete"})
    return sorted(result, key=lambda record: record["path"])


class ArtifactAudit:
    def __init__(self, lineage, project=ROOT):
        self.lineage, self.project = Path(lineage).resolve(), Path(project).resolve()
        self.files, self.datasets, self.errors = {}, {}, []

    def check_file(self, path, expected=None, size=None):
        path = Path(path).resolve()
        if not path.is_relative_to(self.project):
            raise ValueError(f"Reference escapes supplied project: {path}")
        current = path.stat()
        signature = (current.st_size, current.st_mtime_ns, current.st_ino)
        cached = self.files.get(str(path))
        if cached and cached["signature"] != signature:
            raise ValueError(f"Artifact changed during aggregation: {path}")
        if not cached:
            cached = {"sha256": digest(path), "bytes": current.st_size, "signature": signature}
            self.files[str(path)] = cached
        if expected is not None and cached["sha256"] != expected:
            raise ValueError(f"Artifact checksum mismatch: {path}")
        if size is not None and cached["bytes"] != size:
            raise ValueError(f"Artifact byte count mismatch: {path}")
        return cached

    def reference(self, record, *, scientific=True):
        path = resolve_record_path(record["path"], self.project)
        if scientific and not path.is_relative_to(self.lineage):
            raise ValueError("Scientific blob is outside the explicitly supplied new lineage")
        self.check_file(path, record["sha256"], record.get("bytes"))
        return path

    def dataset(self, root, split, expected=None):
        from cdrm.mad_data import FUZZY_TASK, load_dataset
        root = resolve_record_path(root, self.project)
        if not root.is_relative_to(self.lineage):
            raise ValueError("Dataset root escapes supplied new lineage")
        key = (str(root), split)
        if key not in self.datasets:
            dataset = load_dataset(root, FUZZY_TASK, split, verify=True)
            if dataset.manifest["split"] not in ("train", "dev"):
                raise ValueError("Phase one does not include final-test data")
            self.datasets[key] = dataset
            manifest_path = root / FUZZY_TASK / f"{split}.manifest.json"
            self.check_file(manifest_path, dataset.manifest["manifest_sha256"])
            for record in dataset.manifest["files"].values():
                self.check_file(manifest_path.parent / record["name"], record["sha256"])
        dataset = self.datasets[key]
        if expected and dataset.sha256 != expected:
            raise ValueError("Dataset arrays disagree with report identity")
        return dataset

    def audit(self, entry):
        report, path = entry["report"], Path(entry["path"])
        local_errors = []
        def run(check):
            try:
                check()
            except (ValueError, KeyError, FileNotFoundError, TypeError, AssertionError) as error:
                local_errors.append(str(error))
        run(lambda: self.check_file(path, entry["sha256"]))
        identity = report.get("identity", {})
        source_map = report.get("source_sha256", identity.get("source_sha256", {}))
        for name, expected in source_map.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                local_errors.append(f"Nonrelative source snapshot path: {name}")
            else:
                run(lambda name=name, expected=expected: self.check_file(path.parent / "source" / name, expected))
        for key in ("checkpoint", "initial_checkpoint", "data_identity", "side_fixture"):
            if report.get(key):
                run(lambda record=report[key]: self.reference(record))
        for key in ("packets", "checkpoints"):
            for record in report.get(key, {}).values():
                run(lambda record=record: self.reference(record))
        if report.get("resume", {}).get("checkpoint"):
            run(lambda: self.reference(report["resume"]["checkpoint"]))
        if report.get("protocol"):
            run(lambda: self.reference(report["protocol"]))
        if report.get("criteria"):
            run(lambda: self.reference(report["criteria"], scientific=False))
            if report["criteria"]["sha256"] != CRITERIA_SHA:
                local_errors.append("Numerical criteria differs from the retained contract")
        if identity:
            if json_digest(identity) != report.get("identity_sha256"):
                local_errors.append("Calibration identity digest mismatch")
            if identity.get("data_manifest"):
                run(lambda: self.reference(identity["data_manifest"]))
        if report["schema"] == PREP_SCHEMA:
            for split, row in report["splits"].items():
                run(lambda split=split, row=row: self.dataset(report["data_root"], split, row["dataset_sha256"]))
                run(lambda row=row: self.check_file(resolve_record_path(row["manifest_path"], self.project), row["manifest_sha256"]))
            if report.get("epoch_indices"):
                run(lambda: self.reference(report["epoch_indices"]))
        if report["schema"] == NUM_SCHEMA:
            def fixture_check():
                fixture = report["fixture"]
                data = self.dataset(fixture["root"], fixture["split"], fixture["dataset_sha256"])
                batch = data.take(slice(fixture["offset"], fixture["offset"] + fixture["shape"][0]))
                if list(batch.input_ids.shape) != fixture["shape"] or batch.sha256 != fixture["sha256"]:
                    raise ValueError("Numerical fixture offset/shape/array identity differs")
                if int((batch.labels != -100).sum()) != fixture["native_targets"]:
                    raise ValueError("Numerical native loss mask differs")
            run(fixture_check)
        if report["schema"] == TRAIN_SCHEMA and identity:
            def training_data_check():
                import numpy as np
                preparation_path = self.reference(identity["data_manifest"])
                preparation = json.loads(preparation_path.read_text())
                train = self.dataset(preparation["data_root"], "train", identity["data"]["train"]["array_sha256"])
                dev = self.dataset(preparation["data_root"], "dev", identity["data"]["dev"]["array_sha256"])
                if preparation["protocol"]["sha256"] != identity["protocol_sha256"]:
                    raise ValueError("Calibration dataset and run protocol anchors differ")
                if identity["initial_checkpoint_sha256"] != report["initial_checkpoint"]["sha256"]:
                    raise ValueError("Calibration initial checkpoint identity differs")
                batch, per_epoch = identity["physical_batch"], identity["updates_per_epoch"]
                if len(train) != batch * per_epoch or len(dev) != 1280 or train.input_ids.shape[1] != 256:
                    raise ValueError("Calibration data size/batch/length differs")
                order_path = self.reference(preparation["epoch_indices"])
                order = np.load(order_path, allow_pickle=False)
                if order.shape != (50, len(train)) or preparation["shuffle_seed"] != identity["shuffle_seed"]:
                    raise ValueError("Calibration epoch order dimensions or shuffle seed differ")
                for row in report.get("history", []):
                    completed = row["update"] - 1
                    epoch, position = divmod(completed, per_epoch)
                    indices = order[epoch, position * batch:(position + 1) * batch]
                    h = hashlib.sha256(f"array:{indices.dtype}:{indices.shape}:".encode() + indices.tobytes()).hexdigest()
                    if h != row["indices_sha256"] or train.take(indices).sha256 != row["batch_sha256"]:
                        raise ValueError(f"Calibration batch/order hashes differ at update{row['update']}")
            run(training_data_check)
        if report["schema"] == PROFILE_SCHEMA:
            def profile_data_check():
                for split, record in report.get("data", {}).items():
                    matches = [value for (_, stored_split), value in self.datasets.items()
                               if stored_split == split and value.sha256 == record["array_sha256"]
                               and value.manifest["manifest_sha256"] == record["manifest_sha256"]]
                    if len(matches) != 1 or len(matches[0]) != record["examples"] or matches[0].input_ids.shape[1] != record["length"]:
                        raise ValueError("Operational profile dataset is not bound to one verified prepared corpus")
            run(profile_data_check)
        entry["integrity_errors"] = local_errors
        self.errors.extend(f"{path.relative_to(self.lineage)}: {message}" for message in local_errors)
        return local_errors


def numerical_rows(report, path=""):
    rows = []
    initialized = report.get("completed_updates") == 0
    shape = report.get("fixture", {}).get("shape", [None, None])
    for name, comparison in report.get("comparisons", {}).items():
        full = comparison.get("actual_ce_gradients", {})
        side = comparison.get("independent_side_gradients", {})
        adam = comparison.get("adam", {})
        row = {"report": path, "status": report.get("status"), "arm": report.get("arm"),
            "comparison": name, "batch": shape[0], "length": shape[1],
            "checkpoint_updates": report.get("completed_updates"), "initialized": initialized,
            "split": report.get("fixture", {}).get("split"),
            "gradient_relative_l2": full.get("global_parameter_relative_l2"),
            "gradient_global_pass": full.get("global_parameter_l2_pass"),
            "gradient_tensor_l2_failures": full.get("per_tensor_l2_failures", []),
            "gradient_tensor_maximum_failures": full.get("per_tensor_maximum_failures", []),
            "fp32_elementwise_failures": full.get("fp32_elementwise_failures", []),
            "side_relative_l2": side.get("global_parameter_relative_l2"),
            "side_pass": side.get("pass"), "side_fp32_elementwise_failures": side.get("fp32_elementwise_failures", []),
            "side_tensor_l2_failures": side.get("per_tensor_l2_failures", []),
            "side_tensor_maximum_failures": side.get("per_tensor_maximum_failures", []),
            "adam_relative_l2": adam.get("global_delta_relative_l2"), "adam_cosine": adam.get("global_delta_cosine"),
            "adam_guardrail": adam.get("guardrail"), "adam_guardrail_pass": adam.get("guardrail_pass"),
            "adam_plotted_criterion": "initial cosine >=.99" if initialized else "trained relative L2 <=.015625",
            "initial_relative_l2_is_descriptive": initialized,
            "adam_near_zero_error_fraction": adam.get("gradient_near_zero_buckets", {}).get("near_zero", {}).get("fraction_of_delta_error_energy"),
            "absolute_ce_difference": comparison.get("absolute_ce_difference"),
            "machine_screens_pass": comparison.get("machine_screens_pass"),
            "scaling_performed": report.get("scale_check_performed", False),
            "scaling_pass": comparison.get("scaling_pass"),
            "causality_pass": report.get("causality_checks_pass"),
            "candidate_checks_pass": report.get("candidate_checks_pass"),
            "raw_machine_screens_pass": report.get("raw_machine_screens_pass")}
        rows.append(row)
    if not rows:
        rows.append({"report": path, "status": report.get("status"), "arm": report.get("arm"),
                     "batch": shape[0], "length": shape[1], "comparison": None,
                     "error_type": report.get("error_type"), "candidate_checks_pass": None})
    return rows


def numerical_consistency_errors(report):
    errors = []
    if report.get("status") != "diagnostics_complete":
        return errors
    candidate = report.get("comparisons", {}).get("mixed_vs_tiled_fp32")
    if not candidate:
        return ["Completed numerical report lacks candidate comparison"]
    for name, audit in report.get("compiler", {}).items():
        expected_graphs = report["arm"] == "cdrm" and name != "naive_fp32"
        if audit.get("validation_status") != "passed" or audit.get("required") is not True or audit.get("require_graphs") != expected_graphs:
            errors.append(f"Numerical compiler audit incomplete: {name}")
    if not {"tiled_fp32", "mixed"}.issubset(report.get("compiler", {})):
        errors.append("Completed numerical report lacks required compiler arms")
    for name, comparison in report["comparisons"].items():
        adam = comparison.get("adam", {})
        if report["completed_updates"] == 0:
            expected = adam.get("global_delta_cosine") is not None and adam["global_delta_cosine"] >= .99
            if not str(adam.get("guardrail", "")).startswith("initial"):
                errors.append(f"Initial Adam uses the wrong criterion: {name}")
        else:
            expected = adam.get("global_delta_relative_l2", float("inf")) <= .015625
            if not str(adam.get("guardrail", "")).startswith("trained"):
                errors.append(f"Trained Adam uses the wrong criterion: {name}")
        if adam.get("guardrail_pass") != expected:
            errors.append(f"Adam criterion summary is inconsistent: {name}")
        if report.get("scale_check_performed"):
            leaves = comparison.get("output_scaling", {})
            if len(leaves) != 2 or "scaling_pass" not in comparison:
                errors.append(f"Missing requested scaling evidence: {name}")
            for arm, scaling in leaves.items():
                groups = scaling.get("scaling", {})
                equal = (set(groups) == {"0.03125", "32.0"} and
                         all(group and all(row.get("bitwise_normalized_equal") is True for row in group.values()) for group in groups.values())
                         and scaling.get("fixed_forward_logits_bitwise_equal") is True)
                if scaling.get("pass") != equal:
                    errors.append(f"Scaling leaves disagree with summary: {name}/{arm}")
    expected_candidate = bool(candidate.get("machine_screens_pass") and candidate.get("scaling_pass", True)
                              and report.get("causality_checks_pass", True))
    if report.get("candidate_checks_pass") != expected_candidate:
        errors.append("Candidate numerical summary does not match its component flags")
    return errors


def profile_row(report, path=""):
    steady = [row["seconds"] for row in report.get("updates", []) if not row.get("warmup", True)]
    shape = report.get("shape", [None, None])
    return {"report": path, "status": report.get("status"), "arm": report.get("arm"),
            "precision": report.get("precision"), "batch": shape[0], "length": shape[1],
            "fit_success": report.get("status") == "profile_complete",
            "median_seconds": statistics.median(steady) if steady else None,
            "mean_seconds": statistics.mean(steady) if steady else None, "steady_updates": len(steady),
            "allocated_gib": report["training_peak_allocated_bytes"] / 2**30 if report.get("training_peak_allocated_bytes") is not None else None,
            "reserved_gib": report["training_peak_reserved_bytes"] / 2**30 if report.get("training_peak_reserved_bytes") is not None else None,
            "development_seconds": report.get("development", {}).get("seconds"),
            "error_type": report.get("error_type"),
            "runtime_provenance": report.get("runtime"),
            "timing_scope": "Actual checked update helper; excludes compilation warmup, W&B and checkpoint overhead"}


def profile_consistency_errors(report):
    if report.get("status") != "profile_complete":
        return []
    row = profile_row(report)
    errors = []
    if not row["steady_updates"] or not math.isfinite(row["mean_seconds"]) or row["mean_seconds"] <= 0:
        errors.append("Completed profile has no finite positive steady timings")
    for key in ("mean_seconds", "median_seconds", "steady_updates"):
        actual = report.get("timing", {}).get(key)
        if not isinstance(row[key], (float, int)) or not isinstance(actual, (float, int)) or not math.isclose(row[key], actual, rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"Profile timing summary differs from nonwarmup rows: {key}")
    if report.get("compiler", {}).get("required") is not True:
        errors.append("Completed profile lacks required compiler audit")
    return errors


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(child) for key, child in value.items() if key not in {"seconds", "elapsed_seconds"}}
    if isinstance(value, list):
        return [without_timing(child) for child in value]
    return value


def calibration_role(entry, expected_epochs):
    """Only the declared calibration endpoint directory can select an LR."""
    report, directory = entry["report"], Path(entry["path"]).parent
    if (report.get("recovery_comparison") or report.get("arguments", {}).get("reference_final")
            or directory.parent.name == "recovery"):
        return "recovery_reference"
    if directory.parent.name == "calibration" and directory.name.endswith(f"-e{expected_epochs}"):
        return "authoritative_endpoint"
    return "partial_reference"


def calibration_cohort(entries, protocol):
    """Keep reference continuations; select LRs only from declared endpoint runs."""
    expected_epochs = protocol["calibration"]["epochs"]
    groups, errors, recovery = {}, [], []
    for entry in entries:
        report = entry["report"]
        if report.get("schema") != TRAIN_SCHEMA:
            continue
        if report.get("recovery_comparison"):
            recovery.append({"report": entry["path"], **report["recovery_comparison"]})
        identity = report.get("identity", {})
        if not identity:
            continue
        key = (report["arm"], identity["base_lr"], report["precision"])
        if key[0] not in ("cdrm", "seq") or key[1] not in protocol["calibration"]["learning_rates"]:
            errors.append(f"Undeclared calibration arm/LR: {entry['path']}")
        groups.setdefault(key, []).append(entry)
    selected, dev_rows, train_rows = [], [], []
    for arm in ("cdrm", "seq"):
        for lr in protocol["calibration"]["learning_rates"]:
            candidates = groups.get((arm, lr, "bf16"), [])
            if not candidates:
                selected.append({"arm": arm, "lr": lr, "status": "missing", "report": None, "completed_epochs": 0})
                continue
            authoritative = [entry for entry in candidates if calibration_role(entry, expected_epochs) == "authoritative_endpoint"]
            if len(authoritative) > 1:
                errors.append(f"Multiple authoritative calibration endpoints for {arm}/LR{lr}")
            candidates.sort(key=lambda entry: (calibration_role(entry, expected_epochs) == "authoritative_endpoint",
                len(entry["report"].get("history", [])),
                entry["report"].get("status") == "complete", not bool(entry["report"].get("recovery_comparison")), entry["path"]))
            chosen = candidates[-1]
            report, identity = chosen["report"], chosen["report"]["identity"]
            history, development = report.get("history", []), report.get("development", {})
            for prior in candidates:
                old = prior["report"]
                common_updates = min(len(old.get("history", [])), len(history))
                if old["identity"] != identity:
                    errors.append(f"Conflicting calibration identity for {arm}/LR{lr}")
                elif without_timing(old.get("history", [])[:common_updates]) != without_timing(history[:common_updates]):
                    errors.append(f"Conflicting continuation prefix for {arm}/LR{lr}")
                for update, metric in old.get("development", {}).items():
                    if update in development and without_timing(metric) != without_timing(development[update]):
                        errors.append(f"Conflicting development repeat for {arm}/LR{lr}/u{update}")
            batch, per_epoch = identity["physical_batch"], identity["updates_per_epoch"]
            for i, row in enumerate(history, 1):
                expected_lr = 1e-6 + (lr - 1e-6) * (1 + math.cos(math.pi * ((i - 1) // per_epoch) / 50)) / 2
                if (row.get("update"), row.get("epoch"), row.get("batch_in_epoch")) != (i, (i - 1) // per_epoch + 1, (i - 1) % per_epoch):
                    errors.append(f"Calibration update/cursor mismatch: {arm}/LR{lr}/u{i}")
                if not math.isclose(row.get("learning_rate", -1), expected_lr, rel_tol=1e-12, abs_tol=1e-15):
                    errors.append(f"Calibration epoch cosine differs: {arm}/LR{lr}/u{i}")
                train_rows.append({"arm": arm, "lr": lr, "update": i, "epoch": i / per_epoch,
                    "report": chosen["path"], "seconds": row.get("seconds"), "native_ce": row.get("native_loss"),
                    "native_token_accuracy": row.get("native", {}).get("token_accuracy"),
                    "answer_token_accuracy": row.get("answer", {}).get("token_accuracy"),
                    "gradient_norm": row.get("gradient_norm"), "clipped": row.get("clipped")})
            for update in sorted(development, key=int):
                value = development[update]
                if value.get("native") != value.get("answer"):
                    errors.append(f"Development native/answer masks or metric counts differ: {arm}/LR{lr}/u{update}")
                native = value.get("native", {})
                dev_rows.append({"arm": arm, "lr": lr, "update": int(update), "epoch": int(update) / per_epoch,
                    "report": chosen["path"], "native_ce": native.get("ce"), "native_token_accuracy": native.get("token_accuracy"),
                    "sequence_exact_match": native.get("sequence_exact_match"), "targets": native.get("targets"),
                    "examples": native.get("examples"), "evaluation_seconds": value.get("seconds")})
            final_update = expected_epochs * per_epoch
            final = development.get(str(final_update), {}).get("native")
            role = calibration_role(chosen, expected_epochs)
            complete = (role == "authoritative_endpoint" and chosen.get("observation") == "closed_report"
                        and report.get("status") == "complete" and len(history) == final_update and
                        report.get("completed_epochs") == expected_epochs and final is not None)
            selected.append({"arm": arm, "lr": lr, "report": chosen["path"], "status": report.get("status"),
                "role": role, "reference_reports": [{"report": candidate["path"],
                    "role": calibration_role(candidate, expected_epochs),
                    "completed_updates": len(candidate["report"].get("history", []))} for candidate in candidates if candidate is not chosen],
                "batch": batch, "completed_epochs": len(history) // per_epoch,
                "completed_updates": len(history), "frozen_endpoint_complete": complete,
                "endpoint_native": final, "training_seconds": sum(row.get("seconds", 0) for row in history),
                "steady_update_seconds": statistics.mean(row["seconds"] for row in history[1:]) if len(history) > 1 else None,
                "shared_initialization_sha256": identity.get("shared_initialization_sha256"),
                "data_identity": identity.get("data"), "source_sha256": identity.get("source_sha256")})
    complete = all(row.get("frozen_endpoint_complete", False) for row in selected)
    present = [row for row in selected if row["report"]]
    for key in ("shared_initialization_sha256", "data_identity", "source_sha256", "batch"):
        if len({json.dumps(row[key], sort_keys=True) for row in present}) > 1:
            errors.append(f"Paired calibration differs in {key}")
    winners = {}
    if complete and not errors:
        for arm in ("cdrm", "seq"):
            winner = min((row for row in selected if row["arm"] == arm),
                         key=lambda row: (-row["endpoint_native"]["token_accuracy"], row["endpoint_native"]["ce"], row["lr"]))
            winners[arm] = {"lr": winner["lr"], "development_native": winner["endpoint_native"],
                            "selection_scope": "Development calibration only; not a held-out architecture comparison"}
    return {"selected_runs": selected, "development_rows": dev_rows, "training_rows": train_rows,
            "all_six_frozen_endpoints_complete": complete, "selection": winners,
            "errors": errors, "recovery": recovery}


def cost_projections(profiles, calibration, epochs=(25, 50)):
    fits = [row for row in profiles if row["fit_success"] and row.get("usable_for_projection", True) and row["precision"] == "bf16"
            and row["length"] == 300 and row["mean_seconds"] is not None]
    batches = set(row["batch"] for row in fits if row["arm"] == "cdrm") & set(row["batch"] for row in fits if row["arm"] == "seq")
    if not batches:
        return {"status": "unavailable", "reason": "Need successful matched BF16 T300 profiles for both architectures", "rows": []}
    batch = max(batches)
    seconds = {arm: max(row["mean_seconds"] for row in fits if row["arm"] == arm and row["batch"] == batch) for arm in ("cdrm", "seq")}
    actual = {}
    for arm in ("cdrm", "seq"):
        values = [row["steady_update_seconds"] for row in calibration["selected_runs"] if row["arm"] == arm and row.get("batch") == batch and row.get("steady_update_seconds")]
        if values:
            actual[arm] = statistics.mean(values)
    rows = []
    for horizon in epochs:
        base = {"epochs": horizon, "runs": 24, "replicates": 3, "lengths": [64, 128, 256, 300],
                "batch": batch, "updates_total": 24 * 12800 // batch * horizon,
                "excludes": "Compilation, evaluation, W&B, checkpointing and calibration tuning cost"}
        rows.append({**base, "estimate": "T300 timing proxy for all four lengths; upper proxy, not a proven bound",
                     "training_hours": 3 * 4 * (12800 / batch) * horizon * sum(seconds.values()) / 3600})
        if len(actual) == 2:
            rows.append({**base, "estimate": "Measured T256 calibration mean after first update; other three lengths use T300 proxy",
                         "training_hours": 3 * (12800 / batch) * horizon * (sum(actual.values()) + 3 * sum(seconds.values())) / 3600})
    return {"status": "projected_only", "largest_common_profiled_batch": batch,
            "profile_T300_seconds_by_arm": seconds, "calibration_T256_seconds_by_arm": actual, "rows": rows}


def summarize(entries, protocol):
    numerical, profiles, attempts, errors, preparations, counts = [], [], [], [], [], {}
    for entry in entries:
        report, path = entry["report"], entry["path"]
        attempts.append({"report": path, "sha256": entry["sha256"], "schema": report["schema"],
                         "status": report.get("status"), "observation": entry.get("observation"),
                         "error_type": report.get("error_type"), "error": report.get("error"),
                         "integrity_errors": entry.get("integrity_errors", []),
                         "wandb": report.get("wandb")})
        if report["schema"] == NUM_SCHEMA:
            numerical.extend(numerical_rows(report, path))
            errors.extend(f"{path}: {error}" for error in numerical_consistency_errors(report))
        elif report["schema"] == PROFILE_SCHEMA:
            profile_errors = profile_consistency_errors(report)
            profiles.append({**profile_row(report, path),
                             "usable_for_projection": not profile_errors and not entry.get("integrity_errors")})
            errors.extend(f"{path}: {error}" for error in profile_errors)
        elif report["schema"] == PREP_SCHEMA:
            preparations.append({"report": path, "role": report["role"], "length": report["length"], "splits": report["splits"]})
        elif report["schema"] == INIT_SCHEMA:
            for arm, value in report.get("initialization", {}).get("parameter_counts", {}).items():
                if value["total"] != protocol["models"][arm]["parameters"]:
                    errors.append(f"Initialization parameter count differs for {arm}")
                counts[arm] = value
    calibration = calibration_cohort(entries, protocol)
    errors.extend(calibration["errors"])
    return {"schema": SCHEMA, "scope": "Phase one: numerical/operational evidence and development-only T256 LR calibration",
            "final_sweep_performed": False, "final_test_evaluated": False, "model_counts": counts,
            "attempts": attempts, "numerical": numerical, "profiles": profiles,
            "preparations": preparations, "calibration": calibration,
            "cost_projections": cost_projections(profiles, calibration), "consistency_errors": errors,
            "qualification": "Execution completion is separate from numerical pass. Raw flags remain visible; initialized Adam relative L2 is descriptive and its original guard is cosine>=.99. Runtime projections are unmeasured multi-length estimates. Calibration accuracy is not a final held-out comparison."}


def write_csv(path, rows):
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else ["no_completed_observations"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def make_plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outputs = []
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)
    rows = [row for row in report["numerical"] if row.get("comparison") == "mixed_vs_tiled_fp32"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, title, initialized in zip(axes,
        ("gradient_relative_l2", "adam_cosine", "adam_relative_l2"),
        ("Full gradient relative L2 (%)", "Initialization Adam cosine (guard ≥0.99)", "Trained Adam relative L2 (%)"),
        (None, True, False)):
        selected = [row for row in rows if row.get(field) is not None and (initialized is None or row["initialized"] == initialized)]
        for index, row in enumerate(selected):
            value = row[field] if field == "adam_cosine" else 100 * row[field]
            passed = row.get("gradient_global_pass") if field == "gradient_relative_l2" else row.get("adam_guardrail_pass")
            ax.scatter(index, value, c="#167d55" if passed else "#bd3939", marker="o" if row["arm"] == "cdrm" else "s")
        ax.set_xticks(range(len(selected)), [f"{r['arm']} B{r['batch']}/T{r['length']} u{r['checkpoint_updates']}" for r in selected], rotation=65, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.2)
        if field == "adam_cosine": ax.axhline(.99, color="gray", linestyle="--", linewidth=1)
        if field == "adam_relative_l2": ax.axhline(1.5625, color="gray", linestyle="--", linewidth=1)
        if not selected: ax.text(.5, .5, "No completed observations", transform=ax.transAxes, ha="center")
    fig.suptitle("150M-family numerical diagnostics — flags preserved; initial update distance remains descriptive", fontsize=11)
    save(fig, "numerical")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, row in enumerate(report["profiles"]):
        label = f"{row['arm']} {row['precision']} B{row['batch']}/T{row['length']}"
        for ax, field in zip(axes, ("mean_seconds", "allocated_gib")):
            if row.get(field) is not None:
                ax.scatter(index, row[field], c="#167d55" if row["fit_success"] else "#bd3939")
                ax.annotate(label, (index, row[field]), xytext=(0, 5), textcoords="offset points", rotation=45, fontsize=7)
    axes[0].set_ylabel("Steady checked update seconds")
    axes[1].set_ylabel("Peak allocated GiB")
    for ax in axes: ax.grid(alpha=.2); ax.set_xticks([])
    fig.suptitle("Operational fit/profile observations — warmup excluded from timing")
    save(fig, "profiles")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for selected in report["calibration"]["selected_runs"]:
        points = [row for row in report["calibration"]["development_rows"] if row["arm"] == selected["arm"] and row["lr"] == selected["lr"]]
        if not points: continue
        label = f"{selected['arm']} LR{selected['lr']:g} ({selected['status']})"
        axes[0].plot([row["epoch"] for row in points], [100 * row["native_token_accuracy"] for row in points], label=label)
        axes[1].plot([row["epoch"] for row in points], [row["native_ce"] for row in points], label=label)
    calibration = next((row for row in report["preparations"] if row["role"] == "calibration"), None)
    if calibration:
        shortcut = calibration["splits"]["dev"]["baselines"]["query_ignoring_answer_prefix"]["answer_accuracy"]
        axes[0].axhline(shortcut * 100, color="gray", linestyle="--", label="Query-ignoring answer-prefix shortcut")
    axes[0].set_ylabel("Native development answer-token accuracy (%)")
    axes[1].set_ylabel("Native development answer CE")
    for ax in axes:
        ax.set_xlabel("Epochs of native training data"); ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    fig.suptitle("T256 development LR calibration — no final-test or multi-seed comparison")
    save(fig, "calibration-development")
    return outputs


def markdown(report):
    lines = ["# Fuzzy-recall phase-one generated evidence", "", report["scope"] + ".", "", report["qualification"], "",
             f"Integrity/consistency errors: **{len(report.get('integrity_errors', []))}**. "
             f"All six frozen calibration endpoints complete: **{report['calibration']['all_six_frozen_endpoints_complete']}**.", "",
             "Native training uses dense shifted targets including padding. Held-out development scores native masked value tokens; answers are teacher forced. Final data and the four-length main sweep are not included.", ""]
    calibration = next((row for row in report["preparations"] if row["role"] == "calibration"), None)
    if calibration:
        dev = calibration["splits"]["dev"]; coverage = dev["integrity"]["oracle_coverage"]
        shortcut = dev["baselines"]["query_ignoring_answer_prefix"]["answer_accuracy"]
        lines += [f"Calibration dev retains **{coverage['unavailable_tokens']} / {coverage['scored_tokens']}** scored tokens without an earlier matching key. Their native labels remain in every primary denominator. The query-ignoring answer-prefix shortcut reaches **{shortcut:.2%}**; independent uniform eight-value guesses give12.5%. Retrieval coverage is not a hard task ceiling.", ""]
    lines += ["| Arm | Unique parameters | Adapter parameters |", "| --- | ---: | ---: |"]
    for arm, counts in report["model_counts"].items(): lines.append(f"| {arm} | {counts['total']:,} | {counts['adapters']:,} |")
    lines += ["", "| Numerical case | Status | Full gradient L2 | Adam criterion | Adam L2 (descriptive at init) | Pass |", "| --- | --- | ---: | --- | ---: | --- |"]
    def percent(value): return f"{value:.4%}" if value is not None else "unavailable"
    for row in report["numerical"]:
        if row.get("comparison") != "mixed_vs_tiled_fp32": continue
        lines.append(f"| {row['arm']} B{row['batch']}/T{row['length']} u{row['checkpoint_updates']} | {row['status']} | {percent(row['gradient_relative_l2'])} | {row['adam_guardrail']} | {percent(row['adam_relative_l2'])} | {row['candidate_checks_pass']} |")
    lines += ["", "All attempts and full flag lists are retained in report.json and CSVs, including failed/incomplete attempts and strict FP32 coordinate failures.", "",
              "| Arm | LR | Status | Completed epochs | Frozen-endpoint dev accuracy |", "| --- | ---: | --- | ---: | ---: |"]
    for row in report["calibration"]["selected_runs"]:
        value = row.get("endpoint_native")
        accuracy = f"{value['token_accuracy']:.3%}" if value else "pending"
        lines.append(f"| {row['arm']} | {row['lr']:g} | {row['status']} | {row['completed_epochs']} | {accuracy} |")
    lines += ["", "| Projected main sweep | Epochs | Total updates | Training hours |", "| --- | ---: | ---: | ---: |"]
    for row in report["cost_projections"]["rows"]:
        lines.append(f"| {row['estimate']} | {row['epochs']} | {row['updates_total']:,} | {row['training_hours']:.2f} |")
    lines += ["", "Projections cover24 runs and exclude compilation, evaluation, logging, checkpointing and calibration cost. T300 extrapolation is an upper proxy, not a measured or proven bound for the complete sweep.", "",
              "![Numerical diagnostics](numerical.png)", "", "![Operational profiles](profiles.png)", "", "![Development calibration](calibration-development.png)", ""]
    return "\n".join(lines)


def execute(args):
    lineage, output = args.lineage.resolve(), args.output_dir.resolve()
    if not lineage.is_dir() or not output.is_relative_to(lineage) or output.exists():
        raise ValueError("Require an existing explicit lineage and a new output directory within it")
    protocol_path = args.protocol.resolve() if args.protocol else lineage / "decisions/protocol-v1.json"
    if not protocol_path.is_relative_to(lineage): raise ValueError("Protocol must belong to supplied lineage")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "cdrm-fuzzy-prospective-protocol-v1": raise ValueError("Unknown prospective protocol")
    entries = scan_reports(lineage)
    audit = ArtifactAudit(lineage)
    for entry in sorted(entries, key=lambda entry: entry["report"]["schema"] != PREP_SCHEMA): audit.audit(entry)
    report = summarize(entries, protocol)
    report.update(status="summary_complete", lineage=str(lineage), protocol={"path": str(protocol_path), "sha256": digest(protocol_path)},
                  integrity_errors=audit.errors + report["consistency_errors"],
                  verified_files={name: {key: value for key, value in row.items() if key != "signature"} for name, row in audit.files.items()},
                  source_sha256={name: digest(ROOT / name) for name in ("scripts/cdrm_fuzzy_report.py",
                      "tests/test_cdrm_fuzzy_report.py", "scripts/experiment_tracking.py", "cdrm/mad_data.py")})
    report["integrity_pass"] = not report["integrity_errors"]
    output.mkdir(parents=True)
    for name, expected in report["source_sha256"].items():
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
        if digest(destination) != expected: raise ValueError("Reporting source snapshot changed")
    tracker = None
    try:
        for name, rows in (("numerical", report["numerical"]), ("profiles", report["profiles"]),
                           ("calibration-dev", report["calibration"]["development_rows"]),
                           ("calibration-train", report["calibration"]["training_rows"]),
                           ("cost-projections", report["cost_projections"]["rows"]), ("attempts", report["attempts"])):
            write_csv(output / f"{name}.csv", rows)
        images = make_plots(report, output)
        (output / "summary.md").write_text(markdown(report))
        if args.wandb_project:
            from experiment_tracking import OnlineTracker
            tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                                    name=args.wandb_run_name, output_dir=output)
            report["wandb"] = tracker.record
            tracker.start({"evidence": "phase-one aggregate", "protocol_sha256": digest(protocol_path),
                           "scope": report["scope"], "report_sources": [{"path": entry["path"], "sha256": entry["sha256"]} for entry in entries]})
            import wandb
            tracker.log({f"plots/{path.stem}": wandb.Image(str(path)) for path in images if path.suffix == ".png"})
            for row in report["calibration"]["development_rows"]:
                tracker.log({"update": row["update"], f"dev/{row['arm']}/lr{row['lr']:g}/accuracy": row["native_token_accuracy"],
                             f"dev/{row['arm']}/lr{row['lr']:g}/ce": row["native_ce"]})
            tracker.summary({"integrity_pass": report["integrity_pass"],
                             "all_six_calibration_endpoints_complete": report["calibration"]["all_six_frozen_endpoints_complete"],
                             "final_test_evaluated": False, "final_sweep_performed": False,
                             "numerical_candidate_failure_rows": sum(row.get("candidate_checks_pass") is False for row in report["numerical"])})
        for path, value in audit.files.items(): audit.check_file(path, value["sha256"])
    except Exception as error:
        report.update(status="publication_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if tracker: tracker.finish(succeeded=report["status"] == "summary_complete")
        except Exception as error:
            report.update(status="publication_failed", final_sync_error_type=type(error).__name__)
            raise
        finally:
            json_write(output / "report.json", report)
    return report


def main():
    from experiment_tracking import add_wandb_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file(): parser.error("Run CPU reporting inside the project container")
    report = execute(args)
    print(json.dumps({"status": report["status"], "integrity_pass": report["integrity_pass"],
                      "report": str(args.output_dir / "report.json")}))
    if not report["integrity_pass"]: raise SystemExit(2)


if __name__ == "__main__":
    main()
