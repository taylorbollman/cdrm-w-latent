#!/usr/bin/env python3
"""Aggregate retained CDRM records on CPU; never train or evaluate a model.

Discovery follows report schemas and evidence/argument fields, not directory
names. Resumed histories are cumulative and are selected once. Missing/running/
failed jobs are disclosed and never converted into successful result curves.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = "cdrm-reference-validation-v1"
TRAINING = "cdrm-fp32-training-v1"
EXTRAS = {"cdrm.deep_adapter.weight", "cdrm.bridge_adapter.weight"}
TOPOLOGY_FIELDS = {"cdrm_enabled", "cdrm_source", "recurrent_layers", "recurrent_backend", "reference_eager"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def core_sources(source):
    return {key: value for key, value in source.items() if key.startswith("recurrent-transformer/olmo/")}


def model_shape(config):
    return {key: config[key] for key in ("n_layers", "d_model", "n_heads", "mlp_hidden_size", "vocab_size")}


def setting_key(identity):
    """Depth/difficulty/data boundaries precede any search for a paired arm."""
    common_model = {key: value for key, value in identity["model_config"].items() if key not in TOPOLOGY_FIELDS}
    return digest({key: identity[key] for key in ("task", "preset", "seed", "train", "dev", "physical_batch", "shuffle_seed")}
                  | {"model": common_model})


def check_precision(contract, actual, name, *, restored_without_update=False):
    for key, expected in {"float32_matmul_precision": "highest", "tf32_matmul": False,
                          "tf32_cudnn": False, "deterministic_algorithms": True,
                          "flash_sdpa_enabled": False, "memory_efficient_sdpa_enabled": False,
                          "math_sdpa_enabled": True}.items():
        require(contract.get(key) == expected, f"{name}: unsupported precision setting {key}")
    if actual is not None:
        for key in ("parameters", "moments") + (() if restored_without_update else ("gradients",)):
            require(actual.get(key, {}).get("dtypes") == ["torch.float32"], f"{name}: {key} dtype")
            require(actual[key].get("finite") is True, f"{name}: nonfinite {key}")
        if restored_without_update:
            require(actual["gradients"]["tensors"] == 0 and actual["gradients"]["dtypes"] == [], f"{name}: unexpected gradients in an immediate restored pause")
            require(len(actual["missing_gradients"]) == actual["parameters"]["tensors"], f"{name}: inconsistent restored gradient absence")
        else:
            require(not actual.get("missing_gradients"), f"{name}: missing parameter gradients")


def check_metrics(values, name):
    required = ("ce_sum", "targets", "correct", "exact", "examples", "ce", "token_accuracy", "sequence_exact_match")
    require(all(key in values for key in required), f"{name}: incomplete metrics")
    require(values["targets"] > 0 and values["examples"] > 0, f"{name}: empty metrics")
    require(0 <= values["correct"] <= values["targets"] and 0 <= values["exact"] <= values["examples"], f"{name}: invalid counts")
    for key, numerator, denominator in (("ce", "ce_sum", "targets"),
                                         ("token_accuracy", "correct", "targets"),
                                         ("sequence_exact_match", "exact", "examples")):
        require(math.isfinite(values[key]) and math.isclose(values[key], values[numerator] / values[denominator], rel_tol=1e-9, abs_tol=1e-12), f"{name}: inconsistent {key}")


def compact_metrics(values):
    return {key: values[key] for key in ("ce", "token_accuracy", "sequence_exact_match", "targets", "examples")}


def comparisons(values):
    return {"tensors": len(values),
            "elementwise_failures": [name for name, item in values.items() if not item.get("elementwise_pass")],
            "max_abs_error": max((item.get("max_abs_error", 0) or 0 for item in values.values()), default=0),
            "max_relative_l2": max((item.get("rel_l2", 0) or 0 for item in values.values()), default=0)}


class Reporter:
    def __init__(self, run_root, storage_prefix=None):
        self.run_root = Path(run_root).resolve()
        self.storage_prefix = storage_prefix.rstrip("/") if storage_prefix else None
        self.hashes = {}
        self.report_by_dir = {}

    def artifact(self, path, expected=None):
        path = resolve(path).resolve()
        require(path.is_file(), f"Missing retained artifact: {path}")
        if path not in self.hashes:
            self.hashes[path] = file_sha(path)
        value = self.hashes[path]
        require(expected is None or value == expected, f"Artifact hash mismatch: {path}")
        result = {"path": relative(path), "sha256": value, "bytes": path.stat().st_size}
        if self.storage_prefix and path.is_relative_to(self.run_root):
            result["storage_uri"] = self.storage_prefix + "/" + str(path.relative_to(self.run_root))
        return result

    def discover(self):
        accepted, excluded = [], []
        for path in sorted(self.run_root.rglob("report.json")):
            try:
                report = json.loads(path.read_text())
            except json.JSONDecodeError:
                excluded.append({"path": relative(path), "status": "incomplete_json", "reason": "No values consumed"})
                continue
            schema = report.get("format", report.get("schema"))
            if schema not in (VALIDATION, TRAINING):
                continue
            entry = {"path": path, "report": report, "schema": schema}
            self.report_by_dir[path.parent.resolve()] = entry
            status = report.get("status")
            okay = status == "passed" if schema == VALIDATION else status in ("complete", "paused", "screened_early")
            if not okay:
                excluded.append({"artifact": self.artifact(path), "status": status, "evidence": report.get("evidence"),
                                 "arguments": report.get("arguments"), "error_type": report.get("error_type"),
                                 "error": report.get("error"), "reason": "Failed or unfinished reports do not supply result curves"})
                continue
            accepted.append(entry)
        # An active runner may not have emitted report.json yet. Its resolved
        # identity is enough to disclose its existence, never to invent results.
        for path in sorted(self.run_root.rglob("resolved-config.json")):
            if not (path.parent / "report.json").exists():
                excluded.append({"path": relative(path.parent), "status": "no_completed_report",
                                 "reason": "Possibly active; learning-curve JSONL is intentionally not consumed"})
        return accepted, excluded

    def validate_training(self, entry):
        report, label = entry["report"], relative(entry["path"])
        identity = report["identity"]
        require(report.get("identity_sha256") == digest(identity), f"{label}: identity digest")
        require(identity["topology"] == report["arguments"]["topology"], f"{label}: topology mismatch")
        idle = (report["status"] == "paused" and report.get("updates_this_process") == 0
                and report.get("resume", {}).get("loaded_exact") is True)
        require(report.get("final_precision") is not None, f"{label}: missing precision audit")
        check_precision(identity["precision"], report["final_precision"], label, restored_without_update=idle)
        history = report["history"]
        n, batch = identity["train"]["examples"], identity["physical_batch"]
        updates_per_epoch = math.ceil(n / batch)
        require(report["updates_per_epoch"] == updates_per_epoch, f"{label}: updates per epoch")
        completed, epoch = report["completed_updates"], report["completed_epochs"]
        require(len(history) == completed, f"{label}: cumulative history length")
        require(completed == epoch * updates_per_epoch + report["batch_in_epoch"], f"{label}: endpoint counters")
        if report["status"] == "complete":
            require(report["batch_in_epoch"] == 0, f"{label}: incomplete epoch")
            if report.get("termination_reason") != "early_ace":
                require(epoch == report["arguments"]["stop_epochs"], f"{label}: incomplete declared endpoint")
        for index, row in enumerate(history):
            require(row["update"] == index + 1, f"{label}: update ordering")
            require(row["epoch"] == index // updates_per_epoch + 1 and row["batch_in_epoch"] == index % updates_per_epoch, f"{label}: sampler offset")
            require(isinstance(row.get("indices_sha256"), str) and len(row["indices_sha256"]) == 64, f"{label}: missing sampler hash")
            require(math.isfinite(row["native_loss"]) and row["seconds"] >= 0, f"{label}: nonfinite training row")
        development = report["development"]
        require(set(map(int, development)) == set(range(epoch + 1)), f"{label}: missing completed-epoch development")
        for number, row in development.items():
            for kind in ("native", "answer"):
                check_metrics(row[kind], f"{label}: dev {number} {kind}")
        if report.get("termination_reason") == "early_ace" or report["status"] == "screened_early":
            screen = report["screening"]
            require(report["arguments"].get("screen_early_ace") is True and identity["topology"] == "seq"
                    and report.get("run_purpose") == "screening", f"{label}: unauthorized early-screen classification")
            require(screen["consecutive_epochs"] <= epoch <= screen["by_epoch"], f"{label}: early-screen window")
            for e in range(epoch - screen["consecutive_epochs"] + 1, epoch + 1):
                answer = development[str(e)]["answer"]
                require(answer["token_accuracy"] >= screen["token_accuracy"]
                        and answer["sequence_exact_match"] >= screen["sequence_exact_match"], f"{label}: early-ace criterion not met")
        require(identity["accumulation"] is False, f"{label}: accumulation outside supported profile")
        return updates_per_epoch

    def checkpoint(self, record, identity, epochs=None, updates=None):
        import torch
        reference = self.artifact(record["path"], record.get("sha256"))
        packet = torch.load(resolve(record["path"]), map_location="cpu", weights_only=False)
        require(packet["identity"] == identity and packet["identity_sha256"] == digest(identity), f"{reference['path']}: checkpoint identity")
        if epochs is not None:
            require(packet["completed_epochs"] == epochs, f"{reference['path']}: checkpoint epoch")
        if updates is not None:
            require(packet["completed_updates"] == updates, f"{reference['path']}: checkpoint update")
        reference.update(completed_epochs=packet["completed_epochs"], completed_updates=packet["completed_updates"])
        return reference, packet

    def initial_pair(self, seq, cdrm):
        import torch
        left = seq["report"]
        right = cdrm["report"]
        a_ref, a = self.checkpoint(left["roles"]["milestones"]["0"], left["identity"], 0, 0)
        b_ref, b = self.checkpoint(right["roles"]["milestones"]["0"], right["identity"], 0, 0)
        am, bm = a["model"], b["model"]
        require(set(bm) - set(am) == EXTRAS and not (set(am) - set(bm)), "CDRM must add exactly its two adapter weights")
        require(all(torch.equal(value, bm[name]) for name, value in am.items()), "Paired initial backbone weights differ")
        for name, value in bm.items():
            require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), f"Initial non-FP32/nonfinite tensor {name}")
        adapters = [bm[name] for name in sorted(EXTRAS)]
        require(all(bool(torch.count_nonzero(value)) for value in adapters), "Zero initial adapter")
        require(len({value.untyped_storage().data_ptr() for value in adapters}) == 2, "Adapter storage alias")
        width = right["identity"]["model_config"]["d_model"]
        require(all(tuple(value.shape) == (width, width) for value in adapters), "Unexpected adapter dimensions")
        require(right["construction"]["parameter_count"] - left["construction"]["parameter_count"] == 2 * width * width, "Parameter inventory difference")
        require(set(right["construction"]["conversion"]["new"]) == EXTRAS, "Construction adapter audit mismatch")
        return {"seq": a_ref, "cdrm": b_ref, "common_initial_tensors": len(am),
                "common_initial_weights_bitwise_equal": True, "extra_parameters": sorted(EXTRAS),
                "adapters_nonzero_and_distinct": True, "extra_parameter_elements": 2 * width * width}

    def lineage_wall(self, entry):
        total, segments, visited = 0., [], set()
        inherited_limit = None
        while entry is not None:
            path, report = entry["path"].resolve(), entry["report"]
            require(path not in visited, "Cyclic resume lineage")
            visited.add(path)
            seconds = report.get("elapsed_seconds", 0.)
            total += seconds
            segments.append({"report": relative(path), "seconds": seconds,
                             "completed_updates_in_receipt": report.get("completed_updates"),
                             "inherited_updates_limit": inherited_limit,
                             "receipt_contains_post_fork_work": inherited_limit is not None and report.get("completed_updates", 0) > inherited_limit})
            resume = report.get("resume")
            if not resume:
                break
            parent = resolve(resume["path"]).parent.resolve()
            inherited_limit = resume["completed_updates"]
            entry = self.report_by_dir.get(parent)
            require(entry is not None, f"Missing resume ancestor report: {parent}")
        return {"lineage_wall_seconds": total, "segments": segments,
                "scope": "Sum of full selected/ancestor invocation receipts, excluding separate evaluations/benchmarks; an ancestor receipt may include work past a diagnostic fork, explicitly flagged above"}

    def training_summary(self, entry):
        report = entry["report"]
        updates_per_epoch = self.validate_training(entry)
        history = report["history"]
        end = str(report["completed_epochs"])
        monitor_rows = [row for row in history if "state_summaries" in row]
        new_updates = report.get("updates_this_process", report["completed_updates"] - report.get("resume", {}).get("completed_updates", 0))
        data_setting = None
        data_root = report["arguments"].get("data_root")
        if data_root and (resolve(data_root) / "manifest.json").is_file():
            manifest_path = resolve(data_root) / "manifest.json"
            metadata = json.loads(manifest_path.read_text())
            data_setting = {"manifest": self.artifact(manifest_path), "setting": metadata.get("setting")}
        return {"artifact": self.artifact(entry["path"]), "evidence": report["evidence"], "status": report["status"],
                "task": report["identity"]["task"], "topology": report["identity"]["topology"],
                "preset": report["identity"]["preset"], "seed": report["identity"]["seed"],
                "shuffle_seed": report["identity"]["shuffle_seed"], "identity_sha256": report["identity_sha256"],
                "shape": model_shape(report["identity"]["model_config"]),
                "actual_sequence_length": report["identity"]["train"]["length"],
                "setting_id": setting_key(report["identity"]),
                "data_root": report["arguments"].get("data_root"),
                "data_setting": data_setting,
                "completed_epochs": report["completed_epochs"], "completed_updates": report["completed_updates"],
                "updates_this_process": new_updates,
                "classification": "successful_bookkeeping_pause_without_new_updates" if new_updates == 0 and report["status"] == "paused" else "training",
                "batch_in_epoch": report["batch_in_epoch"], "updates_per_epoch": updates_per_epoch,
                "physical_batch": report["identity"]["physical_batch"],
                "training_examples": report["identity"]["train"]["examples"],
                "native_loss_first": history[0]["native_loss"] if history else None,
                "native_loss_last": history[-1]["native_loss"] if history else None,
                "development_initial": compact_metrics(report["development"]["0"]["answer"]),
                "development_last": compact_metrics(report["development"][end]["answer"]),
                "recorded_update_seconds": sum(row["seconds"] for row in history),
                "input_tokens": sum(row["input_tokens"] for row in history),
                "native_targets": sum(row["native_targets"] for row in history),
                "cost": self.lineage_wall(entry), "roles": report["roles"],
                "initial_state_summaries": monitor_rows[0].get("state_summaries") if monitor_rows else None,
                "last_state_summaries": monitor_rows[-1].get("state_summaries") if monitor_rows else None,
                "training_answer_first": compact_metrics(monitor_rows[0]["answer"]) if monitor_rows and "answer" in monitor_rows[0] else None,
                "training_answer_last": compact_metrics(monitor_rows[-1]["answer"]) if monitor_rows and "answer" in monitor_rows[-1] else None,
                "calibration": report.get("calibration"), "screening": report.get("screening"),
                "run_purpose": report.get("run_purpose", report["arguments"].get("run_purpose", "pilot")),
                "termination_reason": report.get("termination_reason", "requested_endpoint" if report["status"] == "complete" else report["status"]),
                "recovery_comparison": report.get("recovery_comparison"),
                "core_source_sha256": digest(core_sources(report["identity"]["source_sha256"])),
                "full_source_sha256": digest(report["identity"]["source_sha256"])}

    def select_pairs(self, entries):
        groups = defaultdict(lambda: defaultdict(list))
        for entry in entries:
            identity = entry["report"]["identity"]
            groups[(identity["task"], identity["preset"], identity["seed"], setting_key(identity))][identity["topology"]].append(entry)
        selected, incomplete = [], []
        for key, arms in sorted(groups.items()):
            if not {"seq", "cdrm"}.issubset(arms):
                example = next(iter(arms.values()))[0]["report"]["identity"]
                incomplete.append({"task": key[0], "preset": key[1], "seed": key[2], "setting_id": key[3],
                                   "shape": model_shape(example["model_config"]), "available_arms": sorted(arms),
                                   "interpretation": "Unpaired calibration/screening only; no architecture comparison"})
                continue
            chosen = {}
            for topology in ("seq", "cdrm"):
                candidates = arms[topology]
                identities = {entry["report"]["identity_sha256"] for entry in candidates}
                require(len(identities) == 1, f"Ambiguous independent {key}/{topology} trajectories with different identities")
                candidates.sort(key=lambda entry: (entry["report"]["completed_updates"], entry["report"]["status"] == "complete", str(entry["path"])))
                chosen[topology] = candidates[-1]
                longest = chosen[topology]["report"]
                for older in candidates[:-1]:
                    old = older["report"]
                    for a, b in zip(old["history"], longest["history"]):
                        require({k: v for k, v in a.items() if k != "seconds"} == {k: v for k, v in b.items() if k != "seconds"}, f"Divergent cumulative trajectory: {older['path']}")
                    for epoch, metrics in old["development"].items():
                        require(all(metrics[k] == longest["development"][epoch][k] for k in ("native", "answer")), f"Divergent inherited development: {older['path']}")
            selected.append(chosen)
        return selected, incomplete

    def pair(self, selected):
        left, right = (selected[arm]["report"] for arm in ("seq", "cdrm"))
        a, b = left["identity"], right["identity"]
        # All experiment/runtime/source fields must match except the explicitly
        # declared topology and model-config flag. Source changes between older
        # NUM/OPS jobs are allowed; changes within a research pair are not.
        common_a = {key: value for key, value in a.items() if key not in ("topology", "model_config")}
        common_b = {key: value for key, value in b.items() if key not in ("topology", "model_config")}
        require(common_a == common_b, f"Research pairing identity mismatch: {a['task']}")
        cfg_a, cfg_b = a["model_config"], b["model_config"]
        require({k: v for k, v in cfg_a.items() if k not in TOPOLOGY_FIELDS} == {k: v for k, v in cfg_b.items() if k not in TOPOLOGY_FIELDS}, "Common backbone configuration differs")
        require(cfg_a["cdrm_enabled"] is False and cfg_b["cdrm_enabled"] is True, "Wrong primary-pair topology")
        require(cfg_b["cdrm_source"] == "deep" and cfg_b["recurrent_layers"] == [] and cfg_a["recurrent_layers"] == [], "Unexpected primary side topology")
        initial = self.initial_pair(selected["seq"], selected["cdrm"])
        epoch = max(set(map(int, left["development"])) & set(map(int, right["development"])))
        updates_per_epoch = left["updates_per_epoch"]
        common_updates = epoch * updates_per_epoch
        data_fields = ("update", "epoch", "batch_in_epoch", "indices_sha256", "input_tokens", "native_targets", "learning_rate")
        hashes = []
        for x, y in zip(left["history"][:common_updates], right["history"][:common_updates]):
            require(all(x[key] == y[key] for key in data_fields), f"Paired stream/schedule diverges at update {x['update']}")
            hashes.append({key: x[key] for key in data_fields})
        require(len(hashes) == common_updates, "Paired history does not reach common epoch")
        endpoints = {}
        for topology, report in (("seq", left), ("cdrm", right)):
            record = report["roles"]["milestones"].get(str(epoch))
            checkpoint = None
            if record is not None:
                checkpoint, packet = self.checkpoint(record, report["identity"], epoch, common_updates)
                del packet
            endpoints[topology] = {"report": relative(selected[topology]["path"]),
                                   "run_status": report["status"], "run_completed_epochs": report["completed_epochs"],
                                   "common_epoch_checkpoint": checkpoint,
                                   "development": compact_metrics(report["development"][str(epoch)]["answer"]),
                                   "recorded_update_seconds_through_common_epoch": sum(row["seconds"] for row in report["history"][:common_updates])}
        return {"task": a["task"], "preset": a["preset"], "seed": a["seed"],
                "setting_id": setting_key(a),
                "physical_batch": a["physical_batch"], "shape": model_shape(cfg_a),
                "actual_sequence_length": a["train"]["length"], "common_completed_epochs": epoch,
                "common_completed_updates": common_updates, "endpoints": endpoints,
                "pairing": {"all_common_identity_fields_equal": True, "source_maps_equal": True,
                            "core_sources_equal": True, "initial_inventory": initial,
                            "train_arrays_equal": True, "dev_arrays_equal": True,
                            "shuffle_seed": a["shuffle_seed"], "per_update_indices_verified": common_updates,
                            "stream_and_schedule_chain_sha256": digest(hashes),
                            "source_sha256": a["source_sha256"], "train": a["train"], "dev": a["dev"]},
                "interpretation": "Single-seed exploratory comparison at equal completed epochs/updates; no automatic advantage claim or training-seed confidence interval"}

    def validation_summary(self, entry):
        report = entry["report"]
        args, fixture, construction = report["arguments"], report["fixture"], report["construction"]
        actual = report.get("final_precision", report.get("actual_step", {}).get("precision"))
        check_precision(report["execution_contract"], actual, str(entry["path"]))
        out = {"artifact": self.artifact(entry["path"]), "evidence": report["evidence"],
               "mode": args["mode"], "status": report["status"], "task": fixture["task"],
               "preset": args["preset"], "topology": args["topology"], "fixture": fixture,
               "shape": model_shape(construction["model_config"]),
               "parameter_count": construction["parameter_count"],
               "backbone_initialization_sha256": construction["backbone_initialization_sha256"],
               "core_source_sha256": digest(core_sources(report["sources"])),
               "full_source_sha256": digest(report["sources"]),
               "execution_contract_sha256": digest(report["execution_contract"]),
               "precision": actual, "elapsed_seconds": report["elapsed_seconds"],
               "peak_allocated_bytes": report["peak_allocated_bytes"], "peak_reserved_bytes": report["peak_reserved_bytes"]}
        if args["mode"] == "benchmark":
            require(report["measurement_compilation"]["steady_compilation_free"], "Benchmark compilation during measurement")
            times = [row["seconds"] for row in report["measured"]]
            require(len(times) == args["steps"] and all(t > 0 for t in times), "Incomplete timing window")
            require(math.isclose(sum(times) / len(times), report["timing"]["mean_seconds"], rel_tol=1e-12), "Timing mean mismatch")
            out.update(timing=report["timing"], measured_steps=len(times), warmup_steps=len(report["warmup"]),
                       startup_peak_allocated_bytes=report["startup_peak_allocated_bytes"],
                       measurement_compilation=report["measurement_compilation"])
        else:
            bypass = report["bypass"]
            summary = comparisons(bypass["gradients"])
            require(not summary["elementwise_failures"] and bypass["logits"]["elementwise_pass"] and report["causality"]["elementwise_pass"], "NUM report contains a failed comparison")
            require(all(report["round_trip"].values()), "NUM checkpoint round trip failed")
            out.update(bypass={"logits": comparisons({"logits": bypass["logits"]}), "gradients": summary,
                               "losses": bypass["losses"], "extra_branch_gradient_max": bypass["extra_branch_gradient_max"]},
                       causality=comparisons({"causality": report["causality"]}),
                       round_trip=report["round_trip"], initial_corrections=report["initial_corrections"],
                       actual_step={key: report["actual_step"][key] for key in ("native_loss", "native_targets", "gradient_norm", "clipped", "seconds")},
                       memory_scope=report["memory_scope"], checkpoint=self.artifact(report["checkpoint"]["path"], report["checkpoint"]["sha256"]))
        return out

    def evaluation_summary(self, entry, research):
        report, label = entry["report"], relative(entry["path"])
        identity = report["identity"]
        require(report["status"] == "complete" and report["arguments"]["mode"] == "evaluate", "Invalid evaluation record")
        require(report["identity_sha256"] == digest(identity), f"{label}: evaluation identity")
        check_precision(identity["precision"], None, label)
        matches = [row for row in research if row["report"]["identity"] == identity]
        require(matches, f"{label}: evaluation has no matching successful training identity")
        metadata = report["checkpoint"]
        reference, packet = self.checkpoint(metadata, identity, metadata["completed_epochs"], metadata["completed_updates"])
        del packet
        require(resolve(report["arguments"]["checkpoint"]).resolve() == resolve(metadata["path"]).resolve(), "Evaluation argument/checkpoint differs")
        for kind in ("native", "answer"):
            check_metrics(report["metrics"][kind], label)
        roles = set()
        for match in latest_runs(matches):
            ledger = match["report"]["roles"]
            for role in ("latest", "best_dev"):
                if ledger.get(role, {}).get("sha256") == reference["sha256"]:
                    roles.add(role)
            for epoch, value in ledger["milestones"].items():
                if value["sha256"] == reference["sha256"]:
                    roles.add(f"milestone_{epoch}")
        return {"artifact": self.artifact(entry["path"]), "task": identity["task"], "topology": identity["topology"],
                "preset": identity["preset"], "seed": identity["seed"], "physical_batch": identity["physical_batch"],
                "setting_id": setting_key(identity), "shape": model_shape(identity["model_config"]),
                "split": report["evaluation_split"], "arguments": report["arguments"],
                "checkpoint": reference, "checkpoint_roles": sorted(roles),
                "evaluation_data": report["evaluation_data"], "identity_sha256": report["identity_sha256"],
                "native": compact_metrics(report["metrics"]["native"]),
                "answer": compact_metrics(report["metrics"]["answer"]), "evaluation_seconds": report["metrics"]["seconds"]}


def benchmark_pairs(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (row["preset"], row["task"], row["fixture"]["batch"], row["fixture"]["length"], row["fixture"]["sha256"])
        require(row["topology"] not in groups[key], f"Ambiguous repeated benchmark for {key}/{row['topology']}")
        groups[key][row["topology"]] = row
    result = []
    for key, arms in sorted(groups.items()):
        if "seq" not in arms:
            continue
        seq = arms["seq"]
        for topology, other in sorted(arms.items()):
            if topology == "seq":
                continue
            for field in ("fixture", "shape", "backbone_initialization_sha256", "core_source_sha256", "execution_contract_sha256"):
                require(seq[field] == other[field], f"Unpaired benchmark {key}: {field}")
            result.append({"preset": key[0], "task": key[1], "batch": key[2], "length": key[3],
                           "topology": topology, "relative_to": "seq",
                           "step_time_ratio": other["timing"]["mean_seconds"] / seq["timing"]["mean_seconds"],
                           "allocated_memory_ratio": other["peak_allocated_bytes"] / seq["peak_allocated_bytes"],
                           "core_source_match": True, "full_source_match": other["full_source_sha256"] == seq["full_source_sha256"],
                           "seq_report": seq["artifact"]["path"], "other_report": other["artifact"]["path"]})
    return result


def curve_rows(entry, limit_epoch=None):
    report = entry["report"]
    identity = report["identity"]
    setting_label = f"{identity['task']} V{identity['model_config']['vocab_size']}"
    data_root = report["arguments"].get("data_root")
    if data_root and (resolve(data_root) / "manifest.json").is_file():
        metadata = json.loads((resolve(data_root) / "manifest.json").read_text())
        setting_label = (metadata.get("setting") or {}).get("name", setting_label)
    base = {"setting_label": setting_label, "task": identity["task"], "preset": identity["preset"], "topology": identity["topology"],
            "seed": identity["seed"], "report": relative(entry["path"]), "evidence": report["evidence"],
            "setting_id": setting_key(identity), "depth": identity["model_config"]["n_layers"],
            "width": identity["model_config"]["d_model"], "heads": identity["model_config"]["n_heads"],
            "length": identity["train"]["length"], "data_sha256": identity["train"]["input_ids"]}
    development, training, cumulative = [], [], [0.]
    for row in report["history"]:
        cumulative.append(cumulative[-1] + row["seconds"])
        if limit_epoch is not None and row["epoch"] > limit_epoch:
            continue
        training.append({**base, **{key: row[key] for key in ("update", "epoch", "native_loss", "native_targets", "input_tokens", "learning_rate", "gradient_norm", "seconds")},
                         "cumulative_recorded_update_seconds": cumulative[-1]})
    for epoch, values in sorted(report["development"].items(), key=lambda item: int(item[0])):
        epoch = int(epoch)
        if limit_epoch is not None and epoch > limit_epoch:
            continue
        development.append({**base, "epoch": epoch, "update": epoch * report["updates_per_epoch"],
                            "cumulative_recorded_update_seconds": cumulative[epoch * report["updates_per_epoch"]],
                            **compact_metrics(values["answer"]), "native_ce": values["native"]["ce"]})
    return development, training


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def make_plots(output_dir, development, training, operational):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"seq": "#2966a3", "cdrm": "#d06329", "r3": "#43875d", "same_depth": "#8768ad"}
    artifacts = []

    def save(fig, name):
        for suffix in ("png", "svg"):
            path = output_dir / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            artifacts.append(path.name)
        plt.close(fig)

    def setting_title(setting):
        first = next(r for r in development if r["setting_id"] == setting)
        return f"{first.get('setting_label', first['task'])} · L{first['depth']} D{first['width']} · seed {first['seed']}"

    settings = sorted({(row["task"], row["setting_id"]) for row in development})
    if settings:
        fig, axes = plt.subplots(len(settings), 3, figsize=(13.5, 3.6 * len(settings)), squeeze=False)
        for index, (task, setting) in enumerate(settings):
            first = next(r for r in development if r["setting_id"] == setting)
            title_prefix = setting_title(setting)
            for topology in ("seq", "cdrm"):
                rows = [row for row in development if row["setting_id"] == setting and row["topology"] == topology]
                for column, (metric, title) in enumerate((("ce", "Answer CE"), ("token_accuracy", "Answer token accuracy"), ("sequence_exact_match", "Direct sequence exact match"))):
                    ax = axes[index, column]
                    ax.plot([r["epoch"] for r in rows], [r[metric] for r in rows], label=topology.upper(), color=colors[topology], linewidth=1.8)
                    ax.set(title=f"{title_prefix}\n{title}", xlabel="Completed epoch")
                    ax.grid(alpha=.2)
                    if column:
                        ymax = max((r[metric] for r in development if r["setting_id"] == setting), default=0)
                        ax.set_ylim(0, min(1.02, max(.25, ymax * 1.1)))
            exact_values = [r["sequence_exact_match"] for r in development if r["setting_id"] == setting]
            if exact_values and all(value == 0 for value in exact_values):
                axes[index, 2].text(.5, .08, "Both 0 at every recorded epoch", transform=axes[index, 2].transAxes,
                                    ha="center", va="bottom", fontsize=9, color="#444444")
        axes[0, 0].legend(frameon=False)
        fig.suptitle("Development at common completed epochs — each training seed shown separately", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .96))
        save(fig, "development-learning-curves")

        fig, axes = plt.subplots(1, len(settings), figsize=(6.5 * len(settings), 3.7), squeeze=False)
        for index, (task, setting) in enumerate(settings):
            for topology in ("seq", "cdrm"):
                rows = [r for r in development if r["setting_id"] == setting and r["topology"] == topology]
                axes[0, index].plot([r["cumulative_recorded_update_seconds"] / 60 for r in rows], [r["ce"] for r in rows], label=topology.upper(), color=colors[topology])
            axes[0, index].set(title=setting_title(setting), xlabel="Cumulative recorded update minutes", ylabel="Development answer CE")
            axes[0, index].grid(alpha=.2)
        axes[0, 0].legend(frameon=False)
        fig.suptitle("Excludes evaluation, data transfer, checkpointing and startup; not a matched-time experiment", fontsize=10)
        fig.tight_layout()
        save(fig, "development-update-time-curves")

        fig, axes = plt.subplots(1, len(settings), figsize=(6.5 * len(settings), 3.7), squeeze=False)
        for index, (task, setting) in enumerate(settings):
            for topology in ("seq", "cdrm"):
                epochs = defaultdict(lambda: [0., 0])
                for row in training:
                    if row["setting_id"] == setting and row["topology"] == topology:
                        epochs[row["epoch"]][0] += row["native_loss"] * row["native_targets"]
                        epochs[row["epoch"]][1] += row["native_targets"]
                values = sorted(epochs.items())
                axes[0, index].plot([e for e, _ in values], [v[0] / v[1] for _, v in values], label=topology.upper(), color=colors[topology])
            axes[0, index].set(title=setting_title(setting), xlabel="Completed epoch", ylabel="Native training CE (target-weighted epoch mean)")
            axes[0, index].grid(alpha=.2)
        axes[0, 0].legend(frameon=False)
        fig.suptitle("Native optimization objective, separate from answer-only development metrics", fontsize=10)
        fig.tight_layout()
        save(fig, "native-training-loss")

    operational = [entry for entry in operational if entry["report"].get("updates_this_process") != 0]
    if operational:
        fig, axes = plt.subplots(len(operational), 2, figsize=(12, 3.2 * len(operational)), squeeze=False)
        for index, entry in enumerate(operational):
            dev, train = curve_rows(entry)
            name = entry["path"].parent.name
            axes[index, 0].plot([r["update"] for r in train], [r["native_loss"] for r in train], color=colors.get(entry["report"]["identity"]["topology"], "black"))
            axes[index, 0].set(title=name, xlabel="Completed update", ylabel="Native training CE")
            axes[index, 1].plot([r["update"] for r in dev], [r["token_accuracy"] for r in dev], label="Answer token accuracy")
            axes[index, 1].plot([r["update"] for r in dev], [r["sequence_exact_match"] for r in dev], label="Direct exact match")
            axes[index, 1].set(xlabel="Completed update", ylabel="Separate OPS monitor", ylim=(0, 1.02))
            for ax in axes[index]:
                ax.grid(alpha=.2)
        axes[0, 1].legend(frameon=False)
        fig.suptitle("OPS diagnostics: reused/subset training and monitoring are not research generalization", fontsize=11)
        fig.tight_layout()
        save(fig, "operational-learning-curves")
    return artifacts


def latest_runs(entries):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["report"]["identity_sha256"]].append(entry)
    selected = []
    for candidates in groups.values():
        candidates.sort(key=lambda entry: (entry["report"]["completed_updates"], entry["report"]["status"] in ("complete", "screened_early"), str(entry["path"])))
        longest = candidates[-1]
        for older in candidates[:-1]:
            for a, b in zip(older["report"]["history"], longest["report"]["history"]):
                require({k: v for k, v in a.items() if k != "seconds"} == {k: v for k, v in b.items() if k != "seconds"}, "Divergent resumed/unpaired cumulative history")
        selected.append(longest)
    return sorted(selected, key=lambda entry: str(entry["path"]))


def make_unpaired_plots(output_dir, entries, historical_depths, screening_depth):
    if not entries:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    artifacts = []
    fig, axes = plt.subplots(len(entries), 3, figsize=(13.5, 3.6 * len(entries)), squeeze=False)
    native_fig, native_axes = plt.subplots(len(entries), 1, figsize=(8, 3.0 * len(entries)), squeeze=False)
    for index, entry in enumerate(entries):
        dev, train = curve_rows(entry)
        first = dev[0]
        purpose = entry["report"].get("run_purpose", entry["report"]["arguments"].get("run_purpose", "pilot"))
        scope = "historical calibration" if first["depth"] in historical_depths else ("screening" if purpose == "screening" else "unpaired " + purpose)
        label = f"{first['task']} · {first['topology'].upper()}{first['depth']} · D{first['width']} H{first['heads']} T{first['length']}\n{scope} · seed {first['seed']} · data {first['data_sha256'][:8]}"
        for column, (metric, name) in enumerate((("ce", "Answer CE"), ("token_accuracy", "Answer token accuracy"), ("sequence_exact_match", "Direct sequence exact match"))):
            ax = axes[index, column]
            ax.plot([row["epoch"] for row in dev], [row[metric] for row in dev], color="#2966a3")
            ax.set(title=label, xlabel="Completed epoch", ylabel=name)
            if column:
                ax.set_ylim(0, min(1.02, max(.25, 1.1 * max(row[metric] for row in dev))))
                if all(row[metric] == 0 for row in dev):
                    ax.text(.5, .08, "0 at every recorded epoch", transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=9, color="#2966a3")
            ax.grid(alpha=.2)
        epochs = defaultdict(lambda: [0., 0])
        for row in train:
            epochs[row["epoch"]][0] += row["native_loss"] * row["native_targets"]
            epochs[row["epoch"]][1] += row["native_targets"]
        rows = sorted(epochs.items())
        ax = native_axes[index, 0]
        ax.plot([epoch for epoch, _ in rows], [values[0] / values[1] for _, values in rows])
        ax.set(title=label, xlabel="Completed epoch", ylabel="Native training CE\n(target-weighted epoch mean)")
        ax.grid(alpha=.2)
    fig.suptitle("Unpaired development records — no CDRM comparison; distinct settings are separate rows", fontsize=11)
    native_fig.suptitle("Native objective: its difference from answer-only dev CE is not a train/dev generalization gap", fontsize=10)
    for figure, name in ((fig, "unpaired-development-curves"), (native_fig, "unpaired-native-training-loss")):
        figure.tight_layout(rect=(0, 0, 1, .975))
        for suffix in ("png", "svg"):
            filename = f"{name}.{suffix}"
            figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
            artifacts.append(filename)
        plt.close(figure)
    return artifacts


def generate(run_root, output_dir, *, storage_prefix=None, require_pairs=False,
             historical_depths=(), screening_depth=None):
    reporter = Reporter(run_root, storage_prefix)
    accepted, excluded = reporter.discover()
    numerical, benchmarks, ops, research, evaluations = [], [], [], [], []
    training_summaries = []
    for entry in accepted:
        report = entry["report"]
        if entry["schema"] == VALIDATION:
            summary = reporter.validation_summary(entry)
            (benchmarks if report["arguments"]["mode"] == "benchmark" else numerical).append(summary)
        elif report["arguments"]["mode"] == "evaluate":
            evaluations.append(entry)
        elif report["evidence"] in ("OPS", "SYN"):
            summary = reporter.training_summary(entry)
            training_summaries.append(summary)
            (ops if report["evidence"] == "OPS" else research).append(entry)
        else:
            raise ValueError(f"Unsupported evidence class: {entry['path']}")
    selections, incomplete = reporter.select_pairs(research)
    if require_pairs:
        require(selections and not incomplete, "Complete primary research pairs required")
    pairs = [reporter.pair(selection) for selection in selections]
    finals = [reporter.evaluation_summary(entry, research) for entry in evaluations]
    for i, a in enumerate(finals):
        for b in finals[i + 1:]:
            if (a["setting_id"], a["split"]) == (b["setting_id"], b["split"]):
                require(a["evaluation_data"] == b["evaluation_data"], "Final evaluation data identities differ")
    paired_identity_ids = {entry["report"]["identity_sha256"] for selected in selections for entry in selected.values()}
    unpaired = [entry for entry in latest_runs(research) if entry["report"]["identity_sha256"] not in paired_identity_ids]
    unpaired_endpoints = []
    for entry in unpaired:
        report = entry["report"]
        reference, packet = reporter.checkpoint(report["roles"]["latest"], report["identity"],
                                                report["completed_epochs"], report["completed_updates"])
        del packet
        unpaired_endpoints.append({"report": relative(entry["path"]), "checkpoint": reference,
                                   "setting_id": setting_key(report["identity"]),
                                   "statement": "Unpaired retained endpoint; no architecture contrast"})
    for row in training_summaries:
        if row["evidence"] == "SYN":
            row["comparison_scope"] = ("historical_calibration_superseded_pair_plan" if row["shape"]["n_layers"] in historical_depths
                                       else "paired_architecture_comparison" if row["identity_sha256"] in paired_identity_ids
                                       else "screening" if row["run_purpose"] == "screening" else "unpaired_research")
    development, training, ops_dev, ops_train = [], [], [], []
    for selected, pair in zip(selections, pairs):
        for entry in selected.values():
            dev, train = curve_rows(entry, pair["common_completed_epochs"])
            development.extend(dev)
            training.extend(train)
    for entry in ops:
        dev, train = curve_rows(entry)
        ops_dev.extend(dev)
        ops_train.extend(train)
    unpaired_dev, unpaired_train = [], []
    for entry in unpaired:
        dev, train = curve_rows(entry)
        unpaired_dev.extend(dev)
        unpaired_train.extend(train)
    summary = {"schema": "cdrm-session-summary-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "run_root": relative(run_root), "generator_sha256": file_sha(__file__),
               "context": {"historical_calibration_depths": list(historical_depths), "screening_depth": screening_depth,
                           "superseded_pair_plan": bool(historical_depths),
                           "statement": "Depths explicitly marked historical are calibration from the superseded pair plan; unpaired SEQ screening supplies no CDRM comparison." if historical_depths else "Unpaired runs are calibration/screening records, not architecture comparisons."},
               "discovery": {"accepted_reports": len(accepted), "excluded_reports": excluded, "incomplete_research_pairs": incomplete},
               "numerical": numerical, "benchmarks": benchmarks, "benchmark_ratios": benchmark_pairs(benchmarks),
               "operational": [row for row in training_summaries if row["evidence"] == "OPS"],
               "research_runs": [row for row in training_summaries if row["evidence"] == "SYN"],
               "primary_pairs": pairs, "explicit_evaluations": finals,
               "selected_unpaired_reports": [relative(entry["path"]) for entry in unpaired],
               "unpaired_endpoint_audits": unpaired_endpoints,
               "notes": ["No GPU work or model evaluation is performed by this generator.",
                         "Training and development histories in resumed reports are cumulative and are not added twice.",
                         "Failed bookkeeping or other failed runs remain excluded, with raw report references.",
                         "Source hashes may differ between historical NUM and OPS records; paired research identities and core benchmark identities must match.",
                         "Only explicitly retained evaluation reports supply final-split model metrics.",
                         "Accuracy is measured on scored answer positions; direct exact match counts whole examples. No training-seed uncertainty is estimated.",
                         "Native dense recall training includes unpredictable next-key and first-value targets; its difference from answer-only development CE is not an ordinary generalization gap.",
                         "Recorded update time excludes external data transfer, evaluation, checkpointing and startup; it is not an end-to-end matched-time experiment."]}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "generated-artifacts.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        require(previous.get("generator") == "scripts/cdrm_report.py", "Output marker belongs to another generator")
        for name in previous["files"]:
            require(Path(name).name == name, "Unsafe previous artifact name")
            if (output_dir / name).exists():
                (output_dir / name).unlink()
    outputs = []
    def csv_file(name, rows):
        write_csv(output_dir / name, rows)
        outputs.append(name)
    csv_file("development.csv", development)
    csv_file("training.csv", training)
    csv_file("operational-development.csv", ops_dev)
    csv_file("operational-training.csv", ops_train)
    csv_file("unpaired-development.csv", unpaired_dev)
    csv_file("unpaired-training.csv", unpaired_train)
    csv_file("benchmarks.csv", [{"task": r["task"], "preset": r["preset"], "topology": r["topology"],
                               "batch": r["fixture"]["batch"], "length": r["fixture"]["length"],
                               "depth": r["shape"]["n_layers"], "vocab_size": r["shape"]["vocab_size"],
                               "fixture_sha256": r["fixture"]["sha256"], "mean_step_seconds": r["timing"]["mean_seconds"], "tokens_per_second": r["timing"]["tokens_per_second"],
                               "peak_allocated_bytes": r["peak_allocated_bytes"], "peak_reserved_bytes": r["peak_reserved_bytes"],
                               "report": r["artifact"]["path"]} for r in benchmarks])
    csv_file("final-evaluations.csv", [{"task": r["task"], "topology": r["topology"], "preset": r["preset"],
                                      "seed": r["seed"], "setting_id": r["setting_id"],
                                      "depth": r["shape"]["n_layers"], "physical_batch": r["physical_batch"],
                                      "data_root": r["arguments"]["data_root"], "split": r["split"], "checkpoint_epochs": r["checkpoint"]["completed_epochs"],
                                      "checkpoint_updates": r["checkpoint"]["completed_updates"],
                                      "roles": ",".join(r["checkpoint_roles"]), **r["answer"],
                                      "checkpoint": r["checkpoint"]["path"], "checkpoint_sha256": r["checkpoint"]["sha256"],
                                      "identity_sha256": r["identity_sha256"], "report": r["artifact"]["path"]} for r in finals])
    csv_file("pairing.csv", [{"task": p["task"], "preset": p["preset"],
                            "seed": p["seed"], "setting_id": p["setting_id"], "depth": p["shape"]["n_layers"],
                            "common_epochs": p["common_completed_epochs"],
                            "common_updates": p["common_completed_updates"], "physical_batch": p["physical_batch"],
                            "shuffle_seed": p["pairing"]["shuffle_seed"], "initial_weights_bitwise_equal": True,
                            "indices_verified": p["pairing"]["per_update_indices_verified"],
                            "stream_chain_sha256": p["pairing"]["stream_and_schedule_chain_sha256"]} for p in pairs])
    outputs.extend(make_plots(output_dir, development, training, ops))
    outputs.extend(make_unpaired_plots(output_dir, unpaired, historical_depths, screening_depth))
    summary["generated_files"] = ["summary.json", *outputs]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    marker.write_text(json.dumps({"generator": "scripts/cdrm_report.py", "files": ["summary.json", *outputs]}, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-prefix")
    parser.add_argument("--require-pairs", action="store_true", help="Fail if any discovered research task lacks a complete primary pair")
    parser.add_argument("--historical-depth", type=int, action="append", default=[], help="Explicitly label this depth as historical calibration from a superseded comparison plan")
    parser.add_argument("--screening-depth", type=int, help="Label unpaired runs at this depth as screening")
    args = parser.parse_args()
    require(Path("/.dockerenv").exists() and Path.cwd() == ROOT, "Run from the explicit CPU project container")
    import torch
    require(not torch.cuda.is_available(), "Use CDRM_DOCKER_GPUS=none; report generation must not use the GPU")
    torch.set_num_threads(1)
    summary = generate(args.run_root, args.output_dir, storage_prefix=args.storage_prefix, require_pairs=args.require_pairs,
                       historical_depths=args.historical_depth, screening_depth=args.screening_depth)
    print(json.dumps({"output_dir": relative(args.output_dir), "numerical": len(summary["numerical"]),
                      "benchmarks": len(summary["benchmarks"]), "operational": len(summary["operational"]),
                      "primary_pairs": len(summary["primary_pairs"]), "explicit_evaluations": len(summary["explicit_evaluations"])}))


if __name__ == "__main__":
    main()
