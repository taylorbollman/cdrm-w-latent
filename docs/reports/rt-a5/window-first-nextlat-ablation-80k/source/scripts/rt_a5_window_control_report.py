#!/usr/bin/env python3
"""Saved-evidence comparison of the same first-window RT with and without NextLat."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

from scripts import rt_a5_depth_order_report as depth
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, _digest_dict, _sha, local_path, metric_rows, training_curve
from scripts.rt_a5_report import finite_number, hash_file, read_input, write_json

ROOT = depth.ROOT
LINEAGE = ROOT / ".runtime/rt-a5/20260915T000224Z-window-first-no-nextlat80k"
REFERENCE_LINEAGE = depth.LINEAGE
SCHEMA = "rt-a5-window-nextlat-control-comparison-v1"
TRAIN_SCHEMA = "rt-a5-window-control-training-v1"
CONTROL = "rt_window2_first_ce"
NEXTLAT = "rt_window2_first"
ARMS = (CONTROL, NEXTLAT)
ENDPOINT = 80000
STEPS = depth.STEPS
LABELS = {CONTROL: "First-window RT: CE only", NEXTLAT: "First-window RT + NextLat"}
COLORS = {CONTROL: "#C47A2B", NEXTLAT: "#24938C"}
SOURCE_SHA = "2c72764fc37343df73cd2fc1e320729be72adb6bd93881666cc1f3754a2a5e73"
ADDITIONS = {"scripts/rt_a5_window_control.py", "scripts/rt_a5_window_control_train.py", "configs/rt_a5_window_control/base.json"}
REPORTING_SOURCES = ("scripts/rt_a5_window_control_report.py", *depth.REPORTING_SOURCES)


def validate_ce_history(history, packet, stop):
    if len(history) != stop or any(row.get("update") != i for i, row in enumerate(history, 1)):
        raise ValueError("CE history must contain every fresh update exactly once")
    for row in history:
        if row.get("examples_seen") != row["update"] * 1024 or not _sha(row.get("order_chain")):
            raise ValueError("CE history word exposure or minibatch order differs")
        for key in ("seconds", "loss", "state_loss", "token_accuracy", "whole_word_exact", "grad_norm"):
            if not finite_number(row.get(key)) or row[key] < 0:
                raise ValueError(f"Nonfinite, missing or negative CE history metric: {key}")
        if row["loss"] != row["state_loss"] or row["token_accuracy"] > 1 or row["whole_word_exact"] > 1:
            raise ValueError("CE state-loss alias or accuracy differs")
        if any(key in row for key in ("latent_loss", "weighted_latent_loss", *depth.budget.DIAGNOSTICS)):
            raise ValueError("CE control must not invent latent-loss or predictor diagnostic metrics")
    seconds = sum(row["seconds"] for row in history)
    if history[-1]["order_chain"] != packet.get("order_chain"):
        raise ValueError("CE report order chain differs from accepted history")
    if (not finite_number(packet.get("train_seconds"))
            or not math.isclose(seconds, packet["train_seconds"], rel_tol=1e-9, abs_tol=1e-5)
            or not finite_number(packet.get("elapsed_seconds")) or packet["elapsed_seconds"] < seconds):
        raise ValueError("CE report timing differs from accepted history")


def validate_ce_evaluations(packet, stop):
    expected = {(step, role) for step in range(500, stop + 1, 500) for role in ROLES}
    keys = [(item.get("update"), item.get("role")) for item in packet["evaluations"]]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing, duplicate or unexpected CE development evaluation")
    if packet.get("one_step_diagnostics") not in (None, []):
        raise ValueError("CE control cannot have predictor diagnostics")
    for item in packet["evaluations"]:
        if item.get("route") != "backbone_only" or item.get("rows") != (102400 if item["update"] in STEPS else 4096):
            raise ValueError("CE evaluation route or sample size differs")
        metric_rows(item, CONTROL)


def shared_backbone_contract(contract):
    result = copy.deepcopy(contract)
    for key in ("schema", "source_sha256", "variant", "objective", "nextlat_config", "predictor_seed",
                "nextlat_enabled", "predictor_registered", "one_step_diagnostics", "training_step",
                "experiment_config", "optimizer"):
        result.pop(key, None)
    return result


def compare_shared_contract(control, nextlat):
    if shared_backbone_contract(control) != shared_backbone_contract(nextlat):
        raise ValueError("Shared backbone, data, runtime or evaluation contract differs")
    if nextlat["optimizer"] != control["optimizer"] + "-all-hybrid":
        raise ValueError("Backbone optimizer hyperparameter recipe differs")
    expected_ce = {"state_loss": "same-position CE mean over B*T", "latent_weight": 0.0,
                   "kl_weight": 0.0, "predicted_state_ce_weight": 0.0}
    expected_joint = {"horizon": 1, "state_loss": "same-position CE mean over B*T",
        "latent_loss": "SmoothL1 beta=1 mean over B*(T-1)*D", "latent_weight": 1.0,
        "kl_weight": 0.0, "predicted_state_ce_weight": 0.0,
        "source_and_embedding_attached": True, "target_detached": True}
    if control.get("objective") != expected_ce or nextlat.get("objective") != expected_joint:
        raise ValueError("Require pure CE versus the unchanged original NextLat joint objective")
    if control.get("nextlat_enabled") is not False or control.get("predictor_registered") is not False:
        raise ValueError("CE control must omit the predictor")
    if control["experiment_config"]["attention"] != nextlat["experiment_config"]["attention"]:
        raise ValueError("Window location or attached recurrence differs")


def compare_initial_packets(control, nextlat):
    """Compare saved CPU tensors and optimizer mappings; instantiate no model."""
    import torch
    left, right = control["model"], nextlat["model"]
    backbone = {name.removeprefix("backbone."): value for name, value in right.items() if name.startswith("backbone.")}
    predictor = {name: value for name, value in right.items() if name.startswith("predictor.")}
    if set(left) != set(backbone) or len(backbone) + len(predictor) != len(right) or not predictor:
        raise ValueError("Control must be exactly the bare reference backbone without a predictor")
    if any(value.device.type != "cpu" or value.dtype != torch.float32 or not torch.isfinite(value).all() for value in right.values()):
        raise ValueError("Reference initial tensors must be finite CPU FP32")
    for name, value in left.items():
        if (value.device.type != "cpu" or value.dtype != torch.float32 or not torch.isfinite(value).all()
                or not torch.equal(value, backbone[name])):
            raise ValueError(f"Initial backbone tensor differs or is not finite CPU FP32: {name}")
    if any(packet.get("completed_updates") != 0 or packet["optimizer"]["state"] for packet in (control, nextlat)):
        raise ValueError("Both initial checkpoints must be untrained with empty Adam state")
    groups_left, groups_right = control["optimizer"]["param_groups"], nextlat["optimizer"]["param_groups"]
    names_left, names_right = control["optimizer_parameter_names"], nextlat["optimizer_parameter_names"]
    if len(groups_left) != len(groups_right) or len(names_left) != len(groups_left) or len(names_right) != len(groups_right):
        raise ValueError("Initial optimizer group mapping differs")
    for groups, names, weights in ((groups_left, names_left, left), (groups_right, names_right, right)):
        flat_ids = [index for group in groups for index in group["params"]]
        flat_names = [name for group in names for name in group]
        if len(flat_ids) != len(set(flat_ids)) or len(flat_names) != len(set(flat_names)) or set(flat_names) != set(weights):
            raise ValueError("Initial optimizer identifiers or names do not cover tensors exactly once")
    seen = []
    for lg, rg, ln, rn in zip(groups_left, groups_right, names_left, names_right):
        if (ln != [name.removeprefix("backbone.") for name in rn if name.startswith("backbone.")]
                or {k: v for k, v in lg.items() if k != "params"} != {k: v for k, v in rg.items() if k != "params"}
                or len(ln) != len(lg["params"]) or len(rn) != len(rg["params"])):
            raise ValueError("Initial optimizer backbone names or hyperparameters differ")
        seen.extend(ln)
    if len(seen) != len(set(seen)) or set(seen) != set(left):
        raise ValueError("Initial optimizer does not cover the bare backbone exactly once")
    return {"exact_backbone_tensor_identity": True, "backbone_parameter_tensors": len(left),
        "backbone_parameters": sum(value.numel() for value in left.values()),
        "nextlat_predictor_parameter_tensors": len(predictor),
        "nextlat_predictor_parameters": sum(value.numel() for value in predictor.values()),
        "same_backbone_optimizer_options": True, "initial_adam_empty": True,
        "control_predictor_registered": False, "execution": "Saved CPU tensors only; no model construction or inference"}


def metric_table(summary):
    rows = [row for arm in ARMS for step in STEPS for role in ROLES
            for row in summary["arms"][arm]["curves"][str(step)][role]]
    expected = {(arm, step, role, length) for arm in ARMS for step in STEPS for role in ROLES
                for length in range(1, (12 if role == "dev" else 36) + 1)}
    keys = [(row["arm"], row["update"], row["role"], row["length"]) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Comparison metrics contain missing, duplicate or out-of-scope rows")
    return rows


def plot_rows(summary, arm, step):
    if arm not in ARMS or step not in STEPS:
        raise ValueError("Plot requires a declared arm and retained checkpoint")
    rows = summary["arms"][arm]["curves"][str(step)]["ood_dev"]
    if [(row["arm"], row["update"], row["role"], row["length"]) for row in rows] != [
            (arm, step, "ood_dev", length) for length in range(1, 37)]:
        raise ValueError("Plot must use the exact shared length-36 prefix rows")
    return rows


def checkpoint_summary(summary):
    result = []
    for arm in ARMS:
        for step in STEPS:
            packet = summary["arms"][arm]
            metric = packet["metrics"][str(step)]
            rows = plot_rows(summary, arm, step)
            result.append({"arm": arm, "update": step, "parameter_count": packet["parameter_count"],
                "dev_whole_word_exact_match": metric["dev"]["whole_word_exact_match"],
                "dev_token_accuracy": metric["dev"]["token_accuracy"], "ood_ce": metric["ood_dev"]["ce"],
                "A36": rows[-1]["A"], "M36": rows[-1]["M"],
                **{f"E{length}": rows[length-1]["E"] for length in depth.budget.KEY_LENGTHS}})
    return result


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}
    def save(fig, name):
        figures[name] = {}
        for suffix in ("png", "pdf"):
            figures[name][suffix] = f"{name}.{suffix}"
            fig.savefig(output / figures[name][suffix], dpi=200, bbox_inches="tight", pad_inches=.18)
        plt.close(fig)
    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
        fig.subplots_adjust(left=.065, right=.985, bottom=.18, top=.76, wspace=.38)
        for axis, key, title in zip(axes, ("E", "A", "M"),
                ("E: every state through t correct", "A: state t correct", "M: mean token accuracy through t")):
            for arm in ARMS:
                rows = plot_rows(summary, arm, ENDPOINT)
                x = [row["length"] for row in rows]
                axis.plot(x, [row[key] for row in rows], color=COLORS[arm], label=LABELS[arm])
                if key != "M":
                    axis.fill_between(x, [row[f"{key}_low95"] for row in rows],
                        [row[f"{key}_high95"] for row in rows], color=COLORS[arm], alpha=.14)
            axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Prefix length t of same length-36 words", title=title)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.axvline(12, color="gray", linestyle=":", alpha=.6)
            axis.grid(alpha=.18)
            axis.legend(fontsize=8)
        axes[1].axhline(1/60, color="gray", linestyle="--", linewidth=.8)
        fig.suptitle("Same first-window RT backbone · 80,000 updates · 102,400 development words\n"
                     "CE only versus CE + NextLat · one seed; control chosen after earlier development results", y=.98)
        save(fig, name)
    for name, lengths in (("whole-word-vs-updates", (36,)), ("exactness-vs-updates", (13, 14, 16))):
        fig, axes = plt.subplots(1, len(lengths), figsize=(8 if len(lengths) == 1 else 14, 4.8), squeeze=False)
        fig.subplots_adjust(left=.09 if len(lengths) == 1 else .065, right=.98, bottom=.18, top=.80, wspace=.35)
        for axis, length in zip(axes[0], lengths):
            for arm in ARMS:
                axis.plot([step/1000 for step in STEPS], [plot_rows(summary, arm, step)[length-1]["E"] for step in STEPS],
                    color=COLORS[arm], marker="o", markersize=4, label=LABELS[arm])
            axis.set(xlabel="Optimizer updates (thousands)", ylabel="Whole-word accuracy" if length == 36 else f"E({length})",
                     ylim=(-.025, 1.025))
            axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.18); axis.legend(fontsize=9)
        fig.suptitle("A5 length-36 whole-word accuracy across training" if lengths == (36,) else "A5 prefix exactness across matched update budgets", y=.96)
        save(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    fig.subplots_adjust(left=.075, right=.98, bottom=.18, top=.80, wspace=.32)
    for axis, xkey, xlabel, divisor in zip(axes, ("update", "training_seconds"),
            ("Optimizer updates (thousands)", "Accumulated training-loop minutes"), (1000, 60)):
        for arm in ARMS:
            points = summary["arms"][arm]["training_curve"]
            axis.plot([point[xkey]/divisor for point in points], [point["state_ce"] for point in points], color=COLORS[arm], label=LABELS[arm])
        axis.set(xlabel=xlabel, ylabel="Training state CE")
        axis.grid(alpha=.18); axis.legend(fontsize=9)
    fig.suptitle("Comparable state CE · nonoverlapping 100-update means\nNextLat also optimizes latent loss; CE control has no latent-loss metric", y=.98)
    save(fig, "training-state-ce")
    return figures


def validate_preflight(preflight, training_sources):
    # Validation also snapshots its own harness; those extra files are not
    # training dependencies. The pinned preflight file binds their hashes.
    if (preflight.get("status") != "passed" or preflight.get("wandb", {}).get("status") != "synced"
            or any(preflight["source_files"].get(name) != digest for name, digest in training_sources.items())):
        raise ValueError("Control preflight did not pass with the frozen training sources")


def validate_protocol(protocol):
    expected = {"schema": "rt-a5-window-control-protocol-v1", "start_update": 0, "endpoint": ENDPOINT,
        "variant": CONTROL, "parameter_count": 6357504, "parameter_tensors": 21,
        "checkpoint_steps": [0, *STEPS], "batch_size": 1024, "training_length": 12, "ood_length": 36,
        "training_words": 800000, "evaluation_rows": 4096, "full_evaluation_rows": 102400, "evaluation_every": 500,
        "backbone_seed": 1234, "data_order_seed": 1234, "evaluation_route": "backbone_only",
        "confirmation_evaluated": False, "latent_rollout_evaluated": False}
    for key, wanted in expected.items():
        if protocol.get(key) != wanted:
            raise ValueError(f"Prospective CE-control protocol differs: {key}")
    sources = protocol["source_files"]
    if len(sources) != 58 or _digest_dict(sources) != SOURCE_SHA or protocol.get("source_sha256") != SOURCE_SHA:
        raise ValueError("Frozen 58-source control lineage differs")
    if not ADDITIONS.issubset(sources) or _digest_dict({k: v for k, v in sources.items() if k not in ADDITIONS}) != depth.SOURCE_SHA:
        raise ValueError("Historical 55 training sources changed")
    contract = protocol["strict_contract"]
    for key, wanted in {"schema": TRAIN_SCHEMA, "variant": CONTROL, "architecture": "rt", "width": 512,
        "source_sha256": SOURCE_SHA, "nextlat_enabled": False, "predictor_registered": False,
        "precision": "fp32", "tf32": False, "compile": False, "cuda_graphs": False,
        "training_step": "scripts.rt_a5_train.train_step (same function object)",
        "evaluation": "scripts.rt_a5_train.evaluate_arrays (same function object)"}.items():
        if contract.get(key) != wanted:
            raise ValueError(f"CE training implementation contract differs: {key}")
    for key, wanted in {"schema": "rt-a5-window-control-config-v1", "variant": CONTROL, "reference_variant": NEXTLAT,
        "n_layers": 2, "window_layer": 0, "window_length": 2, "position_encoding": "alibi",
        "learned_position_parameters": False, "nextlat_enabled": False, "predictor_registered": False}.items():
        if contract["experiment_config"].get(key) != wanted:
            raise ValueError(f"CE control architecture differs: {key}")
    initial = protocol["initialization"]
    for key, wanted in {"schema": "rt-a5-window-control-initialization-v1", "variant": CONTROL,
        "parameter_count": 6357504, "backbone_parameter_count": 6357504, "parameter_tensors": 21,
        "predictor_parameter_count": 0, "predictor_registered": False,
        "exact_reference_backbone_initialization": True, "changed_backbone_parameter_slices": []}.items():
        if initial.get(key) != wanted:
            raise ValueError(f"CE control initialization differs: {key}")
    for record in (protocol["data_manifest"], protocol["preflight"]):
        depth.stopped_window.stopped.verify_file(record)
    preflight = json.loads(local_path(protocol["preflight"]["path"]).read_text())
    validate_preflight(preflight, sources)
    ref = protocol["reference"]
    if ref.get("start_update") != 0 or ref.get("endpoint") != ENDPOINT or ref.get("source_sha256") != depth.SOURCE_SHA:
        raise ValueError("Reference is not the original prospective fresh 80k winning arm")
    if len(ref["source_files"]) != 55 or any(sources.get(k) != v for k, v in ref["source_files"].items()):
        raise ValueError("Reference source subset differs")
    for name in ("report", "config", "protocol", "initial_checkpoint", "final_checkpoint", "retention_readback"):
        depth.stopped_window.stopped.verify_file(ref[name])


def read_control(protocol, *, through_update=ENDPOINT):
    if through_update not in (1000, ENDPOINT):
        raise ValueError("Use final 80k or an explicit read-only 1k preflight")
    directory = local_path(protocol["training_directory"]).resolve()
    raw, report_file = read_input(directory / "report.json")
    packet = json.loads(raw)
    final = through_update == ENDPOINT
    if (packet.get("schema") != TRAIN_SCHEMA or packet.get("status") not in (("complete",) if final else ("running", "complete"))
            or packet.get("start_update") != 0 or packet.get("parent_checkpoint") is not None or packet.get("endpoint") != ENDPOINT):
        raise ValueError("Require a fresh 0-to-80k CE control; no resumed or stopped substitute")
    observed = packet.get("completed_updates", 0)
    if type(observed) is not int or not through_update <= observed <= ENDPOINT:
        raise ValueError("Requested CE control checkpoint is incomplete")
    if final and (observed != ENDPOINT or packet.get("wandb", {}).get("status") != "synced"):
        raise ValueError("Final CE control must complete 80k with synced online evidence")
    if packet.get("confirmation_evaluated") is not False or packet.get("latent_rollout_evaluated") is not False:
        raise ValueError("Confirmation and autonomous latent rollout must remain unevaluated")
    if (packet["contract"] != protocol["strict_contract"] or packet["initialization"] != protocol["initialization"]
            or packet["source_files"] != protocol["source_files"]):
        raise ValueError("Control contract, initialization or frozen source identity changed")
    for relative, wanted in packet["source_files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not _sha(wanted):
            raise ValueError("Invalid frozen source path or hash")
        if hash_file(directory / "source" / path)["sha256"] != wanted:
            raise ValueError(f"Frozen control source changed: {relative}")
    raw, config_file = read_input(directory / "config.json")
    config = json.loads(raw)
    if config != protocol["resolved_args"]:
        raise ValueError("Executed control arguments differ from prospective protocol")
    manifest = hash_file(local_path(config["data_dir"]) / "manifest.json")
    if manifest["sha256"] != packet["contract"]["data_manifest_sha256"]:
        raise ValueError("Control data manifest changed")
    if final:
        raw, history_file = read_input(directory / "history.jsonl")
    else:
        with (directory / "history.jsonl").open("rb") as stream:
            raw = b"".join(stream.readline() for _ in range(observed))
        history_file = {"path": str(directory / "history.jsonl"), "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw), "snapshot": f"immutable prefix through saved update {observed}"}
    history = [json.loads(line) for line in raw.splitlines() if line.strip()]
    validate_ce_history(history, packet, observed)
    validate_ce_evaluations(packet, observed)
    if [item["completed_updates"] for item in packet["checkpoints"]] != [0, *(s for s in STEPS if s <= observed)]:
        raise ValueError("Control retained checkpoint schedule differs")
    checkpoints = {}
    for item in packet["checkpoints"]:
        step = item["completed_updates"]
        if item.get("examples_seen") != step * 1024:
            raise ValueError("Control checkpoint word exposure differs")
        checkpoints[str(step)] = depth.stopped_window.stopped.verify_file(item,
            expected_path=directory / f"checkpoints/step-{step:06d}.pt")
    curves, metrics = {}, {}
    selected = [s for s in STEPS if s <= through_update]
    for step in selected:
        metrics[str(step)] = {role: next(item for item in packet["evaluations"]
            if (item["update"], item["role"]) == (step, role)) for role in ROLES}
        curves[str(step)] = {role: metric_rows(item, CONTROL) for role, item in metrics[str(step)].items()}
    return {"label": LABELS[CONTROL], "contract": packet["contract"], "initialization": packet["initialization"],
        "source_files": packet["source_files"], "checkpoint_updates": selected, "curves": curves, "metrics": metrics,
        "checkpoints": checkpoints, "history": history[:through_update], "training_curve": training_curve(history[:through_update], False),
        "inputs": {"training": {"report": report_file, "history": history_file, "run_config": config_file, "data_manifest": manifest}},
        "wandb": {"training": packet["wandb"]}, "parameter_count": 6357504, "parameter_tensors": 21,
        "selection_kind": "prospectively_fixed80k", "train_seconds": sum(row["seconds"] for row in history[:through_update]),
        "elapsed_seconds": packet["elapsed_seconds"]}


def read_reference(protocol):
    ref = protocol["reference"]
    raw, protocol_file = read_input(local_path(ref["protocol"]["path"]))
    original = json.loads(raw)
    depth.validate_contract(original["arms"][NEXTLAT]["strict_contract"], NEXTLAT)
    packet = depth.read_fresh(NEXTLAT, original)
    if local_path(ref["training_directory"]).resolve() != local_path(original["arms"][NEXTLAT]["directory"]).resolve():
        raise ValueError("Pinned reference directory differs from original training arm")
    checks = ((packet["inputs"]["training"]["report"], ref["report"]),
              (packet["inputs"]["training"]["run_config"], ref["config"]), (protocol_file, ref["protocol"]),
              (packet["checkpoints"]["0"], ref["initial_checkpoint"]),
              (packet["checkpoints"][str(ENDPOINT)], ref["final_checkpoint"]))
    for actual, expected in checks:
        if any(actual[key] != expected[key] for key in ("sha256", "bytes")):
            raise ValueError("Pinned winning NextLat reference changed")
    if packet["source_files"] != ref["source_files"]:
        raise ValueError("Pinned NextLat source snapshot changed")
    packet["label"] = LABELS[NEXTLAT]
    packet["train_seconds"] = sum(row["seconds"] for row in packet["history"])
    return packet


def verify_initial_checkpoints(control, nextlat):
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("Initial checkpoint comparison must not use CUDA")
    records = {CONTROL: control["checkpoints"]["0"], NEXTLAT: nextlat["checkpoints"]["0"]}
    packets = {arm: torch.load(local_path(record["path"]), map_location="cpu", weights_only=False)
               for arm, record in records.items()}
    for arm, expected in ((CONTROL, control), (NEXTLAT, nextlat)):
        packet = packets[arm]
        if (packet.get("schema") != expected["contract"]["schema"]
                or packet.get("contract") != expected["contract"] or packet.get("initialization") != expected["initialization"]
                or packet.get("examples_seen") != 0 or packet.get("completed_updates") != 0):
            raise ValueError("Initial checkpoint contract, metadata or counters differ")
    result = compare_initial_packets(packets[CONTROL], packets[NEXTLAT])
    if (result["backbone_parameters"], result["backbone_parameter_tensors"], result["nextlat_predictor_parameters"],
            result["nextlat_predictor_parameter_tensors"]) != (6357504, 21, 1049600, 4):
        raise ValueError("Initial backbone/predictor parameter counts differ")
    result["checkpoint_inputs"] = records
    if torch.cuda.is_initialized():
        raise RuntimeError("Saved-state verification unexpectedly initialized CUDA")
    return result


def compare_arms(control, nextlat):
    compare_shared_contract(control["contract"], nextlat["contract"])
    if [row["order_chain"] for row in control["history"]] != [row["order_chain"] for row in nextlat["history"][:len(control["history"])]]:
        raise ValueError("Every corresponding minibatch order must match")
    initial, reference = control["initialization"], nextlat["initialization"]
    if (initial["reference_nextlat_initialization"] != reference
            or initial["canonical_sha256"] != reference["canonical_sha256"]
            or initial.get("exact_reference_backbone_initialization") is not True
            or initial.get("changed_backbone_parameter_slices") != []):
        raise ValueError("Control must retain exact winning backbone initialization")
    for relative, digest in nextlat["source_files"].items():
        if control["source_files"].get(relative) != digest:
            raise ValueError("Shared historical source changed")
    return verify_initial_checkpoints(control, nextlat)


def make_summary(protocol_path):
    raw, protocol_file = read_input(local_path(protocol_path))
    protocol = json.loads(raw)
    validate_protocol(protocol)
    arms = {CONTROL: read_control(protocol), NEXTLAT: read_reference(protocol)}
    pairing = compare_arms(arms[CONTROL], arms[NEXTLAT])
    order_chain = arms[CONTROL]["history"][-1]["order_chain"]
    for packet in arms.values():
        packet.pop("history")
    summary = {"schema": SCHEMA, "primary_update": ENDPOINT, "checkpoint_updates": list(STEPS), "arms": arms,
        "protocol": protocol, "protocol_input": protocol_file, "source_files": protocol["source_files"],
        "initial_backbone_verification": pairing, "order_chain": order_chain,
        "confirmation_evaluated": False, "latent_rollout_evaluated": False, "evaluation_route": "backbone_only",
        "scope": "Same first-window/full RT: pure CE versus original NextLat joint training, fixed 80k endpoints; one reused development seed",
        "comparison": {"same_initial_backbone_tensors": True, "same_data_order": True,
            "same_backbone_parameter_budget": True, "same_total_parameter_budget": False, "same_compute_budget": False,
            "same_backbone_optimizer_hyperparameters": True, "global_clipping_inputs_differ": True,
            "both_endpoint_selections": "prospectively_fixed80k", "followup_choice": "adaptive_after_development_review"},
        "budget": {"updates_per_arm": ENDPOINT, "word_presentations_per_arm": ENDPOINT * 1024,
                   "unique_training_words": 800000, "nominal_training_passes_per_arm": 102.4},
        "metric_definitions": {"E": "Every state through t correct", "A": "Only state t correct", "M": "Mean token correctness through t"},
        "training_bin_updates": 100, "intervals": "Pointwise Wilson 95% over words for E/A; not seed or paired-difference uncertainty; no M interval"}
    summary["checkpoint_summary"] = checkpoint_summary(summary)
    return summary


def markdown_report(summary):
    lines = ["# Same first-window RT: CE only versus NextLat", "",
        "Both runs completed their prospectively fixed **80,000-update** endpoints. They use the same two-layer RT backbone: "
        "window-2 attention in the first layer, full recurrent attention in the second. The control trains on state CE only; "
        "the reference trains on state CE plus the original weight-one next-latent SmoothL1 loss. Both evaluate backbone predictions.", "",
        "| Training objective | Total parameters | Backbone parameters | Learned tensors |",
        "| --- | ---: | ---: | ---: |",
        "| CE only | 6,357,504 | 6,357,504 | 21 |",
        "| CE + NextLat | 7,407,104 | 6,357,504 | 25 |", "",
        "All 21 initial backbone tensors were checked for exact equality, mapping the reference's `backbone.` names to the bare control. "
        "The control has no registered predictor. Backbone optimizer parameter groups and hyperparameters match. Removing NextLat "
        "also removes the predictor from the global norm used for clipping; total parameters, joint clipping inputs and compute therefore differ.", "",
        "| Objective at 80k | L12 whole word | E(13) | E(14) | E(16) | E(36) | A(36) | M(36) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["checkpoint_summary"]:
        if row["update"] == ENDPOINT:
            lines.append(f"| {LABELS[row['arm']]} | " + " | ".join(f"{100*row[key]:.4f}%" for key in
                ("dev_whole_word_exact_match", "E13", "E14", "E16", "E36", "A36", "M36")) + " |")
    lines += ["", "| Updates | CE-only E(36) | NextLat E(36) |", "| --- | ---: | ---: |"]
    for step in STEPS:
        lines.append(f"| {step:,} | " + " | ".join(f"{100*plot_rows(summary, arm, step)[-1]['E']:.4f}%" for arm in ARMS) + " |")
    lines += ["", "![Length-36 whole-word accuracy across training](whole-word-vs-updates.png)", "",
        "![Full length curves at 80k](length-full.png)", "", "![Identical rows, boundary view](length-boundary.png)", "",
        "![Prefix exactness across training](exactness-vs-updates.png)", "", "![Comparable training state CE](training-state-ce.png)", "",
        "E(t) requires every state through t correct; A(t) checks only state t; M(t) averages token correctness through t. "
        "All checkpoint evaluations use 102,400 words per development role. Prefix curves use the same length-36 outputs, "
        "and full/boundary figures use identical rows. L12 uses a separate development set. CE-only history contains no latent-loss "
        "metrics; the training plot compares state CE without treating the different total objectives as equivalent.", "",
        "Both arms use width 512, eight heads, GELU FFN width 2048, original Mitchell initialization, ALiBi, full FP32, "
        "the same data/minibatch order, and the same backbone AdamW recipe. Each 80k run represents 81,920,000 word presentations "
        "over 800,000 unique length-12 training words (102.4 nominal passes). The direct attention window preserves attached recurrent "
        "states and gradients; it does not truncate represented history to two tokens.", "",
        "This is a one-seed development comparison. Both endpoints were fixed before their own training; the choice of this control "
        "followed the reference's development results. Repeated use of the same development words is not independent replication. "
        "Pointwise Wilson 95% intervals for E/A describe sampling over words, not seed variability or paired differences. "
        "Zero errors or zero successes mean observations in this sample, not exact population accuracy. "
        "Final confirmation and autonomous latent predictor rollout remain **unevaluated**.", "",
        "The reporter checks frozen source identities, exact initial backbone/optimizer mapping, all retained checkpoint hashes, "
        "finite ordered histories, every matching minibatch hash, objective contracts and evaluation counts. It performs no model inference or training.", "",
        "[Exact metric counts](metrics.csv) · [All checkpoint summaries](checkpoint-summary.csv) · [Plot data](plot-data.json) · [Provenance](report.json)", ""]
    return "\n".join(lines)


def input_records(summary):
    records = [(Path("inputs/protocol.json"), summary["protocol_input"])]
    for arm in ARMS:
        for phase, inputs in summary["arms"][arm]["inputs"].items():
            records += [(Path("inputs") / arm / phase / Path(record["path"]).name, record) for record in inputs.values()]
    for name in ("data_manifest", "preflight"):
        records.append((Path("inputs/control-provenance") / f"{name}.json", summary["protocol"][name]))
    for name in ("protocol", "retention_readback"):
        records.append((Path("inputs/reference-provenance") / f"{name}.json", summary["protocol"]["reference"][name]))
    return records


def run(args):
    output = local_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh CE-versus-NextLat report directory")
    summary = make_summary(args.protocol)
    rows = metric_table(summary)
    output.mkdir(parents=True)
    for relative, record in input_records(summary):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path(record["path"]), target)
        if any(hash_file(target)[key] != record[key] for key in ("sha256", "bytes")):
            raise ValueError("Report input changed during provenance copy")
    summary["reporting_sources"] = {}
    for relative in REPORTING_SOURCES:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        summary["reporting_sources"][relative] = hash_file(target)
    summary["reporting_source_sha256"] = _digest_dict({name: record["sha256"] for name, record in summary["reporting_sources"].items()})
    depth.budget.write_csv(output / "metrics.csv", rows, CSV_COLUMNS)
    with (output / "metrics.csv").open() as stream:
        if list(csv.DictReader(stream)) != [{key: str(row[key]) for key in CSV_COLUMNS} for row in rows]:
            raise ValueError("CSV differs from full/boundary plot rows")
    depth.budget.write_csv(output / "checkpoint-summary.csv", summary["checkpoint_summary"], summary["checkpoint_summary"][0].keys())
    for arm in ARMS:
        points = summary["arms"][arm]["training_curve"]
        depth.budget.write_csv(output / f"training-{arm}.csv", points, points[0].keys())
    write_json(output / "summary.json", summary)
    write_json(output / "plot-data.json", summary)
    figures = plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
        group=args.wandb_group, name="first-window-rt-ce-versus-nextlat-80k-report")
    result = {**summary, "status": "running", "figures": figures}
    try:
        tracker.start({key: summary[key] for key in ("schema", "scope", "primary_update", "comparison", "budget")})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(CSV_COLUMNS), data=[[row[key] for key in CSV_COLUMNS] for row in rows]),
            **{f"report/{name}": wandb.Image(str(output / files["png"])) for name, files in figures.items()}})
        bins = {arm: {point["update"]: point for point in summary["arms"][arm]["training_curve"]} for arm in ARMS}
        for step in range(100, ENDPOINT + 1, 100):
            values = {"update": step}
            for arm in ARMS:
                values.update({f"train/{arm}/{key}": value for key, value in bins[arm][step].items()
                    if key not in ("update", "first_update", "updates_in_bin")})
                if step in STEPS:
                    for length in depth.budget.KEY_LENGTHS:
                        point = plot_rows(summary, arm, step)[length-1]
                        values.update({f"dev/{arm}/prefix_{length}/{key}": point[key] for key in ("E", "A", "M")})
            tracker.log(values)
        tracker.summary({"primary_update": ENDPOINT, "checkpoint_summary": summary["checkpoint_summary"],
            "initial_backbone_verification": summary["initial_backbone_verification"],
            "confirmation_evaluated": False, "latent_rollout_evaluated": False})
        tracker.finish(succeeded=True)
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        try:
            tracker.finish(succeeded=False)
        except Exception:
            pass
        raise
    finally:
        result["wandb"] = tracker.record
        result["artifacts"] = {str(path.relative_to(output)): hash_file(path) for path in sorted(output.rglob("*"))
            if path.is_file() and "wandb" not in path.relative_to(output).parts and path != output / "report.json"}
        write_json(output / "report.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(output), "wandb": tracker.record["run_url"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(LINEAGE / "protocol.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group", default=LINEAGE.name)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
