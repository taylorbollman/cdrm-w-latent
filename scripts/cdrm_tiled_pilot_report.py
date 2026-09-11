#!/usr/bin/env python3
"""CPU-only audit and combined learning report for the bounded three-arm pilot.

Read explicit retained reports and the frozen corpus; never load a model or
checkpoint. Allocation suggestions, operational audits and unchanged numerical
machine flags are separate outputs, not interchangeable acceptance decisions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ARMS = ("seq-fp32", "cdrm-fp32", "cdrm-bf16")
STEPS_PER_EPOCH = 200
MILESTONES = (1000, 2500)
COLORS = {"seq-fp32": "#395f85", "cdrm-fp32": "#398368", "cdrm-bf16": "#c06434"}
PRIMARY = "tiled_bf16_vs_tiled_fp32"
PROTOCOL_SHA256 = "20e5b22b113d8feb6e2614469c1b48af84371ca62b9105edacdd3e6b664efaf7"
CRITERIA_SHA256 = "26b1756dd958e0ab1c916cc51e46b691393eabb5598c645cb0d63b5bfdd0ea20"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_report(path):
    path = Path(path).resolve()
    return json.loads(path.read_text()), {"path": str(path), "sha256": digest(path)}


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def expected_lr(update):
    if update < 1:
        raise ValueError("The reported training LR belongs to an actual positive update")
    completed_epochs_before_update = (update - 1) // STEPS_PER_EPOCH
    return 1e-6 + (5e-4 - 1e-6) * (1 + math.cos(math.pi * completed_epochs_before_update / 200)) / 2


def nontiming(value):
    """The runner's update timing is observational, never part of exact replay."""
    if isinstance(value, dict):
        return {key: nontiming(item) for key, item in value.items()
                if key not in {"seconds", "elapsed_seconds", "training_seconds"}}
    if isinstance(value, list):
        return [nontiming(item) for item in value]
    return value


def false_flags(value, path=""):
    """Retain every explicitly false machine predicate, including nested scopes."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{path}/{key}" if path else key
            if item is False and (key.endswith("pass") or key.endswith("passed")
                                  or key.endswith("equal") or key in ("finite", "numerical_clearance")):
                result[here] = False
            elif isinstance(item, (dict, list)):
                result.update(false_flags(item, here))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(false_flags(item, f"{path}/{index}"))
    return result


def compact(value):
    """Preserve aggregate report content while excluding coordinate-sized tables."""
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()
                if key not in {"rows", "top5_offending_coordinates", "worst_coordinate", "worst_elementwise_coordinate",
                               "source_sha256", "history", "provenance", "model_config"}}
    if isinstance(value, list):
        return value if all(isinstance(item, (str, int, float, bool, type(None))) for item in value) else [compact(item) for item in value]
    return value


def metric_errors(metric, scope):
    errors = []
    required = ("ce_sum", "targets", "correct", "exact", "examples", "ce", "token_accuracy", "sequence_exact_match")
    if any(key not in metric for key in required):
        return [f"{scope}: missing direct metric counts"]
    if any(not isinstance(metric[key], (int, float)) or not math.isfinite(metric[key]) for key in required):
        return [f"{scope}: nonfinite metric counts"]
    if metric["targets"] <= 0 or metric["examples"] <= 0:
        return [f"{scope}: empty metric denominator"]
    if not 0 <= metric["correct"] <= metric["targets"] or not 0 <= metric["exact"] <= metric["examples"]:
        errors.append(f"{scope}: accuracy counts outside denominators")
    expected = {"ce": metric["ce_sum"] / metric["targets"],
                "token_accuracy": metric["correct"] / metric["targets"],
                "sequence_exact_match": metric["exact"] / metric["examples"]}
    for key, result in expected.items():
        if not math.isclose(metric[key], result, rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"{scope}: inconsistent {key}")
    if metric["targets"] != metric["examples"] * 96:
        errors.append(f"{scope}: expected 96 scored targets per example")
    return errors


def audit_history(report, expected_order=None):
    errors, history = [], report.get("history", [])
    completed = report.get("completed_updates", -1)
    if completed != len(history) or not 0 < completed <= 2500:
        errors.append("Full contiguous update-1-origin history does not match completed_updates in 1..2500")
    if report.get("completed_epochs") != completed // STEPS_PER_EPOCH or report.get("batch_in_epoch") != completed % STEPS_PER_EPOCH:
        errors.append("Final epoch/batch position does not match the completed update count")
    for index, row in enumerate(history, 1):
        if row.get("update") != index or row.get("epoch") != (index - 1) // 200 + 1 or row.get("batch_in_epoch") != (index - 1) % 200:
            errors.append(f"Update {index}: update/epoch/batch position differs from the native 200-batch epoch")
        if not math.isclose(row.get("learning_rate", -1), expected_lr(index), rel_tol=1e-12, abs_tol=1e-15):
            errors.append(f"Update {index}: actual LR differs from the original 200-epoch cosine schedule")
        if row.get("native_targets") != 6144 or row.get("input_tokens") != 16384:
            errors.append(f"Update {index}: physical B64/T256/K96 counts differ")
        for key in ("native_loss", "gradient_norm", "seconds"):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                errors.append(f"Update {index}: {key} missing, negative or nonfinite")
        if row.get("ce_dtype") != "torch.float32":
            errors.append(f"Update {index}: aligned CE is not FP32")
        if expected_order and index in expected_order:
            for key in ("batch_sha256", "indices_sha256"):
                if row.get(key) != expected_order[index][key]:
                    errors.append(f"Update {index}: {key} differs from independently reconstructed frozen-corpus order")
        elif not row.get("batch_sha256") or not row.get("indices_sha256"):
            errors.append(f"Update {index}: missing order identity")
        if "answer" in row:
            errors.extend(metric_errors(row["answer"], f"Update {index} answer"))
        precision = row.get("precision")
        if not precision:
            errors.append(f"Update {index}: missing intended gradient/parameter/Adam precision observations")
        else:
            if precision.get("missing_gradients"):
                errors.append(f"Update {index}: missing intended parameter gradients")
            for kind in ("parameters", "gradients", "moments", "step_counters"):
                if precision.get(kind, {}).get("finite") is not True or precision[kind].get("dtypes") != ["torch.float32"]:
                    errors.append(f"Update {index}: {kind} is not reported finite FP32")
            if report.get("arm") in ARMS:
                count = 43 if report["arm"]=="seq-fp32" else 45
                if any(precision.get(kind,{}).get("tensors") != number for kind,number in
                       (("parameters",count),("gradients",count),("moments",2*count),("step_counters",count))):
                    errors.append(f"Update {index}: canonical parameter/gradient/Adam tensor coverage differs")
    development = report.get("development", {})
    required_dev = {0, *range(200, completed + 1, 200)} | ({completed} if completed in MILESTONES else set())
    missing = required_dev - {int(step) for step in development}
    if missing:
        errors.append(f"Missing scheduled development points: {sorted(missing)}")
    for step, row in development.items():
        if int(step) > completed:
            errors.append(f"Development update {step} exceeds completed trajectory")
        for kind in ("native", "answer"):
            errors.extend(metric_errors(row.get(kind, {}), f"Development {step}/{kind}"))
        if row.get("native") != row.get("answer"):
            errors.append(f"Development {step}: native and answer metrics differ for this answer-only task")
    return errors


def trajectory_prefix_errors(earlier, later):
    errors = []
    length = earlier.get("completed_updates", -1)
    if nontiming(earlier.get("history", [])) != nontiming(later.get("history", [])[:length]):
        errors.append("Earlier trajectory is not an exact non-timing prefix of its continuation")
    for key in ("development", "ablation_development"):
        for step, value in earlier.get(key, {}).items():
            if step not in later.get(key, {}) or nontiming(value) != nontiming(later[key][step]):
                errors.append(f"Earlier {key}/{step} was changed or dropped by continuation")
    if earlier.get("identity") != later.get("identity"):
        errors.append("Immutable pilot identity changed between continuation segments")
    return errors


def paired_rows(left, right):
    errors, result = [], []
    for a, b in zip(left.get("history", []), right.get("history", [])):
        for key in ("update", "epoch", "batch_in_epoch", "batch_sha256", "indices_sha256", "learning_rate"):
            if a.get(key) != b.get(key):
                errors.append(f"Paired update {a.get('update')}: {key} differs")
        result.append({"update": a["update"], "fp32_native_ce": a["native_loss"], "bf16_native_ce": b["native_loss"],
                       "bf16_minus_fp32_ce": b["native_loss"] - a["native_loss"]})
    return result, errors


def loss_gap_review(left, right, endpoint):
    rows, errors = paired_rows(left, right)
    rows = [row for row in rows if row["update"] <= endpoint]
    gaps = [row["bf16_minus_fp32_ce"] for row in rows]
    common = sorted(set(left.get("development", {})) & set(right.get("development", {})), key=int)
    dev = [{"update": int(step), "bf16_minus_fp32_ce": right["development"][step]["native"]["ce"] - left["development"][step]["native"]["ce"]}
           for step in common if 0 < int(step) <= endpoint and (int(step) % 200 == 0 or int(step) == endpoint)]
    consecutive = [(a["update"], b["update"]) for a, b in zip(dev, dev[1:])
                   if a["bf16_minus_fp32_ce"] > .02 and b["bf16_minus_fp32_ce"] > .02]
    last = statistics.mean(gaps[-200:]) if gaps else None
    return {"endpoint": endpoint, "paired_updates": len(rows), "pairing_errors": errors,
            "mean_bf16_minus_fp32_ce": statistics.mean(gaps) if gaps else None,
            "last200_mean_bf16_minus_fp32_ce": last,
            "last200_available": len(gaps) >= 200,
            "maximum_absolute_update_gap": max(map(abs, gaps), default=None),
            "development": dev, "consecutive_development_degradation_pairs": consecutive,
            "investigation_triggered": (last is not None and len(gaps) >= 200 and last > .02) or bool(consecutive),
            "interpretation": "A protocol investigation trigger, not a numerical-machine acceptance result"}, rows


def allocation_review(runs, endpoint, modal_token_accuracy):
    seq = runs.get("seq-fp32")
    seq_points = sorted((int(step), row["answer"]) for step, row in (seq or {}).get("development", {}).items()
                        if 0 < int(step) <= 1000 and int(step) % 200 == 0)
    ace_triplets = []
    for index in range(len(seq_points)-2):
        group = seq_points[index:index+3]
        if (group[1][0]-group[0][0] == 200 and group[2][0]-group[1][0] == 200
                and all(row["token_accuracy"] >= .999 and row["sequence_exact_match"] >= .99 for _, row in group)):
            ace_triplets.append(tuple(step for step, _ in group))
    learning = {}
    for arm, report in runs.items():
        initial, final = report.get("development", {}).get("0"), report.get("development", {}).get(str(endpoint))
        if initial is None or final is None:
            continue
        reduced = 1-final["native"]["ce"]/initial["native"]["ce"]
        accuracy_gap = final["answer"]["token_accuracy"] - modal_token_accuracy
        learning[arm] = {"native_ce_reduction_fraction_from_own_initialization_descriptive_only": reduced,
                         "token_accuracy_minus_frozen_modal_baseline": accuracy_gap,
                         "native_ce": final["native"]["ce"], "fixed_ce_learning_threshold": .8*math.log(14),
                         "substantial_learning": final["native"]["ce"] <= .8*math.log(14) or accuracy_gap >= .05,
                         "remaining_errors": final["answer"]["sequence_exact_match"] < 1.}
    substantial = any(row["substantial_learning"] for row in learning.values())
    if ace_triplets:
        suggestion = "Deprioritize this setting; SEQ met the prospectively fixed early-ace rule"
    elif not substantial:
        suggestion = "Inspect supervision/optimization and curves before investing further"
    elif endpoint == 1000:
        suggestion = "Eligible for extension to 2500 if numerical/operational concerns are resolved sufficiently for scoped use"
    elif any(row["remaining_errors"] for row in learning.values()):
        suggestion = "Eligible for the second initialization if task errors remain and no semantic/operational blocker remains; do not condition on which arm won"
    else:
        suggestion = "No remaining task errors at this endpoint; review allocation rather than automatically repeat a solved setting"
    return {"endpoint": endpoint, "early_ace_triplets": ace_triplets,
            "frozen_modal_token_accuracy": modal_token_accuracy, "learning_by_arm": learning,
            "substantial_learning_any_arm": substantial, "allocation_suggestion": suggestion,
            "qualification_review_required": True,
            "scope": "Development allocation only; no final research test or automatic numerical clearance"}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def reconstruct_order(preparation, maximum):
    """Independently rebuild both array-index and actual batch identities on CPU."""
    import numpy as np
    from cdrm.mad_data import load_dataset
    train = load_dataset(Path(preparation["data_roots"]["train_dev"]), "selective-copying", "train")
    if len(train) != 12800 or train.sha256 != preparation["reused_data"]["train"]["array_sha256"]:
        raise ValueError("Frozen inherited training corpus differs from preparation")
    if preparation["training_order"]["shuffle_seed"] != 925704:
        raise ValueError("Pilot shuffle seed differs from the unchanged protocol")
    result = {}
    for epoch in range((maximum + 199)//200):
        permutation = np.random.default_rng(np.random.SeedSequence([925704, epoch])).permutation(12800)
        for batch in range(min(200, maximum-200*epoch)):
            indices = permutation[batch*64:(batch+1)*64]
            index_digest = hashlib.sha256(f"array:{indices.dtype}:{indices.shape}:".encode()+indices.tobytes()).hexdigest()
            batch_digest = hashlib.sha256()
            for name in ("input_ids", "labels", "answer_labels"):
                values = np.ascontiguousarray(getattr(train,name)[indices], dtype="<i8")
                batch_digest.update(name.encode())
                batch_digest.update(json.dumps(list(values.shape)).encode())
                batch_digest.update(values.tobytes())
            result[epoch*200+batch+1] = {"indices_sha256": index_digest, "batch_sha256": batch_digest.hexdigest()}
    if result and result[1]["batch_sha256"] != preparation["training_order"]["initial_batch_sha256"]:
        raise ValueError("Independent first-batch reconstruction disagrees with preparation")
    return result


def snapshot_audit(report, report_path):
    """Verify the immutable source snapshot, not a potentially newer working tree."""
    errors = []
    sources = report.get("identity", {}).get("source_sha256", report.get("source_sha256", {}))
    if not sources:
        return ["Missing source identity"]
    for name, expected in sources.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"Unsafe source snapshot name: {name}")
            continue
        saved = Path(report_path).parent/"source"/path
        if not saved.is_file() or digest(saved) != expected:
            errors.append(f"Missing or changed source snapshot: {name}")
    return errors


def source_difference(left, right):
    """Report inventory differences explicitly; do not silently intersect maps."""
    return [name for name in sorted(set(left) | set(right)) if left.get(name) != right.get(name)]


def ablation_review(report, endpoint):
    value = report.get("ablation_development", {}).get(str(endpoint))
    if value is None:
        return {"available": False, "endpoint": endpoint}, [f"Missing CDRM branch ablation at update {endpoint}"]
    errors = []
    for arm in ("active", "lambda_zero"):
        for kind in ("native", "answer"):
            errors.extend(metric_errors(value.get(arm, {}).get(kind, {}), f"Ablation {endpoint}/{arm}/{kind}"))
    if errors:
        return {"available": True, "endpoint": endpoint, "metrics": value}, errors
    if value.get("training_state_unchanged") is not True:
        errors.append(f"Ablation {endpoint}: no successful unchanged-training-state check")
    active, zero = value["active"], value["lambda_zero"]
    if nontiming(active) != nontiming(report["development"][str(endpoint)]):
        errors.append(f"Active ablation evaluation at {endpoint} differs from the same checkpoint's development metrics")
    return {"available": True, "endpoint": endpoint, "active": compact(active), "lambda_zero": compact(zero),
            "lambda_zero_minus_active_native_ce": zero["native"]["ce"]-active["native"]["ce"],
            "active_minus_lambda_zero_token_accuracy": active["answer"]["token_accuracy"]-zero["answer"]["token_accuracy"],
            "active_minus_lambda_zero_sequence_exact_match": active["answer"]["sequence_exact_match"]-zero["answer"]["sequence_exact_match"],
            "interpretation": "Dependence of this trained CDRM on its branch; separately trained SEQ is the architecture comparison"}, errors


def numerical_attention(report):
    reasons=[]
    pair=report.get("comparisons",{}).get(PRIMARY,{})
    for label in ("actual_ce_gradients","independent_side_gradients"):
        scope=pair.get(label,{})
        if any(not row.get("finite") for row in scope.get("rows",{}).values()) or scope.get("invalid_tensors"):
            reasons.append(f"{label}: missing/nonfinite gradient observation")
        if scope.get("per_tensor_maximum_failures"):
            reasons.append(f"{label}: maximum-error failures")
        if scope.get("per_tensor_l2_failures"):
            reasons.append(f"{label}: per-tensor L2 failures")
    if pair.get("independent_side_gradients",{}).get("pass") is not True:
        reasons.append("Independent unscaled side-gradient screen did not pass")
    if pair.get("adam",{}).get("guardrail_pass") is not True:
        reasons.append("Same-state trained Adam delta guardrail did not pass")
    for scope in ("unscaled_side_states","actual_forward_unscaled_side_states"):
        if any(row.get("prospective_l2_pass") is not True or row.get("prospective_maximum_pass") is not True
               for row in pair.get(scope,{}).values()):
            reasons.append(f"{scope}: unscaled state screen did not pass")
    if report.get("scaling_requested") and report.get("scaling_pass") is not True:
        reasons.append("Requested cotangent scaling check did not pass")
    return {"observed_investigation_triggers":reasons,
            "bf16_actual_ce_global_flag":pair.get("actual_ce_gradients",{}).get("global_parameter_l2_pass") is False,
            "fp32_reference_machine_flag":report.get("comparisons",{}).get("tiled_fp32_vs_naive_fp32",{}).get("machine_screens_pass") is False,
            "interpretation":"Known aggregate/reference flag classes remain visible and require scoped review; no failed result is converted to a pass"}


def numerical_review(paths, preparation, checkpoint_roles, criteria_sha, preparation_sha):
    rows, errors, seen = [], [], set()
    slots = {slot["slot"]: slot for slot in preparation["confirmation"]["slots"]}
    for path in paths:
        report, reference = load_report(path)
        checkpoint_sha = report.get("checkpoint", {}).get("sha256")
        role = checkpoint_roles.get(checkpoint_sha)
        if role is None:
            errors.append(f"Numerical checkpoint is not a declared retained pilot CDRM milestone: {path}")
            expected = None
        else:
            key = f"seed{role['seed']}-u{role['update']:04d}-{role['trajectory']}"
            expected = slots[key]
            if key in seen:
                errors.append(f"Duplicate supplied numerical result for slot {key}; retain attempts explicitly without selecting one")
            seen.add(key)
            fixture = report.get("fixture", {})
            if (fixture.get("sha256") != expected["array_sha256"]
                    or fixture.get("dataset_sha256") != preparation["confirmation"]["dataset_identity"]["array_sha256"]
                    or fixture.get("example_offset") != expected["example_offset"]
                    or fixture.get("shape") != [64, 256]
                    or fixture.get("native_targets") != 6144
                    or report.get("checkpoint", {}).get("completed_updates") != role["update"]):
                errors.append(f"Numerical report differs from the prospectively assigned B64 fixture: {key}")
        if report.get("criteria", {}).get("sha256") != criteria_sha:
            errors.append(f"Numerical report uses a different frozen contract: {path}")
        tensor_ref=report.get("tensor_artifact",{})
        tensor_path=Path(tensor_ref.get("path",""))
        tensor_verified=tensor_path.is_file() and digest(tensor_path)==tensor_ref.get("sha256")
        if not tensor_verified:
            errors.append(f"Saved numerical tensor artifact is missing or its checksum differs: {path}")
        case = Path(path).parent
        launch_path = case.with_name(case.name+".slot-launch.json")
        audit_path = case.with_name(case.name+".slot-audit.json")
        slot_review = {"available":launch_path.is_file() and audit_path.is_file()}
        if slot_review["available"]:
            launch,launch_ref = load_report(launch_path)
            audit,audit_ref = load_report(audit_path)
            slot_review.update(launch=launch_ref,audit=audit_ref,observation=audit)
            if (audit.get("status") != "verified_observation" or audit.get("identity_and_fixture_verified") is not True
                    or audit.get("numerical_clearance") is not False
                    or audit.get("machine_screens_pass") != report.get("machine_screens_pass")
                    or audit.get("numerical_report",{}).get("sha256") != reference["sha256"]
                    or audit.get("launch",{}).get("sha256") != launch_ref["sha256"]
                    or launch.get("checkpoint",{}).get("sha256") != checkpoint_sha
                    or launch.get("preparation",{}).get("sha256") != preparation_sha
                    or launch.get("slot") != expected
                    or launch.get("protocol",{}).get("sha256") != PROTOCOL_SHA256
                    or launch.get("criteria",{}).get("sha256") != criteria_sha):
                errors.append(f"Numerical wrapper role/identity audit disagrees with the retained report: {path}")
        else:
            errors.append(f"Missing independently retained numerical slot launch/completion audit: {path}")
        rows.append({"artifact": reference, "role": role, "allocated_slot": expected,
                     "status": report.get("status"), "machine_screens_pass": report.get("machine_screens_pass"),
                     "numerical_clearance": report.get("numerical_clearance"), "disposition": report.get("disposition"),
                     "criteria": report.get("criteria"), "checkpoint": report.get("checkpoint"),
                     "tensor_artifact":tensor_ref,"tensor_sha256_verified":tensor_verified,
                     "comparisons": compact(report.get("comparisons", {})), "wandb": report.get("wandb"),
                     "slot_audit":slot_review,
                     "review_attention":numerical_attention(report),
                     "all_original_false_machine_predicates": false_flags(report),
                     "scope": "Original flags retained verbatim in meaning; engineering qualification is a separate field"})
    return rows, errors, seen


def plot_summary(output, selected, paired, ablations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout="constrained")
    for seed, cohort in sorted(selected.items()):
        style = "-" if seed == min(selected) else "--"
        for arm, report in cohort.items():
            history = report["history"]
            rolling = [statistics.mean([r["native_loss"] for r in history[max(0,i-49):i+1]]) for i in range(len(history))]
            label = f"{arm}, seed {seed}"
            axes[0,0].plot([r["update"] for r in history], rolling, style, color=COLORS[arm], label=label)
            development = sorted((int(step), row) for step,row in report["development"].items())
            for axis, kind, key in ((axes[0,1],"native","ce"), (axes[1,0],"answer","token_accuracy"),
                                    (axes[1,1],"answer","sequence_exact_match")):
                axis.plot([step for step,_ in development], [row[kind][key] for _,row in development],
                          style, marker="o", markersize=3, color=COLORS[arm], label=label)
        pairs = [row for row in paired if row["seed"] == seed]
        if pairs:
            smoothed = [statistics.mean([r["bf16_minus_fp32_ce"] for r in pairs[max(0,i-199):i+1]]) for i in range(len(pairs))]
            axes[2,0].plot([r["update"] for r in pairs], smoothed, style, label=f"seed {seed}")
    for row in ablations:
        if row["available"]:
            axes[2,1].scatter(row["endpoint"], row["lambda_zero_minus_active_native_ce"],
                              color=COLORS[row["arm"]], marker="o" if row["seed"] == min(selected) else "x",
                              label=f"{row['arm']}, seed {row['seed']}, u{row['endpoint']}")
    settings = (("Training CE, trailing 50 updates", "CE (nats)"), ("Development native CE", "CE (nats)"),
                ("Development answer-token accuracy", "Accuracy"), ("Direct sequence exact match", "Exact match"),
                ("CDRM BF16 minus FP32 CE, trailing 200", "CE difference (nats)"),
                ("Branch use at fixed trained CDRM weights", "Lambda-zero minus active CE (nats)"))
    for axis, (title, label) in zip(axes.flat, settings):
        axis.set(title=title, xlabel="Completed optimizer updates", ylabel=label)
        axis.grid(alpha=.2)
        if axis.lines or axis.collections:
            axis.legend(frameon=False, fontsize=7)
    axes[2,0].axhline(.02, color="#aa3333", linestyle=":", linewidth=1)
    axes[2,1].axhline(0., color="#555555", linewidth=.7)
    if not paired:
        axes[2,0].text(.5,.5,"Matched CDRM precision trajectories pending",ha="center",va="center",transform=axes[2,0].transAxes)
    if not ablations:
        axes[2,1].text(.5,.5,"Trained CDRM branch ablations pending",ha="center",va="center",transform=axes[2,1].transAxes)
    fig.suptitle("Five-block learning pilot · MAD selective copying V16/T256/K96 · development evidence")
    for suffix in ("png", "svg"):
        fig.savefig(output/f"summary.{suffix}", dpi=170)
    plt.close(fig)


def audit_run(report, reference, preparation, expected_order):
    errors = audit_history(report, expected_order)
    errors.extend(snapshot_audit(report, reference["path"]))
    identity = report["identity"]
    if report.get("identity_sha256") != json_digest(identity):
        errors.append("Pilot identity checksum differs")
    if report.get("schema") != "cdrm-tiled-pilot-v1" or report.get("status") != "complete":
        errors.append("Run is not a completed pilot-v1 report")
    arm, seed = report["arm"], int(report["initialization"]["seed"])
    if arm not in ARMS or seed not in (7500, 7501):
        errors.append("Undeclared arm or initialization seed")
        return errors
    initial = preparation["initialization_checkpoints"][str(seed)]
    expected_initialization = (preparation["prior_checkpoints"]["init7500"]["initialization"] if seed == 7500
                               else preparation["fresh_initialization_checks"]["construction"])
    if report["initialization"] != expected_initialization or identity["initialization"] != expected_initialization:
        errors.append("Initialization seeds/weights differ from frozen preparation")
    if (identity["initial_checkpoint_sha256"] != initial["sha256"]
            or report["initial_checkpoint"]["sha256"] != initial["sha256"]
            or identity["backbone_initialization_sha256"] != expected_initialization["backbone_initialization_sha256"]):
        errors.append("Original initialization or shared backbone identity differs")
    origin = (preparation["prior_checkpoints"][f"{arm.split('-')[1]}_u0100"]
              if seed == 7500 and arm.startswith("cdrm") else initial)
    if identity.get("origin", {}).get("sha256") != origin["sha256"]:
        errors.append("Run origin differs from the prescribed original state (SEQ initialization or CDRM continuation)")
    if identity.get("protocol_sha256") != preparation["protocol"]["sha256"]:
        errors.append("Runner protocol differs from prospectively frozen preparation")
    if identity.get("data") != preparation["reused_data"]:
        errors.append("Train/development dataset identity differs from preparation")
    expected = {"physical_batch": 64, "shuffle_seed": 925704, "data_order": "shared_epoch_permutation",
                "accumulation": False, "updates_per_epoch": 200, "maximum_updates": 2500, "eval_every": 200,
                "ablation_updates": [1000,2500]}
    if any(identity.get(key) != value for key,value in expected.items()):
        errors.append("Immutable batching/schedule/stopping protocol differs")
    if not {190,210,1000,2500}.issubset(identity.get("checkpoint_updates", [])):
        errors.append("Required retained checkpoint calendar differs")
    optimizer = {"type":"AdamW","lr":5e-4,"betas":[.9,.98],"eps":1e-8,"weight_decay":0.,
                 "foreach":False,"fused":False,"gradient_clip":1.}
    schedule = {"type":"CosineAnnealingLR","T_max_epochs":200,"eta_min":1e-6,
                "step":"after each completed epoch","warmup":False}
    if identity.get("optimizer") != optimizer or identity.get("schedule") != schedule:
        errors.append("Optimizer or original epoch schedule semantics differ")
    cfg = identity["model_config"]
    cfg_expected = {"n_layers":5,"d_model":128,"n_heads":16,"n_kv_heads":16,"mlp_hidden_size":512,
                    "cdrm_early_layer":1,"cdrm_late_layer":3,"cdrm_epsilon":.1,"cdrm_rho":1.,"cdrm_lambda":.01,
                    "vocab_size":16,"cdrm_backend":"tiled","cdrm_enabled":arm != "seq-fp32",
                    "cdrm_precision_policy":"bf16_fp32_state" if arm == "cdrm-bf16" else "fp32",
                    "recurrent_layers":[],"reference_eager":False,"weight_tying":False}
    if any(cfg.get(key) != value for key,value in cfg_expected.items()) or report["model_config"] != cfg:
        errors.append("Model architecture or explicit precision policy differs")
    logits_dtype = "torch.bfloat16" if arm == "cdrm-bf16" else "torch.float32"
    if any(row.get("logits_dtype") != logits_dtype for row in report.get("history", [])):
        errors.append("Observed training head precision differs from the declared arm")
    parent_sources = (preparation["prior_checkpoints"]["init7500"]["source_sha256"] if seed == 7500
                      else preparation["source_sha256"])
    if identity.get("parent_source_sha256") != parent_sources:
        errors.append("Parent source inventory differs from preparation")
    if any(identity["source_sha256"].get(name) != value for name,value in parent_sources.items()):
        errors.append("Child source inventory changed or omitted a retained parent source")
    resume = report.get("resume", {})
    if resume.get("starting_update", 0) and (not resume.get("loaded_state_exact") or resume.get("warmup_updates") != 0):
        errors.append("Continuation did not report exact restoration without extra warmup updates")
    if report.get("arguments", {}).get("stop_updates") != report["completed_updates"]:
        errors.append("Completed run did not reach its explicit invocation stopping target")
    return errors


def paired_identity_errors(left, right):
    errors = []
    a, b = left["identity"], right["identity"]
    for key in ("initial_checkpoint_sha256", "initial_identity_sha256", "initialization", "backbone_initialization_sha256",
                "source_sha256", "parent_source_sha256", "execution_contract", "data", "physical_batch", "shuffle_seed",
                "data_order", "accumulation", "optimizer", "schedule", "updates_per_epoch", "maximum_updates", "eval_every",
                "checkpoint_updates", "ablation_updates", "protocol_sha256"):
        if a.get(key) != b.get(key):
            errors.append(f"Paired {key} differs")
    configs = [copy.deepcopy(item["model_config"]) for item in (a,b)]
    for cfg in configs:
        for key in ("cdrm_enabled", "cdrm_backend", "cdrm_precision_policy", "reference_eager"):
            cfg.pop(key, None)
    if configs[0] != configs[1]:
        errors.append("Paired model config differs beyond declared fabric and precision controls")
    return errors


def monitor_summary(value):
    history=value.get("history",[])
    signals={name for row in history for name in row.get("parameter_gradient_signals",{})}
    ratios={name for row in history for name in row.get("state_summaries",{})
            if "over" in name and isinstance(row["state_summaries"][name],(int,float))}
    return {"monitored_updates":len(history),"canonical_gradient_monitor_count":len(signals),
            "parameters_without_any_nonzero_observed_gradient":sorted(name for name in signals if not any(
                row.get("parameter_gradient_signals",{}).get(name,{}).get("nonzero") for row in history)),
            "correction_ratio_ranges":{name:{"first":samples[0],"last":samples[-1],"minimum":min(samples),"maximum":max(samples)}
                for name in sorted(ratios) if (samples:=[row["state_summaries"][name] for row in history if name in row.get("state_summaries",{})])},
            "cumulative_input_tokens":sum(row.get("input_tokens",0) for row in history),
            "cumulative_scored_tokens":sum(row.get("native_targets",0) for row in history),
            "recorded_update_seconds":sum(row["seconds"] for row in history),
            "invocation_wall_seconds":value.get("elapsed_seconds"),
            "time_scope":"Recorded model update time includes its compilation/warmup when present; invocation wall time also includes evaluation, logging and retention. No warmed benchmark claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-report",type=Path,required=True)
    parser.add_argument("--run-report",type=Path,action="append",required=True)
    parser.add_argument("--numerical-report",type=Path,action="append",default=[])
    parser.add_argument("--recovery-report",type=Path,action="append",default=[])
    parser.add_argument("--support-report",type=Path,action="append",default=[])
    parser.add_argument("--prior-qualification-report",type=Path,required=True)
    parser.add_argument("--criteria",type=Path,default=Path("docs/reports/cdrm-tiled-bf16/validation-contract.md"))
    parser.add_argument("--reviewed-disposition",required=True)
    parser.add_argument("--require-complete-evidence",action="store_true",
                        help="Require complete matched cohorts, assigned NUM at every used milestone, and epoch-boundary recovery")
    parser.add_argument("--output-dir",type=Path,required=True)
    from experiment_tracking import OnlineTracker, add_wandb_arguments
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project="cdrm-tiled-learning-pilot",wandb_run_name="pilot-combined-summary")
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Run retained reporting in the explicit project CPU container")
    import torch
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("Use CDRM_DOCKER_GPUS=none; reporting may not initialize or use CUDA")
    if not args.wandb_project:
        parser.error("The combined graphable report requires online W&B")
    if args.output_dir.exists():
        raise FileExistsError("Use a new report directory")
    args.output_dir.mkdir(parents=True)
    started = time.monotonic(); tracker = None
    report = {"schema":"cdrm-tiled-pilot-aggregate-v1","status":"running","audit_pass":False,
              "scope":"Bounded development learning and precision comparison; no final task test or blanket numerical clearance",
              "arguments":{key:[str(v) for v in value] if isinstance(value,list) else str(value) if isinstance(value,Path) else value
                           for key,value in vars(args).items()}, "source_sha256":{str(Path(__file__)):digest(__file__)}}
    try:
        (args.output_dir/"cdrm_tiled_pilot_report.py").write_bytes(Path(__file__).read_bytes())
        preparation, preparation_ref = load_report(args.preparation_report)
        prior, prior_ref = load_report(args.prior_qualification_report)
        if preparation["protocol"]["sha256"] != PROTOCOL_SHA256 or digest(args.criteria) != CRITERIA_SHA256:
            raise ValueError("Require the prospectively frozen pilot protocol and unchanged numerical contract")
        report["preparation"] = {"artifact":preparation_ref,"protocol":preparation["protocol"],
                                 "confirmation":preparation["confirmation"],"development_baselines":preparation["development_baselines"]}
        report["qualification"] = {"reviewed_disposition":args.reviewed_disposition,"prior_artifact":prior_ref,
                                   "prior_summary":compact(prior),"all_prior_false_machine_predicates":false_flags(prior),
                                   "policy_changed":False,"numerical_clearance_promoted":False,
                                   "scope":"User-authorized scoped opt-in; kept separate from machine flags and this audit"}
        records = [(value,reference) for value,reference in (load_report(path) for path in args.run_report)]
        maximum = max(value.get("completed_updates",0) for value,_ in records)
        order = reconstruct_order(preparation,maximum+1)
        errors = snapshot_audit(preparation,preparation_ref["path"])
        if preparation.get("status") != "complete":
            errors.append("Preparation did not complete")
        grouped, ledger, run_summaries = {}, {}, []
        for value,reference in records:
            seed,arm = int(value["initialization"]["seed"]),value["arm"]
            run_errors = audit_run(value,reference,preparation,order)
            errors.extend(f"{reference['path']}: {error}" for error in run_errors)
            grouped.setdefault(seed,{}).setdefault(arm,[]).append((value,reference))
            checkpoint_refs = dict(value.get("checkpoints",{}))
            resume = value.get("resume",{})
            if resume.get("checkpoint"):
                checkpoint_refs[str(resume["starting_update"])] = resume["checkpoint"]
            for step,record in checkpoint_refs.items():
                if int(step) in MILESTONES and arm.startswith("cdrm"):
                    ledger[record["sha256"]] = {"seed":seed,"update":int(step),"trajectory":arm.split('-')[1],"checkpoint":record}
            run_summaries.append({"artifact":reference,"seed":seed,"arm":arm,"status":value["status"],
                                  "identity_sha256":value["identity_sha256"],"identity":value["identity"],
                                  "resume":value.get("resume"),"ancestry":value.get("ancestry"),
                                  "completed_updates":value["completed_updates"],"completed_epochs":value["completed_epochs"],
                                  "batch_in_epoch":value["batch_in_epoch"],"checkpoints":value.get("checkpoints"),
                                  "parameter_count":value["construction"]["parameter_count"],
                                  "training_seconds":value["training_seconds"],"invocation_training_seconds":value.get("invocation_training_seconds"),
                                  "clipping_count":value.get("clipping_count"),"final_precision":value.get("final_precision"),
                                  "monitors":monitor_summary(value),
                                  "compiler":value.get("compiler"),"wandb":value.get("wandb"),
                                  "audit_errors":run_errors})
        selected, selected_refs, prefix_reviews = {}, {}, []
        for seed,cohort in grouped.items():
            selected[seed],selected_refs[seed] = {},{}
            for arm,segments in cohort.items():
                segments.sort(key=lambda pair:pair[0]["completed_updates"])
                for (earlier,eref),(later,lref) in zip(segments,segments[1:]):
                    failed = trajectory_prefix_errors(earlier,later)
                    errors.extend(f"{seed}/{arm}: {error}" for error in failed)
                    prefix_reviews.append({"seed":seed,"arm":arm,"earlier":eref,"later":lref,
                                           "exact_nontiming_prefix":not failed,"errors":failed})
                selected[seed][arm],selected_refs[seed][arm] = segments[-1]
        training,development,paired,ablations,cohorts,pending = [],[],[],[],{},[]
        modal = preparation["development_baselines"]["order_ignoring_modal"]["answer_accuracy"]
        for seed,cohort in sorted(selected.items()):
            missing = sorted(set(ARMS)-set(cohort))
            if missing:
                pending.append(f"Seed {seed}: missing architecture/precision arms {missing}")
            for arm,value in cohort.items():
                for row in value["history"]:
                    training.append({"seed":seed,"arm":arm,**{key:row[key] for key in
                        ("update","epoch","batch_in_epoch","native_loss","learning_rate","gradient_norm","clipped","seconds","input_tokens","native_targets")},
                        "answer_token_accuracy":row.get("answer",{}).get("token_accuracy"),
                        "answer_sequence_exact_match":row.get("answer",{}).get("sequence_exact_match")})
                for step,row in value["development"].items():
                    development.append({"seed":seed,"arm":arm,"update":int(step),"native_ce":row["native"]["ce"],
                                        "answer_token_accuracy":row["answer"]["token_accuracy"],
                                        "answer_sequence_exact_match":row["answer"]["sequence_exact_match"],
                                        "direct_exact_sequences":row["answer"]["exact"],"examples":row["answer"]["examples"]})
            available = list(cohort)
            for arm in available[1:]:
                failed = paired_identity_errors(cohort[available[0]],cohort[arm])
                _,trajectory_failed = paired_rows(cohort[available[0]],cohort[arm])
                errors.extend(f"Seed {seed}, {available[0]} vs {arm}: {error}" for error in [*failed,*trajectory_failed])
            endpoints = [step for step in MILESTONES if any(value["completed_updates"] >= step for value in cohort.values())]
            cohort_summary = {"available_arms":available,"latest_completed_updates":{arm:value["completed_updates"] for arm,value in cohort.items()},
                              "latest_artifacts":selected_refs[seed],"milestones":{}}
            for endpoint in endpoints:
                eligible = {arm:value for arm,value in cohort.items() if value["completed_updates"] >= endpoint}
                review = allocation_review(eligible,endpoint,modal)
                review["complete_matched_cohort"] = set(eligible)==set(ARMS)
                if not review["complete_matched_cohort"]:
                    pending.append(f"Seed {seed}, update {endpoint}: no complete matched cohort")
                if {"cdrm-fp32","cdrm-bf16"}.issubset(eligible):
                    gap,pair_rows = loss_gap_review(eligible["cdrm-fp32"],eligible["cdrm-bf16"],endpoint)
                    review["paired_precision_loss"]=gap
                    if endpoint==max(step for step in endpoints if all(cohort[arm]["completed_updates"] >= step for arm in ("cdrm-fp32","cdrm-bf16"))):
                        paired.extend({"seed":seed,**row} for row in pair_rows)
                for arm,value in eligible.items():
                    if arm.startswith("cdrm"):
                        ablation,failed=ablation_review(value,endpoint)
                        errors.extend(f"Seed {seed}/{arm}: {error}" for error in failed)
                        ablations.append({"seed":seed,"arm":arm,**ablation})
                cohort_summary["milestones"][str(endpoint)]=review
            cohorts[str(seed)]=cohort_summary
        numerical,num_errors,seen=numerical_review(args.numerical_report,preparation,ledger,digest(args.criteria),preparation_ref["sha256"])
        errors.extend(num_errors)
        required_slots={f"seed{seed}-u{step:04d}-{arm.split('-')[1]}" for seed,cohort in selected.items()
                        for arm,value in cohort.items() if arm.startswith("cdrm") for step in MILESTONES if value["completed_updates"]>=step}
        pending.extend(f"Missing numerical monitoring slot {slot}" for slot in sorted(required_slots-seen))
        recoveries=[]
        for path in args.recovery_report:
            value,reference=load_report(path)
            verdict=value.get("recovery_comparison",value)
            failed=audit_run(value,reference,preparation,order)
            recovery_seed=int(value["initialization"]["seed"])
            recovery_arm=value["arm"]
            if recovery_arm!="cdrm-bf16":
                failed.append("Required epoch-boundary recovery must exercise CDRM BF16")
            counterpart=selected.get(recovery_seed,{}).get(recovery_arm)
            if counterpart is None or counterpart["identity"]!=value["identity"]:
                failed.append("Recovery identity has no matching retained pilot trajectory")
            errors.extend(f"Recovery {path}: {error}" for error in failed)
            recoveries.append({"artifact":reference,"status":value.get("status"),"resume":value.get("resume"),
                               "completed_updates":value.get("completed_updates"),"comparison":verdict,
                               "identity_matches_pilot":not failed,"audit_errors":failed,
                               "all_original_false_machine_predicates":false_flags(value),"wandb":value.get("wandb")})
        if not any(row["comparison"].get("bitwise_state_and_nontiming_metrics_equal") is True
                   and row["identity_matches_pilot"]
                   and row.get("resume",{}).get("starting_update",200)>=0
                   and row.get("resume",{}).get("starting_update",200)<200<row.get("completed_updates",0) for row in recoveries):
            pending.append("No supplied successful exact recovery across an epoch boundary")
        support=[]
        for path in args.support_report:
            value,reference=load_report(path)
            support.append({"artifact":reference,"summary":compact(value),"all_original_false_machine_predicates":false_flags(value)})
        report.update(runs=run_summaries,prefix_audits=prefix_reviews,cohorts=cohorts,ablations=ablations,
                      numerical=numerical,recovery=recoveries,support=support,pending_evidence=pending,
                      audit_errors=errors,audit_pass=not errors,
                      all_supplied_numerical_machine_screens_pass=all(row["machine_screens_pass"] is True for row in numerical) if numerical else None,
                      complete_evidence=not pending,independent_order_reconstruction={"updates":maximum+1,"corpus":preparation["reused_data"]["train"],
                      "policy":"Rebuilt permutations and actual batch hashes from fixed arrays; no model/global RNG used",
                      "next_batch_scope":"The following batch is independently reconstructed. Exact restored-cursor agreement is supplied by the audited runner/recovery reports; this helper does not load checkpoints."})
        write_csv(args.output_dir/"training.csv",training)
        write_csv(args.output_dir/"development.csv",development)
        write_csv(args.output_dir/"paired-precision.csv",paired)
        write_csv(args.output_dir/"branch-ablation.csv",[{key:value for key,value in row.items() if not isinstance(value,(dict,list))} for row in ablations])
        plot_summary(args.output_dir,selected,paired,ablations)
        for reference in [preparation_ref,prior_ref,*[ref for _,ref in records],
                          *[row["artifact"] for row in numerical],*[row["artifact"] for row in recoveries],
                          *[row["artifact"] for row in support]]:
            if digest(reference["path"])!=reference["sha256"]:
                raise RuntimeError("A retained input report changed during aggregation")
        tracker=OnlineTracker(project=args.wandb_project,entity=args.wandb_entity,group=args.wandb_group,
                              name=args.wandb_run_name,output_dir=args.output_dir)
        tracker.start({"scope":report["scope"],"preparation":preparation_ref,"sources":report["source_sha256"],
                       "reviewed_disposition":args.reviewed_disposition,"run_reports":[ref for _,ref in records]})
        report["wandb"]=tracker.record
        log_rows={}
        for row in training:
            log_rows.setdefault(row["update"],{"update":row["update"]})[f"train/seed{row['seed']}/{row['arm']}/native_ce"]=row["native_loss"]
        for row in development:
            log_rows.setdefault(row["update"],{"update":row["update"]}).update({f"dev/seed{row['seed']}/{row['arm']}/{key}":row[key]
                for key in ("native_ce","answer_token_accuracy","answer_sequence_exact_match")})
        for row in paired:
            log_rows.setdefault(row["update"],{"update":row["update"]})[f"train/seed{row['seed']}/bf16_minus_fp32_ce"]=row["bf16_minus_fp32_ce"]
        for step in sorted(log_rows):
            tracker.log(log_rows[step])
        import wandb
        tracker.log({"summary/learning_curves":wandb.Image(str(args.output_dir/"summary.png"))})
        report["status"]="complete" if not errors and (not args.require_complete_evidence or not pending) else "audit_failed"
        tracker.summary({"audit_pass":not errors,"complete_evidence":not pending,
                         "all_supplied_numerical_machine_screens_pass":report["all_supplied_numerical_machine_screens_pass"],
                         "reviewed_disposition":args.reviewed_disposition,"numerical_clearance_promoted":False})
        text=["This is a bounded development comparison; numerical machine flags and reviewed opt-in qualification remain separate.","",
              f"Audit: {'passed' if not errors else 'failed'}. Complete requested evidence: {not pending}.","",
              "| Seed | Arm | Updates | Dev CE | Token accuracy | Direct sequence exact |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
        for seed,cohort in sorted(selected.items()):
            for arm,value in cohort.items():
                step=value["completed_updates"]; metric=value["development"].get(str(step))
                if metric:
                    text.append(f"| {seed} | {arm} | {step} | {metric['native']['ce']:.6f} | {metric['answer']['token_accuracy']:.4%} | {metric['answer']['sequence_exact_match']:.4%} |")
        text.extend(["",args.reviewed_disposition,"",f"Original numerical machine failures are preserved in [report.json](report.json). [Combined W&B run]({tracker.record['run_url']})."])
        if errors:
            text.extend(["","Audit issues:","",*[f"- {error}" for error in errors]])
        if pending:
            text.extend(["","Pending scope:","",*[f"- {item}" for item in pending]])
        (args.output_dir/"results.md").write_text("\n".join(text)+"\n")
        print(json.dumps({"status":report["status"],"audit_errors":errors,"pending_evidence":pending,
                          "wandb_run_url":tracker.record["run_url"]}),flush=True)
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__)
        raise
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        try:
            if tracker:
                tracker.finish(succeeded=report["status"]=="complete")
        except Exception as error:
            report.update(status="failed",tracking_final_sync_error_type=type(error).__name__)
            raise
        finally:
            save_json(args.output_dir/"report.json",report)
    if report["status"]!="complete":
        raise SystemExit(1)


if __name__=="__main__":
    main()
