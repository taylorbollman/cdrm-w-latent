#!/usr/bin/env python3
"""Report the explicitly user-stopped 80k RT + NextLat checkpoint from saved evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

from scripts import rt_a5_nextlat_budget_report as budget
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import (
    CSV_COLUMNS, ROLES, STEPS, _digest_dict, _sha, local_path, metric_rows,
    normalized_contract, read_training, training_curve,
)
from scripts.rt_a5_report import finite_number, hash_file, read_arm, read_input, write_json

ROOT, LINEAGE = budget.ROOT, budget.LINEAGE
SCHEMA = "rt-a5-nextlat-stopped-budget-comparison-v1"
ENDPOINT = 80000
NEW_STEPS = tuple(s for s in budget.NEW_STEPS if s <= ENDPOINT)
REPORTING_SOURCES = ("scripts/rt_a5_nextlat_stopped_report.py", *budget.REPORTING_SOURCES)


def verify_file(record, expected_path=None):
    path = local_path(record["path"]).resolve()
    if expected_path is not None and path != local_path(expected_path).resolve():
        raise ValueError("Revision evidence path differs from selected input")
    actual = hash_file(path)
    if any(actual[key] != record[key] for key in ("sha256", "bytes")):
        raise ValueError(f"Revision evidence hash/size differs: {path.name}")
    return actual


def validate_revision(revision, protocol_path, directory):
    """Only this explicit stop receipt permits interpretation of a failed trainer report."""
    expected = {"schema": "rt-a5-nextlat-endpoint-revision-v1", "action": "stop_at_retained_checkpoint",
                "original_endpoint": 100000, "accepted_endpoint": ENDPOINT,
                "selection_kind": "user_selected_after_development_review",
                "confirmation_evaluated": False, "latent_rollout_evaluated": False,
                "expected_new_checkpoints": list(NEW_STEPS)}
    for key, value in expected.items():
        if revision.get(key) != value:
            raise ValueError(f"User-stop revision scope differs: {key}")
    if not isinstance(revision.get("user_instruction"), str) or not revision["user_instruction"].strip():
        raise ValueError("Explicit user stop instruction must be retained")
    termination = revision["termination"]
    for key, value in {"signal": "SIGINT", "exit_code": 1, "trainer_process_gone": True,
                       "container_gone": True}.items():
        if termination.get(key) != value:
            raise ValueError(f"Explicit user-stop termination evidence differs: {key}")
    exit_record = verify_file(termination["exit_record"])
    if Path(exit_record["path"]).read_text().strip() != "1":
        raise ValueError("Intentional interrupt exit receipt differs")
    inputs = {"protocol": verify_file(revision["protocol"], protocol_path),
              "report": verify_file(revision["raw_training_report"], directory / "report.json"),
              "history": verify_file(revision["raw_history"], directory / "history.jsonl"),
              "accepted_checkpoint": verify_file(revision["accepted_checkpoint"],
                                                   directory / "checkpoints/step-080000.pt"),
              "termination_exit": exit_record}
    if revision["accepted_checkpoint"].get("completed_updates") != ENDPOINT:
        raise ValueError("Accepted checkpoint update differs")
    return inputs


def validate_stopped_history(history, raw_report, revision):
    """Keep the persisted checkpoint prefix distinct from the uncheckpointed live tail."""
    stopped = raw_report["completed_updates"]
    recorded = revision["raw_training_report"]
    if (raw_report.get("schema") != budget.TRAIN_SCHEMA or raw_report.get("status") != "failed"
            or raw_report.get("error_type") != "KeyboardInterrupt"
            or raw_report.get("wandb", {}).get("status") != "synced_failed_experiment"
            or recorded.get("status") != "failed" or recorded.get("error_type") != "KeyboardInterrupt"
            or recorded.get("wandb_status") != "synced_failed_experiment"
            or recorded.get("completed_updates") != stopped or recorded.get("configured_endpoint") != 100000
            or raw_report.get("endpoint") != 100000 or raw_report.get("start_update") != 10000
            or type(stopped) is not int or not ENDPOINT < stopped < 90000):
        raise ValueError("Raw failure does not match the specifically authorized user interrupt")
    if raw_report.get("confirmation_evaluated") is not False or raw_report.get("latent_rollout_evaluated") is not False:
        raise ValueError("Confirmation and autonomous rollout must remain unevaluated")
    if len(history) != stopped - 10000 or any(row.get("update") != i for i, row in enumerate(history, 10001)):
        raise ValueError("Raw stop history must contain every complete update exactly once")
    accepted = history[:ENDPOINT-10000]
    tail = history[ENDPOINT-10000:]
    # The original exception handler updates completed_updates, but retains the
    # latest checkpoint's train_seconds/order_chain. Validate that exact behavior.
    budget.validate_history(accepted, raw_report, 10000, ENDPOINT)
    # Finite loss/exposure checks for the excluded tail use independently summed
    # history timing; there is no saved model/order counter for that tail.
    tail_seconds = sum(row["seconds"] for row in tail)
    budget.validate_history(tail, {"order_chain": tail[-1]["order_chain"], "train_seconds": tail_seconds,
                                  "elapsed_seconds": raw_report["elapsed_seconds"]}, ENDPOINT, stopped)
    raw_evidence, accepted_evidence, excluded_evidence = (revision[k] for k in
            ("raw_history", "accepted_continuation", "excluded_tail"))
    for record, first, last, count in ((raw_evidence, 10001, stopped, len(history)),
                                     (accepted_evidence, 10001, ENDPOINT, len(accepted)),
                                     (excluded_evidence, ENDPOINT+1, stopped, len(tail))):
        if (record.get("first_update") != first or record.get("updates") != count
                or record.get("last_update", record.get("last_completed_update")) != last):
            raise ValueError("Revision history prefix/tail count boundaries differ")
    if accepted_evidence.get("order_chain") != accepted[-1]["order_chain"]:
        raise ValueError("Revision accepted endpoint data-order chain differs")
    for actual, expected in ((sum(r["seconds"] for r in accepted), accepted_evidence.get("training_seconds")),
                             (tail_seconds, excluded_evidence.get("training_seconds"))):
        if not finite_number(expected) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("Revision accepted or excluded training timing differs")
    if sum(r["seconds"] for r in history) > raw_report["elapsed_seconds"]:
        raise ValueError("Raw job elapsed time must include accepted and excluded updates")
    return accepted, tail


def read_stopped(directory, protocol, revision, evidence):
    raw, report_file = read_input(directory / "report.json")
    packet = json.loads(raw)
    history_raw, history_file = read_input(directory / "history.jsonl")
    if report_file != evidence["report"] or history_file != evidence["history"]:
        raise ValueError("Stopped evidence changed after revision verification")
    history = [json.loads(line) for line in history_raw.splitlines() if line.strip()]
    accepted, tail = validate_stopped_history(history, packet, revision)
    if packet["contract"] != protocol["strict_contract"] or packet["source_files"] != protocol["source_files"]:
        raise ValueError("Stopped run strict training contract or sources differ")
    for relative, expected in packet["source_files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not _sha(expected):
            raise ValueError("Invalid source snapshot path/hash")
        if hash_file(directory / "source" / relative)["sha256"] != expected:
            raise ValueError(f"Stopped run frozen source changed: {relative}")
    config_raw, config_file = read_input(directory / "config.json")
    config = json.loads(config_raw)
    budget.verify_run_config(config, protocol, pilot=False)
    manifest = hash_file(local_path(config["data_dir"]) / "manifest.json")
    if manifest["sha256"] != protocol["strict_contract"]["data_manifest_sha256"]:
        raise ValueError("Stopped run data manifest differs")
    if [r["completed_updates"] for r in packet["checkpoints"]] != list(NEW_STEPS):
        raise ValueError("Stopped run retained checkpoints differ from accepted schedule")
    checkpoints = {}
    for item in packet["checkpoints"]:
        actual = verify_file(item)
        if item.get("examples_seen") != item["completed_updates"]*1024:
            raise ValueError("Retained checkpoint word exposure differs")
        checkpoints[str(item["completed_updates"])] = actual
    if checkpoints[str(ENDPOINT)] != evidence["accepted_checkpoint"]:
        raise ValueError("Accepted checkpoint identity differs from raw report")
    # Every recorded evaluation is still checked, including excluded routine tail
    # diagnostics. Only <=80k evaluations enter the selected model comparison.
    budget.verify_evaluations(packet, 10000, packet["completed_updates"], NEW_STEPS)
    return {"report": packet, "history": accepted, "raw_history": history, "tail": tail,
            "checkpoints": checkpoints, "input_files": {"report": report_file, "history": history_file,
                "run_config": config_file, "data_manifest": manifest}}


def read_pure_references(pilot_dir, run_dir, nextlat_contract, initial, raw_nextlat_history):
    pilot, continuation = read_arm(local_path(pilot_dir), "rt"), read_arm(local_path(run_dir), "rt")
    full_history = budget.verify_pure_continuation(pilot, continuation, 100000)
    if normalized_contract(pilot["report"]["contract"]) != normalized_contract(nextlat_contract):
        raise ValueError("Pure RT shared architecture/data/runtime contract differs")
    if pilot["report"]["contract"].get("optimizer", "") + "-all-hybrid" != nextlat_contract.get("optimizer"):
        raise ValueError("Pure RT reference optimizer recipe differs")
    if pilot["report"]["initialization"] != initial["backbone"]:
        raise ValueError("Pure RT canonical backbone initialization differs")
    # Verifies accepted order and the preserved excluded tail against independent
    # historical evidence, even though tail model outcomes are not compared.
    combined_nextlat = raw_nextlat_history
    if [r["order_chain"] for r in full_history[:len(combined_nextlat)]] != [r["order_chain"] for r in combined_nextlat]:
        raise ValueError("Accepted or excluded per-update data order differs from pure RT")
    sources = pilot["report"]["source_files"]
    if len(sources) != 43 or _digest_dict(sources) != "6f9d55a957bf505aefa1a351c33f0bc76291ff63736bc28b1b618393f46ef6ea":
        raise ValueError("Original pure RT source lineage differs")
    result = {"label": "Pure RT, without NextLat (descriptive matched-budget reference)",
              "curves": {}, "metrics": {}, "checkpoints": {}, "inputs": {}, "wandb": {},
              "source_files": sources, "contract": pilot["report"]["contract"],
              "historical_continuation_endpoint": 100000,
              "accepted_order_chain": full_history[ENDPOINT-1]["order_chain"]}
    for step, phase, record in ((10000, "pilot", pilot), (ENDPOINT, "continuation", continuation)):
        selected = [r for r in record["report"]["checkpoints"] if r["completed_updates"] == step]
        if len(selected) != 1:
            raise ValueError("Missing matched-budget pure RT checkpoint")
        result["checkpoints"][str(step)] = verify_file(selected[0])
        result["curves"][str(step)], result["metrics"][str(step)] = {}, {}
        result["inputs"][phase] = record["input_files"]
        result["wandb"][phase] = record["report"]["wandb"]
        for role in ROLES:
            candidates = [m for m in record["report"]["evaluations"] if (m["update"], m["role"]) == (step, role)]
            if len(candidates) != 1 or candidates[0]["rows"] != budget.ROWS:
                raise ValueError("Missing full matched-budget pure RT evaluation")
            result["metrics"][str(step)][role] = candidates[0]
            result["curves"][str(step)][role] = metric_rows(candidates[0], "rt")
    return result


def checkpoint_summary(summary):
    result = []
    for arm, curves, metrics, steps in (("rt_nextlat", summary["curves"], summary["historical_metrics"], summary["checkpoint_updates"]),
                                      ("rt", summary["pure_rt_reference"]["curves"], summary["pure_rt_reference"]["metrics"], (10000, ENDPOINT))):
        for step in steps:
            dev, ood = metrics[str(step)]["dev"], curves[str(step)]["ood_dev"]
            result.append({"arm": arm, "update": step,
                "selection": "user-selected after development review" if arm == "rt_nextlat" and step == ENDPOINT else "diagnostic/reference",
                "dev_token_accuracy": dev["token_accuracy"], "dev_whole_word_exact_match": dev["whole_word_exact_match"],
                "ood_ce": metrics[str(step)]["ood_dev"]["ce"], "A36": ood[-1]["A"], "M36": ood[-1]["M"],
                **{f"E{t}": ood[t-1]["E"] for t in budget.KEY_LENGTHS}})
    return result


def metric_table(summary):
    rows = [row for step in summary["checkpoint_updates"] for role in ROLES for row in summary["curves"][str(step)][role]]
    rows += [row for step in (10000, ENDPOINT) for role in ROLES for row in summary["pure_rt_reference"]["curves"][str(step)][role]]
    expected = (len(summary["checkpoint_updates"])+2)*48
    if len(rows) != expected or len({(r["arm"],r["update"],r["role"],r["length"]) for r in rows}) != expected:
        raise ValueError("Missing or duplicate accepted metric rows")
    if any(r["update"] > ENDPOINT for r in rows):
        raise ValueError("Excluded tail or unmatched 100k endpoint leaked into comparison")
    return rows


def make_summary(pilot_dir, run_dir, protocol_path, revision_path, pure_pilot_dir, pure_run_dir):
    directory = local_path(run_dir).resolve()
    raw, revision_file = read_input(local_path(revision_path))
    revision = json.loads(raw)
    evidence = validate_revision(revision, protocol_path, directory)
    protocol = json.loads(Path(evidence["protocol"]["path"]).read_text())
    budget.validate_protocol(protocol)
    pilot = read_training(local_path(pilot_dir), "rt_nextlat")
    budget.verify_run_config(json.loads(Path(pilot["input_files"]["run_config"]["path"]).read_text()), protocol, pilot=True)
    budget.validate_history(pilot["history"], pilot["report"], 0, 10000)
    budget.verify_evaluations(pilot["report"], 0, 10000, STEPS)
    continuation = read_stopped(directory, protocol, revision, evidence)
    history = budget.verify_join(pilot, continuation, protocol, ENDPOINT)
    pure = read_pure_references(pure_pilot_dir, pure_run_dir, protocol["strict_contract"],
                                pilot["report"]["initialization"], pilot["history"] + continuation["raw_history"])
    steps = [*STEPS, *NEW_STEPS]
    curves, metrics, checkpoints = {}, {}, {"0": pilot["checkpoints"]["0"]}
    for step in steps:
        source = pilot if step <= 10000 else continuation
        checkpoints[str(step)] = source["checkpoints"][str(step)]
        curves[str(step)], metrics[str(step)] = {}, {}
        for role in ROLES:
            selected = [m for m in source["report"]["evaluations"] if (m["update"],m["role"]) == (step,role)]
            if len(selected) != 1 or selected[0]["rows"] != budget.ROWS:
                raise ValueError("Missing full accepted checkpoint evaluation")
            metrics[str(step)][role] = selected[0]
            curves[str(step)][role] = metric_rows(selected[0], "rt_nextlat")
    report = continuation["report"]
    result = {"schema": SCHEMA, "primary_update": ENDPOINT, "reported_through_update": ENDPOINT,
        "original_planned_endpoint": 100000, "planned_endpoint_reached": False, "accepted_endpoint_completed": True,
        "selection_kind": "user_selected_after_development_review", "checkpoint_updates": steps,
        "scope": "User selected the retained 80k endpoint after reviewing development curves; original prospective plan was 100k. No further training.",
        "confirmation_evaluated": False, "latent_rollout_evaluated": False, "evaluation_route": "backbone_only",
        "contract": protocol["strict_contract"], "initialization": pilot["report"]["initialization"],
        "source_files": protocol["source_files"], "protocol": protocol, "endpoint_revision": revision,
        "revision_input": revision_file, "revision_evidence": evidence,
        "inputs": {"pilot": pilot["input_files"], "continuation": continuation["input_files"]},
        "parent_checkpoint": report["parent_checkpoint"], "order_chain": history[-1]["order_chain"],
        "historical_wandb": {"pilot": pilot["report"]["wandb"], "continuation": report["wandb"]},
        "checkpoints": checkpoints, "historical_metrics": metrics, "curves": curves, "pure_rt_reference": pure,
        "training_curve": training_curve(history, True), "training_bin_updates": 100,
        "budget": {"accepted_updates": ENDPOINT, "additional_accepted_updates": 70000, "original_planned_updates": 100000,
                   "total_word_presentations": ENDPOINT*1024, "total_token_presentations": ENDPOINT*1024*12,
                   "unique_training_words": 800000, "nominal_training_passes": ENDPOINT*1024/800000},
        "stop": {"kind": "user_requested", "raw_status": report["status"], "raw_error_type": report["error_type"],
                 "raw_completed_updates": report["completed_updates"], "excluded_completed_updates": len(continuation["tail"]),
                 "excluded_evaluation_records": sum(m["update"] > ENDPOINT for m in report["evaluations"]),
                 "excluded_one_step_diagnostic_records": sum(m["update"] > ENDPOINT for m in report["one_step_diagnostics"]),
                 "raw_report_counter_scope": "completed_updates includes tail; order_chain and train_seconds describe retained 80k checkpoint"},
        "timing": {"accepted_total_training_seconds": sum(r["seconds"] for r in history),
                   "accepted_continuation_training_seconds": sum(r["seconds"] for r in continuation["history"]),
                   "excluded_tail_training_seconds": sum(r["seconds"] for r in continuation["tail"]),
                   "raw_continuation_training_seconds": sum(r["seconds"] for r in continuation["raw_history"]),
                   "raw_continuation_job_elapsed_seconds": report["elapsed_seconds"],
                   "definition": "Accepted update times exclude the preserved unsaved tail. Raw job elapsed includes evaluation/checkpoint/logging and interruption overhead."},
        "metric_definitions": {"E": "Every state through t correct", "A": "Only state at t correct", "M": "Mean token correctness through t"},
        "intervals": "Pointwise Wilson 95% across words for E/A, not seed or paired-difference uncertainty; no M interval"}
    result["checkpoint_summary"] = checkpoint_summary(result)
    return result


def markdown_report(summary):
    stop, timing = summary["stop"], summary["timing"]
    lines = ["# Original RT + NextLat: user-selected 80k checkpoint", "",
        "The user stopped this continuation at the retained **80,000-update checkpoint after reviewing development curves**. "
        "The original prospective endpoint was 100k; 80k is a retrospective user-selected endpoint, not a prospectively fixed budget. "
        "The original protocol, raw stopped-job report and all history remain unchanged.", "",
        "This is the original Mitchell + ALiBi model: two full-prefix tiled RT blocks, rho1, D512/H8/GELU-FFN2048, "
        "LayerNorm and full-width QK normalization. Backbone 6,357,504 parameters; NextLat predictor 1,049,600; total 7,407,104. "
        "Full FP32/math attention; autocast, TF32, compilation and CUDA graphs disabled. State CE plus weight-one latent SmoothL1 "
        "is unchanged, with only the target latent detached. All accuracy evaluation uses the backbone; no latent predictor rollout.", "",
        "The original 10k checkpoint was resumed exactly with model, Adam, RNG and absolute data-order state. "
        "Accepted training is 70,000 additional updates, **81,920,000 total word presentations** over 800,000 unique length-12 "
        "words, or 102.4 nominal passes. No new training or model evaluation is performed by this reporter.", "",
        "| Model | Updates | L12 token | L12 whole word | E(13) | E(14) | E(16) | M(36) | E(36) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["checkpoint_summary"]:
        if row["arm"] == "rt_nextlat" and row["update"] not in (10000, ENDPOINT):
            continue
        label = "RT + NextLat" if row["arm"] == "rt_nextlat" else "Pure RT, without NextLat (reference)"
        values = [row[k] for k in ("dev_token_accuracy", "dev_whole_word_exact_match", "E13", "E14", "E16", "M36", "E36")]
        lines.append(f"| {label} | {row['update']:,} | " + " | ".join(f"{100*v:.4f}%" for v in values) + " |")
    lines += ["", "Pure RT references are already-existing **10k and 80k** checkpoints without the NextLat objective or predictor. "
        "They have matched update budgets, shared backbone initialization and identical minibatch order. The historical pure RT "
        "job continued to 100k, but its 100k endpoint is not substituted for 80k here. These references were neither retrained nor re-evaluated.", "",
        "E(t) requires every state through t to be correct. A(t) checks only state t, and M(t) averages correctness through t. "
        "OOD results use prefixes of the same 102,400 frozen length-36 development words. Full and boundary figures use exactly "
        "the same rows. L12 development uses a separate short-word set. The 1/60 guessing line applies only to isolated A(t).", "",
        "![Full length curves](length-full.png)", "", "![Boundary view](length-boundary.png)", "",
        "![Exactness versus updates](exactness-vs-updates.png)", "", "![Accepted training losses](training-losses.png)", "",
        "| RT + NextLat checkpoint | E(13) | E(14) | E(16) | M(36) | OOD CE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["checkpoint_summary"]:
        if row["arm"] == "rt_nextlat":
            lines.append(f"| {row['update']:,} | " + " | ".join(f"{100*row[k]:.4f}%" for k in ("E13", "E14", "E16", "M36"))
                         + f" | {row['ood_ce']:.6f} |")
    lines += ["", f"The trainer received the requested SIGINT after update {stop['raw_completed_updates']:,}. "
        f"Its **{stop['excluded_completed_updates']:,} complete uncheckpointed updates after 80k**, plus any interrupted in-flight step, "
        f"are excluded from accepted model results. The {stop['excluded_evaluation_records']} later routine evaluation records and "
        f"{stop['excluded_one_step_diagnostic_records']} later auxiliary diagnostic records remain preserved in raw evidence but are excluded from the comparison.", "",
        "The raw trainer records `failed`/`KeyboardInterrupt` and W&B records `synced_failed_experiment` because of the deliberate "
        "user interruption. This reporter requires the explicit endpoint-revision receipt binding protocol, raw report, raw history, "
        "accepted checkpoint and termination hashes. It does not reinterpret arbitrary failures as completed experiments. "
        "The raw report's completed-update counter includes the unsaved tail; its saved order hash and training-loop time still refer to 80k.", "",
        f"Accepted training-loop time: {timing['accepted_total_training_seconds']/3600:.3f}h total, including "
        f"{timing['accepted_continuation_training_seconds']/3600:.3f}h in this continuation. Excluded tail: "
        f"{timing['excluded_tail_training_seconds']:.3f}s. Raw continuation training-loop time: "
        f"{timing['raw_continuation_training_seconds']/3600:.3f}h; raw job elapsed time: "
        f"{timing['raw_continuation_job_elapsed_seconds']/3600:.3f}h including evaluation/checkpoint/logging/interruption overhead.", "",
        "All reported checkpoint evaluations use 102,400 words per development role. Intermediate checkpoints are diagnostic. "
        "Bands are pointwise Wilson 95% across words for E/A, not seed variability, simultaneous coverage or paired differences. "
        "Zero E means zero successes in this sample. This is one development seed with an endpoint chosen after inspecting development "
        "outcomes; it does not establish convergence or a confirmed generalization gain. Final confirmation and autonomous latent rollout "
        "remain **unevaluated**.", "",
        "[Exact metric counts](metrics.csv) · [Checkpoint summary](checkpoint-summary.csv) · [Accepted training bins](training-curves.csv) · "
        "[Plot data](plot-data.json) · [Provenance](report.json)", ""]
    return "\n".join(lines)


def run(args):
    output = local_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh stopped-checkpoint report directory")
    summary = make_summary(args.pilot_dir, args.run_dir, args.protocol, args.revision, args.pure_pilot_dir, args.pure_run_dir)
    rows = metric_table(summary)
    output.mkdir(parents=True)
    for phase, inputs in summary["inputs"].items():
        for record in inputs.values():
            target = output / "inputs" / phase / Path(record["path"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record["path"], target)
            if any(hash_file(target)[k] != record[k] for k in ("sha256", "bytes")):
                raise ValueError("Stopped report input changed during copy")
    for phase, inputs in summary["pure_rt_reference"]["inputs"].items():
        for key, record in inputs.items():
            if key == "endpoint_checkpoint":
                continue
            target = output / "inputs/pure-rt" / phase / Path(record["path"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record["path"], target)
            if hash_file(target)["sha256"] != record["sha256"]:
                raise ValueError("Pure RT reference evidence changed during copy")
    for record in (summary["revision_input"], summary["revision_evidence"]["protocol"], summary["revision_evidence"]["termination_exit"]):
        target = output / "inputs" / Path(record["path"]).name
        shutil.copyfile(record["path"], target)
        if hash_file(target)["sha256"] != record["sha256"]:
            raise ValueError("Revision evidence changed during copy")
    summary["reporting_sources"] = {}
    for relative in REPORTING_SOURCES:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        summary["reporting_sources"][relative] = hash_file(target)
    summary["reporting_source_sha256"] = _digest_dict({k:v["sha256"] for k,v in summary["reporting_sources"].items()})
    budget.write_csv(output / "metrics.csv", rows, CSV_COLUMNS)
    with (output / "metrics.csv").open() as stream:
        if list(csv.DictReader(stream)) != [{k:str(r[k]) for k in CSV_COLUMNS} for r in rows]:
            raise ValueError("CSV differs from exact full/zoom plot data")
    budget.write_csv(output / "checkpoint-summary.csv", summary["checkpoint_summary"], summary["checkpoint_summary"][0].keys())
    budget.write_csv(output / "training-curves.csv", summary["training_curve"], summary["training_curve"][0].keys())
    write_json(output / "summary.json", summary)
    write_json(output / "plot-data.json", summary)
    figures = budget.plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name="rt-nextlat-original-user-selected80k-report")
    result = {**summary, "status": "running", "figures": figures}
    try:
        tracker.start({key:summary[key] for key in ("schema", "scope", "contract", "budget", "selection_kind", "stop")})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(CSV_COLUMNS), data=[[r[k] for k in CSV_COLUMNS] for r in rows]),
                     **{f"report/{name}":wandb.Image(str(output / files["png"])) for name,files in figures.items()}})
        for point in summary["training_curve"]:
            values = {"update":point["update"], **{f"train/{k}":v for k,v in point.items() if k not in ("update","first_update","updates_in_bin")}}
            if str(point["update"]) in summary["curves"]:
                for length in budget.KEY_LENGTHS:
                    row = summary["curves"][str(point["update"])]["ood_dev"][length-1]
                    values.update({f"dev/ood_prefix_{length}/{key}":row[key] for key in ("E","A","M")})
            tracker.log(values)
        tracker.summary({"accepted_endpoint":ENDPOINT, "original_planned_endpoint":100000,
                         "selection_kind":summary["selection_kind"], "confirmation_evaluated":False,
                         "latent_rollout_evaluated":False, "checkpoint_summary":summary["checkpoint_summary"]})
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
        result["artifacts"] = {str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob("*"))
                               if p.is_file() and "wandb" not in p.relative_to(output).parts and p != output / "report.json"}
        write_json(output / "report.json", result)
    print(json.dumps({"status":result["status"], "output_dir":str(output), "wandb":tracker.record["run_url"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", default=str(ROOT / ".runtime/rt-a5/20260911T191702Z-nextlat/train-rt-nextlat"))
    parser.add_argument("--run-dir", default=str(LINEAGE / "train-rt-nextlat"))
    parser.add_argument("--protocol", default=str(LINEAGE / "protocol.json"))
    parser.add_argument("--revision", default=str(LINEAGE / "endpoint-revision.json"))
    parser.add_argument("--pure-pilot-dir", default=str(ROOT / ".runtime/rt-a5/20260911T154748Z/train-rt"))
    parser.add_argument("--pure-run-dir", default=str(ROOT / ".runtime/rt-a5/20260911T171239Z-rt100k/train-rt"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group", default=LINEAGE.name)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
