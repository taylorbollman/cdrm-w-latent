"""CPU-only continuation lineage, objective, snapshot, and plot consistency guards."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import rt_a5_nextlat_budget_report as report


def history(start, stop):
    return [{"update": i, "examples_seen": i*1024, "order_chain": f"{i:064x}",
             "seconds": .01, "loss": 1., "state_loss": .8, "latent_loss": .2,
             "weighted_latent_loss": .2, "token_accuracy": .5, "whole_word_exact": .25,
             "grad_norm": .5} for i in range(start+1, stop+1)]


def metric(step, role, n=102400):
    length = 12 if role == "dev" else 36
    # Last-prefix zero does not imply isolated accuracy or mean token accuracy zero.
    exact = ([n]*12 + [n//2, n//4, 1] + [0]*21)[:length]
    state = ([n]*12 + [n//2, n//3, 100] + [n//60]*21)[:length]
    return {"update": step, "role": role, "rows": n, "length": length, "tokens": n*length,
            "evaluated_rows": n, "route": "backbone_only", "ce": 1., "per_position_ce": [1.]*length,
            "isolated_state_accuracy": [x/n for x in state], "cumulative_prefix_exactness": [x/n for x in exact],
            "token_accuracy": sum(state)/(n*length), "whole_word_exact_match": exact[-1]/n,
            "final_state_accuracy": state[-1]/n}


def diagnostic(step, role):
    return {"update": step, "role": role, "length": 12 if role == "dev" else 36,
            "rows": 1024, "route": "teacher_conditioned_one_step_diagnostics",
            "loss": 1., "state_loss": .8, "latent_loss": .2, "weighted_latent_loss": .2,
            "diagnostics": {key: .1 for key in report.DIAGNOSTICS}}


def fixture(root, stop=20000, status="running"):
    directory = root / "continuation"
    (directory / "source/scripts").mkdir(parents=True)
    (directory / "source/scripts/fixture.py").write_text("# executed fixture\n")
    sources = {"scripts/fixture.py": report.hash_file(directory / "source/scripts/fixture.py")["sha256"]}
    data = root / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{}\n')
    parent = root / "parent.pt"
    parent.write_bytes(b"saved parent state")
    contract = {"schema": report.TRAIN_SCHEMA, "source_sha256": report._digest_dict(sources),
                "data_manifest_sha256": report.hash_file(data / "manifest.json")["sha256"]}
    protocol = {"strict_contract": contract, "source_files": sources, "data_dir": str(data),
                "parent_checkpoint": str(parent), "parent_checkpoint_sha256": report.hash_file(parent)["sha256"],
                "parent_report_sha256": "c"*64}
    config = {"architecture": "rt", "width": 512, "seed": 1234, "predictor_seed": 1235,
              "predictor_hidden_width": None, "latent_weight": 1., "data_order_seed": 1234,
              "batch_size": 1024, "eval_every": 500, "eval_rows": 4096, "full_eval_rows": 102400,
              "diagnostic_rows": 1024, "log_every": 25, "wandb_project": "rt-a5-state-tracking",
              "updates": 100000, "checkpoint_steps": list(report.NEW_STEPS), "resume": str(parent),
              "data_dir": str(data)}
    rows = history(10000, stop)
    raw_history = b"".join((json.dumps(row)+'\n').encode() for row in rows)
    (directory / "history.jsonl").write_bytes(raw_history)
    checkpoints = []
    for step in report.NEW_STEPS:
        if step <= stop:
            checkpoint = directory / f"step-{step}.pt"
            checkpoint.write_bytes(f"saved checkpoint{step}".encode())
            checkpoints.append({**report.hash_file(checkpoint), "completed_updates": step, "examples_seen": step*1024})
    packet = {"schema": report.TRAIN_SCHEMA, "status": status, "start_update": 10000,
              "completed_updates": stop, "endpoint": 100000, "confirmation_evaluated": False,
              "latent_rollout_evaluated": False, "contract": contract, "source_files": sources,
              "checkpoints": checkpoints, "order_chain": rows[-1]["order_chain"],
              "initialization": {"canonical": "mitchell"},
              "parent_checkpoint": {"path": str(parent), "sha256": protocol["parent_checkpoint_sha256"]},
              "train_seconds": sum(row["seconds"] for row in rows), "elapsed_seconds": stop*.02,
              "wandb": {"status": "synced" if status == "complete" else "running"},
              "evaluations": [metric(step, role, 102400 if step in report.NEW_STEPS else 4096)
                              for step in range(10500, stop+1, 500) for role in report.ROLES],
              "one_step_diagnostics": [diagnostic(step, role) for step in range(10500, stop+1, 500) for role in report.ROLES]}
    report.write_json(directory / "report.json", packet)
    report.write_json(directory / "config.json", config)
    return directory, protocol, packet, raw_history


def test_interim_snapshot_excludes_live_tail_and_never_claims_100k_completion(tmp_path):
    directory, protocol, _, raw = fixture(tmp_path)
    with (directory / "history.jsonl").open("ab") as stream:
        stream.write(b'{"update":20001,"incomplete_live_line":')
    result = report.read_continuation(directory, protocol, 20000)
    assert result["history"][-1]["update"] == 20000
    assert result["input_bytes"]["history"] == raw
    assert "prefix" in result["input_files"]["history"]["snapshot"]
    with pytest.raises(ValueError, match="completed RT"):
        report.read_continuation(directory, protocol, 100000)


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p.update(schema="rt-a5-training-v1"), "completed RT"),
    (lambda p: p.update(confirmation_evaluated=True), "unevaluated"),
    (lambda p: p.update(latent_rollout_evaluated=True), "unevaluated"),
    (lambda p: p.update(completed_updates=19500), "not yet completed"),
    (lambda p: p["evaluations"].pop(), "development evaluation"),
    (lambda p: p["evaluations"][-1].update(rows=4096), "row count"),
    (lambda p: p["one_step_diagnostics"][-1]["diagnostics"].update(latent_rms=float("inf")), "Nonfinite"),
    (lambda p: p.update(train_seconds=1.), "Training-loop time"),
])
def test_incomplete_or_wrong_scope_saved_evidence_rejected(tmp_path, mutate, match):
    directory, protocol, packet, _ = fixture(tmp_path)
    mutate(packet)
    # Deliberately permit malformed JSON numeric values to exercise the reader guard.
    (directory / "report.json").write_text(json.dumps(packet))
    with pytest.raises(ValueError, match=match):
        report.read_continuation(directory, protocol, 20000)


def test_missing_history_corrupt_checkpoint_and_changed_frozen_source_rejected(tmp_path):
    directory, protocol, packet, raw = fixture(tmp_path)
    (directory / "history.jsonl").write_bytes(raw.split(b'\n', 1)[1])
    with pytest.raises(ValueError, match="exactly once"):
        report.read_continuation(directory, protocol, 20000)
    (directory / "history.jsonl").write_bytes(raw)
    checkpoint = Path(packet["checkpoints"][0]["path"])
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checkpoint identity"):
        report.read_continuation(directory, protocol, 20000)
    checkpoint.write_bytes(original)
    (directory / "source/scripts/fixture.py").write_text("changed source")
    with pytest.raises(ValueError, match="source snapshot changed"):
        report.read_continuation(directory, protocol, 20000)


def test_resume_join_rejects_reset_recipe_parent_initialization_or_overlap(tmp_path):
    directory, protocol, packet, _ = fixture(tmp_path)
    continuation = report.read_continuation(directory, protocol, 20000)
    pilot = {"report": {**packet, "start_update": 0, "completed_updates": 10000},
             "history": history(0, 10000), "checkpoints": {"10000": report.hash_file(protocol["parent_checkpoint"])},
             "input_files": {"report": {"sha256": protocol["parent_report_sha256"]}}}
    assert len(report.verify_join(pilot, continuation, protocol, 20000)) == 20000
    for mutation, match in (
        (lambda c: c["report"].update(start_update=0), "exact"),
        (lambda c: c["report"]["contract"].update(optimizer="reset"), "contracts differ"),
        (lambda c: c["report"]["initialization"].update(canonical="identity"), "initialization lineage"),
        (lambda c: c["report"]["parent_checkpoint"].update(sha256="wrong"), "parent checkpoint"),
        (lambda c: c["history"].insert(0, history(9999, 10000)[0]), "exactly once"),
    ):
        changed = copy.deepcopy(continuation)
        mutation(changed)
        with pytest.raises(ValueError, match=match):
            report.verify_join(pilot, changed, protocol, 20000)


def test_joint_loss_and_exposure_guards():
    rows = history(10000, 10003)
    packet = {"order_chain": rows[-1]["order_chain"], "train_seconds": .03, "elapsed_seconds": .1}
    report.validate_history(rows, packet, 10000, 10003)
    for key, value, match in (("loss", 1.2, "Combined loss"), ("latent_loss", float("nan"), "Nonfinite"),
                              ("examples_seen", 123, "exposure"), ("token_accuracy", 2, "outside")):
        changed = copy.deepcopy(rows)
        changed[1][key] = value
        with pytest.raises(ValueError, match=match):
            report.validate_history(changed, packet, 10000, 10003)


def test_full_zoom_csv_share_identical_endpoint_counts_and_fixed100k(tmp_path):
    steps = [*report.STEPS, *report.NEW_STEPS]
    metrics = {str(step): {role: metric(step, role) for role in report.ROLES} for step in steps}
    curves = {str(step): {role: report.metric_rows(metrics[str(step)][role], "rt_nextlat")
                        for role in report.ROLES} for step in steps}
    pure = {str(step): {role: report.metric_rows(metric(step, role), "rt") for role in report.ROLES}
            for step in (10000, 100000)}
    summary = {"checkpoint_updates": steps, "curves": curves, "historical_metrics": metrics,
               "reported_through_update": 100000, "endpoint_completed": True, "scope": "Fixed100k test",
               "pure_rt_reference": {"curves": pure, "metrics": {str(step): metrics[str(step)] for step in (10000, 100000)}},
               "training_curve": report.training_curve(history(0, 100000), True),
               "timing": {"training_seconds": 1000, "continuation_training_seconds": 900}}
    rows = report.metric_table(summary)
    assert len(rows) == 720
    for step in (10000, 100000):
        plotted = report.endpoint_plot_rows(summary, step)
        assert plotted is curves[str(step)]["ood_dev"]
        selected = [r for r in rows if (r["arm"], r["update"], r["role"]) == ("rt_nextlat", step, "ood_dev")]
        assert plotted == selected
        assert plotted[9:18] == [r for r in selected if 10 <= r["length"] <= 18]
        assert plotted[14]["prefix_exact_count"] == 1
        assert plotted[-1]["E"] == 0 and plotted[-1]["A"] > 0 and plotted[-1]["M"] > plotted[-1]["A"]
    summary["checkpoint_summary"] = report.checkpoint_summary(summary)
    text = report.markdown_report(summary)
    assert "Fixed primary endpoint: **100,000**" in text
    assert "without NextLat" in text and "unevaluated" in text
    figures = report.plot_results(summary, tmp_path)
    for paths in figures.values():
        assert (tmp_path / paths["png"]).read_bytes().startswith(b'\x89PNG')
        assert (tmp_path / paths["pdf"]).read_bytes().startswith(b'%PDF')
    assert sum(p["updates_in_bin"] for p in summary["training_curve"]) == 100000
    summary["curves"]["100000"]["ood_dev"].pop()
    with pytest.raises(ValueError, match="missing"):
        report.metric_table(summary)


def test_bad_scalar_and_noninteger_accuracy_rejected():
    value = metric(100000, "ood_dev")
    value["token_accuracy"] = .1
    with pytest.raises(ValueError, match="disagrees"):
        report.metric_rows(value, "rt_nextlat")
    value = metric(100000, "ood_dev")
    value["isolated_state_accuracy"][20] = .123456789
    with pytest.raises(ValueError, match="integer"):
        report.metric_rows(value, "rt_nextlat")
