import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import rt_nextlat_a5_fuzzy_embedding_reordered_queue as driver


def config():
    previous = "old/input/train/checkpoints/step-000452.pt"
    arms = []
    for kind in ("value", "head", "input"):
        start = 452 if kind == "input" else 0
        arm = {"variant": kind, "runtime": f"new/{kind}", "training": f"new/{kind}/train",
               "report": f"reports/{kind}", "expected_start_update": start,
               "parent_checkpoint_sha256": "parent-sha" if start else None,
               "validation_cli": [], "retention_prefix": f"gs://fixture/{kind}/"}
        arm["training_cli"] = ["--output", arm["training"], "--updates", "15000",
                               "--batch-per-task", "2560", "--microbatch", "2560",
                               "--stop-file", "new/STOP"]
        if start:
            arm["training_cli"] += ["--resume", previous]
        arms.append(arm)
    return {"previous_queue": "old", "previous_input_training": "old/input/train",
            "previous_input_runtime": "old/input", "previous_input_checkpoint": previous,
            "previous_input_checkpoint_sha256": "parent-sha", "previous_input_update": 452,
            "baseline_training": "baseline/train", "arms": arms}


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    q = driver.Queue.__new__(driver.Queue)
    q.runtime = tmp_path / "new"
    q.runtime.mkdir()
    q.relative = Path("new")
    q.config = config()
    q.finished, q.current = [], None
    q.status = Mock()
    return q


def test_configuration_requires_order_and_exact_resume(queue):
    queue.verify_arm_configuration()
    queue.config["arms"].reverse()
    with pytest.raises(RuntimeError, match="order"):
        queue.verify_arm_configuration()


@pytest.mark.parametrize("change", ["parent_hash", "start", "resume", "batch", "stop_file"])
def test_configuration_rejects_changed_input_contract(queue, change):
    arm = queue.config["arms"][-1]
    if change == "parent_hash":
        arm["parent_checkpoint_sha256"] = "bad"
    elif change == "start":
        arm["expected_start_update"] = 453
    else:
        flag = {"resume": "--resume", "batch": "--batch-per-task", "stop_file": "--stop-file"}[change]
        arm["training_cli"][arm["training_cli"].index(flag) + 1] = "bad"
    with pytest.raises(RuntimeError):
        queue.verify_arm_configuration()


def test_wait_cancel_does_not_touch_previous(queue, monkeypatch):
    (queue.runtime / "STOP").touch()
    monkeypatch.setattr(driver, "read", Mock(side_effect=AssertionError("must not read")))
    assert queue.wait_for_previous() is False
    queue.status.assert_called_once_with("cancelled_before_variants")


@pytest.mark.parametrize("phase", ["complete", "failed", "cancelled_before_variants",
                                   "stopped_before_next_variant", "stopped_after_preflight"])
def test_wait_rejects_other_terminal_states(queue, monkeypatch, phase):
    monkeypatch.setattr(driver, "read", lambda _: {"phase": phase})
    with pytest.raises(RuntimeError, match="approved retained stop"):
        queue.wait_for_previous()


def test_wait_retained_stop_calls_identity_gate(queue, monkeypatch):
    monkeypatch.setattr(driver, "read", lambda _: {"phase": "stopped_and_retained"})
    queue.verify_previous_input = Mock()
    assert queue.wait_for_previous() is True
    queue.verify_previous_input.assert_called_once()


def parent_fixture(queue):
    report = {"status": "stopped", "start_update": 0, "completed_updates": 452,
              "checkpoints": [{"completed_updates": 452, "sha256": "parent-sha"}]}
    receipt = {"status": "verified", "uri": "gs://fixture/old",
               "members": [{"path": "runtime/train/checkpoints/step-000452.pt", "sha256": "parent-sha"}]}
    return report, receipt


@pytest.mark.parametrize("change", [None, "report_status", "receipt_status", "checkpoint_hash", "local_hash"])
def test_previous_input_retention_gate(queue, monkeypatch, change):
    report, receipt = parent_fixture(queue)
    if change == "report_status": report["status"] = "complete"
    if change == "receipt_status": receipt["status"] = "uploaded"
    if change == "checkpoint_hash": receipt["members"][0]["sha256"] = "bad"
    monkeypatch.setattr(driver, "read", Mock(side_effect=[report, receipt]))
    monkeypatch.setattr(driver, "sha", lambda _: "bad" if change == "local_hash" else "parent-sha")
    if change:
        with pytest.raises(RuntimeError): queue.verify_previous_input()
    else:
        queue.verify_previous_input()
        assert json.loads((queue.runtime / "previous-input-retained.json").read_text())["completed_updates"] == 452


def training_report(start=452):
    return {"status": "complete", "start_update": start, "completed_updates": 15000,
            "parent_checkpoint": {"path": "/workspace/cdrm-w-latent/old/input/train/checkpoints/step-000452.pt",
                                  "sha256": "parent-sha"} if start else None,
            "contract": {"batch_per_task": 2560, "microbatch": 2560, "learning_rate": 0.0001},
            "initialization": {"seed": 1234}}


@pytest.mark.parametrize("change", [None, "parent_hash", "contract", "initialization", "endpoint", "start"])
def test_completed_input_exact_contract(queue, monkeypatch, change):
    report = training_report()
    original = copy.deepcopy(report)
    if change == "parent_hash": report["parent_checkpoint"]["sha256"] = "bad"
    if change == "contract": report["contract"]["learning_rate"] = 0.001
    if change == "initialization": report["initialization"]["seed"] = 42
    if change == "endpoint": report["completed_updates"] = 14999
    if change == "start": report["start_update"] = 0
    monkeypatch.setattr(driver, "read", lambda _: original)
    if change:
        with pytest.raises(RuntimeError): queue.verify_training(queue.config["arms"][-1], report)
    else:
        assert queue.verify_training(queue.config["arms"][-1], report) == 15000


def test_fresh_arm_rejects_inherited_state(queue):
    with pytest.raises(RuntimeError):
        queue.verify_training(queue.config["arms"][0], training_report())
    report = training_report(0)
    assert queue.verify_training(queue.config["arms"][0], report) == 15000


@pytest.mark.parametrize("no_new_input_updates", [False, True])
def test_run_orders_full_boundary_report_retention_before_next_arm(queue, monkeypatch, no_new_input_updates):
    queue.frozen_sources = Mock()
    queue.wait_for_previous = Mock(return_value=True)
    queue.wait_for_baseline = Mock(return_value=True)
    monkeypatch.setattr(driver.shutil, "copy2", lambda *a: None)
    monkeypatch.setattr(driver, "sha", lambda _: "endpoint-sha")
    calls = []
    def read(path):
        path = str(path)
        if path.endswith("ready.json"): return {"passed": True}
        if path.endswith("validation/report.json"): return {"status": "passed"}
        if "reports/" in path: return {"status": "complete"}
        endpoint = 452 if no_new_input_updates and "/input/" in path else 15000
        if "retention/" in path:
            return {"status": "verified", "uri": "gs://fixture/new",
                    "members": [{"path": f"runtime/train/checkpoints/step-{endpoint:06d}.pt", "sha256": "endpoint-sha"}]}
        report = training_report(452 if "/input/" in path else 0)
        if endpoint == 452:
            report.update(status="stopped", completed_updates=452)
        report.update(checkpoints=[{"completed_updates": endpoint, "sha256": "endpoint-sha"}], wandb={})
        return report
    monkeypatch.setattr(driver, "read", read)
    queue.invoke = lambda label, arguments, **kwargs: calls.append((label, arguments, kwargs))
    queue.run()
    expected = ["value-preflight", "value-train", "value-compare", "value-retain",
                "head-preflight", "head-train", "head-compare", "head-retain",
                "input-train", "input-resume-boundary"]
    expected += ["input-retain"] if no_new_input_updates else ["input-compare", "input-retain"]
    assert [c[0] for c in calls] == expected
    assert calls[9][2] == {}
    assert calls[9][1][calls[9][1].index("--expected-update") + 1] == "452"
    if not no_new_input_updates:
        assert calls[-2][1].count("--run") == 4
    assert [x["variant"] for x in queue.finished] == ["value", "head", "input"]
    assert queue.status.call_args.args[0] == ("stopped_and_retained" if no_new_input_updates else "complete")
