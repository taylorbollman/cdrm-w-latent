"""Tracking must not alter RNG state or silently discard online failures."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


_path = Path(__file__).resolve().parents[1] / "scripts/experiment_tracking.py"
_spec = importlib.util.spec_from_file_location("experiment_tracking", _path)
tracking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tracking)


@contextmanager
def preserve_random():
    before = random.getstate()
    try:
        yield
    finally:
        random.setstate(before)


class FakeRun:
    def __init__(self):
        self.settings = SimpleNamespace(mode="online")
        self.url = "https://wandb.ai/taylorbollman/project/runs/test"
        self.id = "test"
        self.summary = {}
        self.rows = []
        self.exit_code = None

    def define_metric(self, *args, **kwargs):
        random.random()

    def get_project_url(self):
        return "https://wandb.ai/taylorbollman/project"

    def log(self, row, step=None):
        random.random()
        self.rows.append((row, step))

    def finish(self, exit_code):
        random.random()
        self.exit_code = exit_code


def test_tracking_preserves_rng_and_logs_separate_resume_update(tmp_path):
    run = FakeRun()
    calls = []

    def initialize(**kwargs):
        random.random()
        calls.append(kwargs)
        return run

    sdk = SimpleNamespace(run=None, init=initialize, Settings=lambda **kwargs: kwargs)
    tracker = tracking.OnlineTracker(project="project", output_dir=tmp_path,
                                     preserve_state=preserve_random)
    before = random.getstate()
    with patch.dict(sys.modules, {"wandb": sdk}):
        tracker.start({"profile": "bf16"})
        tracker.log({"update": 50, "dev/answer_ce": 0.5})
        tracker.log({"update": 51, "train/loss": 0.4})
        tracker.summary({"recovery_exact": True})
        tracker.finish(succeeded=True)
    assert random.getstate() == before
    assert calls[0]["mode"] == "online"
    assert calls[0]["entity"] == "taylorbollman"
    assert [row[0]["update"] for row in run.rows] == [50, 51]
    assert run.exit_code == 0
    assert tracker.record["status"] == "synced"
    assert tracker.record["run_url"] == run.url
    assert run.summary["recovery_exact"] is True


@pytest.mark.parametrize("operation", ["log", "finish"])
def test_online_failures_are_fatal_and_do_not_expose_sdk_error(tmp_path, operation):
    run = FakeRun()
    tracker = tracking.OnlineTracker(project="project", output_dir=tmp_path,
                                     preserve_state=preserve_random)
    tracker._run = run
    tracker.record["status"] = "running"

    def fail(*args, **kwargs):
        random.random()
        raise ConnectionError("Sensitive mock authentication detail")

    setattr(run, operation, fail)
    before = random.getstate()
    with pytest.raises(RuntimeError, match="Required online W&B") as error:
        if operation == "log":
            tracker.log({"train/loss": 0.4})
        else:
            tracker.finish(succeeded=True)
    assert "Sensitive" not in str(error.value)
    assert random.getstate() == before
    assert tracker.record["status"] == "failed"


def test_flatten_only_numeric_metrics_preserves_diagnostic_names():
    report = {"loss": 0.2, "precision": {"finite": True, "dtype": "FP32"},
              "history": [{"loss": 0.3}], "profile": "bf16"}
    assert tracking.scalar_metrics(report, "train") == {
        "train/loss": 0.2, "train/precision/finite": True}


def test_requested_online_run_rejects_offline_fallback(tmp_path):
    run = FakeRun()
    run.settings.mode = "offline"
    sdk = SimpleNamespace(run=None, init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
    tracker = tracking.OnlineTracker(project="project", output_dir=tmp_path)
    with patch.dict(sys.modules, {"wandb": sdk}):
        with pytest.raises(RuntimeError, match="initialization failed"):
            tracker.start({})
    tracker.finish(succeeded=False)
    assert tracker.record["status"] == "failed"
    assert run.exit_code == 1
