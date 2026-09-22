"""Backend scope and before-CUDA determinism configuration, without GPU execution."""
from contextlib import contextmanager
import os
from types import SimpleNamespace

import pytest

from scripts import olmo_f2_graph_backend_probe as probe


@pytest.mark.parametrize("backend", ["flash", "cudnn", "math"])
def test_selected_backend_encompasses_entire_original_probe(monkeypatch, backend):
    events = []
    @contextmanager
    def context(selected):
        events.append(("enter", selected))
        yield
        events.append(("exit", selected))
    args = SimpleNamespace(backend=backend)
    model, report, tracker = object(), {}, object()
    def original(actual_model, actual_args, actual_report, actual_tracker):
        assert actual_model is model and actual_args is args and actual_report is report and actual_tracker is tracker
        assert [event[0] for event in events] == ["enter"]
        events.append(("full_original_probe", None))
    monkeypatch.setattr(probe, "sdpa_kernel", context)
    monkeypatch.setattr(probe.base, "run_probe", original)
    probe.run_backend_probe(model, args, report, tracker)
    assert [event[0] for event in events] == ["enter", "full_original_probe", "exit"]
    assert events[0][1] == {"flash": probe.SDPBackend.FLASH_ATTENTION,
        "cudnn": probe.SDPBackend.CUDNN_ATTENTION, "math": probe.SDPBackend.MATH}[backend]


def test_auto_does_not_force_sdpa_backend(monkeypatch):
    called = []
    monkeypatch.setattr(probe, "sdpa_kernel", lambda _: pytest.fail("Auto cannot force a backend"))
    monkeypatch.setattr(probe.base, "run_probe", lambda *args: called.append(args))
    probe.run_backend_probe(None, SimpleNamespace(backend="auto"), {}, None)
    assert len(called) == 1


def mock_determinism(monkeypatch, *, initialized=False):
    state = {"enabled": False, "calls": []}
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(probe.torch.backends, "cudnn", SimpleNamespace(deterministic=False, benchmark=True))
    def set_algorithms(enabled):
        state["enabled"] = enabled
        state["calls"].append((enabled, os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
    monkeypatch.setattr(probe.torch, "use_deterministic_algorithms", set_algorithms)
    monkeypatch.setattr(probe.torch, "are_deterministic_algorithms_enabled", lambda: state["enabled"])
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    return state


def test_deterministic_workspace_is_set_before_algorithms_and_recorded(monkeypatch):
    state = mock_determinism(monkeypatch)
    record = probe.configure_determinism(True)
    assert state["calls"] == [(True, ":4096:8")]
    assert record == {"deterministic_algorithms": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8"}


def test_late_deterministic_configuration_is_rejected_before_mutating_settings(monkeypatch):
    state = mock_determinism(monkeypatch, initialized=True)
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        probe.configure_determinism(True)
    assert state["calls"] == [] and "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_default_records_nondeterministic_mode_without_inventing_workspace(monkeypatch):
    state = mock_determinism(monkeypatch)
    record = probe.configure_determinism(False)
    assert state["calls"] == [(False, None)]
    assert record["deterministic_algorithms"] is False and record["cublas_workspace_config"] is None
