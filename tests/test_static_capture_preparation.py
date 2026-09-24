"""CPU orchestration tests; fake CUDA contexts never validate GPU execution."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from cdrm.pretrained import static_training
from cdrm.pretrained.static_training import StaticFBTTraining


@pytest.fixture
def preparation(monkeypatch):
    events = []
    state = {"capturing": False, "backwards": 0, "fail_backward": None}

    class Stream:
        def wait_stream(self, other):
            events.append("wait_stream")

    stream = Stream()

    @contextmanager
    def stream_context(value):
        events.append("stream_enter")
        try:
            yield
        finally:
            events.append("stream_exit")

    @contextmanager
    def graph_context(graph, *, stream):
        events.append("graph_enter")
        state["capturing"] = True
        try:
            yield
        finally:
            state["capturing"] = False
            events.append("graph_exit")

    def synchronize():
        assert not state["capturing"]
        events.append("synchronize")

    def empty_cache():
        assert not state["capturing"]
        events.append("empty_cache")

    monkeypatch.setattr(static_training.torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(static_training.torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(static_training.torch.cuda, "stream", stream_context)
    monkeypatch.setattr(static_training.torch.cuda, "graph", graph_context)
    monkeypatch.setattr(static_training.torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(static_training.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(static_training.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(static_training.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(static_training.torch.cuda, "reset_peak_memory_stats",
                        lambda *a, **k: pytest.fail("Runtime must not erase harness peaks"))
    plan = object.__new__(StaticFBTTraining)
    plan.batch = SimpleNamespace(input_ids=SimpleNamespace(device=SimpleNamespace(type="cuda")))
    plan.graph = plan.graph_result = None
    plan.warmup_backward_calls = plan.capture_backward_calls = 0
    plan.gradient_addresses = {"weight": 12345}

    def initialize():
        events.append("initialize")
        # Simulate one completed first backward allocating persistent buffers.
        plan.warmup_backward_calls += 1

    def backward():
        state["backwards"] += 1
        events.append("backward")
        if state["backwards"] == state["fail_backward"]:
            raise RuntimeError("deliberate preparation failure")
        return {"completed_backward": state["backwards"]}

    def validate():
        events.append("validate")
        assert plan.gradient_addresses == {"weight": 12345}

    def observe(phase, event):
        assert not state["capturing"], "Observer ran inside graph capture"
        events.append((phase, event))

    plan.initialize_gradients = initialize
    plan._tensor_backward = backward
    plan.validate_execution = validate
    return plan, observe, events, state


def test_default_capture_keeps_cleanup_and_observation_disabled(preparation):
    plan, _, events, _ = preparation
    plan.capture(warmup=2)
    assert "gc" not in events and "empty_cache" not in events
    assert not any(isinstance(event, tuple) for event in events)
    assert plan.warmup_backward_calls == 3 and plan.capture_backward_calls == 1
    assert plan.graph_result == {"completed_backward": 3}
    assert events.count("backward") == 3


def test_phase_boundaries_surround_sync_cleanup_and_capture_without_tensor_changes(preparation):
    plan, observe, events, _ = preparation
    addresses = plan.gradient_addresses
    plan.capture(warmup=2, release_transient_cache=True, phase_observer=observe)
    assert [event for event in events if isinstance(event, tuple)] == [
        (phase, event)
        for phase in ("gradient_initialization", "warmup", "transient_cleanup", "capture")
        for event in ("begin", "end")]
    assert events.index("stream_exit") < events.index("synchronize") < events.index("gc")
    assert events.index("gc") < events.index("empty_cache") < events.index(("capture", "begin"))
    assert events.index("graph_exit") < events.index(("capture", "end"))
    assert plan.gradient_addresses is addresses
    assert plan.warmup_backward_calls == 3 and plan.capture_backward_calls == 1


def test_warmup_failure_reports_completed_work_without_cleanup_or_graph(preparation):
    plan, observe, events, state = preparation
    state["fail_backward"] = 2
    with pytest.raises(RuntimeError, match="deliberate preparation"):
        plan.capture(warmup=3, release_transient_cache=True, phase_observer=observe)
    assert plan.warmup_backward_calls == 2  # Initialization plus first warmup.
    assert plan.capture_backward_calls == 0 and plan.graph is None
    assert events[-1] == ("warmup", "error")
    assert "stream_exit" in events and "graph_enter" not in events and "gc" not in events


def test_capture_failure_observation_occurs_after_graph_context_exit(preparation):
    plan, observe, events, state = preparation
    state["fail_backward"] = 3
    with pytest.raises(RuntimeError, match="deliberate preparation"):
        plan.capture(warmup=2, phase_observer=observe)
    assert events[-2:] == ["graph_exit", ("capture", "error")]
    assert plan.graph is None and plan.graph_result is None and plan.capture_backward_calls == 0
    assert plan.warmup_backward_calls == 3


def test_secondary_observer_failure_does_not_replace_original_failure(preparation):
    plan, observe, _, state = preparation
    state["fail_backward"] = 2

    def broken_observer(phase, event):
        observe(phase, event)
        if event == "error":
            raise ValueError("memory snapshot also failed")

    with pytest.raises(RuntimeError, match="deliberate preparation") as error:
        plan.capture(warmup=1, phase_observer=broken_observer)
    assert any("memory snapshot also failed" in note for note in error.value.__notes__)


@pytest.mark.parametrize("kwargs", [
    {"release_transient_cache": "yes"}, {"release_transient_cache": 1},
    {"phase_observer": []}, {"phase_observer": False},
])
def test_invalid_observation_options_reject_before_any_cuda_work(kwargs, preparation):
    plan, _, events, _ = preparation
    with pytest.raises(TypeError):
        plan.capture(**kwargs)
    assert not events


def test_cleanup_failure_stops_before_graph_and_preserves_error(preparation, monkeypatch):
    plan, observe, events, _ = preparation

    def failed_cleanup():
        raise RuntimeError("cleanup unavailable")

    monkeypatch.setattr(static_training.torch.cuda, "empty_cache", failed_cleanup)
    with pytest.raises(RuntimeError, match="cleanup unavailable"):
        plan.capture(warmup=1, release_transient_cache=True, phase_observer=observe)
    assert events[-1] == ("transient_cleanup", "error")
    assert plan.graph is None and "graph_enter" not in events
