"""Checkpoint overlap, exclusive storage access, and safe-boundary failures."""
from concurrent.futures import ThreadPoolExecutor
import signal
import threading

import pytest

from scripts.olmo_campaign_loop import LifecycleError, LoopPolicy, StopRequest
from scripts.olmo_pilot_async_loop import run_loop


class LocalCoordinator:
    rank, world_size = 0, 1

    def __init__(self):
        self.calls = []

    def gather(self, value):
        return [value]

    def same(self, phase, value):
        return value

    def call(self, phase, function, *, rank_zero=False):
        self.calls.append((phase, rank_zero))
        try:
            return function()
        except Exception as exc:
            raise LifecycleError(f"{phase}: {exc}") from exc


def scenario(*, policy=None, start=0, restored=False, ready_after=None,
             stop=None, fail=None):
    events, state = [], {"update": start, "clock": 0.0, "pending": None,
                         "accepted": [], "maximum_pending": 0}
    coordinator = LocalCoordinator()

    def update():
        events.append(("update", state["update"] + 1))
        if fail == "update":
            raise RuntimeError("unknown update failure")
        state["update"] += 1
        state["clock"] += 11
        return {"update": state["update"]}

    def log(metrics):
        events.append(("log", metrics["update"]))
        if fail == "log":
            raise OSError("log failure")

    def save(number, reason):
        # Saving and worker storage access must be mutually exclusive.
        assert state["pending"] is None
        events.append(("save", number, reason))
        if fail == "save":
            raise RuntimeError("unknown distributed save failure")
        return {"update": number}

    def submit(receipt):
        assert state["pending"] is None
        events.append(("submit", receipt["update"]))
        if fail == "submit":
            raise OSError("submission failure")
        state["pending"] = dict(receipt)
        state["maximum_pending"] = max(state["maximum_pending"], 1)
        return {"worker": "cpu", "update": receipt["update"]}

    def poll(*, wait=False):
        assert state["pending"] is not None
        events.append(("poll", wait, state["update"]))
        if fail == "worker" and state["update"] >= start + 1:
            raise OSError("worker retention failure")
        if not wait and (ready_after is None
                         or state["update"] < state["pending"]["update"] + ready_after):
            return None
        if wait:
            state["clock"] += 3
        result = state["pending"]
        state["pending"] = None
        return {"receipt": result, "verified": True}

    def accept(record):
        number = record["receipt"]["update"]
        events.append(("accept", number))
        if fail == "accept":
            raise OSError("main publication failure")
        state["accepted"].append(number)
        return number

    kwargs = dict(coordinator=coordinator, policy=policy or LoopPolicy(3),
                  completed=lambda: state["update"], update=update, log=log,
                  save=save, submit_checkpoint=submit, poll_checkpoint=poll,
                  accept_checkpoint=accept, stop=stop or StopRequest(),
                  clock=lambda: state["clock"], restored=restored)
    return kwargs, events, state, coordinator


def test_updates_overlap_pending_retention_without_storage_overlap():
    kwargs, events, state, coordinator = scenario()
    result = run_loop(**kwargs)
    assert events.index(("update", 3)) < events.index(("accept", 0))
    assert events.index(("accept", 0)) < events.index(("save", 3, "update_limit"))
    assert state["maximum_pending"] == 1
    assert state["accepted"] == [0, 3]
    assert result == {"start_update": 0, "completed_update": 3,
        "stop_reason": "update_limit", "checkpoints_submitted": 2,
        "checkpoints_published": 2, "last_saved_update": 3,
        "last_retained_update": 3, "checkpoint_wait_seconds": 6.0,
        "checkpoint_pending": False}
    assert all(rank_zero for _, rank_zero in coordinator.calls)


def test_ready_worker_is_accepted_before_the_next_update():
    kwargs, events, _, _ = scenario(ready_after=1)
    result = run_loop(**kwargs)
    assert events.index(("update", 1)) < events.index(("accept", 0))
    assert events.index(("accept", 0)) < events.index(("update", 2))
    assert result["checkpoint_wait_seconds"] == 3.0  # terminal worker only


def test_backlog_drains_before_next_save_and_never_queues_a_second_worker():
    kwargs, events, state, _ = scenario(policy=LoopPolicy(3, checkpoint_updates=(1, 2)))
    result = run_loop(**kwargs)
    assert [number for kind, number, *_ in events if kind == "accept"] == [0, 1, 2, 3]
    for old, new in ((0, 1), (1, 2), (2, 3)):
        reason = "update_limit" if new == 3 else "cadence"
        assert events.index(("accept", old)) < events.index(("save", new, reason))
    assert state["maximum_pending"] == 1
    assert result["checkpoints_submitted"] == result["checkpoints_published"] == 4


def test_terminal_at_initial_boundary_drains_the_existing_save_once():
    kwargs, events, state, _ = scenario(policy=LoopPolicy(0))
    result = run_loop(**kwargs)
    assert state["accepted"] == [0]
    assert result["completed_update"] == result["last_retained_update"] == 0
    assert not any(event[0] == "update" for event in events)
    assert sum(event[0] == "save" for event in events) == 1
    assert events[-1] == ("accept", 0)


def test_restored_terminal_boundary_has_no_worker_or_duplicate_save():
    kwargs, events, _, _ = scenario(start=4, restored=True, policy=LoopPolicy(4))
    result = run_loop(**kwargs)
    assert events == []
    assert result["last_saved_update"] == result["last_retained_update"] == 4
    assert result["checkpoints_submitted"] == result["checkpoints_published"] == 0
    assert result["checkpoint_wait_seconds"] == 0.0
    assert not result["checkpoint_pending"]


def test_restored_continuation_retains_only_new_final_boundary():
    kwargs, events, state, _ = scenario(start=4, restored=True, policy=LoopPolicy(6))
    result = run_loop(**kwargs)
    assert state["accepted"] == [6]
    assert [event for event in events if event[0] == "save"] == [("save", 6, "update_limit")]
    assert result["start_update"] == 4 and result["completed_update"] == 6


def test_worker_failure_prevents_another_update_and_false_publication():
    kwargs, events, state, _ = scenario(fail="worker")
    with pytest.raises(LifecycleError, match="worker retention failure"):
        run_loop(**kwargs)
    assert state["update"] == 1
    assert state["accepted"] == []
    assert not any(event[0] == "accept" for event in events)
    assert [event for event in events if event[0] == "save"] == [("save", 0, "initial")]


def test_signal_stop_preserves_update_then_drains_before_return():
    stop = StopRequest()
    kwargs, events, state, _ = scenario(stop=stop)
    ordinary_log = kwargs["log"]

    def log(metrics):
        ordinary_log(metrics)
        stop.request(signal.SIGTERM)

    kwargs["log"] = log
    result = run_loop(**kwargs)
    assert state["update"] == 1 and state["accepted"] == [0, 1]
    assert result["stop_reason"] == f"signal:{signal.SIGTERM}"
    assert events[-1] == ("accept", 1)


def test_existing_stop_file_drains_without_update(tmp_path):
    path = tmp_path / "STOP"
    path.touch()
    kwargs, events, state, _ = scenario(stop=StopRequest(path))
    result = run_loop(**kwargs)
    assert result["stop_reason"] == "stop_file"
    assert state["update"] == 0 and state["accepted"] == [0]
    assert not any(event[0] == "update" for event in events)


def test_logging_failure_drains_previous_worker_and_preserves_current_boundary():
    kwargs, events, state, _ = scenario(fail="log")
    with pytest.raises(LifecycleError, match="log failure"):
        run_loop(**kwargs)
    assert state["update"] == 1 and state["accepted"] == [0, 1]
    assert events.index(("accept", 0)) < events.index(("save", 1, "logging_failure"))
    assert events[-1] == ("accept", 1)


def test_unknown_update_failure_does_not_start_emergency_collectives_or_save():
    kwargs, events, state, coordinator = scenario(fail="update")
    with pytest.raises(RuntimeError, match="unknown update failure"):
        run_loop(**kwargs)
    assert events[-1] == ("update", 1)
    assert coordinator.calls[-1][0] == "boundary control"
    assert state["accepted"] == []
    assert state["pending"] == {"update": 0}  # caller must clean up the worker


def test_submission_failure_never_runs_an_update():
    kwargs, events, state, _ = scenario(fail="submit")
    with pytest.raises(LifecycleError, match="submission failure"):
        run_loop(**kwargs)
    assert state["update"] == 0 and state["accepted"] == []
    assert events == [("save", 0, "initial"), ("submit", 0)]


def test_publication_failure_prevents_next_update():
    kwargs, events, state, _ = scenario(fail="accept", ready_after=1)
    with pytest.raises(LifecycleError, match="main publication failure"):
        run_loop(**kwargs)
    assert state["update"] == 1 and state["accepted"] == []
    assert events[-1] == ("accept", 0)


def test_unknown_save_failure_is_not_wrapped_in_extra_collectives():
    kwargs, events, _, coordinator = scenario(fail="save")
    with pytest.raises(RuntimeError, match="unknown distributed save failure"):
        run_loop(**kwargs)
    assert events == [("save", 0, "initial")]
    assert coordinator.calls[-1][0] == "initial checkpoint clock"


def test_cadence_uses_local_submission_clock_without_waiting_for_remote():
    kwargs, events, state, _ = scenario(policy=LoopPolicy(2, checkpoint_seconds=10))
    result = run_loop(**kwargs)
    assert [event for event in events if event[0] == "save"] == [
        ("save", 0, "initial"), ("save", 1, "cadence"), ("save", 2, "update_limit")]
    assert events.index(("update", 1)) < events.index(("accept", 0))
    assert state["accepted"] == [0, 1, 2]
    assert result["checkpoint_wait_seconds"] == 9.0


def test_blocking_reference_publishes_each_save_before_any_following_update():
    kwargs, events, state, _ = scenario(policy=LoopPolicy(2, checkpoint_updates=(1,)))
    result = run_loop(**kwargs, blocking_checkpoints=True)
    for number in (0, 1):
        assert events.index(("accept", number)) < events.index(("update", number + 1))
    assert state["accepted"] == [0, 1, 2]
    assert result["checkpoint_wait_seconds"] == 9.0


@pytest.mark.parametrize("value", [None, [], {"bad": object()}, {"bad": float("nan")},
                                 {"bad": (1, 2)}, {1: "not a string key"}])
def test_submission_must_be_json_object_before_broadcast(value):
    kwargs, _, state, _ = scenario()
    kwargs["submit_checkpoint"] = lambda receipt: value
    with pytest.raises(LifecycleError, match="checkpoint submission"):
        run_loop(**kwargs)
    assert state["update"] == 0


@pytest.mark.parametrize("value", [[], {"bad": object()}, {"bad": float("nan")}])
def test_completed_worker_result_must_be_json_object_before_broadcast(value):
    kwargs, _, state, _ = scenario()
    kwargs["poll_checkpoint"] = lambda **_: value
    with pytest.raises(LifecycleError, match="checkpoint worker poll"):
        run_loop(**kwargs)
    assert state["update"] == 0 and state["accepted"] == []


def test_blocking_poll_must_complete_instead_of_silently_abandoning_work():
    kwargs, _, state, _ = scenario(policy=LoopPolicy(0))
    kwargs["poll_checkpoint"] = lambda **_: None
    with pytest.raises(LifecycleError, match="Blocking checkpoint poll returned no completion"):
        run_loop(**kwargs)
    assert state["accepted"] == []


@pytest.mark.parametrize("value", [True, 7, None])
def test_wrong_publication_update_is_rejected(value):
    kwargs, _, state, _ = scenario(ready_after=0)
    kwargs["accept_checkpoint"] = lambda record: value
    with pytest.raises(LifecycleError, match="wrong committed update"):
        run_loop(**kwargs)
    assert state["update"] == 0


def test_successful_callback_must_advance_exactly_one_optimizer_boundary():
    kwargs, _, state, _ = scenario()
    kwargs["update"] = lambda: {"update": 0}
    with pytest.raises(LifecycleError, match="exactly one optimizer/cursor boundary"):
        run_loop(**kwargs)
    assert state["accepted"] == []


def test_real_background_worker_can_require_training_to_advance_before_finishing():
    kwargs, _, state, _ = scenario(policy=LoopPolicy(1))
    advanced, future = threading.Event(), [None]
    with ThreadPoolExecutor(max_workers=1) as executor:
        def work(receipt):
            if receipt["update"] == 0:
                assert advanced.wait(timeout=5), "training was blocked on the worker"
            return {"receipt": receipt, "verified": True}

        def submit(receipt):
            assert future[0] is None
            future[0] = executor.submit(work, receipt)
            return {"submitted": receipt["update"]}

        def poll(*, wait=False):
            if not wait and not future[0].done():
                return None
            result = future[0].result(timeout=5)
            future[0] = None
            return result

        original_update = kwargs["update"]

        def update():
            result = original_update()
            advanced.set()
            return result

        kwargs.update(submit_checkpoint=submit, poll_checkpoint=poll, update=update)
        result = run_loop(**kwargs)
    assert state["accepted"] == [0, 1]
    assert result["last_retained_update"] == 1 and not result["checkpoint_pending"]


@pytest.mark.parametrize("start", [-1, True, 4])
def test_invalid_start_is_rejected_before_any_save(start):
    kwargs, events, _, _ = scenario(start=start, policy=LoopPolicy(3))
    with pytest.raises(LifecycleError, match="Initial update"):
        run_loop(**kwargs)
    assert events == []


def test_non_boolean_blocking_mode_is_rejected():
    kwargs, events, _, _ = scenario()
    with pytest.raises(LifecycleError, match="blocking_checkpoints must be bool"):
        run_loop(**kwargs, blocking_checkpoints=1)
    assert events == []
