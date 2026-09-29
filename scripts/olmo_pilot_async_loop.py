"""Completed-boundary training loop with one asynchronous checkpoint worker.

Only immutable, synchronously saved files cross into the worker. This loop does
not own CUDA work, model tensors, worker processes, or storage publication. Its
rank-zero callbacks submit/poll one worker and accept its verified publication
into the main report. Every rank observes worker errors before another update.
"""
from __future__ import annotations

import json
import math
import time

from scripts.olmo_campaign_loop import Coordinator, LifecycleError, LoopPolicy, StopRequest


def _control_record(value, phase):
    """Reject tensor/non-JSON worker results before distributed broadcast."""
    if type(value) is not dict:
        raise ValueError(f"{phase} must return a JSON object")
    encoded = json.dumps(value, allow_nan=False)
    if json.loads(encoded) != value:
        raise ValueError(f"{phase} must contain only JSON-native values and string keys")
    return value


def run_loop(*, coordinator, policy, completed, update, log, save,
             submit_checkpoint, poll_checkpoint, accept_checkpoint, stop,
             clock=time.monotonic, restored=False, blocking_checkpoints=False):
    """Advance training while one immutable local checkpoint is retained.

    ``save(update, reason)`` is synchronous and called by all ranks. Rank zero
    calls ``submit_checkpoint(receipt)`` once to launch a CPU-only worker and
    obtain a small JSON submission record. ``poll_checkpoint(wait=False)``
    returns None while it runs; ``wait=True`` must finish or raise. A completed
    JSON record is passed to ``accept_checkpoint``, which publishes it to the
    main report and returns its committed update number.

    Checkpoint workers are drained before any subsequent local save, preventing
    storage validation/publication/pruning from racing another write. Terminal
    boundaries also drain. The cadence clock resets after local save/submission,
    independently of the latest verified remote publication. An update failure
    receives no emergency checkpoint or collective: the caller owns worker
    cleanup and recovery from the last already retained checkpoint.
    ``blocking_checkpoints=True`` drains after each submission for a synchronous
    acceptance reference using the same worker, model, and publication path.
    """
    start = coordinator.same("initial committed update", completed())
    coordinator.same("loop policy", policy)
    coordinator.same("checkpoint transport", blocking_checkpoints)
    if type(blocking_checkpoints) is not bool:
        raise LifecycleError("blocking_checkpoints must be bool")
    if type(start) is not int or not 0 <= start <= policy.max_updates:
        raise LifecycleError("Initial update lies outside this segment")
    last_saved = start if restored else None
    last_retained = start if restored else None
    last_save_time = coordinator.call("initial checkpoint clock", clock, rank_zero=True)
    pending_update = None
    submitted = published = 0
    wait_seconds = 0.0

    def collect(*, wait):
        nonlocal pending_update, last_retained, published, wait_seconds
        if pending_update is None:
            return

        def poll():
            before = clock() if wait else None
            record = poll_checkpoint(wait=wait)
            elapsed = clock() - before if wait else 0.0
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Checkpoint wait clock must advance monotonically")
            if record is None:
                if wait:
                    raise ValueError("Blocking checkpoint poll returned no completion")
            else:
                _control_record(record, "Checkpoint completion")
            return {"record": record, "wait_seconds": elapsed}

        result = coordinator.call("checkpoint worker poll", poll, rank_zero=True)
        wait_seconds += result["wait_seconds"]
        if result["record"] is None:
            return
        def accept():
            retained = accept_checkpoint(result["record"])
            if type(retained) is not int or retained != pending_update:
                raise ValueError("Checkpoint publication returned the wrong committed update")
            return retained

        retained = coordinator.call("checkpoint publication", accept, rank_zero=True)
        last_retained = retained
        pending_update = None
        published += 1

    def checkpoint(reason):
        nonlocal last_saved, last_save_time, pending_update, submitted
        current = coordinator.same("checkpoint committed update", completed())
        if current == last_saved:
            return
        collect(wait=True)
        # The existing distributed save coordinates its own preparation and
        # publication. Unknown collective failures must not trigger a new one.
        receipt = save(current, reason)
        coordinator.call("checkpoint submission",
            lambda: _control_record(submit_checkpoint(receipt), "Checkpoint submission"),
            rank_zero=True)
        pending_update = current
        last_saved = current
        submitted += 1
        last_save_time = coordinator.call("checkpoint clock", clock, rank_zero=True)
        if blocking_checkpoints:
            collect(wait=True)

    if policy.save_initial and not restored:
        checkpoint("initial")
    while True:
        current = coordinator.same("committed update", completed())
        # This is a safe completed optimizer/cursor boundary. A failed worker
        # is reported to every healthy rank before another optimizer update.
        collect(wait=False)
        requests = coordinator.gather(stop.local_reason())

        def decide():
            reason = next((value for value in requests if value), None) or stop.file_reason()
            reason = reason or ("update_limit" if current >= policy.max_updates else None)
            due = (current != last_saved and (reason is not None
                   or current in policy.checkpoint_updates
                   or clock() - last_save_time >= policy.checkpoint_seconds))
            return {"stop": reason, "save": due}

        decision = coordinator.call("boundary control", decide, rank_zero=True)
        if decision["save"]:
            checkpoint(decision["stop"] or "cadence")
        if decision["stop"]:
            collect(wait=True)
            return {"start_update": start, "completed_update": current,
                    "stop_reason": decision["stop"],
                    "checkpoints_submitted": submitted,
                    "checkpoints_published": published,
                    "last_saved_update": last_saved,
                    "last_retained_update": last_retained,
                    "checkpoint_wait_seconds": wait_seconds,
                    "checkpoint_pending": pending_update is not None}
        metrics = update()
        actual = coordinator.same("post-update committed count", completed())
        if actual != current + 1:
            raise LifecycleError("Successful update did not commit exactly one optimizer/cursor boundary")
        try:
            coordinator.call("rank-zero metrics", lambda: log(metrics), rank_zero=True)
        except LifecycleError:
            # The update finished safely. Preserve its local files and verified
            # remote publication before reporting the ordinary logging error.
            checkpoint("logging_failure")
            collect(wait=True)
            raise
