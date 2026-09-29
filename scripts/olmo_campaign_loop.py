"""Host-side lifecycle at completed distributed optimizer boundaries.

This module owns no tensor math or CUDA capture. All ranks call the loop in
lockstep; update/save callbacks use their existing distributed contracts. A
failed collective/update is not recoverable here. Fresh processes restore the
last committed checkpoint. Rank-zero I/O failures are coordinated explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import signal
import time

import torch.distributed as dist
from scripts.olmo_campaign_lifecycle import BoundaryActionError, coordinated_boundary_action


class LifecycleError(RuntimeError):
    """An ordinary host error observed and reported by every healthy rank."""


class Coordinator:
    def __init__(self, group=None):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)

    def gather(self, value):
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def call(self, phase, function, *, rank_zero=False):
        try:
            value = coordinated_boundary_action(phase, function, rank_zero_only=rank_zero, group=self.group)
        except BoundaryActionError as exc:
            raise LifecycleError(str(exc)) from None
        if rank_zero:
            # Small JSON-native control/receipt only; never model tensors.
            return self.gather(value)[0]
        return value

    def same(self, phase, value):
        values = self.gather(value)
        if any(other != values[0] for other in values[1:]):
            raise LifecycleError(f"{phase}: ranks disagree")
        return values[0]


@dataclass(frozen=True)
class LoopPolicy:
    max_updates: int
    checkpoint_seconds: float = 600.0
    checkpoint_updates: tuple[int, ...] = ()
    save_initial: bool = True

    def __post_init__(self):
        if type(self.max_updates) is not int or self.max_updates < 0:
            raise ValueError("max_updates must be a nonnegative integer")
        if (isinstance(self.checkpoint_seconds, bool) or not math.isfinite(self.checkpoint_seconds)
                or not 0 < self.checkpoint_seconds <= 600):
            raise ValueError("Checkpoint cadence must be positive and at most 600 seconds")
        if (any(type(value) is not int or value < 0 for value in self.checkpoint_updates)
                or len(set(self.checkpoint_updates)) != len(self.checkpoint_updates)):
            raise ValueError("Checkpoint update milestones must be distinct nonnegative integers")
        if type(self.save_initial) is not bool:
            raise ValueError("save_initial must be bool")


class StopRequest:
    """Signal handlers set state only; poll and collectives happen at boundaries."""
    def __init__(self, path=None):
        self.path = None if path is None else Path(path)
        self.signum = None

    def request(self, signum=signal.SIGTERM, frame=None):
        self.signum = int(signum)

    @contextmanager
    def installed(self):
        previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
        try:
            for number in previous:
                signal.signal(number, self.request)
            yield self
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)

    def local_reason(self):
        return None if self.signum is None else f"signal:{self.signum}"

    def file_reason(self):
        return "stop_file" if self.path is not None and self.path.exists() else None


def run_loop(*, coordinator, policy, completed, update, log, save, publish_checkpoint,
             stop, retain=None, clock=time.monotonic, restored=False):
    """Run an explicit segment without changing the immutable training plan.

    `completed` returns the committed optimizer-update count. `update` must
    finish optimizer/counter/cursor mutation together, or raise without this
    loop attempting another collective or emergency checkpoint. `save` is an
    all-rank callback. Logging, retention and publication execute on rank zero
    with coordinated errors. Publication follows verified retention if used.
    Stop/time requests take effect after the current optimizer update; the
    checkpoint interval is a deadline checked at boundaries, not a hard bound
    on the duration of an individual update or an I/O operation.
    """
    start = coordinator.same("initial committed update", completed())
    coordinator.same("loop policy", policy)
    if not 0 <= start <= policy.max_updates:
        raise LifecycleError("Initial update lies outside this segment")
    last_saved = start if restored else None
    last_save_time = coordinator.call("initial checkpoint clock", clock, rank_zero=True)
    checkpoints = 0

    def checkpoint(reason):
        nonlocal last_saved, last_save_time, checkpoints
        current = coordinator.same("checkpoint committed update", completed())
        if current == last_saved:
            return
        # Existing distributed checkpoint implementation coordinates its own
        # all-rank preparation/publication. Do not wrap an unknown collective
        # failure in another collective here.
        receipt = save(current, reason)
        if retain is not None:
            remote = coordinator.call("checkpoint retention", lambda: retain(receipt), rank_zero=True)
            receipt = {**receipt, "retention": remote}
        coordinator.call("checkpoint publication", lambda: publish_checkpoint(receipt), rank_zero=True)
        last_saved = current
        last_save_time = coordinator.call("checkpoint clock", clock, rank_zero=True)
        checkpoints += 1

    if policy.save_initial and not restored:
        checkpoint("initial")
    while True:
        current = coordinator.same("committed update", completed())
        requests = coordinator.gather(stop.local_reason())

        def decide():
            reason = next((value for value in requests if value), None) or stop.file_reason()
            reason = reason or ("update_limit" if current >= policy.max_updates else None)
            due = (current != last_saved and (reason is not None
                   or current in policy.checkpoint_updates
                   or clock()-last_save_time >= policy.checkpoint_seconds))
            return {"stop": reason, "save": due}

        decision = coordinator.call("boundary control", decide, rank_zero=True)
        if decision["save"]:
            checkpoint(decision["stop"] or "cadence")
        if decision["stop"]:
            return {"start_update": start, "completed_update": current,
                    "stop_reason": decision["stop"], "checkpoints_published": checkpoints,
                    "last_saved_update": last_saved}
        metrics = update()
        actual = coordinator.same("post-update committed count", completed())
        if actual != current + 1:
            raise LifecycleError("Successful update did not commit exactly one optimizer/cursor boundary")
        try:
            coordinator.call("rank-zero metrics", lambda: log(metrics), rank_zero=True)
        except LifecycleError:
            # The math/update succeeded and every rank is at this safe boundary.
            # Preserve it before reporting the ordinary logging error. A failed
            # update, collective or checkpoint does not use this recovery path.
            checkpoint("logging_failure")
            raise
