#!/usr/bin/env python3
"""Opt-in exception agreement at synchronized, quiescent training boundaries.

This helper is not wired into any existing runner. All ranks must call it in the
same order, outside graph capture and in-flight model collectives. Callbacks must
not themselves require unmatched collectives. It coordinates ordinary Python
exceptions only: dead peers, hung callbacks and process-group failures still need
an external supervisor and process-group timeout. It never retries or rolls back
an update, restores RNG, or advances a data cursor.
"""
from __future__ import annotations

import re
from typing import Callable, TypeVar

import torch.distributed as dist

T = TypeVar("T")
_PHASE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9 _./:-]{0,95}\Z")
_TYPE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{0,79}\Z")


class BoundaryActionError(RuntimeError):
    """All participating ranks rejected a boundary or observed a callback error."""


def _gather(value, group=None):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def coordinated_boundary_action(
    phase: str, callback: Callable[[], T], *, rank_zero_only: bool = False, group=None,
) -> T | None:
    """Run callbacks only after phase agreement; report their errors on all ranks.

    ``phase`` must be a short static label, never user data or a credential.
    ``rank_zero_only`` refers to rank zero *within group*. Results stay local;
    ranks that skip a rank-zero callback receive None. Successful peer callbacks
    may already have side effects when another rank fails. Only exception class
    names enter the common error, deliberately excluding exception messages.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Boundary coordination requires an initialized process group")
    issues = []
    valid_phase = isinstance(phase, str) and _PHASE.fullmatch(phase) is not None
    if not valid_phase:
        issues.append("invalid phase label")
    if type(rank_zero_only) is not bool:
        issues.append("invalid callback ownership")
    if not callable(callback):
        issues.append("callback is not callable")
    descriptor = {"phase": phase if valid_phase else None,
                  "rank_zero_only": rank_zero_only if type(rank_zero_only) is bool else None,
                  "issues": issues}
    descriptors = _gather(descriptor, group)
    if any(item["issues"] for item in descriptors):
        detail = "; ".join(f"rank {rank}: {', '.join(item['issues'])}"
                           for rank, item in enumerate(descriptors) if item["issues"])
        raise BoundaryActionError(f"Boundary descriptor rejected: {detail}")
    if any(item != descriptors[0] for item in descriptors[1:]):
        raise BoundaryActionError("Boundary phase or callback ownership differs across ranks")
    result, error = None, None
    if not rank_zero_only or dist.get_rank(group) == 0:
        try:
            result = callback()
        except Exception as exc:
            name = type(exc).__name__
            error = name if _TYPE.fullmatch(name) else "Exception"
    errors = _gather(error, group)
    if any(item is not None for item in errors):
        detail = "; ".join(f"rank {rank}: {error}"
                           for rank, error in enumerate(errors) if error is not None)
        raise BoundaryActionError(f"{phase}: {detail}") from None
    return result
