"""Deterministic O4 exposure and RT-alpha schedule over a fixed window stream.

There is no cycling, shuffling, model state, or RNG here. Only full consecutive
batches are used. Alpha belongs to the update about to run, so every microbatch
in that update can share one immutable RT mode. Checkpointed completed-update
and valid-token counters identify the same next alpha after a resume.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any, Sequence


SCHEMA = "olmo-lm-exposure-schedule-v1"


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_schedule(window_lengths: Sequence[int], *, batch_size: int,
                   warmup_updates: int = 100, ramp_min_tokens: int = 10_000_000,
                   ramp_min_updates: int = 200) -> dict[str, Any]:
    """Freeze the first full-batch boundaries meeting both exposure conditions.

    After the common alpha-zero LR warmup, the ramp lasts at least the requested
    valid tokens AND updates. The alpha-one phase then lasts at least the actual
    ramp's token AND update exposure, including its boundary overshoot. Remaining
    windows and any incomplete last batch are unused. All returned fields are
    JSON-serializable; ``schedule_sha256`` covers every other plan field.
    """
    batch_size = _integer("batch_size", batch_size, minimum=1)
    warmup_updates = _integer("warmup_updates", warmup_updates)
    ramp_min_tokens = _integer("ramp_min_tokens", ramp_min_tokens, minimum=1)
    ramp_min_updates = _integer("ramp_min_updates", ramp_min_updates, minimum=1)
    lengths = [_integer("window length", length, minimum=1) for length in window_lengths]
    if not lengths:
        raise ValueError("Insufficient data: window_lengths is empty")
    available_updates = len(lengths) // batch_size
    if available_updates <= warmup_updates:
        raise ValueError("Insufficient full batches for the warmup followed by an RT ramp")
    prefix = [0]
    for start in range(0, available_updates * batch_size, batch_size):
        prefix.append(prefix[-1] + sum(lengths[start:start + batch_size]))
    warmup_tokens = prefix[warmup_updates]
    ramp_end = next((boundary for boundary in range(warmup_updates + 1, available_updates + 1)
                     if boundary - warmup_updates >= ramp_min_updates
                     and prefix[boundary] - warmup_tokens >= ramp_min_tokens), None)
    if ramp_end is None:
        raise ValueError("Insufficient data to meet both ramp token and update minima")
    ramp_updates = ramp_end - warmup_updates
    ramp_tokens = prefix[ramp_end] - warmup_tokens
    end = next((boundary for boundary in range(ramp_end + 1, available_updates + 1)
                if boundary - ramp_end >= ramp_updates
                and prefix[boundary] - prefix[ramp_end] >= ramp_tokens), None)
    if end is None:
        raise ValueError("Insufficient data for alpha-one exposure matching the actual ramp tokens and updates")
    plan = {
        "schema": SCHEMA,
        "batch_size": batch_size,
        "target_budget": {"warmup_updates": warmup_updates,
                          "ramp_min_tokens": ramp_min_tokens,
                          "ramp_min_updates": ramp_min_updates},
        "alpha_rule": "before_update_min_actual_ramp_token_and_update_fractions",
        "warmup_updates": warmup_updates,
        "ramp_end_update": ramp_end,
        "total_updates": end,
        "ramp_start_tokens": warmup_tokens,
        "ramp_end_tokens": prefix[ramp_end],
        "total_tokens": prefix[end],
        "actual_ramp_updates": ramp_updates,
        "actual_ramp_tokens": ramp_tokens,
        "alpha1_updates": end - ramp_end,
        "alpha1_tokens": prefix[end] - prefix[ramp_end],
        "phase_input_tokens": {"warmup": warmup_tokens, "ramp": ramp_tokens,
                               "alpha1": prefix[end] - prefix[ramp_end]},
        "phase_updates": {"warmup": warmup_updates, "ramp": ramp_updates, "alpha1": end - ramp_end},
        "batch_token_prefix": prefix[:end + 1],
        "used_windows": end * batch_size,
        "available_windows": len(lengths),
        "unused_windows": len(lengths) - end * batch_size,
        "window_lengths_sha256": _digest(lengths),
        "used_window_lengths_sha256": _digest(lengths[:end * batch_size]),
    }
    return {**plan, "schedule_sha256": _digest(plan)}


def _validate_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("Expected an OLMo LM exposure schedule")
    if plan.get("schedule_sha256") != _digest({key: value for key, value in plan.items() if key != "schedule_sha256"}):
        raise ValueError("Schedule fingerprint changed")
    # Plans originate from build_schedule; these checks also make accidental
    # malformed manually-authored data fail before selecting a batch or mode.
    end, warmup, ramp_end = plan["total_updates"], plan["warmup_updates"], plan["ramp_end_update"]
    prefix = plan["batch_token_prefix"]
    if not 0 <= warmup < ramp_end < end or len(prefix) != end + 1 or prefix[0] != 0:
        raise ValueError("Schedule boundaries or token-prefix geometry are invalid")
    if any(right <= left for left, right in zip(prefix, prefix[1:])):
        raise ValueError("Schedule valid-token prefix must strictly increase")
    if (prefix[warmup] != plan["ramp_start_tokens"] or prefix[ramp_end] != plan["ramp_end_tokens"]
            or prefix[-1] != plan["total_tokens"]
            or ramp_end - warmup != plan["actual_ramp_updates"]
            or prefix[ramp_end] - prefix[warmup] != plan["actual_ramp_tokens"]
            or plan["used_windows"] != plan["batch_size"] * end):
        raise ValueError("Schedule exposure metadata is inconsistent")


def cursor_for_update(plan: dict[str, Any], completed_updates: int) -> int:
    """Index of the next unconsumed window (end cursor is valid when complete)."""
    _validate_plan(plan)
    completed_updates = _integer("completed_updates", completed_updates)
    if completed_updates > plan["total_updates"]:
        raise ValueError("Completed updates exceed the frozen schedule")
    return completed_updates * plan["batch_size"]


def alpha_for_update(plan: dict[str, Any], completed_updates: int, completed_tokens: int) -> float:
    """Alpha for the next update; require exact completed boundary counters.

    The first ramp update starts at zero. At the recorded ramp-end boundary,
    the next update is the first alpha-one update. At the completed final
    boundary this returns one for final evaluation, although no training batch
    remains. Within the ramp the minimum of the two actual-exposure fractions
    is monotone, reaches one at the endpoint, and cannot shorten either budget.
    """
    cursor_for_update(plan, completed_updates)
    completed_updates = int(completed_updates)
    completed_tokens = _integer("completed_tokens", completed_tokens)
    if completed_tokens != plan["batch_token_prefix"][completed_updates]:
        raise ValueError("Completed valid tokens disagree with the prepared-stream update boundary")
    if completed_updates <= plan["warmup_updates"]:
        return 0.0
    if completed_updates >= plan["ramp_end_update"]:
        return 1.0
    token_fraction = (completed_tokens - plan["ramp_start_tokens"]) / plan["actual_ramp_tokens"]
    update_fraction = (completed_updates - plan["warmup_updates"]) / plan["actual_ramp_updates"]
    return min(token_fraction, update_fraction)
