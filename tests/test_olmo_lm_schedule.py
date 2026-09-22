"""Prepared-stream exposure boundaries and stateless before-update RT modes."""

import copy
import json

import pytest

from cdrm.pretrained.lm_schedule import alpha_for_update, build_schedule, cursor_for_update


def test_equal_lengths_exact_minima_and_before_update_endpoints():
    plan = build_schedule([5] * 30, batch_size=2, warmup_updates=2,
                          ramp_min_tokens=30, ramp_min_updates=3)
    assert plan["ramp_end_update"] == 5
    assert plan["total_updates"] == 8
    assert plan["phase_input_tokens"] == {"warmup": 20, "ramp": 30, "alpha1": 30}
    assert plan["phase_updates"] == {"warmup": 2, "ramp": 3, "alpha1": 3}
    alphas = [alpha_for_update(plan, i, tokens) for i, tokens in enumerate(plan["batch_token_prefix"])]
    assert alphas == pytest.approx([0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1, 1])
    assert plan["used_windows"] == 16
    assert plan["unused_windows"] == 14
    assert cursor_for_update(plan, 8) == 16


def test_uneven_lengths_preserve_actual_ramp_overshoot_in_constant_phase():
    # Batch totals: 10 warmup; 8+15 ramp (23, overshooting minimum20);
    # then 8+10+7 alpha-one (25) because two updates only supply18.
    lengths = [4, 6, 3, 5, 7, 8, 4, 4, 5, 5, 3, 4, 100]
    plan = build_schedule(lengths, batch_size=2, warmup_updates=1,
                          ramp_min_tokens=20, ramp_min_updates=2)
    assert plan["ramp_end_update"] == 3
    assert plan["actual_ramp_tokens"] == 23
    assert plan["actual_ramp_updates"] == 2
    assert plan["alpha1_updates"] == 3
    assert plan["alpha1_tokens"] == 25
    assert plan["total_tokens"] == 58
    assert plan["unused_windows"] == 1  # A large partial batch is never used.
    assert alpha_for_update(plan, 2, 18) == pytest.approx(8 / 23)


def test_minimum_updates_can_dominate_tokens_in_both_phases():
    plan = build_schedule([100] * 12, batch_size=1, warmup_updates=0,
                          ramp_min_tokens=50, ramp_min_updates=4)
    assert plan["actual_ramp_tokens"] == 400
    assert plan["actual_ramp_updates"] == 4
    assert plan["alpha1_updates"] == 4
    assert plan["alpha1_tokens"] == 400
    assert plan["total_updates"] == 8


def test_token_threshold_can_dominate_update_threshold():
    plan = build_schedule([10] * 20, batch_size=1, warmup_updates=0,
                          ramp_min_tokens=45, ramp_min_updates=2)
    assert plan["actual_ramp_updates"] == 5
    assert plan["actual_ramp_tokens"] == 50
    assert plan["alpha1_tokens"] == 50
    assert plan["total_updates"] == 10


def test_uneven_alpha_is_monotone_and_bounded_by_both_actual_fractions():
    plan = build_schedule([1, 9, 3, 7, 12, 20, 20, 20, 20, 20], batch_size=1,
                          warmup_updates=0, ramp_min_tokens=30, ramp_min_updates=4)
    alphas = [alpha_for_update(plan, i, n) for i, n in enumerate(plan["batch_token_prefix"])]
    assert alphas == sorted(alphas)
    assert all(0 <= value <= 1 for value in alphas)
    for i in range(plan["ramp_end_update"]):
        assert alphas[i] <= i / plan["actual_ramp_updates"]
        assert alphas[i] <= plan["batch_token_prefix"][i] / plan["actual_ramp_tokens"]
    assert alphas[plan["ramp_end_update"]] == 1


def test_resume_and_json_roundtrip_reproduce_every_alpha_cursor_and_token_count():
    plan = build_schedule([3, 7, 4, 9, 8, 2, 6, 1] * 8, batch_size=2,
                          warmup_updates=2, ramp_min_tokens=45, ramp_min_updates=3)
    restored = json.loads(json.dumps(plan))
    assert restored == plan
    for i, tokens in enumerate(plan["batch_token_prefix"]):
        assert alpha_for_update(restored, i, tokens) == alpha_for_update(plan, i, tokens)
        cursor = cursor_for_update(restored, i)
        assert cursor == i * 2
        assert sum(([3, 7, 4, 9, 8, 2, 6, 1] * 8)[:cursor]) == tokens


@pytest.mark.parametrize("lengths", [[], [0] * 20, [-1] * 20, [1.2] * 20, [True] * 20, [float("nan")] * 20])
def test_invalid_lengths_are_rejected(lengths):
    with pytest.raises(ValueError):
        build_schedule(lengths, batch_size=2, warmup_updates=0, ramp_min_tokens=1, ramp_min_updates=1)


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"batch_size": True}, {"warmup_updates": -1}, {"warmup_updates": 1.5},
    {"ramp_min_tokens": 0}, {"ramp_min_tokens": False}, {"ramp_min_updates": 0},
])
def test_invalid_schedule_settings_are_rejected(kwargs):
    settings = {"batch_size": 1, "warmup_updates": 0, "ramp_min_tokens": 1, "ramp_min_updates": 1}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        build_schedule([10] * 100, **settings)


@pytest.mark.parametrize("lengths,warmup,token_min,update_min,match", [
    ([10], 1, 1, 1, "warmup"),
    ([10] * 4, 1, 100, 1, "ramp token"),
    ([10] * 4, 1, 1, 4, "ramp token"),
    ([10] * 6, 1, 30, 2, "alpha-one"),
])
def test_insufficient_data_is_rejected_without_cycling(lengths, warmup, token_min, update_min, match):
    with pytest.raises(ValueError, match=match):
        build_schedule(lengths, batch_size=1, warmup_updates=warmup,
                       ramp_min_tokens=token_min, ramp_min_updates=update_min)


def test_wrong_resume_counters_and_mutated_schedule_are_rejected():
    plan = build_schedule([10] * 20, batch_size=1, warmup_updates=0,
                          ramp_min_tokens=40, ramp_min_updates=2)
    with pytest.raises(ValueError, match="valid tokens"):
        alpha_for_update(plan, 2, 19)
    with pytest.raises(ValueError):
        alpha_for_update(plan, -1, 0)
    with pytest.raises(ValueError):
        cursor_for_update(plan, plan["total_updates"] + 1)
    altered = copy.deepcopy(plan)
    altered["actual_ramp_tokens"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        alpha_for_update(altered, 2, 20)


def test_plan_hash_records_full_prepared_stream_and_consumed_prefix_separately():
    lengths = [10] * 20
    first = build_schedule(lengths, batch_size=1, warmup_updates=0, ramp_min_tokens=30, ramp_min_updates=2)
    changed = lengths.copy()
    changed[-1] = 11  # Outside this bounded run, but preparation identity changes.
    second = build_schedule(changed, batch_size=1, warmup_updates=0, ramp_min_tokens=30, ramp_min_updates=2)
    assert first["used_window_lengths_sha256"] == second["used_window_lengths_sha256"]
    assert first["window_lengths_sha256"] != second["window_lengths_sha256"]
    assert first["schedule_sha256"] != second["schedule_sha256"]
