"""Independent boundary checks for the A5 warm-start curriculum.

These use the accepted experiment's concrete counters rather than deriving
expectations through the trainer's bookkeeping helpers.
"""

from types import SimpleNamespace

import pytest

from scripts import rt_nextlat_a5_fuzzy_curriculum as curriculum


@pytest.mark.parametrize(
    "phase_update, expected_fuzzy_weight",
    [(0, 0.0), (1, 1 / 6000), (1500, 0.25), (2999, 2999 / 6000),
     (3000, 0.5), (10000, 0.5)],
)
def test_ramp_uses_phase_updates_and_saturates_at_three_thousand(
    phase_update, expected_fuzzy_weight
):
    weights = curriculum.task_weights("mixed-curriculum", phase_update, 3000)
    assert set(weights) == {"a5", "fuzzy"}
    assert weights["fuzzy"] == pytest.approx(expected_fuzzy_weight, abs=1e-15)
    assert weights["a5"] == pytest.approx(1 - expected_fuzzy_weight, abs=1e-15)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("phase_update", [0, 1, 1500, 3000, 10000])
def test_control_keeps_the_full_a5_objective_at_every_phase(phase_update):
    assert curriculum.task_weights("a5-control", phase_update, 3000) == {
        "a5": 1.0, "fuzzy": 0.0
    }


@pytest.mark.parametrize("mode", ["mixed-curriculum", "a5-control"])
@pytest.mark.parametrize(
    "phase, a5_offset, a5_epoch, a5_position, fuzzy_offset, fuzzy_epoch, fuzzy_position",
    [
        (0, 1280000, 1, 480000, 0, 0, 0),
        (1, 1280128, 1, 480128, 128, 0, 128),
        (2500, 1600000, 2, 0, 320000, 25, 0),
        (3000, 1664000, 2, 64000, 384000, 30, 0),
        (10000, 2560000, 3, 160000, 1280000, 100, 0),
    ],
)
def test_parent_a5_stream_continues_while_fuzzy_starts_fresh(
    mode, phase, a5_offset, a5_epoch, a5_position,
    fuzzy_offset, fuzzy_epoch, fuzzy_position,
):
    contract = {
        "mode": mode,
        "batch_per_task": 128,
        "initial_task_offsets": {"a5": 1280000, "fuzzy": 0},
        "streams": {"a5": {"train_rows": 800000}, "fuzzy": {"train_rows": 12800}},
    }
    if mode == "a5-control":
        fuzzy_offset = fuzzy_epoch = fuzzy_position = 0
    assert curriculum.task_offsets(phase, contract) == {
        "a5": a5_offset, "fuzzy": fuzzy_offset
    }
    assert curriculum.cursors(phase, contract) == {
        "a5": {
            "absolute_example_offset": a5_offset, "epoch": a5_epoch, "position": a5_position
        },
        "fuzzy": {
            "absolute_example_offset": fuzzy_offset,
            "epoch": fuzzy_epoch,
            "position": fuzzy_position,
        },
    }


def test_tracking_axes_distinguish_optimizer_updates_from_phase_updates():
    definitions = {}

    def define_metric(name, **kwargs):
        definitions[name] = kwargs

    tracker = SimpleNamespace(
        _run=SimpleNamespace(define_metric=define_metric),
        _call=lambda description, callback: callback(),
    )
    curriculum._tracking_axes(tracker)
    assert "optimizer_update" in definitions
    assert "phase_update" in definitions
    assert definitions["train/*"]["step_metric"] == "optimizer_update"
    assert definitions["dev/*"]["step_metric"] == "optimizer_update"
    assert definitions["phase/*"]["step_metric"] == "phase_update"
