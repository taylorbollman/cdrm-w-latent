"""Saved-record guards for the adaptive six-layer20k continuation comparison."""
import copy
import json

import pytest

from scripts import rt_a5_l1r_depth_extension_report as report


def history(start, stop):
    return [{"update": i, "examples_seen": i * 1024, "order_chain": f"{i:064x}", "seconds": .01,
             "loss": 1.25, "state_loss": 1., "latent_loss": .25, "weighted_latent_loss": .25,
             "grad_norm": 1., "token_accuracy": .5, "whole_word_exact": .1} for i in range(start + 1, stop + 1)]


def test_stitch_preserves_global_updates_and_every_order_hash():
    parent, child, reference = history(0, 10000), history(10000, 20000), history(0, 20000)
    combined = report.stitch_histories(parent, child, reference)
    assert [row["update"] for row in combined] == list(range(1, 20001))
    child[1123]["order_chain"] = "f" * 64
    with pytest.raises(ValueError):
        report.stitch_histories(parent, child, reference)


@pytest.mark.parametrize("change", ["reset", "missing", "duplicate", "loss"])
def test_invalid_continuation_history_is_rejected(change):
    child = history(10000, 20000)
    if change == "reset":
        child[0]["examples_seen"] = 1024
    elif change == "missing":
        child.pop(99)
    elif change == "duplicate":
        child[40] = child[39].copy()
    else:
        child[22]["loss"] = .2
    with pytest.raises(ValueError):
        report.validate_segment(child, 10000, 20000)


@pytest.fixture
def child_fixture():
    path = str(report.ROOT / "test-parent/checkpoints/step-010000.pt")
    source = {"model.py": "a" * 64}
    digest = report._digest_dict(source)
    parent = {"contract": {"source_sha256": digest}, "source_files": source, "initialization": {"seed": 1234}}
    protocol = {"strict_contract": parent["contract"], "source_files": source,
                "source_sha256": digest, "initialization": parent["initialization"],
                "parent_checkpoint": {"path": path, "sha256": "b" * 64}}
    packet = {**copy.deepcopy(parent), "schema": "rt-a5-l1r-depth-training-v1", "status": "complete",
              "start_update": 10000, "completed_updates": 20000, "endpoint": 20000,
              "wandb": {"status": "synced"}, "confirmation_evaluated": False, "latent_rollout_evaluated": False,
              "parent_checkpoint": protocol["parent_checkpoint"].copy(),
              "checkpoints": [{"completed_updates": 15000}, {"completed_updates": 20000}]}
    return packet, {"resume": path}, protocol, parent


def test_exact_resume_contract_is_accepted_without_modifying_base_globals(child_fixture):
    report.validate_child(*child_fixture)
    assert report.base.ENDPOINT == 10000
    assert report.base.STEPS == (1000, 5000, 10000)


@pytest.mark.parametrize("change", ["parent_sha", "parent_path", "resume", "contract", "source", "initialization", "sync"])
def test_resume_mismatches_are_rejected(child_fixture, change):
    packet, config, protocol, parent = child_fixture
    if change == "parent_sha":
        packet["parent_checkpoint"]["sha256"] = "c" * 64
    elif change == "parent_path":
        packet["parent_checkpoint"]["path"] = "wrong.pt"
    elif change == "resume":
        config["resume"] = None
    elif change == "contract":
        packet["contract"]["learning_rate"] = .1
    elif change == "source":
        packet["source_files"]["model.py"] = "d" * 64
    elif change == "initialization":
        packet["initialization"]["seed"] = 9
    else:
        packet["wandb"]["status"] = "running"
    with pytest.raises(ValueError):
        report.validate_child(packet, config, protocol, parent)


def test_15k_is_excluded_even_if_one_arm_has_a_full_evaluation():
    with pytest.raises(ValueError):
        report.selected_evaluations({"evaluations": []}, "six_layers", 15000)
    with pytest.raises(ValueError):
        report.plot_rows({}, "six_layers", 15000)


def test_reference20k_uses_full_words_and_15k_is_routine_only():
    raw = json.loads((report.base.REFERENCE / "report.json").read_text())
    assert {row["rows"] for row in raw["evaluations"] if row["update"] == 15000} == {4096}
    curves, metrics = report.selected_evaluations(raw, "two_layers", 20000)
    assert metrics["ood_dev"]["rows"] == 102400
    summary = {"arms": {"two_layers": {"curves": {"20000": curves}}}}
    rows = report.plot_rows(summary, "two_layers")
    assert len(rows) == 36 and all(row["update"] == 20000 for row in rows)
