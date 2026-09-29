"""Real two-process CPU/Gloo faults and independent checkpoint recovery checks."""
import json

import pytest
import torch

from scripts.olmo_campaign_lifecycle import coordinated_boundary_action
from scripts.olmo_campaign_lifecycle_check import CASES, SECRET_MARKER, run_harness, same_tree


def test_requires_explicit_distributed_context():
    with pytest.raises(RuntimeError, match="initialized process group"):
        coordinated_boundary_action("not initialized", lambda: None)


def test_actual_gloo_faults_preserve_last_checkpoint_and_replay_unsaved_update(tmp_path):
    report = run_harness(tmp_path/"gloo")
    assert all(report["checks"].values())
    assert report["backend"] == "gloo" and report["world_size"] == 2
    assert SECRET_MARKER not in json.dumps(report)
    for rank, rows in enumerate(report["ranks"]):
        by_case = {row["case"]: row for row in rows}
        assert tuple(by_case) == CASES
        assert by_case["success"]["callback_count"] == (1 if rank == 0 else 0)
        assert by_case["success"]["last_published_update"] == 2
        for name in ("log_failure", "persist_failure"):
            row = by_case[name]
            assert row["last_published_update"] == 1 and row["in_memory_completed_updates"] == 2
            assert row["replayed_unsaved_update"]
        for name in ("phase_mismatch", "ownership_mismatch", "invalid_descriptor"):
            assert by_case[name]["callback_count"] == 0
        assert by_case["data_failure"]["in_memory_completed_updates"] == 1
        assert by_case["log_failure"]["error"] == "log update: rank 0: RuntimeError"
        assert by_case["persist_failure"]["error"] == "publish update: rank 0: OSError"
        assert by_case["data_failure"]["error"] == "prepare next data: rank 1: ValueError"
        assert by_case["two_errors"]["error"] == "two callbacks: rank 0: ValueError; rank 1: OSError"
    # Read the actual published/failed temporary files independently. Publication
    # must leave both saved per-rank cursors at1 even though temporary save reached2.
    published = torch.load(tmp_path/"gloo"/"persist_failure.pt", weights_only=True)
    unpublished = torch.load(tmp_path/"gloo"/"persist_failure.unpublished.pt", weights_only=True)
    assert [state["cursor"] for state in published["ranks"]] == [1,1]
    assert [state["cursor"] for state in unpublished["ranks"]] == [2,2]
    for before, after in zip(published["ranks"], unpublished["ranks"]):
        assert not same_tree(before["model"], after["model"])
        assert not same_tree(before["optimizer"], after["optimizer"])
        assert not torch.equal(before["rng"], after["rng"])


def test_harness_refuses_existing_namespace(tmp_path):
    with pytest.raises(FileExistsError):
        run_harness(tmp_path)
