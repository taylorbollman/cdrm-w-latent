import copy
import json

import numpy as np
import pytest
import torch

from scripts import rt_nextlat_embedding_resume_audit as audit


def packet():
    return {"completed_updates": 452, "model": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer": {"state": {0: {"moment": torch.tensor([0.1])}}},
            "rng": {"numpy": ("MT19937", np.array([1, 2], dtype=np.uint32))},
            "examples_seen": {"a5": 1157120, "fuzzy": 1157120},
            "next_cursors": {
                "a5": {"absolute_example_offset": 1157120, "epoch": 1, "position": 357120},
                "fuzzy": {"absolute_example_offset": 1157120, "epoch": 90, "position": 5120}},
            "order_chains": {"a5": "abc", "fuzzy": "def"},
            "contract": {}, "initialization": {}, "optimizer_parameter_names": ["weight"],
            "schema": "test", "finite_state": {"parameters_finite_fp32": True}}


@pytest.mark.parametrize("field", ["model", "optimizer", "rng", "cursor", "sequence_type"])
def test_whole_packet_differences_detect_hidden_resume_changes(field):
    a, b = packet(), packet()
    if field == "model": b["model"]["weight"][0] = 9
    if field == "optimizer": b["optimizer"]["state"][0]["moment"][0] = 9
    if field == "rng": b["rng"]["numpy"][1][0] = 9
    if field == "cursor": b["next_cursors"]["a5"]["position"] += 1
    if field == "sequence_type": b["rng"]["numpy"] = list(b["rng"]["numpy"])
    assert audit.differences(a, b)


def test_whole_packet_exact_resave_and_recognized_evidence(tmp_path):
    parent, child, output = [tmp_path / name for name in
                             ("parent.pt", "child.pt", "recovery-boundary-state-audit.json")]
    state = packet()
    torch.save(state, parent)
    torch.save(copy.deepcopy(state), child)
    proof = audit.audit(parent, child, expected_update=452, parent_sha256=audit.reporting.sha(parent), output=output)
    assert proof["schema"] == "rt-nextlat-a5-fuzzy-recovery-boundary-state-audit-v1"
    assert proof["passed"] and proof["entire_packet_exact"] and proof["differences"] == []
    assert all(proof["field_equality"].values())
    assert json.loads(output.read_text())["matching_examples_seen"] == state["examples_seen"]
    parent_record = {**proof["checkpoints"]["parent"], "verified_local_path": str(parent)}
    lineage = {"checkpoints": {452: parent_record}, "tasks": ["a5", "fuzzy"],
               "report": {"contract": {"batch_per_task": 2560,
                          "streams": {"a5": {"train_rows": 800000}, "fuzzy": {"train_rows": 12800}}}},
               "history": [{"order_chains": state["order_chains"]}]}
    accepted = audit.reporting.boundary_state_proof(
        tmp_path, lineage, proof["checkpoints"]["child"], child, 452)
    assert accepted["receipt"]["entire_packet_exact"] is True
    with pytest.raises(ValueError, match="already exists"):
        audit.audit(parent, child, expected_update=452, parent_sha256=audit.reporting.sha(parent), output=output)


@pytest.mark.parametrize("problem", ["parent_sha", "boundary", "state"])
def test_bad_parent_or_boundary_rejected(tmp_path, problem):
    parent, child, output = [tmp_path / name for name in ("parent.pt", "child.pt", "proof.json")]
    a, b = packet(), packet()
    if problem == "state": b["model"]["weight"][0] = 9
    if problem == "boundary": b["completed_updates"] = 453
    torch.save(a, parent)
    torch.save(b, child)
    expected = "bad" if problem == "parent_sha" else audit.reporting.sha(parent)
    with pytest.raises(ValueError):
        audit.audit(parent, child, expected_update=452, parent_sha256=expected, output=output)
    if problem == "state":
        assert json.loads(output.read_text())["passed"] is False
