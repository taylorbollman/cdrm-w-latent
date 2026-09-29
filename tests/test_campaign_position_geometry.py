"""Independent mask, support and direct-geometry oracles for diagnostic output."""
import copy
import json

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatBatch
from scripts.olmo_campaign_position_geometry import position_geometry, prediction_and_feedback_masks
from scripts.olmo_lm_common import tree_digests


def captured():
    ids = torch.tensor([[2, 3, 4, 5, 0], [7, 8, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, True, False], [True, True, False, False, False]])
    docs = torch.tensor([[0, 0, 1, 1, -1], [2, 2, -1, -1, -1]])
    ce = valid.clone()
    ce[0, 3] = False
    batch = NextLatBatch(ids, valid, docs, ce, valid.clone(), valid.clone())
    h = torch.arange(1, 21, dtype=torch.float32).reshape(2, 5, 2)
    dy = torch.zeros_like(h)
    dy[0, 0] = torch.tensor([1., -1.])
    dy[0, 2, 0] = 2.  # No direct CE target here; total influence still present.
    return {"batch": vars(batch), "pass_hidden_states": (h.clone(), h.clone()),
            "total_incoming_cotangents": (dy.clone(), dy.clone())}


def test_ce_target_alignment_and_feedback_source_destination_at_doc_boundaries():
    batch = NextLatBatch(**captured()["batch"])
    masks = prediction_and_feedback_masks(batch, document_policy="isolated-v1")
    assert masks["direct_ce"].tolist() == [[True, False, False, False, False], [True, False, False, False, False]]
    assert masks["feedback_source_eligible"].tolist() == [[True, False, True, False, False], [True, False, False, False, False]]
    assert masks["feedback_destination_eligible"].tolist() == [[False, True, False, True, False], [False, True, False, False, False]]
    stream = prediction_and_feedback_masks(batch, document_policy="continuous-stream-v1")
    assert stream["direct_ce"][0, 1] and stream["feedback_source_eligible"][0, 1]
    assert stream["feedback_destination_eligible"][0, 2]
    assert not stream["direct_ce"][0, 2]  # target-mask exclusion still applies.


def test_union_support_includes_reference_only_and_actual_only_and_uses_common_geometry():
    ref = captured()
    actual = copy.deepcopy(ref)
    for h, dy in zip(actual["pass_hidden_states"], actual["total_incoming_cotangents"]):
        h[0, 0] += 0.5
        h[1, 0] += 1.
        h[0, 3] += 100.  # Both-zero support; must remain in all-valid geometry.
        dy[0, 0] = 0     # Reference-only support remains included.
        dy[1, 0, 1] = 3. # Actual-only support remains included.
    pins = tree_digests((actual, ref))
    result = position_geometry([actual], [ref])
    assert tree_digests((actual, ref)) == pins
    json.dumps(result)
    for row in result["records"]:
        assert (row["actual_nonzero_positions"], row["reference_nonzero_positions"], row["union_nonzero_positions"]) == (2, 2, 3)
        groups = row["aggregates"]
        assert groups["all_valid"]["positions"] == 6
        assert groups["union_supported"]["positions"] == groups["zero_in_both"]["positions"] == 3
        mask = torch.tensor([[True, False, True, False, False], [True, False, False, False, False]])
        a = actual["pass_hidden_states"][row["pass"]][mask].double()
        b = ref["pass_hidden_states"][row["pass"]][mask].double()
        want = float((a-b).norm()/b.norm())
        assert groups["union_supported"]["hidden"]["relative_l2"] == pytest.approx(want)
        assert result["plot_summaries"][f"record_0_pass_{row['pass']}"]["supported_hidden_relative_l2"] == pytest.approx(want)
        assert groups["all_valid"]["hidden"]["difference_norm"] > groups["union_supported"]["hidden"]["difference_norm"]
        pos = {(p["row"], p["position"]): p for p in row["positions"]}
        assert not pos[0, 2]["direct_ce"] and pos[0, 2]["union_cotangent_nonzero"]
        assert pos[0, 3]["hidden"]["difference_norm"] > 100
        assert not pos[0, 3]["union_cotangent_nonzero"]
        assert pos[0, 0]["reference_cotangent_nonzero"] and not pos[0, 0]["actual_cotangent_nonzero"]
        assert pos[1, 0]["actual_cotangent_nonzero"] and not pos[1, 0]["reference_cotangent_nonzero"]
    assert not any(p["receives_previous_pass"] for p in result["records"][0]["positions"])
    assert not any(p["feeds_next_pass"] for p in result["records"][-1]["positions"])


def test_empty_support_has_explicit_marker_and_no_reference_claim():
    actual = captured()
    for dy in actual["total_incoming_cotangents"]:
        dy.zero_()
    report = position_geometry([actual])
    assert not report["reference_present"]
    for row in report["records"]:
        assert row["reference_nonzero_positions"] is None
        assert row["aggregates"]["union_supported"] == {
            "positions": 0, "empty": True, "hidden": None, "total_incoming_cotangent": None}
        assert row["aggregates"]["zero_in_both"]["positions"] == 6


@pytest.mark.parametrize("mutation", ["mask", "passes", "cotangent", "shape", "nan"])
def test_incompatible_or_incomplete_captures_rejected(mutation):
    ref, actual = captured(), captured()
    if mutation == "mask": actual["batch"]["ce_mask"][0, 1] = False
    elif mutation == "passes": actual["pass_hidden_states"] = actual["pass_hidden_states"][:1]
    elif mutation == "cotangent": actual["total_incoming_cotangents"] = (None, None)
    elif mutation == "shape": actual["pass_hidden_states"] = tuple(h[..., :1] for h in actual["pass_hidden_states"])
    else: actual["pass_hidden_states"][0][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        position_geometry([actual], [ref])
