"""Read-only position observations from captured finite-feedback CE backwards.

No model execution, new autograd operations, loss changes or support filtering
of the actual backward. Active feedback flags assume this diagnostic's beta=1.
"""
from __future__ import annotations

import torch

from cdrm.pretrained.document_policy import feedback_eligibility
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks
from scripts.olmo_campaign_precision_bridge import tensor_geometry
from scripts.olmo_lm_common import tree_digests


def prediction_and_feedback_masks(batch, *, document_policy):
    """CE is indexed by target slot; return masks aligned to hidden positions."""
    ce = torch.zeros_like(batch.valid_mask)
    ce[:, :-1] = build_nextlat_masks(batch, document_policy=document_policy)["ce"]
    eligible = feedback_eligibility(batch.valid_mask, batch.document_ids, document_policy)
    source, destination = torch.zeros_like(ce), torch.zeros_like(ce)
    source[:, :-1], destination[:, 1:] = eligible, eligible
    return {"direct_ce": ce, "feedback_source_eligible": source,
            "feedback_destination_eligible": destination}


def _state_tensors(record):
    hidden = record["pass_hidden_states"]
    cotangents = record["total_incoming_cotangents"]
    if not hidden or len(hidden) != len(cotangents):
        raise ValueError("Position observations need complete matching passes/cotangents")
    shape = record["batch"]["input_ids"].shape
    for h, dy in zip(hidden, cotangents):
        if (not isinstance(h, torch.Tensor) or not isinstance(dy, torch.Tensor)
                or h.ndim != 3 or h.shape[:2] != shape or h.shape != dy.shape
                or h.device.type != "cpu" or dy.device.type != "cpu"
                or not h.is_floating_point() or not dy.is_floating_point()
                or not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(dy).all())):
            raise ValueError("Need finite CPU hidden/cotangent captures with [B,T,D] shapes")
    return hidden, cotangents


def _aggregate(mask, hidden, cotangent, reference_hidden, reference_cotangent):
    count = int(mask.sum())
    if not count:
        return {"positions": 0, "empty": True, "hidden": None, "total_incoming_cotangent": None}
    return {"positions": count, "empty": False,
        "hidden": tensor_geometry(hidden[mask], None if reference_hidden is None else reference_hidden[mask]),
        "total_incoming_cotangent": tensor_geometry(cotangent[mask], None if reference_cotangent is None else reference_cotangent[mask])}


@torch.no_grad()
def position_geometry(records, reference=None, *, document_policy="isolated-v1"):
    """Compare on common masks; exact-zero support is observational, not causal.

With a reference, union support includes a position nonzero in EITHER actual or
reference incoming cotangent. Without one, support describes this case alone.
Hidden positions receive direct CE for the following target; they may also
participate indirectly through a later feedback pass. Full cotangents include
those later paths. The report never treats a missing direct target as zero total
loss influence and never removes positions from the underlying backward.
"""
    if not records or (reference is not None and len(reference) != len(records)):
        raise ValueError("Require nonempty matching physical-record captures")
    rows = []
    for record_index, record in enumerate(records):
        batch = NextLatBatch(**record["batch"])
        masks = prediction_and_feedback_masks(batch, document_policy=document_policy)
        valid = batch.valid_mask
        if valid.device.type != "cpu":
            raise ValueError("Position observations require CPU captures")
        hidden_states, cotangents = _state_tensors(record)
        wanted = None if reference is None else reference[record_index]
        if wanted is not None:
            if tree_digests(record["batch"]) != tree_digests(wanted["batch"]):
                raise ValueError("Position comparison uses different tokens/masks/documents")
            reference_states, reference_cotangents = _state_tensors(wanted)
            if len(reference_states) != len(hidden_states):
                raise ValueError("Position comparison uses different pass counts")
        for pass_index, (hidden, cotangent) in enumerate(zip(hidden_states, cotangents)):
            rh = None if wanted is None else reference_states[pass_index]
            rc = None if wanted is None else reference_cotangents[pass_index]
            if rh is not None and (rh.shape != hidden.shape or rc.shape != cotangent.shape):
                raise ValueError("Position comparison uses different hidden/cotangent shapes")
            actual_support = valid & cotangent.ne(0).any(-1)
            reference_support = torch.zeros_like(valid) if rc is None else valid & rc.ne(0).any(-1)
            union_support = actual_support | reference_support
            selections = {"all_valid": valid, "union_supported": union_support,
                "zero_in_both": valid & ~union_support,
                "direct_ce": masks["direct_ce"], "no_direct_ce": valid & ~masks["direct_ce"]}
            positions = []
            for row, position in valid.nonzero(as_tuple=False).tolist():
                flags = {name: bool(mask[row, position]) for name, mask in masks.items()}
                positions.append({"row": row, "position": position,
                    "document_id": int(batch.document_ids[row, position]), **flags,
                    "receives_previous_pass": pass_index > 0 and flags["feedback_destination_eligible"],
                    "feeds_next_pass": pass_index < len(hidden_states)-1 and flags["feedback_source_eligible"],
                    "actual_cotangent_nonzero": bool(actual_support[row, position]),
                    "reference_cotangent_nonzero": None if rc is None else bool(reference_support[row, position]),
                    "union_cotangent_nonzero": bool(union_support[row, position]),
                    "hidden": tensor_geometry(hidden[row, position], None if rh is None else rh[row, position]),
                    "total_incoming_cotangent": tensor_geometry(cotangent[row, position], None if rc is None else rc[row, position])})
            rows.append({"record": record_index, "pass": pass_index,
                "actual_nonzero_positions": int(actual_support.sum()),
                "reference_nonzero_positions": None if rc is None else int(reference_support.sum()),
                "union_nonzero_positions": int(union_support.sum()),
                "aggregates": {name: _aggregate(mask, hidden, cotangent, rh, rc) for name, mask in selections.items()},
                "positions": positions})
    # Existing W&B scalar flattening deliberately skips lists. Keep a small
    # mapping of graphable summaries; detailed positions stay in local evidence.
    summaries = {}
    for row in rows:
        groups = row["aggregates"]
        summaries[f"record_{row['record']}_pass_{row['pass']}"] = {
            "valid_positions": groups["all_valid"]["positions"],
            "union_supported_positions": groups["union_supported"]["positions"],
            "all_hidden_relative_l2": (groups["all_valid"]["hidden"] or {}).get("relative_l2"),
            "supported_hidden_relative_l2": (groups["union_supported"]["hidden"] or {}).get("relative_l2"),
            "zero_support_hidden_relative_l2": (groups["zero_in_both"]["hidden"] or {}).get("relative_l2"),
            "all_cotangent_relative_l2": (groups["all_valid"]["total_incoming_cotangent"] or {}).get("relative_l2"),
        }
    return {"schema": "olmo-campaign-position-geometry-v1", "document_policy": document_policy,
        "reference_present": reference is not None,
        "support_definition": "valid and any exactly nonzero cotangent element in either precision; actual only when reference absent",
        "qualification": "Zero observed cotangent is not proof of harmlessness; direct CE masks are not total loss influence; active feedback flags assume beta1",
        "records": rows, "plot_summaries": summaries}
