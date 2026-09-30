"""Independent evidence checks must expose denominator and displacement errors."""
import copy
import json
import math

import pytest
import torch

from scripts.olmo_topology_audit import (audit_reports, compare_tensors, file_sha256,
    load_report, load_tensor_artifact, select_update, structural_checks)


def report(world=2, precision="fp32"):
    state = {"model": {"digest": "weights"}, "optimizer": {"digest": "Adam"},
             "scheduler": {"completed": 7}, "counters": {"optimizer_updates": 7, "input_tokens": 100, "microbatches": 28}}
    boundary = {"state": state, "canonical_cursor": {"next_update": 7, "next_chunk": 35},
                "rank_rng": {str(i): {"sha256": str(i) * 64} for i in range(world)}}
    final = copy.deepcopy(boundary)
    final["state"]["counters"].update(optimizer_updates=8, input_tokens=140, microbatches=28 + world * math.ceil(5 / (2 * world)))
    final["state"]["scheduler"] = {"completed": 8}
    final["canonical_cursor"] = {"next_update": 8, "next_chunk": 40}
    return {"schema": "olmo-topology-execution-v1", "status": "completed", "scale": "tiny",
        "precision": precision, "world_size": world, "physical_batch_per_rank": 2,
        "common_origin_manifest_sha256": "a" * 64, "active_parameter_names": ["backbone.weight"],
        "boundaries": {"imported": boundary, "prepared": copy.deepcopy(boundary), "final": final},
        "next_update_input": {"logical_update": 7, "input_tokens": 40,
            "counts": {"ce": 7, "latent": 6, "kl": 5}, "lr_used": [.001],
            "rows": [{"key": f"row{i}", "input_sha256": "b" * 64, "masks_sha256": "c" * 64, "jitter_sha256": "d" * 64}
                     for i in range(5)]}}


def test_changed_rank_common_state_exact_allows_only_final_physical_work_difference():
    assert structural_checks(report(2), report(1), "changed_fp32")["passed"]
    changed = report(1)
    changed["boundaries"]["imported"]["state"]["counters"]["microbatches"] = 14
    assert not structural_checks(report(2), changed, "changed_fp32")["passed"]


def test_final_physical_counter_is_independently_checked_against_allocation():
    actual = report(1)
    actual["boundaries"]["final"]["state"]["counters"]["microbatches"] += 1
    result = structural_checks(report(2), actual, "changed_fp32")
    assert result["checks"]["final_logical_counters_exact"]
    assert not result["checks"]["candidate_physical_counter_increment"]
    assert not result["passed"]


@pytest.mark.parametrize("fault", ["jitter", "mask", "count", "lr", "cursor", "adam", "capture", "rng", "final_counter"])
def test_structural_audit_rejects_semantic_or_preparation_drift(fault):
    reference, actual = report(2), report(1)
    if fault in ("jitter", "mask"):
        actual["next_update_input"]["rows"][0]["jitter_sha256" if fault == "jitter" else "masks_sha256"] = "e" * 64
    elif fault == "count":
        actual["next_update_input"]["counts"]["latent"] -= 1
    elif fault == "lr":
        actual["next_update_input"]["lr_used"] = [.1]
    elif fault == "cursor":
        actual["boundaries"]["imported"]["canonical_cursor"]["next_update"] = 0
    elif fault == "adam":
        actual["boundaries"]["imported"]["state"]["optimizer"] = {"digest": "fresh"}
    elif fault == "capture":
        actual["boundaries"]["prepared"]["state"]["model"] = {"digest": "changed"}
    elif fault == "rng":
        actual["boundaries"]["prepared"]["rank_rng"]["0"] = {"sha256": "e" * 64}
    else:
        actual["boundaries"]["final"]["state"]["counters"]["input_tokens"] += 1
    assert not structural_checks(reference, actual, "changed_fp32")["passed"]


def test_same_topology_requires_exact_end_state_rng_and_physical_counters():
    reference, actual = report(2), report(2)
    assert structural_checks(reference, actual, "same_topology")["passed"]
    actual["boundaries"]["final"]["state"]["optimizer"] = {"digest": "changed"}
    assert not structural_checks(reference, actual, "same_topology")["passed"]


def test_update_difference_is_normalized_by_displacement_not_large_weights():
    initial = {"backbone.weight": torch.tensor([1000., 1000.])}
    reference = {"backbone.weight": initial["backbone.weight"] + torch.tensor([.01, -.01])}
    actual = {"backbone.weight": initial["backbone.weight"] + torch.tensor([.02, -.02])}
    result = compare_tensors(reference, actual, ["backbone.weight"], initial=initial, chunk_elements=1)
    assert result["aggregate"]["relative_l2"] == pytest.approx(1.)
    assert result["aggregate"]["cosine"] == pytest.approx(1.)
    assert result["components"]["backbone"]["relative_l2"] == pytest.approx(1.)


def test_tied_aliases_not_double_counted_and_zero_reference_is_explicit():
    weight = torch.ones(3)
    first = {"backbone.weight": weight, "output_alias": weight}
    result = compare_tensors(first, first, ["backbone.weight"])
    assert result["aggregate"]["elements"] == 3
    assert result["aggregate"]["reference_norm"] == pytest.approx(3 ** .5)
    zero = compare_tensors({"w": torch.zeros(3)}, {"w": weight}, ["w"])
    assert zero["aggregate"]["relative_l2"] is None
    assert zero["aggregate"]["zero_reference_with_nonzero_error"]
    with pytest.raises(ValueError, match="Duplicate"):
        compare_tensors(first, first, ["backbone.weight", "backbone.weight"])


def test_nonfinite_and_shape_mismatch_rejected():
    for value in (torch.tensor([float("nan")]), torch.ones(2)):
        with pytest.raises(ValueError):
            compare_tensors({"w": torch.ones(1)}, {"w": value}, ["w"])


def artifacts(tmp_path, report_value, *, final=1.01, gradient=0.1, prefix=""):
    records = {}
    for name, value in (("initial", 1.), ("final", final), ("gradients", gradient)):
        path = tmp_path / (prefix + name + ".pt")
        torch.save({"backbone.weight": torch.tensor([value])}, path)
        records[name] = {"path": path.name, "sha256": file_sha256(path), "key": None}
    report_value["numeric_artifacts"] = records


def test_bf16_large_actual_update_difference_is_measured_without_automatic_pass(tmp_path):
    reference, actual = report(2, "bf16_mixed"), report(1, "bf16_mixed")
    artifacts(tmp_path, reference, prefix="ref-")
    artifacts(tmp_path, actual, final=1.02, gradient=0.2, prefix="act-")
    result = audit_reports(reference, actual, reference_dir=tmp_path, actual_dir=tmp_path, mode="changed_bf16")
    assert result["structural"]["passed"]
    assert result["acceptance_passed"] is None
    assert result["numerical_acceptance"] == "measured_only"
    assert result["adam_displacement"]["aggregate"]["relative_l2"] == pytest.approx(1.)
    assert result["raw_gradients"]["aggregate"]["relative_l2"] == pytest.approx(1.)


def test_changed_fp32_applies_gradient_and_displacement_budgets(tmp_path):
    reference, actual = report(2), report(1)
    artifacts(tmp_path, reference, prefix="ref-")
    artifacts(tmp_path, actual, prefix="act-")
    assert audit_reports(reference, actual, reference_dir=tmp_path, actual_dir=tmp_path, mode="changed_fp32")["acceptance_passed"]
    artifacts(tmp_path, actual, final=1.02, prefix="act-")
    result = audit_reports(reference, actual, reference_dir=tmp_path, actual_dir=tmp_path, mode="changed_fp32")
    assert result["status"] == "failed"
    assert not result["acceptance_passed"]


def test_evidence_hashes_checked_and_tensor_key_selection_supported(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report()))
    assert load_report(path, file_sha256(path))["status"] == "completed"
    with pytest.raises(ValueError, match="SHA256"):
        load_report(path, "0" * 64)
    state = tmp_path / "state.pt"
    torch.save({"model": {"w": torch.ones(1)}}, state)
    spec = {"path": state.name, "sha256": file_sha256(state), "key": "model"}
    assert torch.equal(load_tensor_artifact(spec, tmp_path, {})["w"], torch.ones(1))
    state.write_bytes(state.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_tensor_artifact(spec, tmp_path, {})


def test_selected_update_overlays_boundary_without_relabeling_completion():
    value = report()
    value["update_evidence"] = {"8": {"next_update_input": {"logical_update": 7}}}
    selected = select_update(value, 8)
    assert selected["next_update_input"] == {"logical_update": 7}
    assert selected["selected_update"] == 8
    assert selected["status"] == "completed"
    assert "selected_update" not in value
    with pytest.raises(ValueError, match="absent"):
        select_update(value, 9)
    value["update_evidence"]["8"]["status"] = "running"
    with pytest.raises(ValueError, match="completion"):
        select_update(value, 8)
