"""CPU checks for exact crossed assembly and unchanged NF observation semantics."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from scripts import olmo_campaign_crossed_precision as diagnostic
from scripts.olmo_campaign_adapted_precision import case_health
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_position_geometry import position_geometry
from scripts.olmo_campaign_precision_bridge import capture_passes, forward_geometry
from scripts.olmo_campaign_precision_components import component_backward, record_gradients
from scripts.olmo_campaign_recurrence_precision import arm_contract, fixture_pins, state_pins
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def tiny():
    model, recipe, _, ids, eos = construct(SimpleNamespace(scale="tiny", length=16), "NF", torch.device("cpu"))
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
        length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
    return model, recipe, fixtures


def mutate_as_adapted(model):
    with torch.no_grad():
        # Change both source groups and deliberately give fusion a saved scale
        # which cannot be recovered by recalibrating the selected embedding.
        next(model.backbone.backbone.parameters()).mul_(1.1)
        model.backbone.fusion.state_proj.weight.add_(.025)
        model.backbone.fusion.token_gate.weight.mul_(.8)
        model.backbone.fusion.output_scale.fill_(.123)


def crossed_sources():
    model, recipe, fixtures = tiny()
    snapshots = {"cold": diagnostic.snapshot_components(model)}
    mutate_as_adapted(model)
    snapshots["adapted"] = diagnostic.snapshot_components(model)
    return model, recipe, fixtures, snapshots


def fingerprint(observed):
    return tree_digests([{key: row[key] for key in ("batch", "token_embeddings", "pass_hidden_states")}
                        for row in observed])


def test_complete_crossed_states_keep_parameter_ownership_predictor_rng_and_saved_scale():
    model, recipe, fixtures, snapshots = crossed_sources()
    identities = {name: id(p) for name, p in model.named_parameters()}
    modes = {name: m.training for name, m in model.named_modules()}
    rng, contract, inputs = torch.get_rng_state().clone(), arm_contract(model, recipe), fixture_pins(fixtures)
    native_first = next(iter(model.backbone.backbone.state_dict()))
    # CPU snapshots must be independently allocated even when the model is CPU.
    assert snapshots["adapted"]["tensors"]["backbone"][native_first].data_ptr() != model.backbone.backbone.state_dict()[native_first].data_ptr()
    assert snapshots["cold"]["tensors"]["backbone"][native_first].data_ptr() != snapshots["adapted"]["tensors"]["backbone"][native_first].data_ptr()
    for backbone_origin, fusion_origin in diagnostic.HYBRIDS.values():
        result = diagnostic.assemble_hybrid(model, snapshots,
            backbone_origin=backbone_origin, fusion_origin=fusion_origin)
        assert all(result["checks"].values())
        assert result["complete_fusion_including_output_scale"]
        for group, origin, module in (("backbone", backbone_origin, model.backbone.backbone),
                                      ("fusion", fusion_origin, model.backbone.fusion)):
            assert tree_digests(module.state_dict()) == snapshots[origin]["tensor_pins"][group]
        assert float(model.backbone.fusion.output_scale) == float(snapshots[fusion_origin]["tensors"]["fusion"]["output_scale"])
        assert float(model.backbone.fusion.output_scale) != float(model.backbone.readout_weight.detach().square().mean().sqrt())
        assert {name: id(p) for name, p in model.named_parameters()} == identities
        assert {name: m.training for name, m in model.named_modules()} == modes
        assert arm_contract(model, recipe) == contract and fixture_pins(fixtures) == inputs
        assert torch.equal(rng, torch.get_rng_state())
        assert all(diagnostic.snapshots_unchanged(snapshots).values())
        json.dumps(result)


def test_real_crossed_backward_retains_diagonal_pass_zero_and_exact_cpu_geometry():
    model, recipe, fixtures = tiny()
    snapshots, diagonal = {}, {}
    for origin in ("cold", "adapted"):
        if origin == "adapted":
            model.zero_grad(set_to_none=True)
            mutate_as_adapted(model)
        snapshots[origin] = diagnostic.snapshot_components(model)
        with capture_passes(model, fixtures) as observed:
            component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
        diagonal[origin] = {"forward_fingerprints": fingerprint(observed)}
    inputs, rng = fixture_pins(fixtures), torch.get_rng_state().clone()
    for backbone_origin, fusion_origin in diagnostic.HYBRIDS.values():
        model.zero_grad(set_to_none=True)
        assembly = diagnostic.assemble_hybrid(model, snapshots,
            backbone_origin=backbone_origin, fusion_origin=fusion_origin)
        with capture_passes(model, fixtures) as observed:
            metrics = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
        gradients, _ = record_gradients(model, scope="CPU hybrid assembly check")
        geometry = forward_geometry(observed, observed)
        assert all(case_health(metrics, gradients, geometry, observed, global_fixture_metadata(model, fixtures)).values())
        assert diagnostic.first_pass_identity(fingerprint(observed), diagonal[backbone_origin])
        assert not diagnostic.first_pass_identity(fingerprint(observed), diagonal[fusion_origin])
        assert gradients["groups"]["backbone"]["norm"] > 0
        assert gradients["groups"]["fusion"]["norm"] > 0
        assert gradients["groups"]["predictor"]["norm"] == 0
        positions = position_geometry(observed, observed)
        assert len(positions["records"]) == 8
        for row in positions["records"]:
            assert row["aggregates"]["all_valid"]["hidden"]["difference_norm"] == 0
            assert row["actual_nonzero_positions"] == row["reference_nonzero_positions"] == row["union_nonzero_positions"]
        assert state_pins(model) == assembly["initial_state"]
        assert fixture_pins(fixtures) == inputs and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("corruption", ["missing_scale", "extra_tensor", "shape", "dtype", "nonfinite",
    "bytes", "requires_grad", "predictor_pin", "target_grad", "target_frozen", "invalid_origin"])
def test_assembly_rejects_bad_sources_and_targets_before_copy(corruption):
    model, _, _, snapshots = crossed_sources()
    source = snapshots["adapted"]["tensors"]["fusion"]
    selected = {"backbone_origin": "cold", "fusion_origin": "adapted"}
    if corruption == "missing_scale":
        del source["output_scale"]
    elif corruption == "extra_tensor":
        source["unexpected"] = torch.tensor(1.)
    elif corruption == "shape":
        source["state_proj.weight"] = source["state_proj.weight"][:1]
    elif corruption == "dtype":
        source["state_proj.weight"] = source["state_proj.weight"].to(torch.bfloat16)
    elif corruption == "nonfinite":
        source["output_scale"].fill_(float("nan"))
    elif corruption == "bytes":
        source["output_scale"].add_(.5)
    elif corruption == "requires_grad":
        source["state_proj.weight"].requires_grad_(True)
    elif corruption == "predictor_pin":
        snapshots["cold"]["original_state_pins"]["predictor"] = {}
    elif corruption == "target_grad":
        next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    elif corruption == "target_frozen":
        next(model.parameters()).requires_grad_(False)
    elif corruption == "invalid_origin":
        selected["fusion_origin"] = "cold"
    if corruption not in ("bytes", "predictor_pin", "target_grad", "target_frozen", "invalid_origin"):
        # Independent integrity/layout checks must still reject structurally bad
        # input whose hash metadata has been made self-consistent in this test.
        snapshots["adapted"]["tensor_pins"] = tree_digests(snapshots["adapted"]["tensors"])
    before = state_pins(model)
    with pytest.raises(ValueError):
        diagnostic.assemble_hybrid(model, snapshots, **selected)
    assert state_pins(model) == before


def test_first_pass_control_ignores_later_states_and_cotangents_but_includes_masks_and_embeddings():
    row = {"batch": {"mask": "one"}, "token_embeddings": {"sha256": "embedding"},
           "pass_hidden_states": [{"sha256": "zero"}, {"sha256": "later"}]}
    reference = {"forward_fingerprints": [row]}
    actual = copy.deepcopy([row])
    actual[0]["pass_hidden_states"][1]["sha256"] = "changed later"
    assert diagnostic.first_pass_identity(actual, reference)
    for field in ("batch", "token_embeddings", "pass_hidden_states"):
        bad = copy.deepcopy(actual)
        if field == "pass_hidden_states":
            bad[0][field][0]["sha256"] = "changed"
        else:
            bad[0][field] = {"changed": True}
        assert not diagnostic.first_pass_identity(bad, reference)


def reference_report():
    return {"schema": "olmo-campaign-adapted-precision-v1", "status": "passed_operational_diagnostic",
        "passed": True, "arm": "NF", "objective": "ce", "aggregate_backwards": 2,
        "physical_batch_backwards": 4, "optimizer_updates": 0, "reference_sha256": diagnostic.REFERENCE_SHA,
        "determinism": {"deterministic_algorithms": True}, "sources": {"source.py": "pinned"},
        "cold_contract_checks": {"state": True}, "import_contract_checks": {"state": True},
        "integrity": {"state": True}, "adapted_import": {"checks": {"state": True}},
        "rows": [{"path": p, "objective": "ce", "passed": True, "health": {"finite": True}}
                 for p in diagnostic.PATHS]}


def pinned_report(tmp_path, monkeypatch, report):
    path = tmp_path/"report.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(diagnostic, "ADAPTED_SHA", sha256_file(path))
    return path


def test_adapted_reference_is_byte_pinned_and_requires_current_source_inventory(tmp_path, monkeypatch):
    report = reference_report()
    path = pinned_report(tmp_path, monkeypatch, report)
    assert diagnostic.load_adapted_reference(path, report["sources"]) == report
    with pytest.raises(ValueError, match="source inventory"):
        diagnostic.load_adapted_reference(path, {"source.py": "changed"})
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="immutable pin"):
        diagnostic.load_adapted_reference(path, report["sources"])


@pytest.mark.parametrize("corruption", ["integrity", "import", "determinism", "missing_endpoint", "health", "updates", "matrix"])
def test_adapted_reference_rejects_unfinished_or_incompatible_evidence(tmp_path, monkeypatch, corruption):
    report = reference_report()
    if corruption == "integrity":
        report["integrity"]["state"] = False
    elif corruption == "import":
        report["adapted_import"]["checks"] = {}
    elif corruption == "determinism":
        report["determinism"]["deterministic_algorithms"] = False
    elif corruption == "missing_endpoint":
        report["rows"].pop()
    elif corruption == "health":
        report["rows"][1]["health"]["finite"] = False
    elif corruption == "updates":
        report["optimizer_updates"] = 1
    elif corruption == "matrix":
        report["reference_sha256"] = "different"
    path = pinned_report(tmp_path, monkeypatch, report)
    with pytest.raises(ValueError):
        diagnostic.load_adapted_reference(path, report["sources"])
