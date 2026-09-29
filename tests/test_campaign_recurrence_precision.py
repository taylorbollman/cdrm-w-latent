"""CPU checks for fixed-arm CE semantics and retained evidence contracts."""
import copy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_precision_bridge import capture_passes, forward_geometry
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, component_backward, record_gradients
from scripts.olmo_campaign_recurrence_precision import (
    ARMS, BF16, FP32, anchor_comparison, arm_contract, first_pass_fingerprints,
    fixture_pins, gradient_norm_summary, load_reference, parse_args, reference_anchor,
    shared_arm_checks, state_pins,
)
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def tiny_arm(arm):
    model, recipe, checkpoint, ids, eos = construct(SimpleNamespace(scale="tiny", length=16), arm, torch.device("cpu"))
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
                length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
    info = {"contract": arm_contract(model, recipe), "source_checkpoint": checkpoint,
            "model_config": model.backbone.backbone.config.to_dict(), "nextlat_config": model.config.to_dict(),
            "recipe": recipe.to_dict(), "initial_state": state_pins(model),
            "production_runtime_flags": {key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS},
            "fixture_inputs": fixture_pins(fixtures), "fixture_metadata": global_fixture_metadata(model, fixtures)}
    return model, recipe, fixtures, info


def test_real_four_arm_controls_and_first_passes_match_while_cotangents_can_differ():
    infos, first_passes, incoming = {}, {}, {}
    for arm in ARMS:
        model, recipe, fixtures, info = tiny_arm(arm)
        infos[arm] = info
        assert all(shared_arm_checks(info, infos["N"], infos.get("NF")).values())
        rng = torch.get_rng_state().clone()
        with capture_passes(model, fixtures) as observed:
            metrics = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
        assert metrics["counts"] == {"ce": 25, "latent": 25, "kl": 21}
        assert metrics["input_tokens"] == 29 and metrics["microbatches"] == 2
        assert len(observed) == 2
        assert all(len(row["pass_hidden_states"]) == (4 if recipe.feedback else 1) for row in observed)
        assert all(g is not None for row in observed for g in row["total_incoming_cotangents"])
        assert all(row["hidden"]["finite"] and row["total_incoming_cotangent"]["finite"]
                   for row in forward_geometry(observed))
        # CE-only still traverses predictor branches, yielding materialized zeros.
        assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.predictor.parameters())
        assert all((p.grad is not None) == recipe.feedback for p in model.backbone.fusion.parameters())
        gradients, _ = record_gradients(model, scope="CPU health only")
        assert gradients["participation_intact"] and gradients["finite"]
        assert gradient_norm_summary(gradients)["all"] >= gradients["groups"]["backbone"]["norm"] > 0
        assert state_pins(model) == info["initial_state"]
        assert fixture_pins(fixtures) == info["fixture_inputs"] and torch.equal(torch.get_rng_state(), rng)
        first_passes[arm] = first_pass_fingerprints(observed)
        incoming[arm] = tree_digests([row["total_incoming_cotangents"][0] for row in observed])
    assert first_passes["N"] == first_passes["NF"]
    assert first_passes["NR"] == first_passes["NFR"]
    assert first_passes["N"] != first_passes["NR"]
    assert incoming["N"] != incoming["NF"] and incoming["NR"] != incoming["NFR"]
    assert infos["N"]["contract"]["parameter_inventory"]["fusion"]["trainable_parameters"] == 0
    assert infos["NF"]["contract"]["parameter_inventory"]["fusion"]["trainable_parameters"] > 0


@pytest.mark.parametrize("arm", ARMS)
def test_ce_only_matches_independent_literal_weighted_loss_backward(arm):
    model, recipe, fixtures, info = tiny_arm(arm)
    actual = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
    actual_gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                        if p.requires_grad and not name.startswith("predictor.")}
    model.zero_grad(set_to_none=True)
    direct_objective = 0.0
    expected_weights = (.5, 1/6, 1/6, 1/6) if "F" in arm else (1.,)
    assert tuple(info["contract"]["ce_pass_weights"]) == expected_weights
    for batches, noises in fixtures:
        for batch, noise in zip(batches, noises):
            result = model.loss_sums(batch, backbone_kwargs={"mode": recipe.mode(),
                "feedback_noise": noise, "right_padded_causal": True})
            assert result.pass_coefficients == expected_weights
            objective = sum(weight*loss.sums["ce"] for weight, loss in zip(expected_weights, result.pass_losses))/25
            objective.backward()
            direct_objective += float(objective.detach())
    assert actual["objective"] == direct_objective
    # The independent objective omits the zero auxiliary graph; active backbone
    # and fusion gradients remain numerically identical.
    for name, p in model.named_parameters():
        if name in actual_gradients:
            torch.testing.assert_close(p.grad, actual_gradients[name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.predictor.parameters())


def test_shared_state_checks_detect_buffer_noise_and_mask_changes():
    _, _, _, reference = tiny_arm("NF")
    model, recipe, fixtures, info = tiny_arm("NFR")
    assert all(shared_arm_checks(info, reference, reference).values())
    with torch.no_grad():
        model.backbone.fusion.output_scale.add_(.001)
    changed = {**info, "initial_state": state_pins(model)}
    assert not shared_arm_checks(changed, reference, reference)["fusion_state_exact"]
    fixtures[0][1][0][0].view(-1)[0] += .01
    changed = {**info, "fixture_inputs": fixture_pins(fixtures)}
    assert not shared_arm_checks(changed, reference, reference)["common_feedback_noise_exact"]
    fixtures[0][0][0].ce_mask[0, 1] = False
    changed = {**info, "fixture_inputs": fixture_pins(fixtures)}
    assert not shared_arm_checks(changed, reference, reference)["tokens_masks_exact"]
    with pytest.raises(ValueError, match="active"):
        arm_contract(model, replace(recipe, predictor_seed=recipe.predictor_seed+1))


def reference_payload():
    rows = []
    for path in (FP32, BF16):
        gradients = {"finite": True, "groups": {"backbone": {"norm": 1.}},
                     "missing_active_gradients": [], "originally_missing_zero_materialized": [], "participation_intact": True}
        if path == BF16:
            gradients.update(geometry={"all": {"relative_l2": .1}}, comparison={
                "relative_l2": .1, "parameter_rows": {"p": {"max_abs": .1}},
                "all_parameters_close": False, "atol": 3e-5, "rtol": 3e-4, "scope": "old scope"})
        rows.append({"path": path, "objective": "ce", "passed": True,
                     "metrics": {"objective": 3.}, "forward_fingerprints": {"hidden": "hash"},
                     "forward_vs_fp32": [{"hidden": {"relative_l2": .01}}],
                     "gradients" if path == FP32 else "gradients_vs_fp32": gradients})
    return {"schema": "olmo-campaign-precision-bridge-v1", "status": "passed_operational_diagnostic", "passed": True,
            "sources": {"source.py": "a"*64}, "determinism": {"deterministic_algorithms": True},
            "integrity": {"weights_unchanged": True, "sources_unchanged": True}, "rows": rows}


def write_reference(tmp_path, payload):
    path = tmp_path/"report.json"
    path.write_text(json.dumps(payload))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_pins_both_precision_endpoints_and_live_sources(tmp_path):
    payload = reference_payload()
    path, sha = write_reference(tmp_path, payload)
    assert load_reference(path, sha, payload["sources"]) == payload
    with pytest.raises(ValueError, match="SHA256"):
        load_reference(path, "0"*64, payload["sources"])
    with pytest.raises(ValueError, match="source pins"):
        load_reference(path, sha, {"source.py": "b"*64})


@pytest.mark.parametrize("mutation", ["failed", "nondeterministic", "state_changed", "missing_fp32", "duplicate_bf16"])
def test_invalid_prior_report_rejected(tmp_path, mutation):
    payload = reference_payload()
    if mutation == "failed": payload["passed"] = False
    elif mutation == "nondeterministic": payload["determinism"]["deterministic_algorithms"] = False
    elif mutation == "state_changed": payload["integrity"]["weights_unchanged"] = False
    elif mutation == "missing_fp32": payload["rows"].pop(0)
    else: payload["rows"].append(copy.deepcopy(payload["rows"][1]))
    path, sha = write_reference(tmp_path, payload)
    with pytest.raises(ValueError):
        load_reference(path, sha, payload["sources"])


@pytest.mark.parametrize("path", [FP32, BF16])
def test_nfr_anchor_checks_exact_geometry_not_only_finite_or_old_budget(path):
    previous = reference_anchor(reference_payload(), path)
    key = "gradients" if path == FP32 else "gradients_vs_fp32"
    gradients = copy.deepcopy(previous[key])
    if path == BF16:
        gradients["comparison"]["scope"] = "new within-arm scope"
    def compare():
        return anchor_comparison(previous["metrics"], previous["forward_fingerprints"], gradients,
                                 previous["forward_vs_fp32"], previous)
    assert compare()["passed"]
    assert "original full gradient vectors were not retained" in compare()["scope"]
    if path == BF16:
        # Both values may fail old budgets; changing the retained error still fails.
        gradients["geometry"]["all"]["relative_l2"] += .001
        assert not compare()["passed"]
    else:
        gradients["groups"]["backbone"]["norm"] += .001
        assert not compare()["passed"]


def test_cli_cannot_expand_into_training_or_unpinned_conditions():
    args = ["--reference-report", "prior.json", "--reference-sha256", "a"*64, "--output-dir", "/tmp/diagnostic"]
    assert parse_args(args).reference_sha256 == "a"*64
    for extra in (["--steps", "2"], ["--length", "1024"], ["--arms", "NFR"],
                  ["--batch-size", "8"], ["--objective", "combined"]):
        with pytest.raises(SystemExit):
            parse_args(args+extra)
    with pytest.raises(SystemExit):
        parse_args(["--reference-report", "prior.json", "--reference-sha256", "A"*64, "--output-dir", "/tmp/diagnostic"])
