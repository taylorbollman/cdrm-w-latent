"""CPU-only fusion scope, gradient oracle and retained NF evidence checks."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.olmo_fbt import FBTConfig, FBTGateProduct
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_precision_bridge import capture_passes
from scripts.olmo_campaign_precision_components import component_backward
from scripts.olmo_campaign_recurrence_precision import arm_contract, first_pass_fingerprints, fixture_pins, state_pins
from scripts.olmo_campaign_fusion_precision import (
    BF16, FP32, CANDIDATE, anchor_comparison, fp32_fusion_forward,
    load_reference, parse_args, reference_anchor, state_contract_checks,
)
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def make_fusion():
    generator = torch.Generator().manual_seed(173)
    return FBTGateProduct(torch.randn(17, 64, generator=generator)*.02, FBTConfig())


def test_override_equals_original_fp32_values_and_gradients_and_leaves_outer_autocast_alone():
    fusion = make_fusion()
    generator = torch.Generator().manual_seed(31)
    hidden = torch.randn(2, 7, 64, generator=generator, requires_grad=True)
    embedding = torch.randn(2, 7, 64, generator=generator, requires_grad=True)
    cotangent = torch.randn(2, 7, 64, generator=generator)
    leaves = (hidden, embedding, *fusion.parameters())
    expected = fusion(hidden, embedding)
    expected_gradients = torch.autograd.grad(expected, leaves, cotangent)
    old_forward = fusion.forward
    old_class_forward = FBTGateProduct.forward
    pins = tree_digests(fusion.state_dict())
    calls = []
    hook = fusion.state_proj.register_forward_pre_hook(lambda module, args: calls.append(torch.is_autocast_enabled("cpu")))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with fp32_fusion_forward(fusion) as evidence:
                result = fusion(hidden, embedding)
                assert torch.is_autocast_enabled("cpu")
                # An unrelated linear operation still executes in outer BF16.
                assert torch.nn.functional.linear(hidden, fusion.state_proj.weight).dtype == torch.bfloat16
            assert torch.is_autocast_enabled("cpu")
        actual_gradients = torch.autograd.grad(result, leaves, cotangent)
    finally:
        hook.remove()
    assert result.dtype == torch.float32
    assert torch.equal(result, expected)
    assert all(torch.equal(actual, reference) for actual, reference in zip(actual_gradients, expected_gradients))
    assert calls == [False]
    assert evidence["restored"] and len(evidence["calls"]) == 1
    assert evidence["calls"][0]["outer_autocast_enabled"]
    assert not evidence["calls"][0]["inner_autocast_enabled"]
    assert evidence["calls"][0]["outer_autocast_restored"]
    assert fusion.forward == old_forward and "forward" not in vars(fusion)
    assert FBTGateProduct.forward is old_class_forward and tree_digests(fusion.state_dict()) == pins


@pytest.mark.parametrize("existing_override", [False, True])
def test_exception_restores_exact_instance_method_and_autocast_state(existing_override):
    fusion = make_fusion()
    if existing_override:
        def fail(hidden, embedding):
            raise RuntimeError("intentional inner error")
        fusion.forward = fail
    original = fusion.forward
    value = torch.ones(1, 2, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises((RuntimeError, ValueError)):
            with fp32_fusion_forward(fusion) as evidence:
                fusion(value if existing_override else value.bfloat16(), value)
        assert torch.is_autocast_enabled("cpu")
    assert evidence["restored"] and fusion.forward == original
    assert ("forward" in vars(fusion)) == existing_override
    with pytest.raises(ValueError, match="masters"):
        with fp32_fusion_forward(make_fusion().bfloat16()):
            pass
    with pytest.raises(TypeError, match="FBTGateProduct"):
        with fp32_fusion_forward(torch.nn.Linear(2, 2)):
            pass


def test_real_nf_candidate_has_six_calls_same_first_pass_and_fixed_state():
    model, recipe, _, ids, eos = construct(SimpleNamespace(scale="tiny", length=16), "NF", torch.device("cpu"))
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
        length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
    state, inputs, rng = state_pins(model), fixture_pins(fixtures), torch.get_rng_state().clone()
    with capture_passes(model, fixtures) as baseline:
        metrics = component_backward(model, recipe, fixtures, precision="bf16_mixed", layout="sparse", objective="ce")
    with fp32_fusion_forward(model.backbone.fusion) as evidence, capture_passes(model, fixtures) as candidate:
        actual = component_backward(model, recipe, fixtures, precision="bf16_mixed", layout="sparse", objective="ce")
    assert actual["counts"] == metrics["counts"] == {"ce": 25, "latent": 25, "kl": 21}
    assert first_pass_fingerprints(candidate) == first_pass_fingerprints(baseline)
    assert tree_digests([r["pass_hidden_states"][1:] for r in candidate]) != tree_digests([
        r["pass_hidden_states"][1:] for r in baseline])
    assert evidence["restored"] and len(evidence["calls"]) == 6
    assert all(row["outer_autocast_enabled"] and not row["inner_autocast_enabled"]
               and row["output_dtype"] == "torch.float32" for row in evidence["calls"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.predictor.parameters())
    assert state_pins(model) == state and fixture_pins(fixtures) == inputs
    assert torch.equal(torch.get_rng_state(), rng)
    # Exact source contract accepts JSON's tuple-to-list conversion but rejects
    # a real state change, such as changed fusion parameters or mode selection.
    current = {"contract": arm_contract(model, recipe), "initial_state": state, "fixture_inputs": inputs}
    prior = json.loads(json.dumps(current))
    assert all(state_contract_checks(current, prior).values())
    prior["contract"]["mode"]["rt_mode"]["selected_layers"] = [0]
    assert not state_contract_checks(current, prior)["contract_exact"]


def payload():
    rows = []
    for path in (FP32, BF16):
        gradients = {"finite": True, "groups": {"backbone": {"norm": 1.}},
            "missing_active_gradients": [], "originally_missing_zero_materialized": [], "participation_intact": True}
        if path == BF16:
            gradients.update(geometry={"all": {"relative_l2": .6}}, comparison={"relative_l2": .6,
                "parameter_rows": {"p": {"max_abs": .1}}, "all_parameters_close": False,
                "atol": 3e-5, "rtol": 3e-4})
        rows.append({"arm": "NF", "path": path, "objective": "ce", "passed": True,
            "metrics": {"objective": 3.}, "forward_fingerprints": {"hidden": "hash"},
            "forward_vs_fp32": [{"hidden": {"relative_l2": .01}}],
            "gradients" if path == FP32 else "gradients_vs_fp32": gradients})
    return {"schema": "olmo-campaign-recurrence-precision-v1", "status": "passed_operational_diagnostic", "passed": True,
        "sources": {"source.py": "a"*64}, "determinism": {"deterministic_algorithms": True},
        "integrity": {"sources_unchanged": True}, "arms": {"NF": {
            "integrity": {"parameters_and_buffers_unchanged": True}, "shared_state_checks": {"common": True}}}, "rows": rows}


def write_reference(tmp_path, report):
    path = tmp_path/"report.json"
    path.write_text(json.dumps(report))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_nf_reference_and_both_endpoint_geometries(tmp_path):
    report = payload()
    path, sha = write_reference(tmp_path, report)
    assert load_reference(path, sha, report["sources"]) == report
    with pytest.raises(ValueError, match="SHA256"):
        load_reference(path, "0"*64, report["sources"])
    with pytest.raises(ValueError, match="source pins"):
        load_reference(path, sha, {"source.py": "b"*64})
    for precision in (FP32, BF16):
        prior = reference_anchor(report, precision)
        gradients = copy.deepcopy(prior["gradients" if precision == FP32 else "gradients_vs_fp32"])
        def compare():
            return anchor_comparison(prior["metrics"], prior["forward_fingerprints"], gradients,
                                     prior["forward_vs_fp32"], prior)
        assert compare()["passed"] and "Prior NF" in compare()["scope"]
        if precision == BF16:
            gradients["geometry"]["all"]["relative_l2"] += .001
        else:
            gradients["groups"]["backbone"]["norm"] += .001
        assert not compare()["passed"]
    with pytest.raises(ValueError, match="no previous"):
        reference_anchor(report, CANDIDATE)


@pytest.mark.parametrize("mutation", ["failed", "nondeterministic", "nf_state_changed", "nf_unmatched", "duplicate", "wrong_arm"])
def test_reference_rejects_missing_integrity_or_nf_endpoint(tmp_path, mutation):
    report = payload()
    if mutation == "failed": report["passed"] = False
    elif mutation == "nondeterministic": report["determinism"]["deterministic_algorithms"] = False
    elif mutation == "nf_state_changed": report["arms"]["NF"]["integrity"]["parameters_and_buffers_unchanged"] = False
    elif mutation == "nf_unmatched": report["arms"]["NF"]["shared_state_checks"]["common"] = False
    elif mutation == "duplicate": report["rows"].append(copy.deepcopy(report["rows"][0]))
    else: report["rows"][0]["arm"] = "NFR"
    path, sha = write_reference(tmp_path, report)
    with pytest.raises(ValueError):
        load_reference(path, sha, report["sources"])


def test_cli_rejects_expanding_scope_or_missing_reference_pin():
    args = ["--reference-report", "prior.json", "--reference-sha256", "a"*64, "--output-dir", "/tmp/fusion"]
    assert parse_args(args).reference_sha256 == "a"*64
    for extra in (["--arm", "NFR"], ["--steps", "2"], ["--length", "1024"], ["--batch-size", "8"]):
        with pytest.raises(SystemExit):
            parse_args(args+extra)
    with pytest.raises(SystemExit):
        parse_args(["--reference-report", "prior.json", "--output-dir", "/tmp/fusion"])
