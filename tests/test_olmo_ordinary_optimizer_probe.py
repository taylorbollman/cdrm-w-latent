"""Explicit tiny CPU fixtures for optimizer-probe wiring, not GPU clearance."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts import olmo_ordinary_optimizer_probe as probe


def model_with_gradients():
    model = torch.nn.Linear(8, 4, bias=True)
    # Duplicate registration represents tied ownership: optimizer must own once.
    model.alias = model.weight
    for parameter in model.parameters():
        parameter.grad = torch.linspace(-.5, .5, parameter.numel()).reshape_as(parameter)
    return model


def test_update_delta_gate_cannot_be_hidden_by_small_whole_weight_error():
    initial = torch.ones(32)
    reference = initial + 1e-5
    candidate = reference + 5e-7
    moment = probe.tensor_metrics(torch.ones(32), torch.ones(32))
    weights = probe.tensor_metrics(candidate, reference)
    updates = probe.tensor_metrics(candidate, reference, initial=initial)
    assert weights["relative_l2"] < probe.BUDGETS["weight_tensor_relative_l2"]
    assert updates["relative_l2"] > probe.BUDGETS["cumulative_update_global_relative_l2"]
    result = probe.comparison_screen(moments={"m": moment}, weights={"w": weights},
        updates={"w": updates}, semantic_matches=True)
    assert not result["passed"]


def test_chunked_metrics_preserve_small_fp32_update_quantization():
    initial = torch.linspace(.01, 2., 37)
    reference = initial + 1e-5
    candidate = reference.clone()
    candidate[4] = torch.nextafter(candidate[4], torch.full_like(candidate[4], float("inf")))
    whole = probe.tensor_metrics(candidate, reference, initial=initial)
    chunks = probe.tensor_metrics(candidate, reference, initial=initial, chunk_size=3)
    for name in whole:
        assert chunks[name] == pytest.approx(whole[name])
    assert whole["max_abs"] > 0
    assert not whole["bitwise_equal"]


def test_nonzero_error_against_zero_moment_is_not_hidden():
    row = probe.tensor_metrics(torch.tensor([1e-30]), torch.zeros(1))
    assert row["relative_l2"] == float("inf")
    assert row["max_relative"] == float("inf")


@pytest.mark.parametrize("category", ["moments", "weights", "updates"])
def test_nonfinite_state_fails_even_with_other_states_equal(category):
    exact = probe.tensor_metrics(torch.ones(3), torch.ones(3))
    args = {"moments": {"m": deepcopy(exact)}, "weights": {"w": deepcopy(exact)},
            "updates": {"w": deepcopy(exact)}, "semantic_matches": True}
    next(iter(args[category].values()))["finite"] = False
    assert not probe.comparison_screen(**args)["passed"]


def test_public_probe_rejects_cpu_before_backward_or_gpu_environment_lookup(monkeypatch):
    def forbidden(*args):
        raise AssertionError("CPU must be rejected before backend execution")
    monkeypatch.setattr(probe, "require_container_gpu", forbidden)
    plan = SimpleNamespace(model=model_with_gradients(), initialize_gradients=forbidden)
    with pytest.raises(ValueError, match="no CPU fallback"):
        probe.compare_fixed_gradients(plan, report={}, persist=forbidden)


def test_fixed_gradient_probe_restores_values_addresses_and_counts_all_updates():
    model = model_with_gradients()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    addresses = {name: (p.data_ptr(), p.grad.data_ptr()) for name, p in model.named_parameters()}
    report = {"physical_optimizer_updates": 7}
    saved = []
    # Explicit CPU orchestration fixture using PyTorch's available CPU AdamW
    # implementations. This does not invoke the public GPU-only entry point.
    result = probe._compare_from_fixed_gradients(model, report=report,
        persist=lambda: saved.append(report["physical_optimizer_updates"]))
    assert result["passed"]
    assert report["physical_optimizer_updates"] == 13
    assert report["optimizer_probe_state_restored"]
    assert saved == [8, 9, 10, 11, 12, 13, 13]
    assert result["physical_optimizer_updates"] == 6
    assert result["optimizer_groups_match_except_fused"]
    assert len(result["arms"]) == 2
    assert result["arms"][0]["ownership"] == result["arms"][1]["ownership"]
    assert sum(len(group) for group in result["arms"][0]["ownership"]) == len(original)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name])
        assert torch.equal(parameter.grad, gradients[name])
        assert (parameter.data_ptr(), parameter.grad.data_ptr()) == addresses[name]


def test_probe_restores_and_records_update_before_scheduler_failure():
    model = model_with_gradients()
    initial = {name: p.clone() for name, p in model.named_parameters()}
    raw = {name: p.grad.clone() for name, p in model.named_parameters()}
    report = {"physical_optimizer_updates": 0}

    def factory(model, *, fused):
        optimizer, scheduler = probe.make_optimizer(model, fused=fused)
        def fail():
            raise RuntimeError("intentional scheduler failure")
        scheduler.step = fail
        return optimizer, scheduler

    with pytest.raises(RuntimeError, match="intentional scheduler failure"):
        probe._compare_from_fixed_gradients(model, report=report, persist=lambda: None,
                                           optimizer_factory=factory)
    assert report["physical_optimizer_updates"] == 1
    assert report["optimizer_probe_state_restored"]
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, initial[name])
        assert torch.equal(parameter.grad, raw[name])


def test_probe_rejects_hidden_optimizer_hyperparameter_change():
    def factory(model, *, fused):
        optimizer, scheduler = probe.make_optimizer(model, fused=fused)
        if fused:
            # A tiny semantic change must fail independently of numerical budgets.
            for group in optimizer.param_groups:
                group["eps"] = 1.0000001e-8
        return optimizer, scheduler

    report = {}
    result = probe._compare_from_fixed_gradients(model_with_gradients(),
        report=report, persist=lambda: None, optimizer_factory=factory)
    assert not result["optimizer_groups_match_except_fused"]
    assert not result["semantic_state_matches"]
    assert not result["passed"]
    assert report["physical_optimizer_updates"] == 6
    assert report["optimizer_probe_state_restored"]


def test_factory_keeps_scheduler_hyperparameters_and_legacy_default():
    for fused in (False, True):
        optimizer, scheduler = probe.make_optimizer(model_with_gradients(), fused=fused)
        assert optimizer.defaults["fused"] is (True if fused else None)
        assert optimizer.defaults["lr"] == 1e-5
        assert optimizer.defaults["betas"] == (.9, .95)
        assert optimizer.defaults["eps"] == 1e-8
        assert optimizer.param_groups[0]["lr"] == 5e-6
        assert scheduler._cdrm_warmup_updates == 2
