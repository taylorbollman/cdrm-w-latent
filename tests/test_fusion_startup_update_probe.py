"""Tiny actual Adam counterfactuals, exact CE oracle and state restoration."""
import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.lm_training import TrainingCounters, _rng_state, _restore_rng
from scripts import olmo_fusion_startup_update_probe as probe
from scripts import olmo_fusion_startup_train as train
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_precision_components import component_backward, RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, BF16
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def tiny():
    model, recipe, _, ids, eos = construct(SimpleNamespace(scale="tiny", length=16), "NF", torch.device("cpu"))
    model.backbone.backbone.attention_precision = "mixed"
    original = {name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS}
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
        length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
    train.freeze_for_startup(model)
    return model, recipe, fixtures, original


def populated():
    model, recipe, fixtures, original = tiny()
    optimizer, scheduler = train.build_optimizer(model)
    batches = tuple(batch for bs,_ in fixtures for batch in bs)
    noises = tuple(noise for _,ns in fixtures for noise in ns)
    train.configure_full_fp32(model)
    train.train_update(model, recipe, batches, noises, optimizer, scheduler, TrainingCounters(), ce_targets=25)
    # CPU fixture uses real moments from its single update, without claiming128.
    saved = copy.deepcopy({"fusion":model.backbone.fusion.state_dict(), "optimizer":optimizer.state_dict(),
                           "scheduler":scheduler.state_dict(), "rng":_rng_state(None)})
    def restore():
        model.backbone.fusion.load_state_dict(saved["fusion"])
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        scheduler.load_state_dict(copy.deepcopy(saved["scheduler"]))
        _restore_rng(copy.deepcopy(saved["rng"]), None)
        return {"counters":{"optimizer_updates":1}, "data_cursor":{"next_update":1}}
    return model, recipe, fixtures, original, optimizer, scheduler, saved, restore


@pytest.mark.parametrize("path,precision", [(FP32,"fp32"), (BF16,"bf16_mixed")])
def test_ce_gradient_matches_original_zero_auxiliary_objective(path, precision):
    model, recipe, fixtures, original = tiny()
    probe.configure_path(model, original, path)
    with probe.sdpa_kernel(probe.SDPBackend.MATH):
        oracle = component_backward(model, recipe, fixtures, precision=precision, layout="sparse", objective="ce")
    gradients = probe.fusion_values(model, gradients=True)
    frozen, rng = train.frozen_state_pins(model), tree_digests(_rng_state(None))
    actual = probe.ce_gradients(model, recipe, fixtures, path=path, original_flags=original)
    assert actual["objective"] == oracle["objective"] and actual["ce_targets"] == 25
    assert tree_digests(probe.fusion_values(model, gradients=True)) == tree_digests(gradients)
    assert train.frozen_state_pins(model) == frozen and tree_digests(_rng_state(None)) == rng
    assert all(parameter.grad is None for name,parameter in model.named_parameters() if name not in train.FUSION_NAMES)


@pytest.mark.parametrize("path", [FP32,BF16])
def test_backward_context_remains_active_with_checkpoint_replay(monkeypatch,path):
    model, recipe, fixtures, original = tiny()
    seen, active = [], []
    previous = probe.sdpa_kernel
    @contextmanager
    def scope(backend):
        with previous(backend):
            active.append(backend)
            try:
                yield
            finally:
                active.pop()
    def gradient_hook(gradient):
        assert active == [probe.SDPBackend.MATH]
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.backends.cuda.flash_sdp_enabled()
        seen.append(True)
    monkeypatch.setattr(probe, "sdpa_kernel", scope)
    handles = [parameter.register_hook(gradient_hook) for parameter in train.assert_fusion_only(model)]
    try:
        with torch.autocast("cpu",dtype=torch.bfloat16):
            probe.ce_gradients(model, recipe, fixtures, path=path, original_flags=original)
            assert torch.is_autocast_enabled("cpu")
    finally:
        for handle in handles:
            handle.remove()
    assert len(seen) == 4 and not active and not torch.is_autocast_enabled("cpu")


def test_three_real_adam_calls_restore_boundary_and_isolate_common_history(monkeypatch):
    model, recipe, fixtures, original, optimizer, scheduler, saved, restore = populated()
    start = probe.fusion_values(model)
    calls, observed = [], []
    original_step = optimizer.step
    def step(*args, **kwargs):
        calls.append(True)
        return original_step(*args, **kwargs)
    monkeypatch.setattr(optimizer, "step", step)
    result = probe.measure_updates(model, recipe, fixtures, optimizer, scheduler, restore=restore,
                                   original_flags=original, publish=observed.append)
    assert len(calls) == 3 and result["optimizer_calls"] == 3
    assert [row["case"] for row in result["rows"]] == list(probe.CASES)
    assert all(result["restoration"].values())
    assert len(observed) == 3
    for row in observed:
        assert all(value == 1 for value in row["saved_steps"].values())
        assert all(value == 2 for value in row["candidate_steps"].values())
        assert row["same_initial_boundary"] and row["finite"]
    zero, fp32, bf16 = observed
    assert zero["observations"]["raw_gradient"]["all"]["norm"] == 0
    assert zero["observations"]["actual_master_delta"]["all"]["norm"] > 0
    assert zero["observations"]["delta_beyond_zero_gradient_adam"]["all"]["norm"] == 0
    assert zero["observations"]["delta_without_common_decay"]["all"]["norm"] > 0
    assert fp32["observations"]["delta_beyond_zero_gradient_adam"]["all"]["norm"] > 0
    assert bf16["bf16_vs_fp32"]["actual_master_delta"]["all"]["relative_l2"] is not None
    assert tree_digests(probe.fusion_values(model)) == tree_digests(start)
    assert tree_digests(optimizer.state_dict()) == tree_digests(saved["optimizer"])
    assert tree_digests(scheduler.state_dict()) == tree_digests(saved["scheduler"])
    assert tree_digests(_rng_state(None)) == tree_digests(saved["rng"])


def test_actual_fp32_master_delta_matches_independent_optimizer_oracle():
    model, recipe, fixtures, original, optimizer, scheduler, _, restore = populated()
    initial = probe.fusion_values(model)
    probe.ce_gradients(model, recipe, fixtures, path=FP32, original_flags=original)
    torch.nn.utils.clip_grad_norm_(train.assert_fusion_only(model),1.,foreach=False,error_if_nonfinite=True)
    optimizer.step(); scheduler.step()
    expected = probe.difference(probe.fusion_values(model),initial)
    model.zero_grad(set_to_none=True); restore()
    result = probe.measure_updates(model,recipe,fixtures,optimizer,scheduler,restore=restore,
                                   original_flags=original,publish=lambda row:None)
    fp32 = result["rows"][1]
    assert fp32["tensor_pins"]["actual_master_delta"] == tree_digests(expected)


def test_failed_observation_restores_checkpoint_without_further_optimizer_step(monkeypatch):
    model, recipe, fixtures, original, optimizer, scheduler, saved, restore = populated()
    calls = []
    step = optimizer.step
    def count(*args, **kwargs):
        calls.append(True); return step(*args,**kwargs)
    monkeypatch.setattr(optimizer,"step",count)
    def reject(row):
        raise RuntimeError("observation failed")
    with pytest.raises(RuntimeError,match="observation failed"):
        probe.measure_updates(model,recipe,fixtures,optimizer,scheduler,restore=restore,
                              original_flags=original,publish=reject)
    assert len(calls) == 1
    assert tree_digests(model.backbone.fusion.state_dict()) == tree_digests(saved["fusion"])
    assert tree_digests(optimizer.state_dict()) == tree_digests(saved["optimizer"])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_rejects_unpopulated_adam_and_wrong_fixture_count():
    model, recipe, fixtures, original = tiny()
    optimizer,_ = train.build_optimizer(model)
    with pytest.raises(ValueError,match="populated"):
        probe.moment_values(model,optimizer)
    with pytest.raises(ValueError,match="two held-out"):
        probe.ce_gradients(model,recipe,fixtures[:1],path=FP32,original_flags=original)


def test_mismatched_adam_step_and_counter_rejected_before_candidate_update(monkeypatch):
    model,recipe,fixtures,original,optimizer,scheduler,_,restore = populated()
    def wrong_counter():
        restored = restore()
        restored["counters"]["optimizer_updates"] = 2
        return restored
    monkeypatch.setattr(optimizer,"step",lambda: (_ for _ in ()).throw(AssertionError("candidate ran")))
    with pytest.raises(ValueError,match="Adam steps"):
        probe.measure_updates(model,recipe,fixtures,optimizer,scheduler,restore=wrong_counter,
                              original_flags=original,publish=lambda row:None)


def test_geometry_handles_zero_reference_without_fabricated_ratio():
    result = probe.geometry({"x":torch.ones(2)}, {"x":torch.zeros(2)})
    assert result["all"]["relative_l2"] is None and result["all"]["cosine"] is None
    assert result["all"]["difference_norm"] == pytest.approx(2**.5)
