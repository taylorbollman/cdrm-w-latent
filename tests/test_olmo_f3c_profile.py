"""Observer attribution must preserve native RT gradients and global callables."""
import pytest
import torch

from cdrm.pretrained import olmo_tiled
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.static_training import StaticFBTTraining
from scripts import olmo_f3c_profile as profile


def tiny_plan(length, combined):
    torch.set_num_threads(1)
    torch.manual_seed(921)
    base = olmo_tiled.OLMoTiledRTForCausalLM(OLMoConfig.tiny(),
        attention_backend="math", attention_precision="fp32",
        ordinary_activation_checkpointing=True)
    model = FBTNextLatLM(OLMoFBT(base), NextLatConfig(model_dim=base.config.model_dim,
                        proj_factor=2, vocab_chunk_size=4), enabled=combined).train()
    ids = torch.arange(2, 2 + length).expand(2, -1).clone()
    valid = torch.ones_like(ids, dtype=torch.bool)
    batch = NextLatBatch(ids, valid, torch.arange(2)[:, None].expand_as(ids))
    mode = FBTMode(enabled=combined, num_passes=2, beta=.4, rt_mode=RTMode((0,), .37))
    return StaticFBTTraining(model, batch, mode=mode)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("length", [3, 5, 8])
def test_annotations_preserve_full_loss_and_every_gradient_with_correct_phase_counts(length, combined):
    plan = tiny_plan(length, combined)
    reference = profile.snapshot_backward(plan.model, plan.backward())
    parameters = {name: value.detach().clone() for name, value in plan.model.named_parameters()}
    rng = torch.get_rng_state().clone()
    original_grad = torch.autograd.grad
    original_helpers = {name: getattr(olmo_tiled, name) for name in profile.HELPER_PHASES}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        with profile.phase_annotations() as counts:
            actual = plan.backward()
    assert profile.check_observer_neutrality(plan.model, actual, reference, plan.active_names)["passed"]
    assert all(counts[name] == expected for name, expected in profile.expected_backward_calls(length).items())
    event_names = {row["name"] for row in profile.profile_rows(trace)}
    assert set(profile.expected_backward_calls(length)) <= event_names
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, parameters[name]) for name, value in plan.model.named_parameters())
    assert torch.autograd.grad is original_grad
    assert all(getattr(olmo_tiled, name) is fn for name, fn in original_helpers.items())


def test_external_autograd_call_is_unlabeled_and_observers_restore_after_exception():
    original_grad, original_mm = torch.autograd.grad, olmo_tiled._mm
    originals = {name: getattr(olmo_tiled, name) for name in profile.HELPER_PHASES}
    x = torch.tensor(3., requires_grad=True)
    with pytest.raises(RuntimeError, match="test interruption"):
        with profile.phase_annotations() as counts:
            assert torch.autograd.grad(x.square(), x)[0].item() == 6
            assert not counts
            raise RuntimeError("test interruption")
    assert torch.autograd.grad is original_grad
    assert olmo_tiled._mm is original_mm
    assert all(getattr(olmo_tiled, name) is fn for name, fn in originals.items())


def test_unrecognized_rt_vjp_signature_fails_instead_of_silently_misattributing():
    namespace = {"__name__": olmo_tiled.__name__, "torch": torch}
    exec("def backward(x):\n return torch.autograd.grad((x.square(),x.sin()),x)\n", namespace)
    original_grad = torch.autograd.grad
    with pytest.raises(RuntimeError, match="Unrecognized native RT VJP signature"):
        with profile.phase_annotations():
            namespace["backward"](torch.tensor(1., requires_grad=True))
    assert torch.autograd.grad is original_grad


def test_neutrality_gate_detects_changed_gradient_loss_and_ownership():
    plan = tiny_plan(3, True)
    result = plan.backward()
    reference = profile.snapshot_backward(plan.model, result)
    names = tuple(plan.active_names)
    assert not profile.check_observer_neutrality(plan.model, result, reference, names[:-1])["passed"]
    name, parameter = next((n, p) for n, p in plan.model.named_parameters() if p.grad is not None)
    parameter.grad.flatten()[0] += 1
    check = profile.check_observer_neutrality(plan.model, result, reference, names)
    assert not check["passed"] and name in check["nonidentical_gradients"]
    parameter.grad.copy_(reference["gradients"][name])
    reference["losses"]["pass0/ce"] += 1
    check = profile.check_observer_neutrality(plan.model, result, reference, names)
    assert not check["passed"] and check["nonidentical_losses"] == ["pass0/ce"]


@pytest.mark.parametrize("extra", [["--variant", "cast_once"], ["--batch-size", "256"], ["--length", "1024"]])
def test_cli_rejects_unplanned_profile_scope_before_device_initialization(extra):
    with pytest.raises(SystemExit):
        profile.parse_args(["--output-dir", "unused", *extra])
