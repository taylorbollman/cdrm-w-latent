"""CPU preparation checks; simulated rank averaging is not a DDP/NCCL test."""
import copy
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.distributed_training import (
    ObjectiveForwardAdapter, ddp_normalized_objective, sum_objective_counts,
)
from cdrm.pretrained.fbt_training import FBTNextLatLM, aggregate_pass_losses
from cdrm.pretrained.lm_training import (
    LMTrainingConfig, TERMS, build_adamw, optimizer_step, parameter_layout,
)
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, NextLatLosses
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.static_training import normalized_objective


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(912)
    torch.set_num_threads(1)


def make_model(*, nextlat=True, checkpoint=True, gamma=.4):
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=checkpoint,
        reuse_rope=True, kv_only_writes=True)
    return FBTNextLatLM(OLMoFBT(base), NextLatConfig(model_dim=base.config.model_dim,
        proj_factor=2, lambda_latent=.3, lambda_kl=.7, vocab_chunk_size=4),
        enabled=nextlat, gamma=gamma).train()


def make_batches():
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 8, 9, 2, 1, 1], [4, 7, 1, 1, 1, 1]])
    valid = torch.tensor([[True]*6, [True]*4+[False]*2, [True]*2+[False]*4])
    docs = torch.arange(3)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    ce, latent, kl = [valid.clone() for _ in range(3)]
    ce[0, 2] = False
    latent[0, 3] = False
    kl[0, 4] = False
    latent[1:] = False  # The second simulated rank has no auxiliary targets.
    kl[1:] = False
    batch = NextLatBatch(ids, valid, docs, ce, latent, kl)
    def rows(part):
        return NextLatBatch(**{name: tensor[part] for name, tensor in vars(batch).items()})
    return [rows(slice(0, 1)), rows(slice(1, 3))]


def parameter_grads(model):
    return {name: None if p.grad is None else p.grad.clone()
            for name, p in model.named_parameters()}


def compare_grads(left, right, *, atol=2e-6, rtol=3e-4):
    assert left.keys() == right.keys()
    for name in left:
        if left[name] is None or right[name] is None:
            assert left[name] is right[name] is None, name
        else:
            torch.testing.assert_close(left[name], right[name], atol=atol, rtol=rtol, msg=name)


MODES = [(f, r, n) for f in (False, True) for r in (False, True) for n in (False, True)]


@pytest.mark.parametrize("fbt,rt,nextlat", MODES)
@pytest.mark.parametrize("checkpoint", [False, True])
def test_simulated_rank_average_preserves_canonical_clipped_adam_update(fbt, rt, nextlat, checkpoint):
    reference = make_model(nextlat=nextlat, checkpoint=checkpoint)
    ranks = [copy.deepcopy(reference) for _ in range(2)]
    batches = make_batches()
    mode = FBTMode(enabled=fbt, num_passes=3, beta=.35,
                   rt_mode=RTMode((0, 1) if rt else (), .37))
    kwargs = {"mode": mode}
    counts = sum_objective_counts([reference.counts(batch) for batch in batches])
    optimizers = [build_adamw(model, lr=1e-4, eps=1e-4) for model in [reference, *ranks]]
    expected_grads = {}
    handle = optimizers[0].register_step_pre_hook(
        lambda optimizer, args, kw: expected_grads.update(parameter_grads(reference)))
    expected = optimizer_step(reference, optimizers[0], batches,
        config=LMTrainingConfig(max_grad_norm=1.), backbone_kwargs=kwargs)
    handle.remove()
    results = []
    for model, batch in zip(ranks, batches):
        result = ObjectiveForwardAdapter(model)(batch, global_counts=counts,
                                               world_size=2, backbone_kwargs=kwargs)
        result["objective"].backward()
        results.append(result)
    # Deliberately emulate only the mathematics of DDP's default averaging.
    # Real hooks, collectives and asynchronous bucket behavior remain untested.
    rank_parameters = [dict(model.named_parameters()) for model in ranks]
    for name in rank_parameters[0]:
        gradients = [parameters[name].grad for parameters in rank_parameters]
        attached = [gradient for gradient in gradients if gradient is not None]
        average = None if not attached else sum(attached) / 2
        for parameters in rank_parameters:
            parameters[name].grad = None if average is None else average.clone()
    for model, optimizer in zip(ranks, optimizers[1:]):
        norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1., foreach=False)
        assert float(norm) == pytest.approx(expected["gradient_norm_before_clip"], rel=1e-5)
        compare_grads(parameter_grads(model), expected_grads)
        optimizer.step()
        for (name, actual), (_, target) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(actual, target, atol=3e-7, rtol=3e-5, msg=name)
            actual_state, target_state = optimizer.state.get(actual, {}), optimizers[0].state.get(target, {})
            assert actual_state.keys() == target_state.keys()
            for key in target_state:
                torch.testing.assert_close(actual_state[key], target_state[key], atol=3e-7, rtol=5e-5, msg=name+key)
    assert counts == expected["counts"]
    assert float(sum(result["objective"].detach() for result in results) / 2) == pytest.approx(
        expected["objective"], rel=2e-6)
    for term in TERMS:
        assert sum(float(result["loss_sums"][term]) for result in results) == pytest.approx(
            expected["loss_sums"][term], rel=2e-6, abs=2e-7)


@pytest.mark.parametrize("fbt,rt,nextlat", MODES)
def test_single_rank_adapter_is_exact_canonical_objective_and_gradient(fbt, rt, nextlat):
    model = make_model(nextlat=nextlat)
    reference = copy.deepcopy(model)
    batch = make_batches()[0]
    mode = FBTMode(enabled=fbt, num_passes=3, beta=.35, rt_mode=RTMode((0,) if rt else ()))
    result = reference.loss_sums(batch, backbone_kwargs={"mode": mode})
    expected = normalized_objective(result)
    expected.backward()
    adapter = ObjectiveForwardAdapter(model)
    forward_calls = []
    hook = adapter.register_forward_hook(lambda *args: forward_calls.append(True))
    actual = adapter(batch, global_counts=model.counts(batch), backbone_kwargs={"mode": mode})
    hook.remove()
    assert forward_calls == [True]
    assert torch.equal(actual["objective"], expected)
    actual["objective"].backward()
    compare_grads(parameter_grads(model), parameter_grads(reference), atol=0, rtol=0)
    assert actual["pass_coefficients"] == result.pass_coefficients
    assert actual["counts"] == result.counts
    assert actual["objective_weights"] == result.weights
    def assert_detached(value):
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad and value.grad_fn is None
        elif isinstance(value, dict):
            for child in value.values(): assert_detached(child)
        elif isinstance(value, (tuple, list)):
            for child in value: assert_detached(child)
    assert_detached({k: v for k, v in actual.items() if k != "objective"})


@pytest.mark.parametrize("fbt,rt,nextlat", MODES)
def test_fp32_simulated_rank_gradients_match_concatenated_batch(fbt, rt, nextlat):
    reference = make_model(nextlat=nextlat)
    ranks = [copy.deepcopy(reference) for _ in range(2)]
    batches = make_batches()
    batch = NextLatBatch(**{name: torch.cat([getattr(part, name) for part in batches])
                           for name in vars(batches[0])})
    mode = FBTMode(enabled=fbt, num_passes=2, beta=.35, rt_mode=RTMode((0,) if rt else ()))
    normalized_objective(reference.loss_sums(batch, backbone_kwargs={"mode": mode})).backward()
    counts = sum_objective_counts([reference.counts(part) for part in batches])
    for rank, part in zip(ranks, batches):
        output = ObjectiveForwardAdapter(rank)(part, global_counts=counts, world_size=2,
                                              backbone_kwargs={"mode": mode})
        output["objective"].backward()
    grads = [parameter_grads(rank) for rank in ranks]
    average = {}
    for name in grads[0]:
        used = [values[name] for values in grads if values[name] is not None]
        average[name] = sum(used)/2 if used else None
    compare_grads(average, parameter_grads(reference), atol=3e-6, rtol=5e-4)


def test_globally_active_term_with_zero_local_targets_preserves_zero_and_used_parameter_scope():
    model = make_model()
    batch = make_batches()[0]
    empty = torch.zeros_like(batch.valid_mask)
    empty_batch = replace(batch, ce_mask=empty, latent_mask=empty, kl_mask=empty)
    mode = FBTMode(rt_mode=RTMode((0,)))
    result = ObjectiveForwardAdapter(model)(empty_batch, global_counts=model.counts(batch),
                                          world_size=2, backbone_kwargs={"mode": mode})
    assert result["counts"] == dict.fromkeys(TERMS, 0)
    assert result["objective"].requires_grad
    assert result["objective"].item() == 0
    result["objective"].backward()
    assert all(p.grad is None or not bool(p.grad.any()) for p in model.parameters())
    assert all(p.grad is None for p in model.predictor.parameters())


def test_globally_empty_auxiliary_terms_are_omitted_without_creating_predictor_gradients():
    model = make_model()
    batch = make_batches()[0]
    empty = torch.zeros_like(batch.valid_mask)
    batch = replace(batch, latent_mask=empty, kl_mask=empty)
    counts = model.counts(batch)
    assert counts["ce"] and counts["latent"] == counts["kl"] == 0
    result = ObjectiveForwardAdapter(model)(batch, global_counts=counts)
    result["objective"].backward()
    assert all(p.grad is None for p in model.predictor.parameters())


@pytest.mark.parametrize("beta,passes,gamma", [(0., 3, 1.), (1., 1, 1.), (1., 3, 0.)])
def test_zero_weight_and_inactive_fusion_preserve_canonical_gradient_participation(beta, passes, gamma):
    model = make_model(gamma=gamma)
    reference = copy.deepcopy(model)
    batch = make_batches()[0]
    mode = FBTMode(num_passes=passes, beta=beta, rt_mode=RTMode((0,)))
    normalized_objective(reference.loss_sums(batch, backbone_kwargs={"mode": mode})).backward()
    output = ObjectiveForwardAdapter(model)(batch, global_counts=model.counts(batch),
                                           backbone_kwargs={"mode": mode})
    output["objective"].backward()
    compare_grads(parameter_grads(model), parameter_grads(reference), atol=0, rtol=0)


@pytest.mark.parametrize("passes,gamma", [(1, 2.), (2, .7), (3, 0.), (4, .4)])
def test_pass_gamma_does_not_multiply_global_denominators(passes, gamma):
    leaves = [torch.tensor(float(i+1), requires_grad=True) for i in range(passes)]
    local = {"ce": 3, "latent": 2, "kl": 1}
    weights = {"ce": 1., "latent": .3, "kl": .7}
    records = [NextLatLosses({"ce": x, "latent": 2*x, "kl": 3*x}, local, weights) for x in leaves]
    combined = aggregate_pass_losses(records, gamma=gamma)
    global_counts = {"ce": 7, "latent": 5, "kl": 4}
    objective = ddp_normalized_objective(combined, global_counts=global_counts, world_size=2)
    factor = 2 * (1/7 + .3*2/5 + .7*3/4)
    expected = factor * (leaves[0] + (gamma * sum(leaves[1:]) / (passes-1) if passes > 1 else 0))
    torch.testing.assert_close(objective, expected)
    objective.backward()
    assert leaves[0].grad.item() == pytest.approx(factor)
    for leaf in leaves[1:]:
        assert leaf.grad.item() == pytest.approx(factor * gamma / (passes-1))


def test_adapter_preserves_parameter_tying_native_names_runtime_flags_and_training_state():
    model = make_model().eval()
    layout = parameter_layout(model)
    state_keys = tuple(model.state_dict())
    pointers = [p.data_ptr() for p in model.parameters()]
    adapter = ObjectiveForwardAdapter(model)
    assert not adapter.training and not model.training
    assert adapter.model is model
    assert tuple(model.state_dict()) == state_keys
    assert parameter_layout(model) == layout
    assert [p.data_ptr() for p in adapter.parameters()] == pointers
    assert {key.removeprefix("model.") for key in adapter.state_dict()} == set(state_keys)
    assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
    assert sum(p is model.backbone.readout_weight for p in adapter.parameters()) == 1
    core = model.backbone.backbone
    assert core.ordinary_activation_checkpointing
    assert core.reuse_rope and core.kv_only_writes
    adapter.train()
    assert model.training and adapter.training


def simple_result():
    x = torch.tensor(2., requires_grad=True)
    return NextLatLosses({t: x for t in TERMS}, dict.fromkeys(TERMS, 1), dict.fromkeys(TERMS, 1.))


@pytest.mark.parametrize("world_size", [0, -1, True, 1.5, "2"])
def test_invalid_world_sizes_reject(world_size):
    with pytest.raises(ValueError, match="world_size"):
        ddp_normalized_objective(simple_result(), global_counts=dict.fromkeys(TERMS, 2), world_size=world_size)


@pytest.mark.parametrize("counts", [{"ce": 1}, {**dict.fromkeys(TERMS, 1), "other": 1},
    {"ce": -1, "latent": 1, "kl": 1}, {"ce": True, "latent": 1, "kl": 1},
    {"ce": 1., "latent": 1, "kl": 1}])
def test_invalid_counts_reject(counts):
    with pytest.raises(ValueError, match="counts"):
        ddp_normalized_objective(simple_result(), global_counts=counts, world_size=2)
    with pytest.raises(ValueError, match="counts"):
        sum_objective_counts([counts])


def test_invalid_global_contracts_reject_before_forward():
    model = make_model()
    batch = make_batches()[0]
    adapter = ObjectiveForwardAdapter(model)
    with pytest.raises(ValueError, match="exceeds"):
        adapter(batch, global_counts=dict.fromkeys(TERMS, 0))
    empty = torch.zeros_like(batch.valid_mask)
    with pytest.raises(ValueError, match="no valid positively weighted"):
        adapter(replace(batch, ce_mask=empty, latent_mask=empty, kl_mask=empty),
                global_counts=dict.fromkeys(TERMS, 0))
    disabled = make_model(nextlat=False)
    with pytest.raises(ValueError, match="Disabled"):
        ObjectiveForwardAdapter(disabled)(batch, global_counts={**disabled.counts(batch), "latent": 1})
    with pytest.raises(ValueError, match="At least one"):
        sum_objective_counts([])
    with pytest.raises(TypeError, match="FBTNextLatLM"):
        ObjectiveForwardAdapter(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf"), True, "1"])
def test_invalid_objective_weights_reject(weight):
    result = simple_result()
    result.weights["ce"] = weight
    with pytest.raises(ValueError, match="weights"):
        ddp_normalized_objective(result, global_counts=dict.fromkeys(TERMS, 2), world_size=2)


def test_nonscalar_or_missing_sums_reject():
    for replacement in (torch.ones(2), 1.):
        result = simple_result()
        result.sums["ce"] = replacement
        with pytest.raises(ValueError, match="scalar"):
            ddp_normalized_objective(result, global_counts=dict.fromkeys(TERMS, 2), world_size=2)
    result = simple_result()
    del result.sums["ce"]
    with pytest.raises(ValueError, match="scalar"):
        ddp_normalized_objective(result, global_counts=dict.fromkeys(TERMS, 2), world_size=2)


def test_changed_forward_counts_reject(monkeypatch):
    model = make_model()
    batch = make_batches()[0]
    result = model.loss_sums(batch)
    result.counts["ce"] += 1
    monkeypatch.setattr(model, "loss_sums", lambda *args, **kwargs: result)
    with pytest.raises(ValueError, match="precomputed"):
        ObjectiveForwardAdapter(model)(batch, global_counts=model.counts(batch))
