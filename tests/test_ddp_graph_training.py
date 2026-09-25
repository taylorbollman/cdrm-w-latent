"""Prepared objective/ownership guards; CPU tests do not validate NCCL capture."""
import copy
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.ddp_graph_training import PreparedDDPObjective, DDPGraphTraining
from cdrm.pretrained.distributed_training import ObjectiveForwardAdapter, sum_objective_counts
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.lm_training import LMTrainingConfig
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTMode
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(715)
    torch.set_num_threads(1)


def model_and_batches(nextlat=True):
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=True,
        reuse_rope=True, kv_only_writes=True)
    model = FBTNextLatLM(OLMoFBT(base), NextLatConfig(model_dim=base.config.model_dim,
        proj_factor=2, vocab_chunk_size=4, lambda_latent=.3, lambda_kl=.7),
        enabled=nextlat, gamma=.4).train()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    batch = NextLatBatch(ids, valid, torch.zeros_like(ids), valid.clone(), valid.clone(), valid.clone())
    ce = valid.clone(); ce[0, 1:3] = False
    second = replace(batch, input_ids=ids.roll(1, -1), ce_mask=ce,
                     latent_mask=torch.zeros_like(valid), kl_mask=torch.zeros_like(valid))
    return model, (batch, second)


def grads(model):
    return {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()}


def check_grads(left, right):
    assert left.keys() == right.keys()
    for name in left:
        if left[name] is None or right[name] is None:
            assert left[name] is right[name] is None, name
        else:
            torch.testing.assert_close(left[name], right[name], atol=3e-6, rtol=4e-4, msg=name)


@pytest.mark.parametrize("fbt,rt,nextlat", [(f,r,n) for f in (False,True) for r in (False,True) for n in (False,True)])
@pytest.mark.parametrize("rank", [0, 1])
def test_prepared_rank_objective_and_vjp_match_canonical(fbt, rt, nextlat, rank):
    model, batches = model_and_batches(nextlat)
    reference = copy.deepcopy(model)
    mode = FBTMode(enabled=fbt, num_passes=3, beta=.35, rt_mode=RTMode((0, 1) if rt else (), .37))
    global_counts = sum_objective_counts([model.counts(batch) for batch in batches])
    prepared = PreparedDDPObjective(model, batches[rank], mode=mode,
        global_counts=global_counts, world_size=2)
    expected = ObjectiveForwardAdapter(reference)(batches[rank], global_counts=global_counts,
        world_size=2, backbone_kwargs={"mode": mode})
    actual = prepared()
    torch.testing.assert_close(actual["objective"], expected["objective"], atol=2e-6, rtol=2e-6)
    actual["objective"].backward(); expected["objective"].backward()
    check_grads(grads(model), grads(reference))
    assert actual["pass_coefficients"] == expected["pass_coefficients"]
    for term in global_counts:
        assert not actual["loss_sums"][term].requires_grad
        torch.testing.assert_close(actual["loss_sums"][term], expected["loss_sums"][term], atol=2e-6, rtol=2e-6)
    assert {id(p) for p in prepared.parameters()} == {id(p) for p in model.parameters()}
    assert len(list(prepared.parameters())) == len(list(model.parameters()))
    assert all(n.startswith("model.") for n, _ in prepared.named_parameters())


def prepared_setup():
    model, batches = model_and_batches()
    mode = FBTMode(enabled=False, rt_mode=RTMode((0,)))
    adapter = PreparedDDPObjective(model, batches[0], mode=mode,
        global_counts=model.counts(batches[0]), world_size=1)
    return adapter, batches[0]


def active_names(adapter):
    adapter()["objective"].backward()
    names = tuple(n for n, p in adapter.model.named_parameters() if p.grad is not None)
    adapter.model.zero_grad(set_to_none=True)
    return names


def test_fixed_mask_validation_precedes_token_buffer_write():
    adapter, batch = prepared_setup()
    original = adapter.plan.batch.input_ids.clone()
    changed_mask = batch.ce_mask.clone(); changed_mask[0, 1] = False
    with pytest.raises(ValueError, match="ce_mask changed"):
        adapter.load_batch(replace(batch, input_ids=batch.input_ids+1, ce_mask=changed_mask))
    assert torch.equal(adapter.plan.batch.input_ids, original)
    adapter.load_batch(replace(batch, input_ids=batch.input_ids.roll(1, -1)))
    assert torch.equal(adapter.plan.batch.input_ids, batch.input_ids.roll(1, -1))


def test_forward_does_not_call_eager_loss_or_validation(monkeypatch):
    adapter, _ = prepared_setup()
    def forbidden(*args, **kwargs):
        raise AssertionError("host/eager validation called inside prepared forward")
    monkeypatch.setattr(adapter.model, "loss_sums", forbidden)
    monkeypatch.setattr(adapter.plan, "validate_execution", forbidden)
    monkeypatch.setattr(adapter, "validate_execution", forbidden)
    result = adapter()
    assert result["objective"].requires_grad
    result["objective"].backward()


@pytest.mark.parametrize("change", ["counts", "world_size", "coefficient", "mode", "eval", "parameters"])
def test_prepared_execution_contract_rejects_mutation(change):
    adapter, _ = prepared_setup()
    if change == "counts": adapter._global_counts = (1,2,3)
    elif change == "world_size": adapter.world_size = 2
    elif change == "coefficient": adapter._coefficients = (("ce",1.),)
    elif change == "mode": adapter.plan.mode = FBTMode(enabled=True)
    elif change == "eval": adapter.eval()
    else:
        adapter.model.predictor.mlp[0].weight = torch.nn.Parameter(adapter.model.predictor.mlp[0].weight.clone())
    with pytest.raises(ValueError): adapter.validate_execution()


@pytest.mark.parametrize("names", [[], ["missing"], ["backbone.fusion.projection.weight"]*2, "model"])
def test_active_parameter_declaration_rejects_invalid_names(names):
    adapter, _ = prepared_setup()
    with pytest.raises(ValueError): DDPGraphTraining(adapter, expected_active_names=names)


def test_freeze_preserves_globally_unused_none_and_rejects_gradient_replacement():
    adapter, _ = prepared_setup()
    expected = active_names(adapter)
    runtime = DDPGraphTraining(adapter, expected_active_names=expected)
    adapter()["objective"].backward()
    runtime._freeze_gradients()
    runtime.validate_execution()
    assert runtime.active_names == expected
    unused = [(n,p) for n,p in adapter.model.named_parameters() if n not in expected]
    assert unused and all(p.grad is None for _,p in unused)
    # No fake zero gradient may be attached to inactive fusion parameters.
    unused[0][1].grad = torch.zeros_like(unused[0][1])
    with pytest.raises(ValueError, match="gradient buffers"):
        runtime.validate_execution()
    unused[0][1].grad = None
    runtime.validate_execution()
    name = expected[0]; parameter = dict(adapter.model.named_parameters())[name]
    parameter.grad = parameter.grad.clone()
    with pytest.raises(ValueError, match="gradient buffers"):
        runtime.validate_execution()


def test_declared_participation_is_checked_after_warmup():
    adapter, _ = prepared_setup()
    expected = active_names(adapter)
    runtime = DDPGraphTraining(adapter, expected_active_names=expected[:-1])
    adapter()["objective"].backward()
    with pytest.raises(ValueError, match="declared global parameter participation"):
        runtime._freeze_gradients()


def test_tensor_backward_invokes_wrapped_forward_and_overwrites(monkeypatch):
    adapter, _ = prepared_setup()
    runtime = DDPGraphTraining(adapter, expected_active_names=active_names(adapter))
    calls = []
    class Wrapped(torch.nn.Module):
        def forward(self):
            calls.append("forward")
            return adapter()
    runtime.ddp = Wrapped()
    runtime._tensor_backward()
    initial = grads(adapter.model)
    runtime._tensor_backward()
    assert calls == ["forward", "forward"]
    check_grads(grads(adapter.model), initial)
    assert runtime.adapter.plan.warmup_backward_calls == 0
    assert runtime.adapter.plan.gradient_addresses is None


@pytest.mark.parametrize("field,value", [("bucket_cap_mb",0), ("bucket_cap_mb",float("nan")),
                                          ("gradient_as_bucket_view",1)])
def test_invalid_runner_options(field, value):
    adapter, _ = prepared_setup()
    with pytest.raises((ValueError,TypeError)):
        DDPGraphTraining(adapter, expected_active_names=active_names(adapter), **{field:value})


def test_cpu_capture_does_not_silently_fallback():
    adapter, _ = prepared_setup()
    runtime = DDPGraphTraining(adapter, expected_active_names=active_names(adapter))
    with pytest.raises(ValueError, match="requires CUDA"):
        runtime.capture()
    assert runtime.ddp is None and runtime.graph is None
    with pytest.raises(ValueError, match="Capture"):
        runtime.backward()


def test_failed_plan_cannot_be_reused():
    adapter, _ = prepared_setup()
    runtime = DDPGraphTraining(adapter, expected_active_names=active_names(adapter))
    runtime._failed = True
    with pytest.raises(RuntimeError, match="recreate"):
        runtime.load_batch(adapter.plan.batch)


def test_runner_contract_rejects_mutated_global_active_set():
    adapter, _ = prepared_setup()
    runtime = DDPGraphTraining(adapter, expected_active_names=active_names(adapter))
    runtime.expected_active_names = runtime.expected_active_names[:-1]
    with pytest.raises(ValueError, match="runner settings"):
        runtime.validate_execution()


def test_no_grad_context_cannot_execute_plan():
    adapter, _ = prepared_setup()
    with torch.no_grad(), pytest.raises(ValueError, match="grad-enabled"):
        adapter.validate_execution()


@pytest.mark.parametrize("warmup", [0, 1, 10, True, 11.0])
def test_capture_rejects_insufficient_ddp_warmup_before_cuda_calls(monkeypatch, warmup):
    adapter, _ = prepared_setup()
    runtime = DDPGraphTraining(adapter, expected_active_names=active_names(adapter))
    monkeypatch.setattr(DDPGraphTraining, "device", property(lambda self: torch.device("cuda:0")))
    with pytest.raises(ValueError, match="At least 11"):
        runtime.capture(warmup=warmup)
    assert not runtime._capture_started


@pytest.mark.parametrize("mismatch", [None, "configuration", "counts"])
def test_rank_configuration_agreement_precedes_count_collective(monkeypatch, mismatch):
    adapter, _ = prepared_setup()
    runtime = DDPGraphTraining(adapter, expected_active_names=active_names(adapter))
    dist = torch.distributed
    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(dist, "get_world_size", lambda group: 1)
    def gather(records, shared, group):
        calls.append("gather")
        records[0] = dict(shared)
        if mismatch == "configuration": records[0]["gamma"] += 1
    def reduce(counts, group):
        calls.append("counts")
        if mismatch == "counts": counts.add_(1)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    if mismatch:
        with pytest.raises(ValueError): runtime._validate_distributed_contract()
    else:
        runtime._validate_distributed_contract()
    assert calls == (["gather"] if mismatch == "configuration" else ["gather", "counts"])


def test_tensor_diagnostic_outputs_are_all_detached():
    adapter, _ = prepared_setup()
    output = adapter()
    def check(value):
        if isinstance(value, torch.Tensor): assert not value.requires_grad
        elif isinstance(value, dict):
            for child in value.values(): check(child)
        elif isinstance(value, (tuple,list)):
            for child in value: check(child)
    check({key:value for key,value in output.items() if key != "objective"})
