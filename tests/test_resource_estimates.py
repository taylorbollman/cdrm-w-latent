"""Resource estimates checked against ownership and executed tiny matrix ops."""
from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cdrm.pretrained.nextlat import NextLatConfig, NextLatPredictor, _ce_chunk, _kl_chunk, _chunked_sum
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.resource_estimates import (
    LossWork, architecture_parameter_counts, estimate_training_resources,
    parameter_inventory,
)


class MatrixCounter(TorchDispatchMode):
    """Count real eager mm/bmm calls, independently of the analytic formulas."""
    def __init__(self):
        super().__init__()
        self.dense = self.attention = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.mm.default:
            a, b = args[:2]
            self.dense += 2*a.shape[0]*a.shape[1]*b.shape[1]
        elif func is torch.ops.aten.bmm.default:
            a, b = args[:2]
            self.attention += 2*a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
        return func(*args, **(kwargs or {}))


def estimate(**kwargs):
    return estimate_training_resources(OLMoConfig.tiny(), batch_size=2,
        sequence_length=5, mode=FBTMode(enabled=False, rt_mode=RTMode(())), **kwargs)


def values(result, prefix):
    chosen = [c for c in result.components if c.name.startswith(prefix)]
    return sum(c.minimum for c in chosen), sum(c.maximum for c in chosen)


def test_native_shared_counts_match_existing_parameter_ledger():
    c = OLMoConfig.native_1b()
    result = architecture_parameter_counts(c, fbt=True, nextlat=NextLatConfig(c.model_dim))
    assert result == {"backbone": 1176764416, "fusion": 8388608,
        "nextlat_training_only": 82726912, "training_architecture": 1267879936,
        "deployable_inference": 1185153024}


@pytest.mark.parametrize("bias", [False, True])
def test_parameter_formulas_match_actual_tiny_modules(bias):
    torch.set_num_threads(1)
    c = OLMoConfig.tiny()
    n = NextLatConfig(c.model_dim, bias=bias)
    core = OLMoFBT(OLMoTiledRTForCausalLM(c))
    predictor = NextLatPredictor(n)
    result = architecture_parameter_counts(c, fbt=True, nextlat=n)
    assert result["backbone"] == sum(p.numel() for p in core.backbone.parameters())
    assert result["fusion"] == sum(p.numel() for p in core.fusion.parameters())
    assert result["nextlat_training_only"] == sum(p.numel() for p in predictor.parameters())


def test_inventory_deduplicates_aliases_and_distinguishes_frozen_unused_weights():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(3, 4))
    model.alias = model.weight
    model.unused = torch.nn.Parameter(torch.ones(7), requires_grad=False)
    model.weight.grad = torch.zeros_like(model.weight)
    optimizer = torch.optim.AdamW([model.weight])
    got = parameter_inventory(model, optimizer=optimizer,
        executed_names={"alias", "weight"}, inference_names={"alias"})
    assert got == {"registered_unique": 19, "resident_parameter_bytes": 76,
        "trainable": 12, "gradient_participating": 12, "optimizer_owned": 12,
        "executed_declared": 12, "deployable_inference_declared": 12}
    with pytest.raises(ValueError, match="Unknown"):
        parameter_inventory(model, executed_names={"missing"})
    other = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    with pytest.raises(ValueError, match="outside"):
        parameter_inventory(model, optimizer=other)


@pytest.mark.parametrize("length", [3, 5, 8, 33, 65])
@pytest.mark.parametrize("backward_memory", ["materialized", "recompute"])
@pytest.mark.parametrize("kv_only_writes", [False, True])
def test_rt_estimate_matches_executed_dense_and_attention_matmuls(length, backward_memory, kv_only_writes):
    torch.set_num_threads(1)
    c = replace(OLMoConfig.tiny(), num_layers=1, max_context_length=128)
    model = OLMoTiledRTForCausalLM(c, attention_backend="math", attention_precision="fp32",
        backward_memory=backward_memory, kv_only_writes=kv_only_writes)
    x = torch.randn(2, length, c.model_dim, requires_grad=True)
    with MatrixCounter() as observed:
        model(inputs_embeds=x, mode=RTMode((0,)), return_logits=False).last_hidden_state.square().sum().backward()
    got = estimate_training_resources(c, batch_size=2, sequence_length=length,
        mode=FBTMode(enabled=False, rt_mode=RTMode((0,))), backward_memory=backward_memory,
        kv_only_writes=kv_only_writes)
    # Everything in the RT block is included except explicitly excluded
    # pointwise arithmetic. Loss readouts are not part of this direct-block test.
    dense = sum(c.minimum for c in got.components if c.name.startswith("rt_") and not c.name.startswith("rt_attention_"))
    assert observed.dense == dense
    assert observed.attention == values(got, "rt_attention_")[0]


def test_default_backward_accounting_stays_materialized():
    default = estimate()
    assert default.to_dict() == estimate(backward_memory="materialized").to_dict()
    assert default.to_dict()["backward_memory"] == "materialized"
    assert not any(c.name == "rt_attention_probability_recompute" for c in default.components)


def test_default_writer_accounting_stays_full_qkv():
    default = estimate()
    assert default.to_dict() == estimate(kv_only_writes=False).to_dict()
    assert default.to_dict()["kv_only_writes"] is False


@pytest.mark.parametrize("enabled,passes", [(False, 1), (True, 1), (True, 2), (True, 3)])
@pytest.mark.parametrize("backward_memory", ["materialized", "recompute"])
def test_kv_only_savings_include_local_and_batched_vjps_for_every_invocation(enabled, passes, backward_memory):
    c = replace(OLMoConfig.tiny(), num_layers=4)
    mode = FBTMode(enabled=enabled, num_passes=passes, rt_mode=RTMode((0, 2, 3), alpha=0.37))
    arguments = dict(batch_size=2, sequence_length=5, mode=mode,
        nextlat=NextLatConfig(c.model_dim), accumulation_steps=3, backward_memory=backward_memory)
    reference = estimate_training_resources(c, **arguments)
    candidate = estimate_training_resources(c, kv_only_writes=True, **arguments)
    calls = 3 * (passes-1 if enabled else 1)
    # Each omitted forward Q projection saves 2*N*D². Forward, batched
    # reconstruction, local forward+input VJP, and batched input+weight VJPs
    # therefore save 2+2+4+4 = 12*N*D² per RT call, including accumulation.
    expected_savings = 12*2*5*c.model_dim**2*calls*3
    assert reference.matrix_flops_minimum - candidate.matrix_flops_minimum == expected_savings
    assert reference.matrix_flops_maximum - candidate.matrix_flops_maximum == expected_savings
    assert candidate.parameter_counts == reference.parameter_counts
    assert candidate.input_tokens_per_update == reference.input_tokens_per_update
    assert candidate.pass_token_work_per_update == reference.pass_token_work_per_update
    assert candidate.objective_positions_across_passes == reference.objective_positions_across_passes
    assert values(candidate, "rt_attention_") == values(reference, "rt_attention_")
    assert candidate.to_dict()["kv_only_writes"] is True


@pytest.mark.parametrize("enabled,passes,alpha", [(False, 1, 1.0), (True, 1, 1.0),
    (True, 2, 1.0), (True, 3, 1.0), (True, 3, 0.0)])
def test_recompute_work_counts_every_selected_rt_invocation_and_accumulation(enabled, passes, alpha):
    c = replace(OLMoConfig.tiny(), num_layers=4)
    mode = FBTMode(enabled=enabled, num_passes=passes, rt_mode=RTMode((0, 2, 3), alpha=alpha))
    arguments = dict(batch_size=2, sequence_length=5, mode=mode,
                     nextlat=NextLatConfig(c.model_dim), accumulation_steps=3)
    reference = estimate_training_resources(c, **arguments)
    candidate = estimate_training_resources(c, backward_memory="recompute", **arguments)
    calls = 3 * (passes-1 if enabled else 1)
    additional = 2*2*c.model_dim*(5*4//2 + 5*5)*calls*3
    assert candidate.rt_block_calls_per_microbatch == calls
    assert values(candidate, "rt_attention_probability_recompute") == (additional, additional)
    assert candidate.matrix_flops_minimum - reference.matrix_flops_minimum == additional
    assert candidate.matrix_flops_maximum - reference.matrix_flops_maximum == additional
    assert candidate.parameter_counts == reference.parameter_counts
    assert candidate.input_tokens_per_update == reference.input_tokens_per_update
    assert candidate.pass_token_work_per_update == reference.pass_token_work_per_update
    assert candidate.objective_positions_across_passes == reference.objective_positions_across_passes
    assert candidate.to_dict()["backward_memory"] == "recompute"


@pytest.mark.parametrize("checkpointing", [False, True])
def test_ordinary_dense_estimate_brackets_actual_checkpoint_early_stop(checkpointing):
    torch.set_num_threads(1)
    c = replace(OLMoConfig.tiny(), num_layers=1)
    model = OLMoTiledRTForCausalLM(c, attention_backend="math",
        ordinary_activation_checkpointing=checkpointing)
    x = torch.randn(2, 5, c.model_dim, requires_grad=True)
    with MatrixCounter() as observed:
        model(inputs_embeds=x, mode=RTMode(()), return_logits=False).last_hidden_state.square().sum().backward()
    got = estimate_training_resources(c, batch_size=2, sequence_length=5,
        mode=FBTMode(enabled=False, rt_mode=RTMode(())), ordinary_checkpointing=checkpointing)
    lower, upper = values(got, "ordinary_dense_")
    assert lower <= observed.dense <= upper
    lower, upper = values(got, "ordinary_attention_")
    assert lower <= observed.attention <= upper


@pytest.mark.parametrize("kind", ["ce", "kl"])
def test_readout_estimates_match_real_checkpoint_recompute_and_detachments(kind):
    torch.set_num_threads(1)
    c = OLMoConfig.tiny()
    state = torch.randn(4, c.model_dim, requires_grad=True)
    weight = torch.randn(c.vocab_size, c.model_dim, requires_grad=True)
    teacher = torch.randn_like(state)
    targets = torch.tensor([1, 2, 3, 4])
    with MatrixCounter() as observed:
        if kind == "ce":
            loss = _chunked_sum(_ce_chunk, state, weight, targets, 3, weight_second=True)
        else:
            loss = _chunked_sum(_kl_chunk, state, teacher, weight.detach(), 3, weight_second=False)
        loss.backward()
    config = NextLatConfig(c.model_dim) if kind == "kl" else None
    work = LossWork(4) if kind == "ce" else LossWork(0, 0, 4, 4)
    got = estimate(nextlat=config, loss_work=work)
    assert observed.dense == values(got, kind + "_readout_")[0]
    assert (weight.grad is None) is (kind == "kl")


def test_k_passes_share_parameters_but_count_bootstrap_and_all_loss_work():
    c = OLMoConfig.tiny()
    n = NextLatConfig(c.model_dim)
    kwargs = dict(batch_size=2, sequence_length=5, nextlat=n, loss_work=LossWork(3, 5, 4, 6))
    k2 = estimate_training_resources(c, mode=FBTMode(num_passes=2, rt_mode=RTMode((0,))), **kwargs)
    k3 = estimate_training_resources(c, mode=FBTMode(num_passes=3, rt_mode=RTMode((0,))), **kwargs)
    assert k2.parameter_counts == k3.parameter_counts
    assert (k2.ordinary_block_calls_per_microbatch, k2.rt_block_calls_per_microbatch) == (3, 1)
    assert (k3.ordinary_block_calls_per_microbatch, k3.rt_block_calls_per_microbatch) == (4, 2)
    assert k3.input_tokens_per_update == k2.input_tokens_per_update == 10
    assert k3.pass_token_work_per_update == 30
    assert k3.objective_positions_per_update == {"ce": 3, "latent": 5, "kl": 4, "predictor": 6}
    assert k3.objective_positions_across_passes["ce"] == 9
    assert values(k3, "ce_readout_")[0]*2 == values(k2, "ce_readout_")[0]*3


def test_accumulation_scales_arithmetic_and_exposure_but_not_parameters():
    single = estimate()
    accumulated = estimate(accumulation_steps=3)
    assert accumulated.matrix_flops_minimum == 3*single.matrix_flops_minimum
    assert accumulated.matrix_flops_maximum == 3*single.matrix_flops_maximum
    assert accumulated.parameter_counts == single.parameter_counts
    assert accumulated.input_tokens_per_update == 30


def test_full_document_masks_and_disabled_auxiliaries():
    n = NextLatConfig(32, lambda_latent=0)
    assert LossWork.full_document(2, 5, n) == LossWork(8, 0, 6, 6)
    assert LossWork.full_document(2, 1, n) == LossWork(0)
    assert LossWork.full_document(2, 2, n) == LossWork(2)
    with pytest.raises(ValueError, match="union"):
        LossWork(1, 4, 5, 3)
    with pytest.raises(ValueError, match="counts"):
        estimate(loss_work=LossWork(100))
    with pytest.raises(ValueError, match="require NextLat"):
        estimate(loss_work=LossWork(0, 1, 0, 1))
    with pytest.raises(ValueError, match="Zero-weight"):
        estimate(nextlat=n, loss_work=LossWork(1, 1, 0, 1))


def test_k1_fbt_is_ordinary_and_beta_zero_omits_fusion_work():
    c = OLMoConfig.tiny()
    for mode in (FBTMode(num_passes=1, rt_mode=RTMode((0,))),
                 FBTMode(num_passes=2, beta=0, rt_mode=RTMode((0,)))):
        got = estimate_training_resources(c, batch_size=2, sequence_length=5, mode=mode)
        assert values(got, "fusion_") == (0, 0)
        assert got.rt_block_calls_per_microbatch == mode.num_passes-1


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"sequence_length": True},
    {"accumulation_steps": -1}, {"sequence_length": 33}, {"ordinary_checkpointing": 1},
    {"backward_memory": "unknown"}, {"backward_memory": None}, {"backward_memory": True},
    {"kv_only_writes": 1}, {"kv_only_writes": None}, {"kv_only_writes": "true"}])
def test_bad_shape_and_execution_arguments_reject(kwargs):
    arguments = dict(batch_size=2, sequence_length=5,
        mode=FBTMode(enabled=False, rt_mode=RTMode(())))
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        estimate_training_resources(OLMoConfig.tiny(), **arguments)
