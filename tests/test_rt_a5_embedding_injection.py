"""Bounded CPU structure and attached-gradient checks for embedding routing."""
import math

import pytest
import torch

from olmo.model import OLMoRecurrentAutogradBlock, OLMoRecurrentBlockTiled
from scripts.rt_a5_common import fp32_context
from scripts.rt_a5_embedding_injection import build_model, configuration, make_projection
from scripts.rt_a5_l1r_depth import build_model as build_depth_model
from scripts.rt_a5_nextlat import _parameter_sha256, nextlat_objective
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def small(**kwargs):
    return build_model(width=64, backend="naive", predictor_hidden_width=256, **kwargs)


def inputs():
    return torch.tensor([[2, 4, 8, 1], [3, 7, 2, 6]], dtype=torch.long)


def test_actual_width_four_layer_count_and_exact_shared_initialization():
    baseline = build_depth_model(n_layers=4)
    rng = torch.get_rng_state().clone()
    model = build_model()
    assert torch.equal(rng, torch.get_rng_state())
    original, actual = dict(baseline.named_parameters()), dict(model.named_parameters())
    assert set(actual) - set(original) == {"backbone.embedding_projection.weight"}
    assert all(torch.equal(value, actual[name]) for name, value in original.items())
    assert sum(p.numel() for p in model.parameters()) == 13964800
    assert sum(p.numel() for p in model.backbone.parameters()) == 12915200
    assert len(actual) == 44 and len(list(model.backbone.parameters())) == 40
    assert model.nextlat_initialization["shared_four_layer_model_sha256"] == _parameter_sha256(baseline)
    assert model.nextlat_initialization["predictor_sha256"] == "3a1fccdd63a53e50328c010b03ecd81d7cf8782229b63cf1c843d1c658731e58"
    assert model.nextlat_initialization["baseline_parameter_tensors_changed"] == []
    assert model.nextlat_initialization["canonical_sha256"] != baseline.nextlat_initialization["canonical_sha256"]
    assert model.nextlat_config == baseline.nextlat_config
    assert "injection_coefficient" not in actual
    assert len(list(model.named_parameters(remove_duplicate=False))) == len(actual)


@pytest.mark.parametrize("backend", ["naive", "tiled"])
def test_four_blocks_positions_and_owned_views(backend):
    model = build_model(width=128, backend=backend)
    blocks = model.backbone.transformer.blocks
    assert len(blocks) == 4
    assert type(blocks[0]) is (WindowTwoRecurrentAutogradBlock if backend == "naive" else WindowTwoRecurrentBlockTiled)
    assert all(type(block) is (OLMoRecurrentAutogradBlock if backend == "naive" else OLMoRecurrentBlockTiled)
               for block in blocks[1:])
    assert [block.layer_id for block in blocks] == [0, 1, 2, 3]
    assert model.backbone.config.alibi and not model.backbone.config.rope
    assert "wpe" not in model.backbone.transformer and "emb_norm" not in model.backbone.transformer
    assert model.backbone.config.embedding_dropout == 0 and not model.backbone.config.scale_emb_init
    for block in blocks:
        assert block.pre_attention_block.kv_proj is block.kv_proj
        assert block.pre_attention_block.q_proj is block.q_proj
        assert block.post_attention_block.ff_proj is block.ff_proj


def test_projection_draw_is_exact_independent_and_shared_by_variants():
    rng = torch.get_rng_state().clone()
    projection = make_projection(64, 1236)
    assert torch.equal(rng, torch.get_rng_state())
    generator = torch.Generator().manual_seed(1236)
    expected = torch.empty(64, 64).normal_(mean=0, std=1 / math.sqrt(64), generator=generator)
    assert torch.equal(projection.weight, expected) and projection.bias is None
    first, other = small(), small(seed=8, predictor_seed=9)
    assert torch.equal(first.backbone.embedding_projection.weight, other.backbone.embedding_projection.weight)
    changed = small(projection_seed=7)
    assert _parameter_sha256(first.backbone.transformer) == _parameter_sha256(changed.backbone.transformer)
    assert _parameter_sha256(first.predictor) == _parameter_sha256(changed.predictor)
    value = small(variant="value")
    assert _parameter_sha256(first) == _parameter_sha256(value)
    assert first.nextlat_initialization["shared_four_layer_model_sha256"] == value.nextlat_initialization["shared_four_layer_model_sha256"]


@pytest.mark.parametrize("variant", ["input", "value"])
def test_coefficient_zero_is_exact_full_nextlat_baseline_outputs_and_gradients(variant):
    baseline = build_depth_model(width=64, n_layers=4, backend="naive", predictor_hidden_width=256)
    model = small(coefficient=0, variant=variant)
    x = inputs()
    with fp32_context("cpu"):
        expected = nextlat_objective(baseline, x, x)
        actual = nextlat_objective(model, x, x)
        for key in ("logits", "loss", "state_loss", "latent_loss"):
            assert torch.equal(actual[key], expected[key]), key
        expected["loss"].backward()
        actual["loss"].backward()
    actual_parameters = dict(model.named_parameters())
    for name, parameter in baseline.named_parameters():
        assert parameter.grad is not None and torch.equal(parameter.grad, actual_parameters[name].grad), name
    assert model.backbone.embedding_projection.weight.grad is None
    assert not model.backbone.transformer.blocks[1]._forward_pre_hooks


def test_input_addition_is_only_before_block1_and_existing_normalization():
    model = small()
    blocks = model.backbone.transformer.blocks
    seen = {}
    hooks = [blocks[0].register_forward_hook(lambda _m, _args, out: seen.update(first_output=out[0])),
             blocks[1].attn_norm.register_forward_pre_hook(lambda _m, args: seen.setdefault("norm_inputs", []).append(args[0]))]
    # This independent pre-hook executes before the call-local injection hook.
    hooks.append(blocks[1].register_forward_pre_hook(lambda _m, args: seen.update(second_input=args[0])))
    x = inputs()
    try:
        with fp32_context("cpu"):
            model(x)
    finally:
        for hook in hooks:
            hook.remove()
    assert torch.equal(seen["second_input"], seen["first_output"])
    expected = seen["first_output"] + .02 * model.backbone.embedding_projection(model.backbone.transformer.wte(x))
    assert torch.equal(seen["norm_inputs"][0], expected)
    assert all(not block._forward_pre_hooks for block in blocks)


@pytest.mark.parametrize("variant", ["input", "value"])
def test_active_projection_and_original_embedding_gradients_final_latents_and_causality(variant):
    model = small(variant=variant)
    x = inputs()
    seen = {}
    hook = model.backbone.transformer.ln_f.register_forward_hook(lambda _m, _args, out: seen.update(hidden=out))
    with fp32_context("cpu"):
        output = model.backbone(x, return_pre_logits=True)
        assert output.pre_logits is seen["hidden"]
        hook.remove()
        result = nextlat_objective(model, x, x)
        result["loss"].backward()
        grad = model.backbone.embedding_projection.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert torch.count_nonzero(model.backbone.transformer.wte.weight.grad) > 0
        changed = x.clone()
        changed[:, -1] += 1
        with torch.no_grad():
            original_logits, changed_logits = model(x).logits, model(changed).logits
        assert torch.equal(original_logits[:, :-1], changed_logits[:, :-1])
        assert torch.equal(original_logits, output.logits)
    assert not model.backbone._injection_in_progress
    assert not model.backbone.transformer.blocks[1]._forward_pre_hooks


def test_scope_cleanup_after_exception_and_no_predictor_on_normal_forward():
    model = small()
    block = model.backbone.transformer.blocks[1]
    def fail(*_args):
        raise RuntimeError("deliberate block failure")
    hook = block.register_forward_hook(fail)
    with fp32_context("cpu"), pytest.raises(RuntimeError, match="deliberate"):
        model(inputs())
    hook.remove()
    assert not block._forward_pre_hooks and not model.backbone._injection_in_progress
    predictor_hook = model.predictor.register_forward_pre_hook(fail)
    try:
        with fp32_context("cpu"):
            assert model(inputs()).logits.shape == (2, 4, 60)
    finally:
        predictor_hook.remove()


@pytest.mark.parametrize("kwargs", [{"variant": "other"}, {"coefficient": float("nan")},
                                    {"coefficient": -1}, {"projection_seed": True}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        configuration(**kwargs)


def test_unsupported_outer_forward_modes_are_rejected():
    model = small()
    for options in ({"use_cache": True}, {"input_embeddings": torch.zeros(2, 4, 64)},
                    {"past_key_values": []}):
        with pytest.raises(ValueError, match="raw token IDs"):
            model.backbone(inputs(), **options)
