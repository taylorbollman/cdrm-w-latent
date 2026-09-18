"""Focused checks for the combined RT+NextLat initialization/position pilot."""
from dataclasses import asdict
import math

import pytest
import torch

from scripts.rt_a5_common import canonical_parameter_sha256, configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_nextlat import _parameter_sha256, build_nextlat_model, nextlat_objective
from scripts.rt_a5_nextlat_variant import FixedSinusoidalPositions, build_variant_model, variant_configuration


@pytest.fixture(autouse=True)
def bounded_cpu_runtime():
    torch.set_num_threads(1)
    configure_fp32_runtime()


@pytest.mark.parametrize("width", [128, 512])
def test_exact_parameter_changes_noise_and_truthful_provenance(width):
    torch.random.default_generator.manual_seed(91)
    rng = torch.get_rng_state().clone()
    baseline = build_nextlat_model("rt", width=width, backend="naive")
    variant = build_variant_model(width=width, backend="naive")
    assert torch.equal(rng, torch.get_rng_state())
    assert variant.nextlat_config == baseline.nextlat_config
    assert variant.nextlat_initialization["baseline_initialization"] == baseline.nextlat_initialization
    expected_changed = {f"backbone.transformer.blocks.{i}.{name}"
                        for i in range(2) for name in ("kv_proj.weight", "attn_out.weight")}
    assert list(variant.state_dict()) == list(baseline.state_dict())
    actual_changed = {name for name, value in baseline.state_dict().items()
                      if not torch.equal(value, variant.state_dict()[name])}
    assert actual_changed == expected_changed
    generator = torch.Generator().manual_seed(1236)
    for index, block in enumerate(variant.backbone.transformer.blocks):
        original = baseline.backbone.transformer.blocks[index]
        assert torch.equal(block.kv_proj.weight[:width], original.kv_proj.weight[:width])
        for value in (block.kv_proj.weight[width:], block.attn_out.weight):
            expected = torch.eye(width) + (1 / math.sqrt(width)) * torch.randn(width, width, generator=generator)
            assert torch.equal(value, expected)
    assert variant.nextlat_initialization["canonical_sha256"] == canonical_parameter_sha256(variant.backbone)
    assert variant.nextlat_initialization["canonical_sha256"] != baseline.nextlat_initialization["canonical_sha256"]
    assert variant.nextlat_initialization["model_parameter_sha256"] == _parameter_sha256(variant)
    assert variant.nextlat_initialization["predictor_sha256"] == _parameter_sha256(baseline.predictor)
    assert variant.nextlat_initialization["parameter_count"] == sum(p.numel() for p in variant.parameters())
    before, after = asdict(baseline.backbone.config), asdict(variant.backbone.config)
    assert {name for name in before if before[name] != after[name]} == {"alibi"}
    assert not variant.backbone.config.alibi and not variant.backbone.config.rope
    assert all(not block.config.alibi and not block.config.rope
               for block in variant.backbone.transformer.blocks)
    assert {item["parameter"] for item in variant.nextlat_initialization["changed_parameter_slices"]} == expected_changed


def test_repeatable_factory_and_fixed_noise_independent_of_backbone_seed():
    first = build_variant_model(width=128, backend="naive")
    second = build_variant_model(width=128, backend="naive")
    assert first.nextlat_initialization == second.nextlat_initialization
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items())
    other = build_variant_model(width=128, seed=999, backend="naive")
    for first_block, other_block in zip(first.backbone.transformer.blocks, other.backbone.transformer.blocks):
        assert torch.equal(first_block.kv_proj.weight[128:], other_block.kv_proj.weight[128:])
        assert torch.equal(first_block.attn_out.weight, other_block.attn_out.weight)
    assert not torch.equal(first.backbone.transformer.wte.weight, other.backbone.transformer.wte.weight)
    assert variant_configuration(512)["initialization"]["noise_std"] == 1 / math.sqrt(512)


def test_sinusoids_match_independent_formula_and_are_length_independent():
    module = FixedSinusoidalPositions(128)
    positions = torch.tensor([[0, 1, 11, 35]], dtype=torch.long)
    result = module(positions)
    expected = torch.tensor([[[func(position / 10000.0 ** (2 * i / 128))
                               for i in range(64) for func in (math.sin, math.cos)]
                              for position in positions[0].tolist()]], dtype=torch.float32)
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(result[0, 0, 0::2], torch.zeros(64))
    assert torch.equal(result[0, 0, 1::2], torch.ones(64))
    short = module(torch.arange(12).unsqueeze(0))
    long = module(torch.arange(36).unsqueeze(0))
    assert torch.equal(short, long[:, :12])
    assert result.dtype == torch.float32 and not result.requires_grad
    assert list(module.parameters()) == [] and module.state_dict() == {}


def test_exact_input_addition_raw_nextlat_conditioning_and_no_alibi(monkeypatch):
    model = build_variant_model(width=128, backend="naive")
    tokens = torch.tensor([[0, 1, 2, 3], [4, 0, 7, 9]])
    block_inputs, predictor_inputs = [], []
    def forbid_alibi(*_args, **_kwargs):
        raise AssertionError("ALiBi must not be used by the sinusoidal pilot")
    monkeypatch.setattr(model.backbone, "get_alibi_attention_bias", forbid_alibi)
    def observe_block(_module, args, kwargs):
        assert kwargs["attention_bias"] is None
        block_inputs.append(args[0].detach().clone())
    block_hook = model.backbone.transformer.blocks[0].register_forward_pre_hook(observe_block, with_kwargs=True)
    predictor_hook = model.predictor.register_forward_pre_hook(
        lambda _module, args: predictor_inputs.append(tuple(value.detach().clone() for value in args)))
    try:
        with fp32_context("cpu"):
            result = nextlat_objective(model, tokens, tokens)
    finally:
        block_hook.remove()
        predictor_hook.remove()
    expected = model.backbone.transformer.wte(tokens) + model.backbone.transformer.wpe(
        torch.arange(tokens.shape[1]).unsqueeze(0))
    assert torch.equal(block_inputs[0], expected)
    assert torch.equal(predictor_inputs[0][1], model.backbone.transformer.wte(tokens[:, 1:]))
    assert torch.isfinite(result["loss"])


@pytest.mark.parametrize("length", [12, 36])
def test_causality_finite_joint_gradients_and_backbone_only_inference(length):
    model = build_variant_model(width=128, backend="naive").eval()
    generator = torch.Generator().manual_seed(71)
    tokens = torch.randint(0, 60, (2, length), generator=generator)
    labels = torch.randint(0, 60, tokens.shape, generator=generator)
    changed = tokens.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 60
    with fp32_context("cpu"):
        result = nextlat_objective(model, tokens, labels)
        result["loss"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with torch.no_grad():
            original = model(tokens).logits
            altered = model(changed).logits
            prefix = model(tokens[:, :6]).logits
    torch.testing.assert_close(original[:, :6], altered[:, :6], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(original[:, :6], prefix, rtol=2e-5, atol=2e-6)
    def forbid_predictor(_module, _args):
        raise AssertionError("Ordinary inference must bypass the auxiliary predictor")
    handle = model.predictor.register_forward_pre_hook(forbid_predictor)
    try:
        with torch.no_grad(), fp32_context("cpu"):
            assert torch.equal(model(tokens).logits, original)
    finally:
        handle.remove()


def test_positions_have_no_optimizer_or_persistent_checkpoint_state():
    model = build_variant_model(width=128, backend="naive")
    optimizer = make_optimizer(model)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len(set(map(id, parameters))) == len(list(model.parameters()))
    assert all("wpe" not in name for name, _ in model.named_parameters())
    assert all("wpe" not in name for name in model.state_dict())
    restored = build_variant_model(width=128, backend="naive")
    restored.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.tensor([[0, 7, 8, 9]])
    with torch.no_grad(), fp32_context("cpu"):
        assert torch.equal(model(tokens).logits, restored(tokens).logits)


@pytest.mark.parametrize("kwargs", [{"architecture": "seq"}, {"width": 65}, {"seed": -1}, {"predictor_seed": -1}])
def test_unsupported_variant_requests_rejected(kwargs):
    with pytest.raises(ValueError):
        build_variant_model(**kwargs)
