"""Focused semantic checks for Mitchell positions and two-record RT attention."""
from dataclasses import asdict

import pytest
import torch

from olmo.model import OLMoRecurrentAutogradBlock, OLMoRecurrentBlockTiled
from scripts.rt_a5_common import canonical_parameter_sha256, configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_nextlat import _parameter_sha256, build_nextlat_model, nextlat_objective
from scripts.rt_a5_nextlat_variant import FixedSinusoidalPositions
from scripts.rt_a5_window import (WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled,
                                  build_window_model, window_configuration, window_two_attention_bias)


@pytest.fixture(autouse=True)
def bounded_cpu_runtime():
    torch.set_num_threads(1)
    configure_fp32_runtime()


@pytest.mark.parametrize("position", ["alibi", "sinusoidal"])
@pytest.mark.parametrize("window", [None, 2])
@pytest.mark.parametrize("width", [128, 512])
def test_all_learned_tensors_remain_original_mitchell(position, window, width):
    torch.random.default_generator.manual_seed(913)
    rng = torch.get_rng_state().clone()
    original = build_nextlat_model("rt", width=width, backend="naive")
    model = build_window_model(width=width, backend="naive", position_encoding=position,
                               second_layer_window=window)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.nextlat_config == original.nextlat_config
    assert model.nextlat_initialization["baseline_initialization"] == original.nextlat_initialization
    assert model.nextlat_initialization["changed_parameter_slices"] == []
    assert list(model.state_dict()) == list(original.state_dict())
    assert list(dict(model.named_parameters())) == list(dict(original.named_parameters()))
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.state_dict().items())
    assert model.nextlat_initialization["model_parameter_sha256"] == _parameter_sha256(original)
    assert canonical_parameter_sha256(model.backbone) == original.nextlat_initialization["canonical_sha256"]
    if width == 512:
        assert model.nextlat_initialization["canonical_sha256"] == "0380e2fd4cdd4db63ce6d732a0834ad054e185c2276da55d733839276d8c55ad"
        assert sum(p.numel() for p in model.parameters()) == 7407104
    assert model.backbone.a5_initialization == original.backbone.a5_initialization
    before, after = asdict(original.backbone.config), asdict(model.backbone.config)
    assert {name for name in before if before[name] != after[name]} == ({"alibi"} if position == "sinusoidal" else set())
    assert not model.backbone.config.rope
    assert all(block.config.alibi == (position == "alibi") for block in model.backbone.transformer.blocks)
    assert type(model.backbone.transformer.blocks[0]) is OLMoRecurrentAutogradBlock
    expected = WindowTwoRecurrentAutogradBlock if window else OLMoRecurrentAutogradBlock
    assert type(model.backbone.transformer.blocks[1]) is expected
    assert ("wpe" in model.backbone.transformer) == (position == "sinusoidal")
    assert all("wpe" not in name for name in model.state_dict())
    assert all("wpe" not in name for name, _ in model.named_parameters())


def test_tiled_factory_uses_existing_backwards_and_canonical_names_without_running_cuda():
    model = build_window_model(width=128, second_layer_window=2)
    assert type(model.backbone.transformer.blocks[0]) is OLMoRecurrentBlockTiled
    assert type(model.backbone.transformer.blocks[1]) is WindowTwoRecurrentBlockTiled
    block = model.backbone.transformer.blocks[1]
    assert block.pre_attention_block.q_proj is block.q_proj
    assert block.pre_attention_block.kv_proj is block.kv_proj
    assert block.post_attention_block.ff_proj is block.ff_proj
    parameters = [p for group in make_optimizer(model).param_groups for p in group["params"]]
    assert len(parameters) == len(set(map(id, parameters))) == len(list(model.parameters()))


@pytest.mark.parametrize("length", [1, 2, 3, 12, 36])
def test_mask_exactly_self_and_previous_preserving_incoming_bias(length):
    x = torch.zeros(2, length, 128)
    incoming = torch.randn(1, 2, length, length, generator=torch.Generator().manual_seed(88))
    untouched = incoming.clone()
    result = window_two_attention_bias(x, incoming)
    assert torch.equal(incoming, untouched)
    for query in range(length):
        for key in range(length):
            if key in (query - 1, query):
                assert torch.equal(result[..., query, key], incoming[..., query, key])
            else:
                assert torch.isneginf(result[..., query, key]).all()
    default = window_two_attention_bias(x)
    assert torch.equal(default.diagonal(dim1=-2, dim2=-1), torch.zeros(1, 1, length))
    assert torch.isfinite(default).sum().item() == 2 * length - 1


def test_alibi_full_is_exact_original_model_and_sinusoids_added_once(monkeypatch):
    baseline = build_nextlat_model("rt", width=128, backend="naive")
    unchanged = build_window_model(width=128, backend="naive", position_encoding="alibi")
    sinusoidal = build_window_model(width=128, backend="naive", second_layer_window=2)
    tokens = torch.tensor([[0, 7, 3, 9, 1], [2, 1, 5, 0, 4]])
    with fp32_context("cpu"):
        assert torch.equal(baseline(tokens).logits, unchanged(tokens).logits)
    def forbid_alibi(*_args, **_kwargs):
        raise AssertionError("Sinusoidal models must not invoke ALiBi")
    monkeypatch.setattr(sinusoidal.backbone, "get_alibi_attention_bias", forbid_alibi)
    block_inputs, predictor_inputs = [], []
    first = sinusoidal.backbone.transformer.blocks[0].register_forward_pre_hook(
        lambda _module, args: block_inputs.append(args[0].detach().clone()))
    predictor = sinusoidal.predictor.register_forward_pre_hook(
        lambda _module, args: predictor_inputs.append(args[1].detach().clone()))
    try:
        with fp32_context("cpu"):
            objective = nextlat_objective(sinusoidal, tokens, tokens)
    finally:
        first.remove()
        predictor.remove()
    assert isinstance(sinusoidal.backbone.transformer.wpe, FixedSinusoidalPositions)
    expected = sinusoidal.backbone.transformer.wte(tokens) + sinusoidal.backbone.transformer.wpe(
        torch.arange(tokens.shape[1]).unsqueeze(0))
    assert torch.equal(block_inputs[0], expected)
    assert torch.equal(predictor_inputs[0], sinusoidal.backbone.transformer.wte(tokens[:, 1:]))
    assert torch.isfinite(objective["loss"])


@pytest.mark.parametrize("position", ["alibi", "sinusoidal"])
def test_window_preserves_first_two_tokens_but_changes_later_attention(position):
    full = build_window_model(width=128, backend="naive", position_encoding=position)
    window = build_window_model(width=128, backend="naive", position_encoding=position, second_layer_window=2)
    tokens = torch.tensor([[0, 7, 3, 9, 1], [2, 1, 5, 0, 4]])
    with torch.no_grad(), fp32_context("cpu"):
        full_logits, window_logits = full(tokens).logits, window(tokens).logits
    assert torch.equal(full_logits[:, :2], window_logits[:, :2])
    assert not torch.equal(full_logits[:, 2:], window_logits[:, 2:])


@pytest.mark.parametrize("position", ["alibi", "sinusoidal"])
def test_window_joint_gradients_causality_and_backbone_only_inference(position):
    model = build_window_model(width=128, backend="naive", position_encoding=position, second_layer_window=2).eval()
    generator = torch.Generator().manual_seed(44)
    tokens = torch.randint(0, 60, (2, 12), generator=generator)
    labels = torch.randint(0, 60, tokens.shape, generator=generator)
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 60
    with fp32_context("cpu"):
        objective = nextlat_objective(model, tokens, labels)
        objective["loss"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with torch.no_grad():
            original = model(tokens).logits
            altered = model(changed).logits
            prefix = model(tokens[:, :5]).logits
    torch.testing.assert_close(original[:, :5], altered[:, :5], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(original[:, :5], prefix, rtol=2e-5, atol=2e-6)
    def forbid_predictor(*_args, **_kwargs):
        raise AssertionError("Evaluation must use the backbone alone")
    handle = model.predictor.register_forward_pre_hook(forbid_predictor)
    try:
        with torch.no_grad(), fp32_context("cpu"):
            assert torch.equal(model(tokens).logits, original)
    finally:
        handle.remove()


@pytest.mark.parametrize("kwargs", [{"architecture": "seq"}, {"width": 65}, {"seed": -1},
                                     {"predictor_seed": -1}, {"position_encoding": "rope"},
                                     {"second_layer_window": 1}, {"second_layer_window": True},
                                     {"second_layer_window": 2.0}])
def test_unsupported_requests_rejected(kwargs):
    with pytest.raises(ValueError):
        build_window_model(**kwargs)
