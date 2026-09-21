"""Bounded activation retention and independent predictor initialization."""

from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.nextlat import (
    NextLatBatch, NextLatConfig, NextLatLM, NextLatPredictor,
    compute_nextlat_loss_sums,
)
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(20260921)


def test_predictor_construction_is_rng_isolated_and_seeded_separately():
    config = NextLatConfig(model_dim=32, seed=713)
    state = torch.random.get_rng_state().clone()
    first = NextLatPredictor(config)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.rand(37)
    second = NextLatPredictor(config)
    different = NextLatPredictor(replace(config, seed=714))
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    assert not torch.equal(first.mlp[0].weight, different.mlp[0].weight)
    assert torch.equal(first.norm_x.weight, torch.ones(64))


@pytest.mark.parametrize("enabled,coefficients", [(False, (1.0, 1.0)), (True, (0.0, 0.0)), (True, (1.0, 1.0))])
def test_wrapping_preserves_checkpoint_and_only_registers_predictor_once(enabled, coefficients):
    backbone = OLMoForCausalLM(OLMoConfig.tiny(), attention_backend="math")
    before = {name: value.clone() for name, value in backbone.state_dict().items()}
    config = NextLatConfig(model_dim=32, lambda_latent=coefficients[0], lambda_kl=coefficients[1])
    wrapper = NextLatLM(backbone, config, enabled=enabled)
    for key, value in before.items():
        torch.testing.assert_close(value, wrapper.backbone.state_dict()[key], atol=0, rtol=0)
    predictor_present = enabled and any(coefficients)
    assert (wrapper.predictor is not None) == predictor_present
    assert len(wrapper.state_dict()) == len(before) + (4 if predictor_present else 0)
    assert wrapper.backbone.readout_weight is wrapper.backbone.token_embeddings.weight
    assert all(name.startswith(("backbone.", "predictor.")) for name in wrapper.state_dict())


def test_vocab_chunks_save_no_vocabulary_activations_until_backward():
    # More valid positions than a chunk, and V distinct from predictor/input
    # widths, so accidental retention of logits/logprobs is unambiguous.
    batch_size, length, width, vocab = 2, 11, 32, 211
    config = NextLatConfig(model_dim=width, vocab_chunk_size=3)
    hidden = torch.randn(batch_size, length, width, requires_grad=True)
    embeddings = torch.randn_like(hidden, requires_grad=True)
    weight = torch.randn(vocab, width, requires_grad=True)
    ids = torch.randint(vocab, (batch_size, length))
    batch = NextLatBatch(ids, torch.ones_like(ids, dtype=torch.bool), torch.zeros_like(ids))
    predictor = NextLatPredictor(config)
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        losses = compute_nextlat_loss_sums(hidden, embeddings, weight, batch, predictor, config)
    assert losses.counts == {"ce": 20, "latent": 20, "kl": 18}
    assert not any(len(shape) >= 2 and shape[-1] == vocab for shape in saved_shapes), saved_shapes
    gradients = torch.autograd.grad(losses.total, (hidden, embeddings, weight, *predictor.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


def test_lm_predictor_geometry_matches_released_one_billion_recipe():
    config = NextLatConfig(model_dim=2048)
    assert config.hidden_dim == 6528
    assert config.lambda_latent == config.lambda_kl == 1
    assert config.bias is False and config.norm_eps == 1e-5
    assert config.to_dict() == NextLatConfig.from_dict(config.to_dict()).to_dict()
    # Do not allocate the ~83M predictor just to inspect its parameter count.
    assert 4096 * 6528 + 6528 * 6528 + 6528 * 2048 + 4096 == 82_726_912
