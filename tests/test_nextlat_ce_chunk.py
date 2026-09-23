"""CE-only chunking keeps the selected full-vocabulary objective and ownership.

CPU checks cover semantics and static execution, not CUDA graph capture or
native BF16 reduction tolerances. Chunk sizes 3 and 7 exercise multiple and
partial chunks without requiring a long recurrent scan.
"""
import copy
from dataclasses import replace
from itertools import product

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import (
    NextLatBatch, NextLatConfig, NextLatPredictor, compute_nextlat_loss_sums,
)
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.static_nextlat import (
    PreparedNextLatLayout, compute_static_nextlat_loss_sums,
)
from cdrm.pretrained.static_training import StaticFBTTraining


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(438)


def batch_fixture():
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19], [23, 29, 31, 37, 41, 43, 1, 1]])
    valid = torch.tensor([[True] * 8, [True] * 6 + [False] * 2])
    docs = torch.tensor([[0] * 8, [1] * 6 + [-1] * 2])
    ce, latent, kl = valid.clone(), valid.clone(), valid.clone()
    ce[0, :2] = False
    ce[1, 3] = False
    latent[0, 3] = False
    kl[1, 4] = False
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def evaluate(hidden, embeddings, readout, batch, predictor, config, *, static, enabled=True):
    if static:
        layout = PreparedNextLatLayout.from_batch(batch, config, enabled=enabled)
        return compute_static_nextlat_loss_sums(hidden, embeddings, readout, batch.input_ids,
            predictor, config, layout, enabled=enabled)
    return compute_nextlat_loss_sums(hidden, embeddings, readout, batch, predictor, config, enabled=enabled)


def assert_gradients(actual, expected, *, atol=3e-7, rtol=3e-5):
    assert len(actual) == len(expected)
    for index, (left, right) in enumerate(zip(actual, expected)):
        assert (left is None) == (right is None), index
        if left is not None:
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=f"gradient {index}")


def test_old_configuration_roundtrip_keeps_exact_serialized_fields_and_defaults():
    legacy = dict(model_dim=32, proj_factor=1.6, bias=False, dropout=0.0, norm_eps=1e-5,
        init_std=.02, lambda_latent=1.0, lambda_kl=1.0, seed=20260921, vocab_chunk_size=32)
    config = NextLatConfig.from_dict(legacy)
    assert config == NextLatConfig(model_dim=32)
    assert config.ce_chunk_size is None
    assert config.effective_ce_chunk_size == 32
    assert config.to_dict() == legacy
    shared = replace(config, vocab_chunk_size=128)
    assert shared.effective_ce_chunk_size == 128
    explicit = replace(shared, ce_chunk_size=2048)
    assert explicit.vocab_chunk_size == 128
    assert explicit.effective_ce_chunk_size == 2048
    assert explicit.to_dict() == {**legacy, "vocab_chunk_size": 128, "ce_chunk_size": 2048}
    assert NextLatConfig.from_dict(explicit.to_dict()) == explicit


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.0, "7"])
def test_invalid_ce_override_is_rejected(invalid):
    with pytest.raises(ValueError, match="ce_chunk_size"):
        NextLatConfig(model_dim=32, ce_chunk_size=invalid)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("chunk", [None, 1, 7, 2048])
def test_ce_matches_dense_full_vocabulary_loss_hidden_and_readout_gradients(static, chunk):
    config = NextLatConfig(model_dim=32, vocab_chunk_size=3, ce_chunk_size=chunk)
    batch = batch_fixture()
    hidden = (torch.randn(2, 8, 32) * .2).requires_grad_()
    readout = (torch.randn(67, 32) * .1).requires_grad_()
    expected_h, expected_w = (t.detach().clone().requires_grad_() for t in (hidden, readout))
    result = evaluate(hidden, torch.zeros_like(hidden), readout, batch, None, config,
        static=static, enabled=False)
    # Explicit selected pairs avoid relying on production mask construction.
    selected = [(0, i) for i in range(1, 7)] + [(1, i) for i in (0, 1, 3, 4)]
    dense_logits = F.linear(torch.stack([expected_h[r, t] for r, t in selected]), expected_w)
    targets = torch.stack([batch.input_ids[r, t+1] for r, t in selected])
    expected = F.cross_entropy(dense_logits.float(), targets, reduction="sum")
    assert result.counts == {"ce": 10, "latent": 0, "kl": 0}
    torch.testing.assert_close(result.sums["ce"], expected, rtol=2e-6, atol=2e-6)
    actual_grads = torch.autograd.grad(result.sums["ce"], (hidden, readout))
    expected_grads = torch.autograd.grad(expected, (expected_h, expected_w))
    assert_gradients(actual_grads, expected_grads)
    # Output rows beyond the tokenizer vocabulary still participate in CE.
    assert actual_grads[1][61:].abs().sum() > 0
    assert torch.count_nonzero(actual_grads[0][1, 5:]) == 0


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("term", ["latent", "kl"])
def test_ce_override_keeps_auxiliary_values_and_every_gradient_bitwise(static, term):
    batch = batch_fixture()
    config = NextLatConfig(model_dim=32, vocab_chunk_size=3)
    predictor = NextLatPredictor(config)
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    embeddings = torch.randn_like(hidden, requires_grad=True)
    readout = (torch.randn(67, 32) * .1).requires_grad_()
    parameters = (hidden, embeddings, readout, *predictor.parameters())
    baseline = evaluate(hidden, embeddings, readout, batch, predictor, config, static=static)
    override = evaluate(hidden, embeddings, readout, batch, predictor,
        replace(config, ce_chunk_size=7), static=static)
    assert override.counts == baseline.counts == {"ce": 10, "latent": 11, "kl": 9}
    torch.testing.assert_close(override.sums[term], baseline.sums[term], rtol=0, atol=0)
    original_grad = torch.autograd.grad(baseline.sums[term], parameters, allow_unused=True)
    override_grad = torch.autograd.grad(override.sums[term], parameters, allow_unused=True)
    assert_gradients(override_grad, original_grad, atol=0, rtol=0)
    assert override_grad[2] is None  # auxiliary readout remains detached
    assert override_grad[1].abs().sum() > 0  # conditioning embeddings stay attached


@pytest.mark.parametrize("static", [False, True])
def test_zero_selected_targets_keep_absent_readout_gradient(static):
    batch = batch_fixture()
    batch = replace(batch, ce_mask=torch.zeros_like(batch.valid_mask))
    config = NextLatConfig(model_dim=32, vocab_chunk_size=3, ce_chunk_size=7)
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    readout = torch.randn(67, 32, requires_grad=True)
    result = evaluate(hidden, torch.zeros_like(hidden), readout, batch, None, config,
        static=static, enabled=False)
    assert result.counts == {"ce": 0, "latent": 0, "kl": 0}
    h_grad, w_grad = torch.autograd.grad(result.total, (hidden, readout), allow_unused=True)
    assert torch.count_nonzero(h_grad) == 0
    assert w_grad is None


def test_static_layout_rejects_chunk_override_changed_after_preparation():
    batch = batch_fixture()
    config = NextLatConfig(model_dim=32, vocab_chunk_size=3)
    layout = PreparedNextLatLayout.from_batch(batch, config)
    with pytest.raises(ValueError, match="configuration"):
        layout.validate_execution(batch.input_ids, replace(config, ce_chunk_size=7), enabled=True)


@pytest.mark.parametrize("rt,fbt,nextlat", list(product((False, True), repeat=3)))
def test_all_eight_features_keep_losses_counts_and_every_gradient_across_static_ce_chunks(rt, fbt, nextlat):
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        ordinary_activation_checkpointing=True, attention_precision="fp32",
        backward_memory="recompute")
    config = NextLatConfig(model_dim=32, vocab_chunk_size=3)
    reference = FBTNextLatLM(OLMoFBT(base), config, enabled=nextlat).train()
    actual = copy.deepcopy(reference)
    actual.config = replace(config, ce_chunk_size=7)
    batch = batch_fixture()
    mode = FBTMode(enabled=fbt, num_passes=2, rt_mode=RTMode((0, 1) if rt else ()))
    expected = reference.loss_sums(batch, backbone_kwargs={"mode": mode})
    plan = StaticFBTTraining(actual, batch, mode=mode)
    result = plan.loss_sums()
    assert result.counts == expected.counts == {
        "ce": 10, "latent": 11 if nextlat else 0, "kl": 9 if nextlat else 0}
    assert result.weights == expected.weights
    assert result.pass_coefficients == expected.pass_coefficients == ((1., 1.) if fbt else (1.,))
    for term in ("ce", "latent", "kl"):
        torch.testing.assert_close(result.sums[term], expected.sums[term], atol=3e-6, rtol=3e-6)
    expected.total.backward()
    result.total.backward()
    assert_gradients([p.grad for p in actual.parameters()], [p.grad for p in reference.parameters()])
    assert actual.backbone.readout_weight is actual.backbone.token_embeddings.weight
    assert all((p.grad is not None) == fbt for p in actual.backbone.fusion.parameters())
