"""Bounded CPU checks for the new task interface and changed small architecture."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from olmo.config import ModelConfig
from olmo.model import OLMo
from cdrm.rt_nextlat_tasks import (
    TaskNextLat, build_model, encode_inputs, read_configuration, task_logits, task_loss,
)
from scripts.rt_a5_common import fp32_context
from scripts.rt_a5_nextlat import NextLatDynamicsModel


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def small_configuration():
    config = read_configuration()
    config["backbone"]["max_sequence_length"] = 16
    return config


def test_model_preserves_canonical_core_and_expands_tables_without_rng_side_effects():
    config = small_configuration()
    torch.manual_seed(41)
    rng = torch.random.get_rng_state().clone()
    model = build_model(config, seed=9, backend="naive")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.initialization["parameter_count"] == 479616
    assert model.initialization["backbone_parameter_count"] == 413824
    assert model.initialization["predictor_parameter_count"] == 65792
    assert model.backbone.config.n_heads == 16
    assert model.backbone.config.d_model // model.backbone.config.n_heads == 8
    assert model.backbone.transformer.blocks[0].attention_window == 2
    assert not hasattr(model.backbone.transformer.blocks[1], "attention_window")
    assert "wpe" not in model.backbone.transformer
    raw = copy.deepcopy(config["backbone"])
    raw.update(vocab_size=60, pad_token_id=0, block_type="sequential", recurrent_backend="naive")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(9)
        canonical = OLMo(ModelConfig(**raw))
    actual = dict(model.backbone.named_parameters())
    for name, value in canonical.named_parameters():
        if name.endswith("att_proj.weight"):
            prefix = name.removesuffix("att_proj.weight")
            candidate = torch.cat((actual[prefix + "q_proj.weight"], actual[prefix + "kv_proj.weight"]))
        else:
            candidate = actual[name]
        if name in ("transformer.wte.weight", "transformer.ff_out.weight"):
            candidate = candidate[:60]
        assert torch.equal(candidate, value), name
    other = build_model(config, seed=9, fuzzy_seed=4321, backend="naive")
    for name, value in model.named_parameters():
        candidate = dict(other.named_parameters())[name]
        if name in ("backbone.transformer.wte.weight", "backbone.transformer.ff_out.weight"):
            assert torch.equal(value[:60], candidate[:60])
            assert not torch.equal(value[60:], candidate[60:])
        else:
            assert torch.equal(value, candidate), name


class IndependentLatents(nn.Module):
    """Independent position leaves expose target-role stop-gradient precisely."""
    def __init__(self):
        super().__init__()
        self.h = nn.Parameter(torch.randn(2, 4, 4))
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(76, 4), "ff_out": nn.Linear(4, 76, bias=False),
        })

    def forward(self, input_ids, return_pre_logits=False):
        return SimpleNamespace(logits=self.transformer.ff_out(self.h),
                               pre_logits=self.h if return_pre_logits else None)


def independent_model():
    torch.manual_seed(13)
    return TaskNextLat(IndependentLatents(), NextLatDynamicsModel(4, 8))


def test_native_aligned_local_ce_and_backbone_only_readout():
    model = independent_model()
    local = torch.tensor([[15, 15, 0, 7], [15, 1, 2, 8]])
    inputs = encode_inputs(local)
    assert torch.equal(inputs, local + 60)
    assert torch.equal(encode_inputs(local, "a5"), local)
    labels = torch.tensor([[15, 0, 7, 9], [1, 2, 8, 3]])
    called = []
    handle = model.predictor.register_forward_pre_hook(lambda *_: called.append(True))
    with fp32_context("cpu"):
        logits = task_logits(model, inputs)
        result = task_loss(model, inputs, labels, latent_weight=0)
        torch.testing.assert_close(result["ce"], F.cross_entropy(logits.flatten(0, 1), labels.flatten()))
        assert torch.equal(result["ce"], result["loss"])
        assert result["latent"].item() == 0
        assert task_logits(model, inputs, "a5").shape == (2, 4, 60)
        result["loss"].backward()
    handle.remove()
    assert not called
    assert all(p.grad is None for p in model.predictor.parameters())
    head_gradient = model.backbone.transformer.ff_out.weight.grad
    assert torch.count_nonzero(head_gradient[:60]) == 0
    assert torch.count_nonzero(head_gradient[60:]) > 0


def test_nextlat_keeps_padding_transitions_independent_of_ce_mask_and_stops_only_target():
    model = independent_model()
    inputs = encode_inputs(torch.tensor([[0, 15, 1, 2], [3, 15, 4, 2]]))
    labels = torch.tensor([[15, 1, 2, 7], [15, 4, 2, 8]])
    masked_labels = labels.clone()
    masked_labels[:, :3] = -100
    result = task_loss(model, inputs, labels)
    masked = task_loss(model, inputs, masked_labels)
    assert torch.equal(result["latent"], masked["latent"])
    assert not torch.equal(result["ce"], masked["ce"])
    hidden = model.backbone.h
    predicted = model.predictor(hidden[:, :-1], model.backbone.transformer.wte(inputs[:, 1:]))
    error = (predicted - hidden[:, 1:].detach()).abs()
    independent_loss = torch.where(error < 1, error.square() / 2, error - 0.5).sum() / (2 * 3 * 4)
    torch.testing.assert_close(result["latent"], independent_loss)
    result["latent"].backward()
    assert hidden.grad[:, :-1].abs().sum() > 0
    assert torch.count_nonzero(hidden.grad[:, -1]) == 0
    embedding_gradient = model.backbone.transformer.wte.weight.grad
    assert embedding_gradient[75].abs().sum() > 0  # Native left-padding transition participates.
    assert torch.count_nonzero(embedding_gradient[[60, 63]]) == 0  # Only first inputs, not next inputs.
    assert model.backbone.transformer.ff_out.weight.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.predictor.parameters())


def test_causality_and_example_and_forward_isolation():
    model = build_model(small_configuration(), backend="naive").eval()
    inputs = encode_inputs(torch.tensor([[15, 0, 7, 1, 8], [15, 2, 9, 3, 10]]))
    changed = inputs.clone()
    changed[:, 3:] = 74
    with torch.no_grad(), fp32_context("cpu"):
        original = task_logits(model, inputs)
        modified = task_logits(model, changed)
        alone = task_logits(model, inputs[:1])
        repeated = task_logits(model, inputs)
    torch.testing.assert_close(original[:, :3], modified[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(original[:1], alone, rtol=3e-6, atol=1e-6)
    torch.testing.assert_close(original, repeated, rtol=0, atol=0)


def test_dense_fuzzy_microbatch_gradient_accumulation_matches_logical_batch():
    model = build_model(small_configuration(), backend="naive")
    reference = build_model(small_configuration(), backend="naive")
    inputs = encode_inputs(torch.tensor([[15, 0, 7, 1], [15, 2, 9, 3]]))
    labels = torch.tensor([[0, 7, 1, 8], [2, 9, 3, 10]])
    with fp32_context("cpu"):
        task_loss(reference, inputs, labels)["loss"].backward()
        for index in range(2):
            (0.5 * task_loss(model, inputs[index:index+1], labels[index:index+1])["loss"]).backward()
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert actual.grad is not None and expected.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=2e-6, msg=name)


def test_default_length_configuration_is_not_legacy_a5_limit():
    config = read_configuration()
    assert config["backbone"]["max_sequence_length"] == 1024
    assert config["backbone"]["mlp_hidden_size"] == 512
    assert config["predictor_hidden_width"] == 128
    assert config["latent_weight"] == 1.0
