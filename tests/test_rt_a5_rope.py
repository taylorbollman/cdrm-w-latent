"""Bounded correctness checks for the single positional-encoding ablation."""
from dataclasses import asdict
import math

import pytest
import torch

from olmo.model import OLMoSequentialBlock
from scripts.rt_a5_common import (
    build_model, canonical_parameter_sha256, configure_fp32_runtime,
    fp32_context, parameter_count, task_loss,
)
from scripts.rt_a5_rope import CONFIG_DELTA, build_rope_model, rope_model_config


@pytest.fixture(autouse=True)
def bounded_cpu_runtime():
    torch.set_num_threads(1)
    configure_fp32_runtime()


@pytest.mark.parametrize("width", [64, 128, 512])
def test_only_pe_config_changes_with_exact_original_initialization(width):
    torch.random.default_generator.manual_seed(91)
    rng = torch.get_rng_state().clone()
    original = build_model("seq", width=width, seed=1234)
    rope = build_rope_model(width=width, seed=1234)
    assert torch.equal(rng, torch.get_rng_state())
    before, after = asdict(original.config), asdict(rope.config)
    assert {key: {"from": before[key], "to": after[key]}
            for key in before if before[key] != after[key]} == CONFIG_DELTA
    assert list(original.state_dict()) == list(rope.state_dict())
    for name, value in original.state_dict().items():
        assert torch.equal(value, rope.state_dict()[name]), name
    assert canonical_parameter_sha256(rope) == canonical_parameter_sha256(original)
    assert rope.a5_initialization["paired_base_initialization"] == original.a5_initialization
    assert sum(p.numel() for p in rope.parameters()) == parameter_count(width)
    assert {p.data_ptr() for p in original.parameters()}.isdisjoint(
        p.data_ptr() for p in rope.parameters())
    assert all(type(block) is OLMoSequentialBlock and hasattr(block, "rotary_emb")
               and block.config.reference_eager for block in rope.transformer.blocks)
    assert "wpe" not in rope.transformer
    if width == 512:
        assert rope.a5_initialization["canonical_sha256"] == (
            "0380e2fd4cdd4db63ce6d732a0834ad054e185c2276da55d733839276d8c55ad")


@pytest.mark.parametrize("length", [12, 36])
def test_forward_backward_causality_and_effective_rope(length):
    rope = build_rope_model(width=128, seed=1234).eval()
    original = build_model("seq", width=128, seed=1234).eval()
    generator = torch.Generator().manual_seed(717)
    inputs = torch.randint(0, 60, (2, length), generator=generator)
    labels = torch.randint(0, 60, inputs.shape, generator=generator)
    changed = inputs.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 60
    with fp32_context("cpu"):
        logits = rope(inputs).logits
        loss = task_loss(logits, labels)
        loss.backward()
        with torch.no_grad():
            other = rope(changed).logits
            prefix = rope(inputs[:, :6]).logits
            baseline = original(inputs).logits
    assert logits.dtype == torch.float32 and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in rope.parameters())
    torch.testing.assert_close(logits[:, :6], other[:, :6], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(logits[:, :6], prefix, rtol=2e-5, atol=2e-6)
    assert (logits - baseline).abs().max() > 1e-3


@pytest.mark.parametrize("length", [12, 36])
def test_existing_attention_applies_manual_rope_after_qk_norm(length):
    model = build_rope_model(width=128, seed=1234).eval()
    block = model.transformer.blocks[0]
    generator = torch.Generator().manual_seed(92)
    q, k, v = [torch.randn(2, length, 128, generator=generator) for _ in range(3)]
    with torch.no_grad(), fp32_context("cpu"):
        actual, cache = block.attention(q, k, v)
        # Independent position formula and explicit normalized attention. Norms
        # operate on full D before head reshaping, as in the original SEQ.
        qh = block.q_norm(q).view(2, length, 2, 64).transpose(1, 2)
        kh = block.k_norm(k).view(2, length, 2, 64).transpose(1, 2)
        vh = v.view(2, length, 2, 64).transpose(1, 2)
        angles = torch.outer(torch.arange(length).float(),
                             10000.0 ** (-torch.arange(0, 64, 2).float() / 64))
        angles = torch.cat((angles, angles), dim=-1)[None, None]
        def rotate(value):
            left, right = value.chunk(2, dim=-1)
            return value * angles.cos() + torch.cat((-right, left), dim=-1) * angles.sin()
        qr, kr = rotate(qh), rotate(kh)
        scores = qr @ kr.transpose(-1, -2) / math.sqrt(64)
        scores.masked_fill_(~torch.ones(length, length, dtype=torch.bool).tril(), -torch.inf)
        expected = (scores.softmax(dim=-1) @ vh).transpose(1, 2).reshape(2, length, 128)
        expected = block.attn_out(expected)
    assert cache is None
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_metadata_seed_and_configuration_are_stable():
    first = build_rope_model(width=64, seed=1234)
    second = build_rope_model(width=64, seed=1234)
    assert first.a5_initialization == second.a5_initialization
    assert canonical_parameter_sha256(first) != canonical_parameter_sha256(
        build_rope_model(width=64, seed=1235))
    config = rope_model_config()
    assert config.rope_theta == 10000 and config.rope_full_precision
    assert first.a5_initialization["position_encoding"]["value_rotation"] is False
    with pytest.raises(ValueError, match="seed"):
        build_rope_model(seed=-1)
    with pytest.raises(ValueError, match="width"):
        build_rope_model(width=65)
