"""Independent FP64 arithmetic and directional derivatives for adjudication."""
import math
import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from scripts.olmo_tiled_roundoff import fp64_block_oracle, roundoff_check, _fixed_native_rope_constants, _rotate64


def fixture():
    torch.manual_seed(71)
    config = OLMoConfig(model_dim=8, num_heads=2, mlp_intermediate_size=16,
                        num_layers=1, vocab_size=67, tokenizer_vocab_size=61, eos_token_id=60)
    layer = OLMoBlock(config)
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([[3, 7, 12, 16], [19, 24, 27, 31]])
    valid = torch.tensor([[True, True, False, True], [False, False, True, True]])
    return layer, x, positions, valid


def test_alpha_zero_oracle_matches_parallel_fp64_sdpa_ordinary_block():
    layer, x, positions, _ = fixture()
    valid = torch.ones(x.shape[:2], dtype=torch.bool)
    actual, _ = fp64_block_oracle(layer, x, positions, valid, 0.0)
    c = layer.config
    wq, wo, wf, wd = [p.detach().double() for p in (layer.att_proj.weight, layer.attn_out.weight,
                                                 layer.ff_proj.weight, layer.ff_out.weight)]
    normalized = F.layer_norm(x, (c.model_dim,), eps=c.layer_norm_eps)
    q, k, v = [v.reshape(2, 4, c.num_heads, c.head_dim).transpose(1, 2)
               for v in F.linear(normalized, wq).chunk(3, dim=-1)]
    cosine, sine = _fixed_native_rope_constants(positions, c.head_dim, c.rope_freq_constant)
    q, k = _rotate64(q, cosine, sine), _rotate64(k, cosine, sine)
    attended = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape_as(x)
    residual = x + F.linear(attended, wo)
    up, gate = F.linear(F.layer_norm(residual, (c.model_dim,), eps=c.layer_norm_eps), wf).chunk(2, -1)
    expected = residual + F.linear(F.silu(gate) * up, wd)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * probe).sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * probe).sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
def test_fp64_fractional_recurrence_cache_cotangents_match_directional_finite_difference(alpha):
    layer, x, positions, valid = fixture()
    output, cache = fp64_block_oracle(layer, x, positions, valid, alpha)
    probes = [torch.randn_like(value) for value in (output, *cache)]
    loss = sum((value * probe).sum() for value, probe in zip((output, *cache), probes))
    gradient = torch.autograd.grad(loss, x)[0]
    direction = torch.randn_like(x); direction /= direction.norm()
    def loss_at(value):
        out, pair = fp64_block_oracle(layer, value, positions, valid, alpha)
        return sum((item * probe).sum() for item, probe in zip((out, *pair), probes))
    step = 1e-5
    difference = (loss_at(x.detach() + step * direction) - loss_at(x.detach() - step * direction)) / (2 * step)
    torch.testing.assert_close(difference, (gradient * direction).sum(), atol=2e-8, rtol=2e-8)
    assert torch.isfinite(gradient).all()


def test_roundoff_report_keeps_original_screen_and_does_not_modify_source_layer():
    layer, x, positions, valid = fixture()
    x = x.detach().float()
    for p in layer.parameters():
        p.grad = torch.full_like(p, 0.125)
    before = [(p.detach().clone(), p.grad.clone(), p.requires_grad) for p in layer.parameters()]
    go = torch.randn_like(x)
    gk = torch.randn(2, layer.config.num_heads, 4, layer.config.head_dim)
    gv = torch.randn_like(gk)
    report = roundoff_check(layer, x, positions, valid, go, gk, gv, 0.37)
    assert report["original_elementwise_screen"]["atol"] == 2e-6
    assert report["original_elementwise_screen"]["rtol"] == 3e-4
    assert report["original_elementwise_screen"]["coordinates"] == x.numel()
    assert report["input_gradient"]["tiled_vs_fp64"]["relative_l2"] < 1e-5
    assert report["input_gradient"]["scan_vs_fp64"]["relative_l2"] < 1e-5
    assert report["worst_budget_coordinates"]
    assert all(math.isfinite(row["oracle_fp64"]) for row in report["worst_budget_coordinates"])
    for parameter, (data, grad, required) in zip(layer.parameters(), before):
        torch.testing.assert_close(parameter, data, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, grad, rtol=0, atol=0)
        assert parameter.requires_grad == required


def test_oracle_rejects_accidental_fp32_execution():
    layer, x, positions, valid = fixture()
    with pytest.raises(ValueError, match="float64"):
        fp64_block_oracle(layer, x.float(), positions, valid, 0.37)
