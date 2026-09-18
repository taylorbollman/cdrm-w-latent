"""Bounded semantics and FP32 gradient checks for the new permanent-value path.

CPU tests exercise the independent ordinary-autograd reference. The explicit
CUDA test must be scheduled after the active experiment, inside the container.
"""
import json
import math

import pytest
import torch

from scripts.rt_a5_common import configure_fp32_runtime
from scripts.rt_a5_l1r_depth import build_model
from scripts.rt_a5_value_bypass import (
    ValueBypassRecurrentAutogradBlock, ValueBypassRecurrentBlockTiled,
    _PermanentValueProjection, replace_value_bypass_block,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def block(backend="naive", width=64):
    return build_model(width=width, n_layers=4, backend=backend,
                       predictor_hidden_width=256 if width == 64 else None).backbone.transformer.blocks[1]


def manual_scan(original, x, addition, bias):
    """Independent write-site oracle: add only after splitting permanent KV."""
    batch, length, width = x.shape
    heads, head_dim = original.config.n_heads, width // original.config.n_heads
    q, ki, vi = original.pre_attention_block(x)
    keys, values, outputs = [], [], []
    for t in range(length):
        k = torch.cat(keys + [ki[:, :, t:t+1]], dim=2)
        v = torch.cat(values + [vi[:, :, t:t+1]], dim=2)
        logits = (q[:, :, t:t+1] @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        if bias is not None:
            logits = logits + bias[:, :, t:t+1, :t+1]
        attention = (logits.softmax(-1) @ v).transpose(1, 2).reshape(batch, 1, width)
        out = original.post_attention_block(x[:, t:t+1], original.attn_out(attention))
        outputs.append(out)
        pk, pv = original.kv_proj(original.attn_norm(out)).chunk(2, -1)
        if original.k_norm is not None:
            pk = original.k_norm(pk)
        pv = pv + addition[:, t:t+1]
        keys.append(pk.reshape(batch, 1, heads, head_dim).transpose(1, 2))
        values.append(pv.reshape(batch, 1, heads, head_dim).transpose(1, 2))
    return torch.cat(outputs, dim=1)


def draw(width=64, length=5, batch=2, device="cpu"):
    generator = torch.Generator().manual_seed(743)
    x = torch.randn(batch, length, width, generator=generator).to(device).requires_grad_()
    raw = torch.randn(batch, length, width, generator=generator).to(device).requires_grad_()
    weight = (torch.randn(width, width, generator=generator) / math.sqrt(width)).to(device).requires_grad_()
    bias = torch.randn(1, width // 64, length, length, generator=generator).to(device) * .1
    adjoint = torch.randn(batch, length, width, generator=generator).to(device)
    return x, raw, weight, bias, adjoint


def gradients(model, inputs, *, manual=False):
    x, raw, weight, bias, adjoint = inputs
    addition = .02 * torch.nn.functional.linear(raw, weight)
    out = manual_scan(model, x, addition, bias) if manual else model(x, attention_bias=bias, value_bypass=addition)[0]
    (out * adjoint).sum().backward()
    result = {name: p.grad for name, p in model.named_parameters()}
    result.update(x=x.grad, raw_embeddings=raw.grad, projection_weight=weight.grad)
    assert all(value is not None and torch.isfinite(value).all() for value in result.values())
    return out.detach(), result


@pytest.mark.parametrize("backend", ["naive", "tiled"])
def test_replacement_preserves_weights_names_owned_views_and_rng(backend):
    original = block(backend)
    state = torch.get_rng_state().clone()
    replaced = replace_value_bypass_block(original)
    assert torch.equal(state, torch.get_rng_state())
    assert type(replaced) is (ValueBypassRecurrentAutogradBlock if backend == "naive" else ValueBypassRecurrentBlockTiled)
    assert list(replaced.state_dict()) == list(original.state_dict())
    assert all(torch.equal(value, replaced.state_dict()[name]) for name, value in original.state_dict().items())
    assert replaced.pre_attention_block.kv_proj is replaced.kv_proj
    assert replaced.post_attention_block.ff_proj is replaced.ff_proj
    assert len(list(replaced.parameters())) == len({id(p) for p in replaced.parameters()})


def test_write_proxy_changes_only_permanent_value_and_checks_call_count():
    original = block()
    x, raw, weight, _, _ = draw(length=3)
    addition = .02 * torch.nn.functional.linear(raw, weight)
    proxy = _PermanentValueProjection(original, addition)
    qkv_before = original.pre_attention_block(x)
    with pytest.raises(ValueError, match="every token"):
        proxy.finish()
    for t in range(3):
        expected = original.kv_proj(x[:, t:t+1])
        got = proxy(x[:, t:t+1])
        assert torch.equal(got[..., :64], expected[..., :64])
        assert torch.equal(got[..., 64:], expected[..., 64:] + addition[:, t:t+1])
    proxy.finish()
    assert all(torch.equal(a, b) for a, b in zip(qkv_before, original.pre_attention_block(x)))
    with pytest.raises(ValueError, match="order/shape"):
        proxy(x[:, :1])


def test_naive_value_path_matches_independent_scan_and_all_gradients():
    original = block()
    replaced = replace_value_bypass_block(original)
    actual, actual_grads = gradients(replaced, draw())
    expected, expected_grads = gradients(original, draw(), manual=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert actual_grads.keys() == expected_grads.keys()
    for name in actual_grads:
        torch.testing.assert_close(actual_grads[name], expected_grads[name], rtol=3e-5, atol=3e-6, msg=name)
    assert torch.count_nonzero(actual_grads["raw_embeddings"][:, -1]) == 0
    assert torch.count_nonzero(actual_grads["raw_embeddings"][:, :-1]) > 0
    assert torch.count_nonzero(actual_grads["projection_weight"]) > 0


def test_zero_bypass_is_original_and_last_write_has_no_future_consumer():
    original = block()
    replaced = replace_value_bypass_block(original)
    x, raw, weight, bias, _ = draw()
    with torch.no_grad():
        baseline = original(x, attention_bias=bias)[0]
        assert torch.equal(baseline, replaced(x, attention_bias=bias)[0])
        assert torch.equal(baseline, replaced(x, attention_bias=bias, value_bypass=torch.zeros_like(x))[0])
        addition = .02 * torch.nn.functional.linear(raw, weight)
        a = replaced(x, attention_bias=bias, value_bypass=addition)[0]
        changed = addition.clone(); changed[:, -1] += 100
        b = replaced(x, attention_bias=bias, value_bypass=changed)[0]
        assert torch.equal(a, b)
        assert torch.equal(a[:, :1], baseline[:, :1])
        assert not torch.equal(a[:, 1:], baseline[:, 1:])


def test_per_call_write_state_is_not_shared_between_forwards():
    original = block()
    replaced = replace_value_bypass_block(original)
    x, raw, weight, bias, _ = draw()
    addition = .02 * torch.nn.functional.linear(raw, weight)
    first = replaced(x, attention_bias=bias, value_bypass=addition)[0]
    second = replaced(x, attention_bias=bias, value_bypass=-addition)[0]
    torch.testing.assert_close(first, manual_scan(original, x, addition, bias))
    torch.testing.assert_close(second, manual_scan(original, x, -addition, bias))


def test_cpu_tiled_path_and_wrong_target_are_rejected():
    tiled = replace_value_bypass_block(block("tiled"))
    x, _, _, bias, _ = draw()
    with pytest.raises(ValueError, match="requires CUDA"):
        tiled(x, attention_bias=bias, value_bypass=torch.zeros_like(x))
    wrong = build_model(width=64, n_layers=4, backend="naive", predictor_hidden_width=256).backbone.transformer.blocks[2]
    with pytest.raises(ValueError, match="index1"):
        replace_value_bypass_block(wrong)


@pytest.mark.parametrize("length", [3, 12, 36])
def test_cuda_tiled_value_gradients_match_naive(length):
    """Run explicitly after current training, on the project-container GPU."""
    from pathlib import Path
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        pytest.fail("CUDA comparison must run inside the project container")
    if not torch.cuda.is_available():
        pytest.skip("Explicit CUDA comparison was not requested on this CPU-only invocation")
    configure_fp32_runtime()
    tiled = replace_value_bypass_block(block("tiled", width=512)).cuda()
    naive = replace_value_bypass_block(block("naive", width=512)).cuda()
    naive.load_state_dict(tiled.state_dict(), strict=True)
    actual, actual_grads = gradients(tiled, draw(width=512, length=length, device="cuda"))
    expected, expected_grads = gradients(naive, draw(width=512, length=length, device="cuda"))
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    errors = {}
    for name in actual_grads:
        a, b = actual_grads[name], expected_grads[name]
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5, msg=name)
        errors[name] = {"max_abs": (a-b).abs().max().item(), "relative_l2": ((a-b).norm()/b.norm().clamp_min(1e-12)).item()}
    assert torch.count_nonzero(actual_grads["raw_embeddings"][:, -1]) == 0
    assert torch.count_nonzero(actual_grads["raw_embeddings"][:, :-1]) > 0
    assert torch.count_nonzero(actual_grads["projection_weight"]) > 0
    print(json.dumps({"length": length, "width": 512, "batch": 2, "gradients_checked": len(errors), "errors": errors}))
