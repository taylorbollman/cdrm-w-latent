"""CPU observer/leaf contracts, not CUDA Flash numerical qualification."""
import math

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained import olmo
from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_campaign_attention_local import (
    attention_context, capture_attention, fixed_leaf, layout_record, local_vjp, parse_args,
)
from scripts.olmo_campaign_ddp_probe import fixture_for_update
from scripts.olmo_campaign_precision_components import component_backward
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_only_threads():
    torch.set_num_threads(1)


def model_fixture():
    torch.manual_seed(518)
    recipe = CampaignRecipe("NFR", sequence_length=8, rt_layers=(0,))
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32",
        ordinary_activation_checkpointing=True, tile_backend="eager", backward_tile_backend="eager",
        backward_memory="recompute", reuse_rope=True, kv_only_writes=True)
    model = build_campaign_model(base, recipe).train()
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0, length=8,
        token_ids=list(range(2, 59)), eos_id=60, batch_size=1) for rank in range(2)]
    return model, recipe, fixtures


def test_observer_preserves_checkpointed_model_outputs_gradients_rng_and_excludes_replay():
    model, recipe, fixtures = model_fixture()
    original = olmo.F
    before = torch.get_rng_state().clone()
    expected = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    with capture_attention(model, fixtures, layers=(1,), passes=(0, 3)) as observed:
        assert olmo.F is not original
        assert torch.nn.functional is original
        actual = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
    assert olmo.F is original and torch.nn.functional is original
    assert actual == expected
    assert torch.equal(torch.get_rng_state(), before)
    assert observed["forward_calls"] == 2
    assert observed["original_sdpa_calls"] == 8
    assert observed["excluded_recompute_sdpa_calls"] > 0
    assert len(observed["sites"]) == 4
    assert {(s["record"], s["pass"], s["layer"]) for s in observed["sites"]} == {
        (0, 0, 1), (0, 3, 1), (1, 0, 1), (1, 3, 1)}
    for site in observed["sites"]:
        assert site["cotangent_calls"] == 1
        assert set(site["tensors"]) == {"query", "key", "value", "output", "cotangent"}
        assert all(t.device.type == "cpu" and not t.requires_grad for t in site["tensors"].values())
    assert all(torch.equal(p.grad, gradients[name]) for name, p in model.named_parameters() if p.grad is not None)


def test_proxy_and_outer_hooks_restore_when_checkpointed_forward_raises(monkeypatch):
    model, recipe, fixtures = model_fixture()
    original = olmo.F
    hooks = (len(model.backbone._forward_pre_hooks), len(model.backbone._forward_hooks))

    def fail(*args, **kwargs):
        raise RuntimeError("injected ordinary-block failure")

    monkeypatch.setattr(model.backbone.backbone.layers[1], "forward", fail)
    with pytest.raises(RuntimeError, match="injected ordinary-block failure"):
        with capture_attention(model, fixtures, layers=(1,), passes=(0, 3)):
            component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
    assert olmo.F is original and torch.nn.functional is original
    assert (len(model.backbone._forward_pre_hooks), len(model.backbone._forward_hooks)) == hooks


def attention_fixture():
    generator = torch.Generator().manual_seed(202)
    packed = torch.randn(2, 4, 3*2*8, generator=generator).bfloat16()
    qkv = tuple(v.view(2, 4, 2, 8).transpose(1, 2) for v in packed.split(16, dim=-1))
    cotangent = torch.randn(2, 4, 2, 8, generator=generator).bfloat16().transpose(1, 2)
    cotangent[1, :, 2:] = 0
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(*qkv, is_causal=True)
    tensors = dict(zip(("query", "key", "value"), qkv))
    tensors.update(output=output, cotangent=cotangent)
    return {"key": "CPU fixed attention", "tensors": {k: t.clone() for k, t in tensors.items()},
            "layouts": {k: layout_record(t) for k, t in tensors.items()},
            "valid_queries": torch.tensor([[True]*4, [True, True, False, False]]),
            "attention": {"attn_mask": None, "dropout_p": 0., "is_causal": True,
                          "scale": None, "enable_gqa": False}}


def test_fixed_leaves_preserve_gapped_strides_values_and_independent_storage():
    site = attention_fixture()
    original = site["tensors"]["value"]
    layout = site["layouts"]["value"]
    assert layout["storage_offset"] > 0
    leaf = fixed_leaf(original, layout, device="cpu", dtype=torch.float32, requires_grad=True)
    assert leaf.is_leaf and leaf.requires_grad and leaf.dtype == torch.float32
    assert leaf.stride() == tuple(layout["stride"]) and leaf.storage_offset() == 0
    assert leaf.data_ptr() != original.data_ptr()
    assert torch.equal(leaf, original.float())
    (leaf*2).sum().backward()
    assert torch.equal(leaf.grad, torch.full_like(leaf, 2.))
    assert original.grad is None


def test_common_cotangent_fp32_vjp_matches_independent_attention_oracle_and_keeps_inputs_fixed():
    site = attention_fixture()
    before = tree_digests(site["tensors"])
    result = local_vjp(site, path="fp32_math", device="cpu")
    q, k, v = (site["tensors"][name].float().detach().clone().requires_grad_() for name in ("query", "key", "value"))
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    probabilities = (q @ k.transpose(-2, -1) / math.sqrt(8)).masked_fill(~causal, -torch.inf).softmax(-1)
    output = probabilities @ v
    expected = torch.autograd.grad(output, (q, k, v), site["tensors"]["cotangent"].float())
    torch.testing.assert_close(result["values"]["output"], output.detach(), atol=3e-7, rtol=2e-6)
    for name, gradient in zip(("query", "key", "value"), expected):
        torch.testing.assert_close(result["values"][name], gradient, atol=8e-7, rtol=2e-5)
        assert result["input_layouts"][name]["stride"] == site["layouts"][name]["stride"]
        assert result["input_digests"][name] == tree_digests(site["tensors"][name].float())
        assert torch.count_nonzero(result["values"][name][1, :, 2:]) == 0
    assert result["cotangent_digest"] == tree_digests(site["tensors"]["cotangent"].float())
    assert before == tree_digests(site["tensors"])
    assert result["finite"]
    assert all(result["common_value_checks"].values())


def test_bf16_math_uses_same_represented_values_and_cotangent_without_precision_fallback():
    site = attention_fixture()
    result = local_vjp(site, path="bf16_math", device="cpu")
    assert result["finite"]
    assert all(result["common_value_checks"].values())
    assert all(t.dtype == torch.bfloat16 for t in result["values"].values())
    for name in ("query", "key", "value"):
        assert result["input_digests"][name] == tree_digests(site["tensors"][name])
    assert result["cotangent_digest"] == tree_digests(site["tensors"]["cotangent"])
    assert torch.equal(result["values"]["output"], site["tensors"]["output"])
    with pytest.raises(ValueError, match="no CPU fallback"):
        local_vjp(site, path="bf16_flash", device="cpu")
    site["tensors"]["cotangent"] = site["tensors"]["cotangent"].float()
    with pytest.raises(ValueError, match="BF16 Q/K/V"):
        local_vjp(site, path="fp32_math", device="cpu")


def test_logit_context_excludes_padded_queries_and_respects_causality():
    site = attention_fixture()
    site["tensors"]["query"].zero_()
    # Huge logits at ignored queries must not enter a valid-query summary.
    site["tensors"]["query"][1, :, 2:] = 1000
    context = attention_context(site)
    assert context["valid_query_heads"] == 12
    assert context["logit_row_range"] == {"min": 0., "mean": 0., "max": 0.}
    assert context["max_attention_probability"]["max"] == 1.
    assert context["max_attention_probability"]["min"] == .25
    expected = (1+.5+1/3+.25+1+.5)/6
    assert context["max_attention_probability"]["mean"] == pytest.approx(expected)


@pytest.mark.parametrize("extra", [["--steps", "5"], ["--layers", "all"], ["--length", "1024"]])
def test_cli_stays_bounded(extra):
    with pytest.raises(SystemExit):
        parse_args(["--reference-report", "reference.json", "--reference-sha256", "a"*64,
                    "--output-dir", "/tmp/attention-local", *extra])
