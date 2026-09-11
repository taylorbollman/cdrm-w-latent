"""Explicit CPU unit tests for fixed-state precision diagnostic controls."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cdrm_precision_probe as probe


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.att_proj = torch.nn.Linear(4, 12, bias=False)
        self.attn_out = torch.nn.Linear(4, 4, bias=False)

    @classmethod
    def _cast_attn_bias(cls, bias, dtype):
        return bias.to(dtype)

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, **kwargs):
        return q * .2 + k * .3 + v * .5

    def forward(self, x, attention_bias=None):
        q, k, v = self.att_proj(x).chunk(3, dim=-1)
        bias = self._cast_attn_bias(attention_bias, q.dtype)
        out = self._scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0., is_causal=False)
        return x + self.attn_out(out), None


class Side(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.deep_adapter = torch.nn.Linear(4, 4, bias=False)
        self.bridge_adapter = torch.nn.Linear(4, 4, bias=False)

    def forward(self, early, late, owner, attention_bias, output_states=False):
        shared = owner.att_proj(early)[..., :4]
        self.last_projection_dtype = shared.dtype
        proposed = shared + self.deep_adapter(late - early)
        return late + self.config.cdrm_lambda * self.bridge_adapter(proposed), None


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(cdrm_lambda=.01)
        self.transformer = torch.nn.ModuleDict({"blocks": torch.nn.ModuleList([Block() for _ in range(5)]),
                                                "ln_f": torch.nn.Identity(),
                                                "ff_out": torch.nn.Linear(4, 2, bias=False)})
        self.cdrm = Side(self.config)

    def forward(self, x):
        bias = torch.zeros(1, 1, 2, 2)
        early = None
        for index, block in enumerate(self.transformer.blocks):
            x, _ = block(x, attention_bias=bias)
            if index == 1:
                early = x
            if index == 3:
                x, _ = self.cdrm(early, x, self.transformer.blocks[1], bias)
        return self.transformer.ff_out(self.transformer.ln_f(x))


def test_capture_preserves_forward_gradients_and_restores_class_lookup():
    torch.manual_seed(871)
    model = Toy()
    copy_model = copy.deepcopy(model)
    x = torch.randn(2, 2, 4, requires_grad=True)
    copy_x = x.detach().clone().requires_grad_()
    expected = copy_model(copy_x)
    expected.square().sum().backward()
    capture = {}
    with probe.intervention(model, probe.ARMS["current_fp32"], capture):
        observed = model(x)
        observed.square().sum().backward()
    assert torch.equal(expected, observed) and torch.equal(x.grad, copy_x.grad)
    assert not probe.exact_tree({n: p.grad for n, p in model.named_parameters()},
                                {n: p.grad for n, p in copy_model.named_parameters()})
    probe.validate_capture(capture)
    for block in model.transformer.blocks:
        assert all(name not in block.__dict__ for name in ("forward", "_cast_attn_bias", "_scaled_dot_product_attention"))
    assert set(capture["blocks"]["0"]["sdpa"]) >= {"q_gradient", "k_gradient", "v_gradient", "output_gradient", "bias_before_cast"}


def test_precision_islands_apply_to_ordinary_calls_and_leave_shared_children_alone():
    torch.manual_seed(872)
    model = Toy()
    capture = {}
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with probe.intervention(model, probe.ARMS["bf16_block_1_fp32"], capture):
            output = model(torch.randn(2, 2, 4, requires_grad=True))
        output.float().square().sum().backward()
    assert capture["blocks"]["1"]["sdpa"]["q"].dtype == torch.float32
    assert capture["blocks"]["0"]["sdpa"]["q"].dtype == torch.bfloat16
    assert model.cdrm.last_projection_dtype == torch.bfloat16
    assert identities == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert "forward" not in model.transformer.blocks[1].att_proj.__dict__


def test_exception_restores_lambda_reduction_flags_and_bypass_method():
    model = Toy()
    reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    arm = probe.Arm("combined-test", False, lambda_zero=True, bypass=True, reduction_off=True)
    with pytest.raises(RuntimeError, match="intentional"):
        with probe.intervention(model, arm, {}):
            assert model.config.cdrm_lambda == 0.
            assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            raise RuntimeError("intentional")
    assert model.config.cdrm_lambda == .01
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == reduction
    assert "forward" not in model.cdrm.__dict__


def test_bypass_and_zero_match_backbone_but_native_adam_none_is_distinct():
    torch.manual_seed(873)
    model = Toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., foreach=False, fused=False)
    x = torch.randn(2, 2, 4)
    model(x).square().sum().backward()
    optimizer.step()  # Nonzero adapter moments are necessary for this distinction.
    initial = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    packets, steps = {}, {}
    for name in ("fp32_lambda_zero", "fp32_bypass"):
        model.load_state_dict(initial[0])
        optimizer.load_state_dict(copy.deepcopy(initial[1]))
        model.zero_grad(set_to_none=True)
        with probe.intervention(model, probe.ARMS[name]):
            output = model(x)
            output.square().sum().backward()
        packets[name] = {"parameters": {n: probe.cpu(p.grad) for n, p in model.named_parameters()}, "logits": probe.cpu(output)}
        steps[name] = probe.step_packet(model, optimizer, 1.)
    assert torch.equal(packets["fp32_lambda_zero"]["logits"], packets["fp32_bypass"]["logits"])
    for name in probe.ADAPTERS:
        assert packets["fp32_bypass"]["parameters"][name] is None
        assert not torch.count_nonzero(packets["fp32_lambda_zero"]["parameters"][name])
        assert not torch.equal(steps["fp32_lambda_zero"]["weights"][name], steps["fp32_bypass"]["weights"][name])
    model.load_state_dict(initial[0])
    optimizer.load_state_dict(copy.deepcopy(initial[1]))
    probe.restore_raw_gradients(model, packets["fp32_bypass"], zero_unused_adapters=True)
    assert not probe.exact_tree(steps["fp32_lambda_zero"], probe.step_packet(model, optimizer, 1.))


def test_selection_always_includes_exact_and_matching_function_references():
    selected = probe.selected_arms(["bf16_bypass"])
    assert selected[:2] == probe.BASELINES
    assert set(selected) == {*probe.BASELINES, "fp32_bypass", "bf16_bypass", "fp32_lambda_zero", "bf16_lambda_zero"}
    assert not any(name.startswith("bf16_block_") for name in probe.selected_arms(None))
    assert "bf16_alibi_fp32" not in probe.selected_arms(None)
    assert "bf16_attention_fp32" not in probe.selected_arms(None)
    with pytest.raises(ValueError, match="Unknown"):
        probe.selected_arms(["unfrozen-new-arm"])


def test_capture_rejects_missing_or_duplicate_native_adjoints():
    model = Toy()
    capture = {}
    with probe.intervention(model, probe.ARMS["current_fp32"], capture):
        model(torch.randn(2, 2, 4, requires_grad=True)).sum().backward()
    capture["blocks"]["2"]["sdpa"]["gradient_hook_calls"]["q"] = 2
    with pytest.raises(AssertionError, match="Missing/duplicate"):
        probe.validate_capture(capture)


def test_per_arm_compiler_total_retains_failure_and_all_previous_counts():
    records = {"first": {"passed": True, "counters": {"stats": {"unique_graphs": 4}}},
               "second": {"passed": False, "counters": {"stats": {"unique_graphs": 2}, "unimplemented": {"limit": 1}}}}
    summary = probe.compiler_totals(records)
    assert summary["counters"]["stats"]["unique_graphs"] == 6
    assert summary["counters"]["unimplemented"]["limit"] == 1
    assert not summary["all_arms_pass"] and summary["failed_arms"] == ["second"]


@pytest.fixture
def ordinary_model():
    from cdrm_tiled_common import cpu_initial_model
    model, _ = cpu_initial_model(7500)
    return model


def inputs_and_bias():
    generator = torch.Generator().manual_seed(874)
    x = torch.randn(2, 5, 128, generator=generator, requires_grad=True)
    positions = torch.arange(5)
    bias = -(positions[:, None] - positions[None, :]).abs().float()[None, None] / 7
    return x, bias.expand(1, 16, 5, 5).contiguous()


def test_attention_region_reproduces_existing_fp32_block_and_derivatives(ordinary_model):
    block = ordinary_model.transformer.blocks[0]
    reference = copy.deepcopy(block)
    x, bias = inputs_and_bias()
    ref_x = x.detach().clone().requires_grad_()
    expected, _ = reference(ref_x, attention_bias=bias)
    actual, _ = probe.ordinary_attention_fp32(block, x, attention_bias=bias)
    cotangent = torch.randn(actual.shape, generator=torch.Generator().manual_seed(875))
    expected.backward(cotangent)
    actual.backward(cotangent)
    assert torch.equal(expected, actual) and torch.equal(ref_x.grad, x.grad)
    assert not probe.exact_tree({n: p.grad for n, p in reference.named_parameters()},
                                {n: p.grad for n, p in block.named_parameters()})


def test_complete_attention_fp32_keeps_ordinary_mlp_bf16(ordinary_model):
    block = ordinary_model.transformer.blocks[0]
    x, bias = inputs_and_bias()
    mlp_dtypes = []
    hook = block.ff_out.register_forward_hook(lambda module, args, output: mlp_dtypes.append(output.dtype))
    capture = {}
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with probe.intervention(ordinary_model, probe.ARMS["bf16_attention_fp32"], capture):
                result, _ = block(x, attention_bias=bias)
            result.square().sum().backward()
    finally:
        hook.remove()
    assert capture["blocks"]["0"]["sdpa"]["q"].dtype == torch.float32
    assert capture["blocks"]["0"]["sdpa"]["mask"].dtype == torch.float32
    assert mlp_dtypes == [torch.bfloat16]
    assert "forward" not in block.att_proj.__dict__ and "forward" not in block.ff_out.__dict__


def test_alibi_fp32_keeps_exact_bf16_query_key_value_operands(ordinary_model):
    captures = {}
    x, bias = inputs_and_bias()
    for name in ("current_bf16", "bf16_alibi_fp32"):
        capture = {}
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with probe.intervention(ordinary_model, probe.ARMS[name], capture):
                ordinary_model.transformer.blocks[0](x, attention_bias=bias)
        captures[name] = capture["blocks"]["0"]["sdpa"]
    native, controlled = captures.values()
    assert native["mask"].dtype == torch.bfloat16
    assert controlled["mask"].dtype == torch.float32
    assert all(torch.equal(native[name], controlled[name]) for name in ("q", "k", "v"))
    assert all(controlled[name].dtype == torch.bfloat16 for name in ("q", "k", "v"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with probe.intervention(ordinary_model, probe.ARMS["bf16_alibi_fp32"]):
            unobserved, _ = ordinary_model.transformer.blocks[0](x, attention_bias=bias)
        with probe.intervention(ordinary_model, probe.ARMS["bf16_alibi_fp32"], {}):
            observed, _ = ordinary_model.transformer.blocks[0](x, attention_bias=bias)
    assert torch.equal(unobserved, observed)
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Explicit CUDA-only primitive test")
def test_cuda_alibi_control_preserves_effective_fp32_mask(ordinary_model):
    from stage_a_common import require_cuda_container
    require_cuda_container()
    block = ordinary_model.transformer.blocks[0]
    q = torch.zeros(1, 1, 2, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    v[:, :, 1] = 1
    # BF16 rounds 64.2 to64. A common shift leaves FP32 attention unchanged,
    # making the actual mask precision visible in the returned BF16 output.
    mask = torch.tensor([[[[64., 64.2], [64., 64.2]]]], device="cuda", dtype=torch.float32)
    with torch.autocast("cuda", enabled=False):
        strict = block._scaled_dot_product_attention(q, k, v, attn_mask=mask)
        rounded = block._scaled_dot_product_attention(q, k, v, attn_mask=mask.bfloat16())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        native = block._scaled_dot_product_attention(q, k, v, attn_mask=mask)
        with probe.intervention(ordinary_model, probe.ARMS["bf16_alibi_fp32"]):
            controlled = block._scaled_dot_product_attention(q, k, v, attn_mask=block._cast_attn_bias(mask, q.dtype))
        capture = {}
        with probe.intervention(ordinary_model, probe.ARMS["bf16_alibi_fp32"], capture):
            observed = block._scaled_dot_product_attention(q, k, v, attn_mask=block._cast_attn_bias(mask, q.dtype))
    assert all(t.dtype == torch.bfloat16 for t in (q, k, v, strict, rounded, native, controlled, observed))
    assert torch.equal(native, rounded) and not torch.equal(native, strict)
    assert torch.equal(controlled, strict) and torch.equal(observed, strict)
    assert capture["blocks"]["0"]["sdpa"]["operation_autocast_disabled_by_control"]
