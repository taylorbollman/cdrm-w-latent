"""Explicit CPU policy checks and separately selected CUDA wrapper-parity tests."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from olmo.cdrm import persistent_kv3, post3, pre3
from olmo.config import ActivationCheckpointingStrategy, ModelConfig
from olmo.exceptions import OLMoConfigurationError
from olmo.model import OLMo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cdrm_precision_probe as archived_probe


def config(**overrides):
    values = json.loads((ROOT / "configs/cdrm/base_d128_5.json").read_text())
    values.update(d_model=32, n_heads=4, n_kv_heads=4, mlp_hidden_size=128,
                  n_layers=12, cdrm_early_layer=3, cdrm_late_layer=8,
                  max_sequence_length=32, ordinary_attention_precision_policy="fp32")
    values.update(overrides)
    return ModelConfig(**values)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(9581)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with sdpa_kernel(SDPBackend.MATH):
        yield


def tensors_equal(expected, actual):
    assert expected.dtype == actual.dtype and expected.shape == actual.shape
    assert torch.equal(expected.detach().contiguous().reshape(-1).view(torch.uint8),
                       actual.detach().contiguous().reshape(-1).view(torch.uint8))


def test_default_and_serialized_legacy_configs_preserve_existing_fp32_path():
    original = json.loads((ROOT / "configs/cdrm/base_d128_5.json").read_text())
    assert "ordinary_attention_precision_policy" not in original
    assert ModelConfig(**original).ordinary_attention_precision_policy == "legacy"
    assert ModelConfig(**asdict(config())).ordinary_attention_precision_policy == "fp32"
    legacy = OLMo(config(ordinary_attention_precision_policy="legacy"))
    selected = OLMo(config())
    selected.load_state_dict(legacy.state_dict(), strict=True)
    assert list(legacy.state_dict()) == list(selected.state_dict())
    tokens = torch.randint(0, 16, (2, 5))
    # Construction restores global SDPA flags, so set the reference runtime
    # backend after constructing both models, as the numerical harness does.
    with sdpa_kernel(SDPBackend.MATH):
        left, right = legacy(tokens).logits, selected(tokens).logits
    tensors_equal(left, right)
    cotangent = torch.randn_like(left)
    left_grads = torch.autograd.grad(left, tuple(legacy.parameters()), cotangent)
    right_grads = torch.autograd.grad(right, tuple(selected.parameters()), cotangent)
    for first, second in zip(left_grads, right_grads):
        tensors_equal(first, second)


@pytest.mark.parametrize("overrides", [
    {"ordinary_attention_precision_policy": "automatic"},
    {"recurrent_layers": [3]},
    {"recurrent_precision_policy": "bf16_fp32_state"},
    {"norm_after": True}, {"alibi": False}, {"rope": True},
    {"flash_attention": True}, {"n_kv_heads": 1}, {"block_group_size": 2},
    {"attention_dropout": .1}, {"residual_dropout": .1}, {"embedding_dropout": .1},
    {"clip_qkv": 1.}, {"precision": torch.bfloat16},
])
def test_policy_rejects_unsupported_configuration(overrides):
    with pytest.raises(OLMoConfigurationError):
        config(**overrides).validate_ordinary_attention_precision()


def test_all_twelve_ordinary_attention_regions_are_fp32_mlp_and_head_bf16():
    net = OLMo(config())
    canonical = [(name, id(value)) for name, value in net.named_parameters()]
    observed = {}
    hooks = []
    for i, block in enumerate(net.transformer.blocks):
        for name in ("attn_norm", "att_proj", "q_norm", "k_norm", "attn_out", "ff_norm", "ff_proj", "act", "ff_out"):
            def receive(module, inputs, output, key=(i, name)):
                observed[key] = (inputs[0].dtype, output.dtype, torch.is_autocast_enabled("cpu"))
            hooks.append(getattr(block, name).register_forward_hook(receive))
    hooks.append(net.transformer.ff_out.register_forward_hook(
        lambda module, inputs, output: observed.update(head=output.dtype)))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = net(torch.randint(0, 16, (2, 5))).logits
        torch.autograd.grad(logits.float().square().sum(), tuple(net.parameters()))
    finally:
        for hook in hooks:
            hook.remove()
    for i in range(12):
        for name in ("attn_norm", "att_proj", "q_norm", "k_norm", "attn_out"):
            assert observed[i, name] == (torch.float32, torch.float32, False), (i, name)
        assert observed[i, "ff_norm"] == (torch.float32, torch.float32, True)
        assert observed[i, "ff_proj"] == (torch.float32, torch.bfloat16, True)
        for name in ("act", "ff_out"):
            assert observed[i, name] == (torch.bfloat16, torch.bfloat16, True)
    assert observed["head"] == torch.bfloat16
    assert canonical == [(name, id(value)) for name, value in net.named_parameters()]


def test_explicit_policy_forces_math_sdpa_for_seq(monkeypatch):
    net = OLMo(config(n_layers=1, cdrm_early_layer=0, cdrm_late_layer=0))
    from torch.nn import attention
    original = attention.sdpa_kernel
    selected = []
    def record(backends, **kwargs):
        selected.append(backends)
        return original(backends, **kwargs)
    monkeypatch.setattr(attention, "sdpa_kernel", record)
    net(torch.randint(0, 16, (1, 3)))
    assert selected == [SDPBackend.MATH]


def test_shared_fabric_helpers_retain_independent_bf16_policy_on_cpu():
    # Explicit primitive unit check; no CUDA scan or CPU fallback is attempted.
    net = OLMo(config(cdrm_enabled=True, cdrm_backend="tiled", cdrm_precision_policy="bf16_fp32_state"))
    names = list(net.named_parameters(remove_duplicate=False))
    assert len(names) == len({id(value) for _, value in names})
    assert set(dict(net.cdrm.named_parameters())) == {"deep_adapter.weight", "bridge_adapter.weight"}
    owner = net.transformer.blocks[3]
    source = torch.randn(2, 5, 32, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        q, k, v = pre3(owner, source)
        pk, pv = persistent_kv3(owner, source)
        dense_read = owner.attn_out(source)
        proposed = post3(owner, source, dense_read)
    assert all(value.dtype == torch.bfloat16 for value in (q, k, v, pk, pv, dense_read))
    assert proposed.dtype == torch.float32
    assert all("forward" not in child.__dict__ for child in (owner.att_proj, owner.attn_out, owner.ff_proj, owner.ff_out))
    gradients = torch.autograd.grad(q.float().sum() + k.float().sum() + v.float().sum() + pk.float().sum()
                                   + pv.float().sum() + proposed.sum(), (source, *owner.parameters()))
    assert all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.parametrize("kwargs", [
    {"use_cache": True}, {"past_key_values": []},
    {"attention_mask": torch.ones(1, 3)}, {"attention_bias": torch.zeros(1, 1, 3, 3)},
    {"doc_lens": torch.tensor([[3]])}, {"max_doc_lens": [3]},
])
def test_explicit_policy_rejects_unsupported_model_inputs(kwargs):
    net = OLMo(config())
    with pytest.raises(OLMoConfigurationError, match="independent inputs"):
        net(torch.randint(0, 16, (1, 3)), **kwargs)


def test_runtime_guards_reject_checkpointing_half_parameters_and_fp16_autocast():
    net = OLMo(config())
    with pytest.raises(OLMoConfigurationError, match="checkpointing"):
        net.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    tokens = torch.randint(0, 16, (1, 3))
    with torch.autocast("cpu", dtype=torch.float16):
        with pytest.raises(OLMoConfigurationError, match="outer BF16"):
            net(tokens)
    net.bfloat16()
    with pytest.raises(OLMoConfigurationError, match="FP32 residuals and parameters"):
        net(tokens)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires explicitly selected CUDA container")


@CUDA
@pytest.mark.parametrize("cdrm", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_cuda_five_block_policy_matches_preserved_wrapper(cdrm, mixed):
    assert Path("/.dockerenv").exists(), "CUDA tests require the project container"
    torch.use_deterministic_algorithms(True)
    kwargs = dict(n_layers=5, d_model=128, n_heads=16, n_kv_heads=16, mlp_hidden_size=512,
                  cdrm_early_layer=1, cdrm_late_layer=3, cdrm_enabled=cdrm,
                  cdrm_backend="tiled", cdrm_precision_policy="bf16_fp32_state" if mixed else "fp32",
                  reference_eager=False)
    reference = OLMo(config(**kwargs, ordinary_attention_precision_policy="legacy")).cuda()
    selected = OLMo(config(**kwargs)).cuda()
    selected.load_state_dict(reference.state_dict(), strict=True)
    tokens = torch.randint(0, 16, (2, 17), device="cuda")
    cotangent = torch.randn(2, 17, 16, device="cuda")
    def reset_and_configure():
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        archived_probe.configure_compiled_helpers(True)

    def audit(name):
        # Emit each complete audit before resetting counters for the next arm.
        raw = archived_probe.compiler_audit(False, require_graphs=cdrm)
        print(json.dumps({"cdrm": cdrm, "mixed": mixed, "arm": name, "compiler": raw}), flush=True)
        archived_probe.compiler_audit(True, require_graphs=cdrm)

    reset_and_configure()
    with archived_probe.intervention(reference, archived_probe.ARMS["bf16_attention_fp32"]):
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=mixed, dtype=torch.bfloat16):
            first = reference(tokens, output_cdrm_states=cdrm)
        first_gradients = torch.autograd.grad(first.logits, tuple(reference.parameters()), cotangent)
    audit("preserved_wrapper")
    reset_and_configure()
    with torch.autocast("cuda", enabled=mixed, dtype=torch.bfloat16):
        second = selected(tokens, output_cdrm_states=cdrm)
    second_gradients = torch.autograd.grad(second.logits, tuple(selected.parameters()), cotangent)
    audit("model_option")
    tensors_equal(first.logits, second.logits)
    for left, right in zip(first_gradients, second_gradients):
        tensors_equal(left, right)
    if cdrm:
        assert list(first.cdrm_states) == list(second.cdrm_states)
        for key, value in first.cdrm_states.items():
            if torch.is_tensor(value):
                tensors_equal(value, second.cdrm_states[key])
            else:
                assert value == second.cdrm_states[key]
    assert all(value.grad is None for value in selected.parameters())
