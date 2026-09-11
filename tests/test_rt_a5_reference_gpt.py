"""Bounded equivalence to SHA-pinned upstream model classes, with our harness.

Set A5_NEXTLAT_REFERENCE_DIR to a retained pristine source checkout/directory.
Otherwise the two exact model files are fetched from their immutable raw URLs.
Only model classes are AST-loaded; no upstream trainer, optimizer or Fabric
setup is imported. The unused fused-loss mixin is replaced by a no-op mixin.
"""
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from scripts.rt_a5_common import fp32_context, make_optimizer, task_loss
from scripts.rt_a5_reference_gpt import (
    REFERENCE_COMMIT, REFERENCE_SOURCES, ReferenceGPT, ReferenceRotaryEmbedding,
    build_reference_gpt, reference_config, reference_parameter_count, reference_parameter_sha256,
)
from scripts.rt_a5_train import train_step


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


@pytest.fixture(scope="module")
def upstream():
    source_directory = os.environ.get("A5_NEXTLAT_REFERENCE_DIR")
    sources = {}
    for relative, expected_hash in REFERENCE_SOURCES.items():
        if source_directory:
            content = (Path(source_directory) / relative).read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/JaydenTeoh/NextLat/{REFERENCE_COMMIT}/{relative}"
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
        assert hashlib.sha256(content).hexdigest() == expected_hash, relative
        sources[relative] = content.decode()
    module = ModuleType("a5_pinned_upstream_model_oracle")
    module.__dict__.update(torch=torch, nn=nn, F=F, dataclass=dataclass,
                           Optional=Optional, Tuple=Tuple, List=List, Dict=Dict, Any=Any)
    class LossMixin:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    module.FusedCrossEntropyLoss = LossMixin
    sys.modules[module.__name__] = module
    selected = {
        "models/model_base.py": {"DocumentRelativePositions", "RotaryPositionEmbedding", "LayerNorm", "SwiGLU"},
        "models/model_gpt.py": {"GPTConfig", "CausalSelfAttention", "MLP", "Block", "Transformer"},
    }
    for relative, names in selected.items():
        parsed = ast.parse(sources[relative], filename=relative)
        classes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names]
        assert {node.name for node in classes} == names
        code = compile(ast.Module(body=classes, type_ignores=[]), filename=relative, mode="exec")
        exec(code, module.__dict__)
    yield module
    del sys.modules[module.__name__]


def source_model(upstream, width=512, seed=1234):
    config = upstream.GPTConfig(n_layer=2, n_head=width // 64, n_embd=width,
                                vocab_size=60, block_size=1024, dropout=0, bias=False,
                                eos_token_id=60, context_length=0, use_fused=False)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        result = upstream.Transformer(config).to(dtype=torch.float32)
    return result


@pytest.mark.parametrize("width,expected", [(128, 441984), (512, 6486528)])
def test_source_initialization_order_parameter_names_and_counts_match(upstream, width, expected):
    torch.random.default_generator.manual_seed(61)
    rng_before = torch.random.get_rng_state().clone()
    model = build_reference_gpt(width=width)
    oracle = source_model(upstream, width=width)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert reference_parameter_count(width) == expected
    assert sum(p.numel() for p in model.parameters()) == expected
    assert set(model.state_dict()) == set(oracle.state_dict())
    for name, value in model.state_dict().items():
        assert torch.equal(value, oracle.state_dict()[name]), name
    assert model.a5_initialization["canonical_sha256"] == reference_parameter_sha256(oracle)
    assert model.a5_initialization["reference_sources"] == REFERENCE_SOURCES
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(torch.equal(p, torch.ones_like(p)) for name, p in model.named_parameters()
               if "ln_" in name or "norm.weight" in name)


@pytest.mark.parametrize("length", [12, 36])
def test_copied_primary_weights_match_actual_source_logits_and_final_norm(upstream, length):
    model = build_reference_gpt(seed=55)
    oracle = source_model(upstream, seed=56)
    oracle.load_state_dict(model.state_dict(), strict=True)
    inputs = (torch.arange(2 * length).reshape(2, length) * 7) % 60
    with fp32_context("cpu"), torch.no_grad():
        result = model(inputs, return_pre_logits=True)
        logits, hidden = oracle(inputs, return_hidden_states=True)
    torch.testing.assert_close(result.logits, logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(result.pre_logits, hidden, rtol=2e-5, atol=2e-6)
    assert torch.equal(result.logits.argmax(-1), logits.argmax(-1))
    assert result.logits.shape == (2, length, 60)


@pytest.mark.parametrize("length", [12, 36])
def test_source_gradients_and_one_unchanged_common_adam_update(upstream, length):
    model = build_reference_gpt(width=128, seed=71)
    oracle = source_model(upstream, width=128, seed=72)
    oracle.load_state_dict(model.state_dict(), strict=True)
    class OutputAdapter(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
        def forward(self, x):
            return SimpleNamespace(logits=self.original(x))
    adapter = OutputAdapter(oracle)
    inputs = (torch.arange(2 * length).reshape(2, length) * 11) % 60
    labels = (inputs + torch.arange(length)) % 60
    actual_optimizer, source_optimizer = make_optimizer(model), make_optimizer(adapter)
    # train_step is the existing historical function, using the same CE/clip/Adam path.
    actual = train_step(model, actual_optimizer, inputs, labels)
    expected = train_step(adapter, source_optimizer, inputs, labels)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], rel=2e-5, abs=2e-6), key
    for name, parameter in model.named_parameters():
        reference = dict(oracle.named_parameters())[name]
        assert parameter.grad is not None and reference.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=2e-5, atol=2e-6, msg=name)
        torch.testing.assert_close(parameter, reference, rtol=2e-5, atol=2e-7, msg=name)
    actual_states = actual_optimizer.state_dict()["state"]
    source_states = source_optimizer.state_dict()["state"]
    assert actual_states.keys() == source_states.keys()
    for parameter_id, actual_state in actual_states.items():
        for key, value in actual_state.items():
            torch.testing.assert_close(value, source_states[parameter_id][key], rtol=2e-5, atol=2e-7)


def test_rope_uses_source_frequencies_positions_and_noninterleaved_halves(upstream):
    ours = ReferenceRotaryEmbedding(1025, 64)
    oracle = upstream.RotaryPositionEmbedding(max_seq_len=1025, head_dim=64)
    positions = torch.arange(1, 37).unsqueeze(0).expand(2, -1)
    ours_rope, source_rope = ours(positions), oracle(positions)
    assert ours.max_seq_len == oracle.max_seq_len == 1280
    assert torch.equal(ours_rope[0], source_rope[0])
    assert torch.equal(ours_rope[1], source_rope[1])
    q = torch.randn(2, 2, 36, 64)
    k = torch.randn_like(q)
    for actual, expected in zip(ours.apply(q, k, ours_rope), oracle.apply(q, k, source_rope)):
        assert torch.equal(actual, expected)
    # The first coordinate uses angle1 at the first input; no implicit BOS state.
    assert ours_rope[0][0, 0, 0] == torch.tensor(1.0).cos()


def test_prefix_causality_and_no_cross_word_context_or_identity_masking():
    model = build_reference_gpt(width=128)
    inputs = (torch.arange(72).reshape(2, 36) * 13) % 60
    inputs[0, :4] = 0
    changed = inputs.clone()
    changed[0, 12:] = (changed[0, 12:] + 1) % 60
    changed[1] = (changed[1] + 2) % 60
    with fp32_context("cpu"), torch.no_grad():
        original = model(inputs).logits
        truncated = model(inputs[:, :12]).logits
        altered = model(changed).logits
    torch.testing.assert_close(original[:, :12], truncated, rtol=2e-5, atol=2e-6)
    assert torch.equal(original[0, :12], altered[0, :12])
    assert torch.isfinite(original).all()
    # Labels never enter the model API; shared unshifted CE includes identity positions.
    logits = original.detach().requires_grad_()
    task_loss(logits, inputs).backward()
    assert (logits.grad[0, :4, 0] < 0).all()


def test_metadata_is_stable_serializable_and_unpaired_from_mitchell_family():
    model = build_reference_gpt(width=128, seed=1234)
    duplicate = build_reference_gpt(width=128, seed=1234)
    different = build_reference_gpt(width=128, seed=1235)
    assert model.config == duplicate.config == reference_config(128)
    assert model.a5_initialization == duplicate.a5_initialization
    assert model.a5_initialization["canonical_sha256"] != different.a5_initialization["canonical_sha256"]
    assert model.a5_initialization["initialization_family"] == "released_gpt_normal_0.02"
    assert model.token_embedding.weight is not model.lm_head.weight
    assert not any("q_norm" in name or "k_norm" in name or "predictor" in name for name, _ in model.named_parameters())
    json.dumps(model.config)
    json.dumps(model.a5_initialization)
    assert reference_config()["mlp_hidden_width"] == 1408
    assert reference_config()["rope_position_start"] == 1
    assert model.rotary_embedding.cos_lookup is None
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert reference_parameter_sha256(restored) == reference_parameter_sha256(model)


def test_invalid_model_and_input_contracts_are_rejected():
    with pytest.raises(ValueError, match="multiple of 64"):
        build_reference_gpt(width=100)
    with pytest.raises(ValueError, match="nonnegative"):
        build_reference_gpt(seed=-1)
    model = build_reference_gpt(width=128)
    with pytest.raises(TypeError, match="int64"):
        model(torch.zeros(1, 12, dtype=torch.int32))
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.zeros(1, 0, dtype=torch.long))
    with pytest.raises(ValueError, match="context cap"):
        model(torch.zeros(1, 1025, dtype=torch.long))
