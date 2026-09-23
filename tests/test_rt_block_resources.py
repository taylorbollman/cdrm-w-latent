"""Isolated-stack ledger checked against parameters and executed matrix ops."""
from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from cdrm.pretrained.olmo_author import author_tiled_recurrent_layer
from cdrm.pretrained.olmo_rope import build_rope_tables
from cdrm.pretrained.olmo_tiled import tiled_recurrent_layer
from cdrm.pretrained.rt_block_resources import estimate_rt_block_resources


class MatrixCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.dense = self.attention = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.mm.default:
            a, b = args[:2]
            self.dense += 2 * a.shape[0] * a.shape[1] * b.shape[1]
        elif func is torch.ops.aten.bmm.default:
            a, b = args[:2]
            self.attention += 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("backend,arm,backward", [
    ("author", "both", "materialized"), ("native", "control", "materialized"),
    ("native", "rope", "recompute"), ("native", "both", "materialized"),
    ("native", "both", "recompute"),
])
@pytest.mark.parametrize("length", [1, 3, 5, 8])
def test_forward_and_backward_counts_match_executed_matrix_work(backend, arm, backward, length):
    torch.set_num_threads(1)
    torch.manual_seed(924)
    config = replace(OLMoConfig.tiny(), num_layers=1)
    layer = OLMoBlock(config)
    x = torch.randn(2, length, config.model_dim, requires_grad=True)
    positions = torch.arange(length).expand(2, -1)
    tables = build_rope_tables(positions, config.head_dim, config.rope_freq_constant)
    valid = torch.ones_like(positions, dtype=torch.bool)
    estimate = estimate_rt_block_resources(config, batch_size=2, sequence_length=length,
        backend=backend, native_arm=arm, native_backward=backward)
    with MatrixCounter() as counter:
        if backend == "author":
            output = author_tiled_recurrent_layer(layer, x, tables, compiled_helpers=False)
        else:
            output, _ = tiled_recurrent_layer(layer, x, alpha=1, past=None,
                query_positions=positions, key_positions=positions, key_valid=valid,
                attention_precision="fp32", backward_memory=backward, reuse_rope=arm != "control",
                kv_only_writes=arm == "both")
        forward = {"dense": counter.dense, "attention": counter.attention}
        output.backward(torch.randn_like(output))
    total = {"dense": counter.dense, "attention": counter.attention}
    for family in ("dense", "attention"):
        assert estimate["matrix_breakdown"]["forward"][family] == forward[family]
        assert estimate["matrix_breakdown"]["total"][family] == total[family]
        assert estimate["matrix_breakdown"]["backward"][family] == total[family] - forward[family]


@pytest.mark.parametrize("backend", ["native", "author"])
@pytest.mark.parametrize("layers", [1, 2, 6])
def test_stack_parameter_ownership_and_linear_layer_multiplicity(backend, layers):
    config = replace(OLMoConfig.tiny(), num_layers=layers)
    modules = torch.nn.ModuleList(OLMoBlock(config) for _ in range(layers))
    arguments = dict(batch_size=2, sequence_length=5, backend=backend)
    one = estimate_rt_block_resources(replace(config, num_layers=1), **arguments)
    stack = estimate_rt_block_resources(config, **arguments)
    assert stack["unique_parameters"] == sum(p.numel() for p in modules.parameters())
    assert stack["unique_parameters"] == layers * stack["parameters_per_layer"]
    assert stack["block_calls"] == layers
    for key in ("forward_matrix_flops", "backward_matrix_flops", "total_matrix_flops"):
        assert stack[key] == layers * one[key]
    assert stack["total_matrix_flops"] == stack["forward_matrix_flops"] + stack["backward_matrix_flops"]
    assert stack["input_tokens"] == one["input_tokens"]


def test_rope_only_does_not_change_matrix_arithmetic_and_author_ignores_native_options():
    config = OLMoConfig.tiny()
    arguments = dict(batch_size=2, sequence_length=5)
    control = estimate_rt_block_resources(config, **arguments, backend="native", native_arm="control")
    rope = estimate_rt_block_resources(config, **arguments, backend="native", native_arm="rope")
    both = estimate_rt_block_resources(config, **arguments, backend="native", native_arm="both")
    assert control["matrix_breakdown"] == rope["matrix_breakdown"]
    assert control["total_matrix_flops"] - both["total_matrix_flops"] == 12 * 2 * 5 * config.model_dim**2 * config.num_layers
    old = estimate_rt_block_resources(config, **arguments, backend="author", native_arm="control", native_backward="recompute")
    new = estimate_rt_block_resources(config, **arguments, backend="author", native_arm="both", native_backward="materialized")
    assert old == new
    assert old["backward_memory"] == "materialized" and old["native_arm"] is None


@pytest.mark.parametrize("change", [
    {"batch_size": 0}, {"batch_size": True}, {"batch_size": 1.5},
    {"sequence_length": 0}, {"sequence_length": True}, {"sequence_length": 1.5},
    {"backend": "author_scan"}, {"backend": None}, {"native_arm": "unknown"},
    {"native_backward": "unknown"},
])
def test_invalid_inputs_rejected(change):
    kwargs = dict(batch_size=2, sequence_length=5, backend="native")
    kwargs.update(change)
    with pytest.raises(ValueError):
        estimate_rt_block_resources(OLMoConfig.tiny(), **kwargs)


def test_configuration_type_is_explicit():
    with pytest.raises(TypeError):
        estimate_rt_block_resources({}, batch_size=2, sequence_length=5, backend="native")
