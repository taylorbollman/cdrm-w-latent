"""CPU-only preflight checks; numerical/replay correctness is checked on GPU.

The CUDA-shaped token stand-in has no storage and must never be materialized.
It lets unsupported model contracts fail before allocating capture buffers.
"""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from olmo.config import BlockType, ModelConfig
from olmo.model import OLMo
from rt_cuda_graph import CapturedRTBackward


class CUDAInputSpec:
    device = torch.device('cuda:0')

    def __init__(self, shape=(2, 8), dtype=torch.long):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype

    def clone(self):
        raise AssertionError('Rejected capture input must not allocate static storage')


@pytest.fixture(autouse=True)
def forbid_cuda_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU contract tests must never initialize or execute CUDA')

    for name in ('_lazy_init', 'CUDAGraph', 'Stream', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def model():
    # Real RT module ownership/configuration, with small CPU parameters. No
    # recurrent forward is attempted: every case must fail during preflight.
    config = ModelConfig(d_model=16, n_heads=2, mlp_hidden_size=32, n_layers=2,
                         vocab_size=16, embedding_size=16, max_sequence_length=8,
                         block_type=BlockType.recurrent, init_device='cpu',
                         alibi=True, rope=False, weight_tying=False,
                         include_bias=False, attention_layer_norm=True,
                         attention_dropout=0., residual_dropout=0., embedding_dropout=0.,
                         recurrent_precision_policy='bf16_fp32_state', precision=None)
    return OLMo(config).train()


@pytest.mark.parametrize('dtype', [torch.long, torch.float32])
def test_cpu_tokens_rejected_before_accessing_model_or_initializing_cuda(dtype):
    with pytest.raises(ValueError, match='CUDA int64'):
        CapturedRTBackward(object(), torch.zeros((2, 8), dtype=dtype))


@pytest.mark.parametrize('spec,options', [
    (CUDAInputSpec(dtype=torch.int32), {}),
    (CUDAInputSpec((0, 8)), {}),
    (CUDAInputSpec((8,)), {}),
    (CUDAInputSpec((2, 1)), {}),
    (CUDAInputSpec(), {'head_chunk_size': 0}),
    (CUDAInputSpec(), {'warmup': 1}),
])
def test_invalid_capture_inputs_rejected_before_model_access_or_static_storage(spec, options):
    with pytest.raises(ValueError):
        CapturedRTBackward(object(), spec, **options)


def test_evaluation_or_disabled_autograd_cannot_start_capture(model):
    model.eval()
    with pytest.raises(ValueError, match='grad-enabled training'):
        CapturedRTBackward(model, CUDAInputSpec())
    model.train()
    with torch.no_grad(), pytest.raises(ValueError, match='grad-enabled training'):
        CapturedRTBackward(model, CUDAInputSpec())


def test_python_callbacks_rejected_without_running_them_or_mutating_gradients(model):
    parameter = next(model.parameters())
    original_gradient = torch.ones_like(parameter)
    parameter.grad = original_gradient

    def unexpected_callback(*args, **kwargs):
        raise AssertionError('Preflight must not execute model callbacks')

    block = model.transformer.blocks[0]
    handle = block.register_forward_hook(unexpected_callback)
    try:
        with pytest.raises(ValueError, match='module hooks'):
            CapturedRTBackward(model, CUDAInputSpec())
    finally:
        handle.remove()
    block._recurrent_precision_observer = unexpected_callback
    with pytest.raises(ValueError, match='precision observers'):
        CapturedRTBackward(model, CUDAInputSpec())
    assert parameter.grad is original_gradient
    torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


@pytest.mark.parametrize('violation,message', [
    ('outer_checkpoint', 'outer checkpointing'),
    ('ordinary_layer', 'all layers to be tiled recurrent'),
    ('legacy_precision', 'explicit autocast'),
    ('tied_head', 'untied head'),
])
def test_unsupported_model_contracts_fail_before_capture_allocation(model, violation, message):
    if violation == 'outer_checkpoint':
        # The public setter already rejects this; test the capture guard also
        # protects a model loaded or mutated with an incompatible strategy.
        model.activation_checkpointing_strategy = 'whole_layer'
    elif violation == 'ordinary_layer':
        model.transformer.blocks[0] = torch.nn.Identity()
    elif violation == 'legacy_precision':
        model.config.recurrent_precision_policy = 'legacy'
    elif violation == 'tied_head':
        model.config.weight_tying = True
    with pytest.raises(ValueError, match=message):
        CapturedRTBackward(model, CUDAInputSpec())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_parameters_on_wrong_device_fail_without_materializing_cuda_input(model):
    with pytest.raises(ValueError, match='FP32 parameters on the input CUDA device'):
        CapturedRTBackward(model, CUDAInputSpec())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_legacy_requires_explicit_opt_in_and_matching_layer_policy(model):
    import copy
    model.config.recurrent_precision_policy = 'legacy'
    for block in model.transformer.blocks:
        block.config.recurrent_precision_policy = 'legacy'
    # Explicit opt-in passes precision guards, then rejects CPU parameters.
    with pytest.raises(ValueError, match='FP32 parameters on the input CUDA device'):
        CapturedRTBackward(model, CUDAInputSpec(), precision_policy='legacy')
    block = model.transformer.blocks[0]
    block.config = copy.deepcopy(block.config)
    block.config.recurrent_precision_policy = 'bf16_fp32_state'
    with pytest.raises(ValueError, match='Every recurrent layer'):
        CapturedRTBackward(model, CUDAInputSpec(), precision_policy='legacy')


def test_unknown_policy_rejected_before_allocation(model):
    with pytest.raises(ValueError, match='explicitly selected precision policy'):
        CapturedRTBackward(model, CUDAInputSpec(), precision_policy='all_bf16')
