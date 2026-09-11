"""CPU contracts for the independent terminal-write credit diagnostic."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rt_precision_credit import (backbone_gradients, credit_summary,
                                 normalized_tensors, terminal_cotangent, write_observer)


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('CPU contracts must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)


def test_terminal_cotangent_has_only_common_final_position_credit_and_private_rng():
    before = torch.random.get_rng_state().clone()
    first = terminal_cotangent((3, 16, 64), seed=401)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(first, terminal_cotangent((3, 16, 64), seed=401))
    assert first.dtype == torch.float32 and first.shape == (3, 16, 64)
    assert torch.count_nonzero(first[:, :-1]) == 0
    assert bool((first[:, -1].norm(dim=1) > 0).all())
    with pytest.raises(ValueError):
        terminal_cotangent((3, 1, 64), seed=401)


def test_observer_ignores_uninitialized_buffers_and_clones_consumed_write_credit():
    records = {}
    observer = write_observer(1, records)
    observer('backward.buffers', {'gs': torch.full((2,), float('nan'))})
    assert records == {}
    k, v = torch.tensor([.1]), torch.tensor([.3])
    observer('backward.attention_adjoint', {'k_grad': k, 'v_grad': v}, token_index=4)
    k.zero_(); v.zero_()
    assert records['layer_1/token_4/k'].item() == pytest.approx(.1)
    assert records['layer_1/token_4/v'].item() == pytest.approx(.3)
    with pytest.raises(AssertionError, match='Duplicate'):
        observer('backward.attention_adjoint', {'k_grad': k, 'v_grad': v}, token_index=4)


@pytest.mark.parametrize('bad', [torch.tensor([float('nan')]), torch.tensor([1.], dtype=torch.bfloat16)])
def test_observer_rejects_nonfinite_or_wrong_dtype_credit(bad):
    with pytest.raises(FloatingPointError):
        write_observer(0, {})('backward.attention_adjoint', {'k_grad': bad, 'v_grad': bad}, token_index=0)


def write_fixture():
    return {f'layer_{layer}/token_{token}/{kind}': torch.tensor([0. if token == 2 else .1])
            for layer in range(2) for token in range(3) for kind in ('k', 'v')}


def test_credit_checks_every_layer_token_without_requiring_terminal_write_to_learn():
    values = write_fixture()
    assert credit_summary(values, layers=2, length=3)['pass']
    values['layer_1/token_1/k'].zero_()
    values['layer_1/token_1/v'].zero_()
    result = credit_summary(values, layers=2, length=3)
    assert result['earlier_write_nonzero_failures'] == ['layer_1/token_1']
    assert not result['pass']
    values = write_fixture()
    values['layer_0/token_2/v'].fill_(1e-20)
    result = credit_summary(values, layers=2, length=3)
    assert result['terminal_write_zero_failures'] == ['layer_0/token_2']
    assert not result['pass']


def test_missing_or_nonfinite_write_credit_is_not_silently_omitted():
    values = write_fixture()
    del values['layer_0/token_0/k']
    with pytest.raises(AssertionError, match='coverage'):
        credit_summary(values, layers=2, length=3)
    values = write_fixture()
    values['layer_1/token_1/v'].fill_(float('inf'))
    with pytest.raises(FloatingPointError):
        credit_summary(values, layers=2, length=3)


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Linear(2, 3, bias=False)
        self.transformer = torch.nn.ModuleDict({'ff_out': torch.nn.Linear(3, 4, bias=False)})


def test_backbone_gradients_distinguish_structurally_unused_head_from_missing_credit():
    model = ProbeModel()
    model.used(torch.ones(1, 2)).sum().backward()
    gradients = backbone_gradients(model)
    assert set(gradients) == {'used.weight'}
    assert model.transformer.ff_out.weight.grad is None
    model.used.weight.grad = None
    with pytest.raises(AssertionError, match='Missing backbone'):
        backbone_gradients(model)
    model.used.weight.grad = torch.ones_like(model.used.weight)
    model.transformer.ff_out.weight.grad = torch.zeros_like(model.transformer.ff_out.weight)
    with pytest.raises(AssertionError, match='unexpectedly used head'):
        backbone_gradients(model)


@pytest.mark.parametrize('scale', [0., -1., float('inf'), float('nan')])
def test_invalid_scale_does_not_generate_undefined_normalized_gradients(scale):
    with pytest.raises(ValueError):
        normalized_tensors({'x': torch.ones(3)}, scale)


def test_power_of_two_normalization_preserves_values_without_mutating_packet():
    value = torch.tensor([.25, -1., 2.])
    scaled = {'x': value * 32}
    result = normalized_tensors(scaled, 32)
    assert torch.equal(result['x'], value)
    assert torch.equal(scaled['x'], value * 32)
