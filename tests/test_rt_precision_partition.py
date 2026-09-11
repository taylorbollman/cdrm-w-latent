"""CPU contracts for the stricter FP32 partition acceptance screen."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rt_precision_compare import OPTIMIZER_OPTIONS
from rt_precision_partition import partition_comparison


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('CPU contracts must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)


def packet_fixture():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(.2)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER_OPTIONS)
    starting = {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict())}
    model.weight.grad = torch.tensor([[1e-10, -1e-10]])
    gradient = model.weight.grad.clone()
    optimizer.step()
    packet = {'arm': 'C', 'model': copy.deepcopy(model.state_dict()),
              'optimizer': copy.deepcopy(optimizer.state_dict()), 'optimizer_parameter_names': ['weight'],
              'raw_gradients': {'weight': gradient}, 'clipped_gradients': {'weight': gradient.clone()},
              'loss': torch.tensor(2.), 'gradient_norm': gradient.norm(),
              'clip_coefficient': torch.tensor(1.), 'layer_output_samples': {}}
    return starting, packet


def test_identical_fp32_partition_packets_pass_all_gradient_and_adam_checks():
    starting, packet = packet_fixture()
    result = partition_comparison(starting, packet, copy.deepcopy(packet), trained=False)
    assert result['partition_validation_pass']
    assert all(row['exact'] for row in result['same_precision_coordinate_checks'].values())


def test_tiny_wrong_gradient_cannot_hide_under_coordinate_absolute_tolerance():
    starting, packet = packet_fixture()
    altered = copy.deepcopy(packet)
    altered['raw_gradients']['weight'] *= 2
    result = partition_comparison(starting, packet, altered, trained=False)
    assert result['same_precision_coordinate_checks']['raw_gradients']['pass']
    assert result['same_precision_norm_review']['requires_review']
    assert not result['partition_validation_pass']


def test_wrong_global_loss_denominator_fails_partition_validation():
    starting, packet = packet_fixture()
    altered = copy.deepcopy(packet)
    altered['loss'] *= 2
    result = partition_comparison(starting, packet, altered, trained=False)
    assert not result['same_precision_scalar_checks']['loss']['pass']
    assert not result['partition_validation_pass']


def test_extra_adam_step_is_hard_failure():
    starting, packet = packet_fixture()
    altered = copy.deepcopy(packet)
    next(iter(altered['optimizer']['state'].values()))['step'] += 1
    with pytest.raises(AssertionError, match='different step counts'):
        partition_comparison(starting, packet, altered, trained=False)
