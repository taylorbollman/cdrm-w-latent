"""Bounded checks for the new10k/25k report gates; no model or evaluation."""
import hashlib

import pytest

from scripts import rt_a5_six_layer_head_25k_report as head
from scripts import rt_a5_six_layer_value_25k_report as value


def fixture(module, start, endpoint, end=None):
    end = endpoint if end is None else end
    protocol = {'start_update': start, 'endpoint': endpoint, 'variant': 'linear'}
    report = {'schema': module.TRAIN_SCHEMA, 'start_update': start, 'endpoint': endpoint,
        'completed_updates': end, 'status': 'complete', 'wandb': {'status': 'synced'},
        'requested_endpoint_reached': True, 'confirmation_evaluated': False,
        'latent_rollout_evaluated': False}
    if module is value:
        report['injection_coefficient'] = value.coefficient(end, 'linear')
    else:
        report['embedding_head_enabled'] = True
    return report, protocol


@pytest.mark.parametrize('module', [value, head])
@pytest.mark.parametrize('start,endpoint', [(0, 10000), (10000, 25000)])
def test_complete_only_at_revised_actual_endpoint(module, start, endpoint):
    report, protocol = fixture(module, start, endpoint)
    assert module.terminal(report, protocol) == endpoint
    report['completed_updates'] -= 1
    with pytest.raises(ValueError, match='actual requested endpoint'):
        module.terminal(report, protocol)
    stale, old_protocol = fixture(module, 10000, 50000)
    with pytest.raises(ValueError, match='Unknown'):
        module.terminal(stale, old_protocol)


@pytest.mark.parametrize('module', [value, head])
def test_bound_stop_before25k_retains_actual_scope(module, tmp_path):
    report, protocol = fixture(module, 10000, 25000, 15001)
    stop = tmp_path / 'STOP_AFTER_UPDATE'
    stop.write_text('Explicit user stop\n')
    protocol['resolved_args'] = {'stop_file': str(stop)}
    report.update(status='stopped', requested_endpoint_reached=False,
        stop_request={'reason': 'user_stop_file', 'observed_after_update': 15001,
            'path': str(stop), 'sha256': hashlib.sha256(stop.read_bytes()).hexdigest()})
    assert module.terminal(report, protocol) == 15001
    report['stop_request']['observed_after_update'] = 15000
    with pytest.raises(ValueError, match='bound graceful user stop'):
        module.terminal(report, protocol)


def test_linear_warmup_stays_global20k_despite25k_training_limit():
    assert value.coefficient(10000, 'linear') == .005
    assert value.coefficient(10001, 'linear') == .01 * (10001 / 20000)
    assert value.coefficient(20000, 'linear') == value.coefficient(25000, 'linear') == .01
    report, protocol = fixture(value, 10000, 25000)
    report['injection_coefficient'] = .005
    with pytest.raises(ValueError, match='coefficient differs'):
        value.terminal(report, protocol)
