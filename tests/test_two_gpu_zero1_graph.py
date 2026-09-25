"""CPU checks of graph-probe accounting, selection and terminal lifetimes."""
from types import SimpleNamespace
import weakref

import pytest
import torch

from scripts import olmo_two_gpu_zero1_graph as harness


def record(owned, states):
    return {'inventory': {'local_owned_names': owned, 'local_state_bytes_by_device': {'cuda:0': 100},
                          'outer_state_bytes_by_device': {}, 'consolidation_cache_present': False},
            'local_moment_digests': states, 'replicated': {'weights': 'fixed', 'gradient': 'fixed'},
            'moment_schema_valid': True, 'finite_local_moments': True,
            'step_counters_match': True, 'fp32_local_state': True}


def test_replica_summary_checks_partition_without_requiring_sharded_moments_equal():
    records = [record(['weight', 'inactive'], {'weight': 'rank0 moment hash'}),
               record(['bias'], {'bias': 'different rank1 moment hash'})]
    result = harness.summarize_replica_records(records, ['weight', 'bias', 'inactive'], ['weight', 'bias'])
    assert result['passed']
    assert result['global_optimizer_state_bytes'] == 200
    assert result['active_moment_partition_exact']
    assert result['replicated_parameters_gradients_scheduler_counters_exact']


@pytest.mark.parametrize('fault', ['replica', 'missing_state', 'duplicate_owner', 'wrong_owner',
                                 'schema', 'nonfinite', 'dtype', 'step', 'duplicate_outer_state', 'cache'])
def test_replica_summary_rejects_wrong_or_unhealthy_state(fault):
    records = [record(['weight'], {'weight': 'a'}), record(['bias'], {'bias': 'b'})]
    if fault == 'replica': records[1]['replicated']['weights'] = 'changed'
    elif fault == 'missing_state': records[1]['local_moment_digests'] = {}
    elif fault == 'duplicate_owner': records[1]['inventory']['local_owned_names'].append('weight')
    elif fault == 'wrong_owner':
        records[0]['local_moment_digests'], records[1]['local_moment_digests'] = {'bias': 'b'}, {'weight': 'a'}
    elif fault == 'schema': records[1]['moment_schema_valid'] = False
    elif fault == 'nonfinite': records[1]['finite_local_moments'] = False
    elif fault == 'dtype': records[1]['fp32_local_state'] = False
    elif fault == 'step': records[1]['step_counters_match'] = False
    elif fault == 'duplicate_outer_state': records[1]['inventory']['outer_state_bytes_by_device'] = {'cuda:1': 100}
    elif fault == 'cache': records[1]['inventory']['consolidation_cache_present'] = True
    assert not harness.summarize_replica_records(records, ['weight', 'bias'], ['weight', 'bias'])['passed']


def test_replica_summary_refuses_empty_inventory():
    with pytest.raises(ValueError, match='empty'):
        harness.summarize_replica_records([], [], [])


@pytest.mark.parametrize('fault', [None, 'missing_moment', 'moment_shape', 'step_shape',
                                 'fractional_step', 'stale_step', 'nonfinite', 'dtype'])
def test_actual_local_adam_state_health_detects_bad_storage_and_counters(fault):
    parameter = torch.nn.Parameter(torch.ones(2))
    state = {'step': torch.tensor(3.), 'exp_avg': torch.ones(2), 'exp_avg_sq': torch.ones(2)}
    if fault == 'missing_moment': del state['exp_avg_sq']
    elif fault == 'moment_shape': state['exp_avg'] = torch.ones(3)
    elif fault == 'step_shape': state['step'] = torch.tensor([3.])
    elif fault == 'fractional_step': state['step'] = torch.tensor(3.5)
    elif fault == 'stale_step': state['step'] = torch.tensor(2.)
    elif fault == 'nonfinite': state['exp_avg_sq'][0] = float('nan')
    elif fault == 'dtype': state['exp_avg'] = state['exp_avg'].to(torch.bfloat16)
    check = harness.local_state_health({parameter: state}, 3)
    assert all(check.values()) == (fault is None)


class Lifetime:
    pass


def test_terminal_release_drops_graph_outputs_before_cleanup_and_keeps_reducer():
    events = []
    runtime = SimpleNamespace(graph=Lifetime(), graph_result=Lifetime(), ddp=object(), _capture_started=True,
        validate_execution=lambda: events.append('validate'), _addresses=lambda: {'p': 123})
    graph, result = weakref.ref(runtime.graph), weakref.ref(runtime.graph_result)
    reducer = runtime.ddp
    def cleanup():
        assert graph() is None and result() is None
        events.append('cleanup')
    check = harness.release_graph_for_eager(runtime, synchronize=lambda: events.append('sync'), cleanup=cleanup)
    assert events == ['validate', 'sync', 'cleanup', 'validate']
    assert runtime.ddp is reducer and runtime._capture_started is True
    assert runtime.graph is runtime.graph_result is None
    assert check == {'graph_released': True, 'same_ddp_reducer': True, 'gradient_addresses_preserved': True}


def test_terminal_release_rejects_gradient_buffer_replacement():
    pointer = [123]
    runtime = SimpleNamespace(graph=Lifetime(), graph_result=Lifetime(), ddp=object(),
        validate_execution=lambda: None, _addresses=lambda: {'p': pointer[0]})
    with pytest.raises(AssertionError, match='persistent gradient'):
        harness.release_graph_for_eager(runtime, synchronize=lambda: None,
                                        cleanup=lambda: pointer.__setitem__(0, 456))


def test_terminal_release_requires_existing_graph():
    runtime = SimpleNamespace(graph=None, validate_execution=lambda: None)
    with pytest.raises(ValueError, match='existing'):
        harness.release_graph_for_eager(runtime)


@pytest.mark.parametrize('extra', [['--length', '256'], ['--batch-size', '0'], ['--batch-size', '513'],
                                 ['--stage', 'integration', '--batch-size', '9'], ['--case', 'ordinary']])
def test_cli_bounded_actual_scope(extra):
    with pytest.raises(SystemExit):
        harness.parse_args(['--case', 'rt', '--output-dir', str(harness.ROOT/'.runtime/zero1-unit'), *extra])


def test_cli_tiny_integration_and_bucket_view_are_explicit():
    args = harness.parse_args(['--case', 'combined', '--tiny', '--length', '8', '--stage', 'integration',
        '--bucket-view', '--output-dir', str(harness.ROOT/'.runtime/zero1-unit')])
    assert args.length == 8 and args.bucket_view and args.stage == 'integration'
