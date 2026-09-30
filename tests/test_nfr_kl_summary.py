"""NFR summary aggregation and interpretation guards, with synthetic records."""
import copy
import json

import pytest

from scripts import olmo_nfr_kl_summary as summary
from scripts.olmo_nfr_kl_tracking import validate_summary
from test_kl_continuation_summary import fixture as nf_fixture


def fixture():
    control, reduced, parent = nf_fixture()
    parent['arm'] = 'NFR'
    for report in (control, reduced):
        report['arm'] = 'NFR'
        report['sources'] = {str(n): str(n) for n in range(214)} | {'scope.json': 'scope-pin'}
        report['nfr_scope'] = {'sha256': 'scope-pin', 'declaration': 'same'}
        report['configuration']['parameters'] = copy.deepcopy(report['configuration']['parameters'])
        report['configuration']['parameters']['mode'] = {'rt_mode': {'selected_layers': [0, 15], 'alpha': 1.},
            'num_passes': 4, 'enabled': True, 'beta': 1., 'feedback_jitter': .02}
    pins = {name: {'sha256': name + '-pin'} for name in ('parent', 'control', 'reduced', 'audit', 'scope')}
    audit = {'schema': 'olmo-nfr-kl-audit-v1', 'comparison_kind': 'paired_fork', 'passed': True,
        'failures': [], 'checks': [{'passed': True}], 'inputs': copy.deepcopy(pins)}
    return control, reduced, parent, audit, pins


def test_matched_origin_raw_terms_scoped_timing_and_no_objective_totals():
    result = summary.summarize(*fixture()); validate_summary(result)
    arm = result['arms']['KL0.1']
    assert result['arm'] == 'NFR' and result['f64_context'] is None
    assert arm['clipping']['clipped_updates'] == 32
    assert arm['rates']['all33_64']['compute_regions']['seconds'] == 32 * 22
    assert arm['rates']['all33_64']['compute_plus_materialization']['seconds'] == 32 * 24
    assert arm['rates']['all33_64']['complete_update_callback']['seconds'] == 32 * 31
    assert arm['memory_sample_summary_gib']['reserved_gib'] == 50.
    assert arm['parameters']['optimizer_owned'] == 6
    assert result['comparison'][0]['raw_mean_deltas_by_pass'][0]['ce'] == 0
    assert result['comparison'][1]['raw_mean_deltas_by_pass'][0]['ce'] == pytest.approx(-.1)
    assert 'weighted_objective_branch_specific' not in json.dumps(result)


@pytest.mark.parametrize('mutation', ['NF', 'failed_audit', 'wrong_pin', 'RT_disabled', 'running', 'missing_update', 'mismatched_data'])
def test_incomplete_or_unmatched_pair_rejected(mutation):
    control, reduced, parent, audit, pins = fixture()
    if mutation == 'NF': reduced['arm'] = 'NF'
    if mutation == 'failed_audit': audit['passed'] = False
    if mutation == 'wrong_pin': audit['inputs']['reduced']['sha256'] = 'other'
    if mutation == 'RT_disabled': reduced['configuration']['parameters']['mode']['rt_mode']['selected_layers'] = []
    if mutation == 'running': reduced['status'] = 'running'
    if mutation == 'missing_update': reduced['updates'].pop('40')
    if mutation == 'mismatched_data': reduced['observations']['40']['rank_data'][0]['valid_tokens'] += 1
    with pytest.raises(ValueError): summary.summarize(control, reduced, parent, audit, pins)


def f_context(result):
    _, _, parent, _, _ = fixture()
    evaluation = copy.deepcopy(parent['evaluations'][0]); evaluation['after_update'] = 64
    evaluation['panels']['dev-main']['result']['enabled'] = {'ce': True, 'latent': False, 'kl': False}
    publication = {'counters': {'optimizer_updates': 64}, 'metadata': {'configuration': {
        'execution_identity': {'payload': {'arm': 'F',
            'recipe': {'sequence_length': 1024, 'effective_valid_tokens': 524288}, 'model_contract': {
            'weights': {'ce': 1., 'latent': 0., 'kl': 0.}, 'component_parameters': {'predictor': 0},
            'mode': {'num_passes': 4, 'rt_mode': {'selected_layers': []}, 'enabled': True, 'beta': 1.}}}}}}}
    return evaluation, publication, result['arms']['KL1']['development'][-1]


def test_f64_only_context_and_exact_development_membership():
    values = f_context(summary.summarize(*fixture()))
    result = summary.f64_context(*values)
    assert result['after_update'] == 64 and result['passes'][0]['ce'] == 3.
    assert result['ce_gap_vs_first_pass'] == [0., 1., 2., 3.]
    values[0]['after_update'] = 128
    with pytest.raises(ValueError, match='update64'): summary.f64_context(*values)
    values[0]['after_update'] = 64
    values[0]['panels']['dev-main']['membership_sha256'] = 'other'
    with pytest.raises(ValueError, match='membership'): summary.f64_context(*values)


@pytest.mark.parametrize('mutation', ['FBT_disabled', 'predictor_present', 'auxiliary_enabled'])
def test_f64_requires_active_feedback_but_no_predictor_or_auxiliary_loss(mutation):
    values = f_context(summary.summarize(*fixture()))
    contract = values[1]['metadata']['configuration']['execution_identity']['payload']['model_contract']
    if mutation == 'FBT_disabled': contract['mode']['enabled'] = False
    if mutation == 'predictor_present': contract['component_parameters']['predictor'] = 1
    if mutation == 'auxiliary_enabled': contract['weights']['kl'] = 1.
    with pytest.raises(ValueError, match='F-only'): summary.f64_context(*values)


def test_cli_pins_snapshots_and_outputs_without_objective_columns(tmp_path):
    control, reduced, parent, audit, _ = fixture()
    result = summary.summarize(control, reduced, parent, audit, fixture()[-1])
    evaluation, publication, _ = f_context(result)
    parent_data = json.dumps(parent).encode(); parent_pin = summary.common.sha(parent_data)
    for report in (control, reduced):
        report['branch']['parent_report_sha256'] = report['parent_report_sha256'] = parent_pin
    values = {'parent': parent, 'control': control, 'reduced': reduced,
              'f64-evaluation': evaluation, 'f64-publication': publication}
    arguments = []
    for label, value in values.items():
        data = json.dumps(value).encode(); path = tmp_path/(label+'.json'); path.write_bytes(data)
        pin = summary.common.sha(data)
        if label in audit['inputs']: audit['inputs'][label]['sha256'] = pin
        arguments += ['--'+label, str(path), pin]
    audit_data = json.dumps(audit).encode(); audit_path = tmp_path/'audit.json'; audit_path.write_bytes(audit_data)
    arguments += ['--audit', str(audit_path), summary.common.sha(audit_data)]
    output = tmp_path/'summary'
    summary.main(arguments + ['--output', str(output)])
    assert len(list(output.glob('*.pdf'))) == 4
    assert len(list(output.glob('*.png'))) == 4
    assert 'objective' not in (output/'training.csv').read_text().splitlines()[0]
    assert 'objective' not in (output/'development.csv').read_text().splitlines()[0]
    stored = json.loads((output/'report.json').read_text())
    assert stored['f64_context']['after_update'] == 64
    assert (output/'input-snapshot/parent.json').read_bytes() == parent_data
    for name, entry in stored['sources'].items():
        assert summary.common.sha((output/'source-snapshot'/name).read_bytes()) == entry
    arguments[2] = '0'*64
    with pytest.raises(ValueError, match='Input SHA'):
        summary.main(arguments + ['--output', str(tmp_path/'bad')])
    assert not (tmp_path/'bad').exists()


def test_tracking_guard_rejects_unfinished_pair():
    result = summary.summarize(*fixture())
    result['arms']['KL0.1']['terminal_cloud_update'] = 48
    with pytest.raises(ValueError, match='retained endpoints'): validate_summary(result)
