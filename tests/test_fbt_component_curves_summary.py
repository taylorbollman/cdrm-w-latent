"""A plot may combine conditions, never input/mask/precision mismatches."""
import json

import pytest

from scripts import olmo_fbt_component_curves_summary as summary


def record(arm, update):
    metrics = {'ce': 3., 'relative_delta_rms': None}
    return {'schema': summary.SCHEMAS[0 if arm == 'F' else 1], 'status': 'completed',
        'arm': arm, 'kl_weight': None if arm == 'F' else 1., 'after_update': update,
        'membership_sha256': 'a'*64, 'index_manifest_sha256': 'b'*64,
        'result': {'input_tokens': 8192, 'ce_targets': 8184, 'policy': 'common_fp32_no_jitter_v1',
            'beta': 1., 'passes': [{'pass': 1, 'regions': {'all': {'metrics': metrics}}}]}}


def write(tmp_path, name, value):
    path = tmp_path/name; path.write_text(json.dumps(value)); return path


def test_same_update_in_different_conditions_is_explicit_and_valid(tmp_path):
    paths = [write(tmp_path, f'{arm}.json', record(arm, 32)) for arm in ('NFR', 'F', 'NF')]
    actual = summary.load_inputs(paths)
    assert [r['arm'] for r in actual] == ['F', 'NF', 'NFR']
    assert actual[1]['label'] == 'NF update 32 KL 1'


@pytest.mark.parametrize('change', ['membership', 'targets', 'precision', 'beta', 'running', 'passes', 'duplicate'])
def test_invalid_comparisons_are_rejected(tmp_path, change):
    first = record('F', 32); second = record('NF', 32)
    if change == 'membership': second['membership_sha256'] = 'c'*64
    elif change == 'targets': second['result']['ce_targets'] -= 1
    elif change == 'precision': second['result']['policy'] = 'bf16'
    elif change == 'beta': second['result']['beta'] = .5
    elif change == 'running': second['status'] = 'running'
    elif change == 'passes': second['result']['passes'][0]['pass'] = 2
    elif change == 'duplicate': second = first
    with pytest.raises(ValueError):
        summary.load_inputs([write(tmp_path, 'first.json', first), write(tmp_path, 'second.json', second)])
