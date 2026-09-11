"""CPU-only checks of native-mask partitioning and denominator semantics."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cdrm_fuzzy_ceiling_eval as ceiling
from cdrm.mad_data import MadDataset


def test_oracle_subsets_partition_native_targets_without_inventing_supervision():
    labels = np.array([[7, 8, -100], [-100, 9, 10], [11, -100, -100]])
    available = np.array([[True, False, False], [False, True, True], [False, False, False]])
    masks = ceiling.masks_for(labels, available)
    assert np.array_equal(masks['oracle_available'] | masks['oracle_unavailable'], masks['native'])
    assert not np.any(masks['oracle_available'] & masks['oracle_unavailable'])
    prediction = np.array([[7, 2, 4], [1, 9, 2], [11, 3, 2]])
    scores = {key: ceiling.subset_metrics(prediction, labels, mask) for key, mask in masks.items()}
    assert scores['native']['targets'] == 5
    assert scores['native']['errors'] == scores['oracle_available']['errors'] + scores['oracle_unavailable']['errors'] == 2
    assert scores['oracle_available']['examples_with_targets'] == 2
    assert scores['oracle_available']['examples_without_targets'] == 1
    assert scores['oracle_available']['exact'] == 1
    assert scores['oracle_available']['sequence_exact_match'] == .5
    assert scores['oracle_unavailable']['examples_with_targets'] == 2
    assert scores['oracle_unavailable']['exact'] == 1


def test_empty_subset_has_no_vacuously_correct_examples_or_nan_rates():
    labels = np.array([[7, -100], [8, -100]])
    result = ceiling.subset_metrics(labels, labels, np.zeros_like(labels, dtype=bool), np.zeros_like(labels, dtype=np.float32))
    assert result['targets'] == result['exact'] == result['examples_with_targets'] == 0
    assert result['examples_without_targets'] == 2
    assert result['token_accuracy'] is result['sequence_exact_match'] is result['ce'] is None
    assert result['ce_sum'] == 0.


def test_subset_ce_excludes_ignored_and_other_subset_positions():
    labels = np.array([[7, 8, -100], [9, -100, -100]])
    mask = np.array([[True, False, False], [True, False, False]])
    losses = np.array([[.5, 99., np.nan], [1.5, np.nan, np.nan]], dtype=np.float32)
    result = ceiling.subset_metrics(labels, labels, mask, losses)
    assert result['ce_sum'] == 2. and result['ce'] == 1.


def test_prediction_flips_distinguish_corrections_regressions_and_still_wrong():
    labels = np.array([[7, 8, 9, -100], [10, -100, -100, -100], [11, -100, -100, -100]])
    bf16 = np.array([[1, 8, 1, 3], [10, 2, 3, 4], [2, 2, 3, 4]])
    fp32 = np.array([[7, 2, 2, 9], [10, 8, 8, 8], [11, 2, 3, 4]])
    mask = labels != -100
    result = ceiling.prediction_changes(bf16, fp32, labels, mask)
    assert result['prediction_flips'] == 4
    assert result['bf16_wrong_fp32_right'] == 2
    assert result['bf16_right_fp32_wrong'] == 1
    assert result['both_wrong_prediction_changed'] == 1
    assert result['exact_examples_gained_in_fp32'] == 1
    assert result['exact_examples_lost_in_fp32'] == 0
    assert result['token_accuracy_fp32_minus_bf16'] == pytest.approx(.2)


@pytest.mark.parametrize('available', [np.array([[True, True]]), np.array([[1, 0]]), np.zeros((2, 2), dtype=bool)])
def test_availability_rejects_new_targets_nonboolean_and_wrong_shape(available):
    with pytest.raises(ValueError):
        ceiling.masks_for(np.array([[7, -100]]), available)


def test_native_replay_requires_exact_metrics_without_timing_comparison():
    expected = {'native': {'ce': 2., 'correct': 5}, 'answer': {'ce': 2., 'correct': 5}, 'seconds': 99.}
    actual = {'native': {'ce': 2., 'correct': 5}, 'answer': {'ce': 2., 'correct': 5}}
    assert ceiling.require_native_replay(actual, expected)
    actual['native']['ce'] = np.nextafter(2., 3.)
    with pytest.raises(AssertionError, match='BF16 native'):
        ceiling.require_native_replay(actual, expected)


@pytest.fixture
def authority(tmp_path):
    labels = np.full((1280, 256), -100, dtype=np.int64)
    labels[:, 0] = 7
    dataset = MadDataset(np.zeros_like(labels), labels, labels.copy(), [{} for _ in labels],
                         {'task': ceiling.common.TASK, 'split': 'dev', 'vocab_size': 16,
                          'manifest_sha256': 'retained-development-manifest',
                          'native_training_objective': 'dense_next_token_including_native_padding'})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}\n')
    source = tmp_path / 'frozen.py'
    source.write_text('unchanged = True\n')
    identity = {'arm': 'seq', 'precision': 'bf16', 'physical_batch': 128, 'length': 256,
                'updates_per_epoch': 100, 'maximum_epochs': 10,
                'model_config': {'ordinary_attention_precision_policy': 'fp32'},
                'source_sha256': {str(source): ceiling.common.file_digest(source)},
                'data': {'dev': ceiling.common.dataset_identity(dataset)},
                'data_manifest': ceiling.common.reference(manifest)}
    endpoint = {'native': {'correct': 1280}, 'answer': {'correct': 1280}}
    checkpoint = {'format': ceiling.common.FORMAT, 'identity': identity,
                  'identity_sha256': ceiling.common.json_digest(identity), 'arm': 'seq', 'precision': 'bf16',
                  'model_config': identity['model_config'], 'completed_updates': 1000,
                  'completed_epochs': 10, 'batch_in_epoch': 0, 'development': {'1000': endpoint}}
    reference = {'path': 'authoritative-u01000.pt', 'sha256': 'exact-checkpoint-sha', 'bytes': 123}
    report = {**copy.deepcopy(checkpoint), 'schema': 'cdrm-fuzzy-calibration-v1', 'status': 'complete',
              'checkpoints': {'1000': reference},
              'arguments': {'output_dir': 'retained/calibration/seq-lr5e-4-e10', 'reference_final': None}}
    return checkpoint, report, reference, dataset, source


def test_authority_accepts_only_matching_endpoint_and_frozen_data(authority):
    checkpoint, report, reference, dataset, _ = authority
    assert ceiling.validate_authority(checkpoint, report, reference, dataset) == report['development']['1000']
    bad_reference = {**reference, 'sha256': 'different-checkpoint'}
    with pytest.raises(ValueError, match='authoritative epoch-10 checkpoint'):
        ceiling.validate_authority(checkpoint, report, bad_reference, dataset)
    dataset.labels[0, 0] = 8
    dataset.answer_labels[0, 0] = 8
    with pytest.raises(ValueError, match='same full retained'):
        ceiling.validate_authority(checkpoint, report, reference, dataset)


def test_authority_rejects_earlier_endpoint_and_modified_source(authority):
    checkpoint, report, reference, dataset, source = authority
    earlier = copy.deepcopy(checkpoint)
    earlier['completed_updates'] = 999
    with pytest.raises(ValueError, match='epoch-10 endpoint'):
        ceiling.validate_authority(earlier, report, reference, dataset)
    source.write_text('unchanged = False\n')
    with pytest.raises(RuntimeError, match='source identity changed'):
        ceiling.validate_authority(checkpoint, report, reference, dataset)


@pytest.mark.parametrize('change', ['recovery_comparison', 'reference_final', 'wrong_output_scope'])
def test_recovery_or_noncalibration_endpoint_cannot_be_used_as_authority(authority, change):
    checkpoint, report, reference, dataset, _ = authority
    if change == 'recovery_comparison':
        report['recovery_comparison'] = {'bitwise_state_and_nontiming_metrics_equal': True}
    elif change == 'reference_final':
        report['arguments']['reference_final'] = 'calibration/authoritative.pt'
    else:
        report['arguments']['output_dir'] = 'retained/recovery/seq-lr5e-4-e10'
    with pytest.raises(ValueError, match='not a recovery control'):
        ceiling.validate_authority(checkpoint, report, reference, dataset)
