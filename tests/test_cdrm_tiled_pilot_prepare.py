"""CPU checks of prospective slot isolation and legacy checkpoint compatibility."""
import copy
import dataclasses
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cdrm_tiled_pilot_prepare as prepare
import cdrm_tiled_common as common


def test_confirmation_roles_are_disjoint_and_leave_reserved_examples_unused():
    allocation = prepare.confirmation_allocation()
    assert allocation['seed'] == 925801 and allocation['examples'] == 768
    roles = {(slot['model_seed'], slot['completed_updates'], slot['checkpoint_trajectory'])
             for slot in allocation['slots']}
    assert roles == {(seed, step, precision) for seed in (7500, 7501)
                     for step in (1000, 2500) for precision in ('fp32', 'bf16')}
    seen = set()
    for slot in allocation['slots']:
        assigned = set(range(slot['example_offset'], slot['example_stop']))
        assert len(assigned) == slot['physical_batch'] == 64
        assert not seen & assigned
        seen.update(assigned)
    assert seen == set(range(512))
    assert allocation['reserved']['example_offset'] == 512
    assert allocation['reserved']['example_stop'] == 768
    assert allocation['slots'][0]['slot'] == 'seed7500-u1000-fp32'
    assert allocation['slots'][-1]['slot'] == 'seed7501-u2500-bf16'


def test_retained_or_overlapping_output_paths_are_rejected(tmp_path):
    lineage, prior = tmp_path / 'new', tmp_path / 'old'
    prepare.validate_output_paths(lineage / 'prepare', lineage / 'data/confirmation', lineage=lineage, prior=prior)
    with pytest.raises(ValueError, match='dedicated pilot'):
        prepare.validate_output_paths(prior / 'prepare', lineage / 'data', lineage=lineage, prior=prior)
    with pytest.raises(ValueError, match='separate sibling'):
        prepare.validate_output_paths(lineage / 'prepare', lineage / 'prepare/data', lineage=lineage, prior=prior)
    (lineage / 'prepare').mkdir(parents=True)
    with pytest.raises(FileExistsError, match='Refusing'):
        prepare.validate_output_paths(lineage / 'prepare', lineage / 'data', lineage=lineage, prior=prior)


def test_fresh_seed_checkpoint_preserves_optimizer_and_original_sources(tmp_path):
    # No model forward, GPU execution, original data access or retained writes.
    assert not torch.cuda.is_available(), 'Use the explicitly selected CPU container'
    original, _ = common.cpu_initial_model(1907)
    optimizer = common.optimizer_for(original)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    prior = {'model_config': dataclasses.asdict(original.config),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'identity': {'source_sha256': {'original/source.py': 'retained-source-hash'}}}
    prior_snapshot = copy.deepcopy(prior)
    freeze = tmp_path / 'freeze.json'
    freeze.write_text('{"test": true}\n')
    parent = {'path': 'original/init.pt', 'sha256': 'retained-checkpoint-hash', 'identity_sha256': 'retained-identity-hash'}
    packet, checks = prepare.fresh_initialization(7501, prior_initial=prior, tracked_sources={},
        freeze_path=freeze, data={'train': {'identity': 'same'}, 'dev': {'identity': 'same'}},
        data_root=Path('original/data'), confirmation_root=Path('new/data/confirmation'), parent_record=parent)
    assert common.state_digest(prior) == common.state_digest(prior_snapshot)
    assert packet['format'] == common.INIT_FORMAT and packet['completed_updates'] == 0
    assert packet['identity_sha256'] == common.json_digest(packet['identity'])
    assert packet['identity']['parent_source_sha256'] == prior['identity']['source_sha256']
    assert packet['identity']['seeds']['shuffle'] == 925704
    assert checks['legacy_num_loader_compatible'] and not packet['optimizer']['state']
    assert packet['initialization']['seed'] == 7501
    # Build the matching ordinary backbone independently and compare every tensor.
    common.seed_cpu(7501)
    from olmo.model import OLMo
    seq_config = common.config('naive', 'fp32')
    seq_config.cdrm_enabled = False
    seq = OLMo(seq_config)
    extras = set(packet['model']) - set(seq.state_dict())
    assert extras == {'cdrm.deep_adapter.weight', 'cdrm.bridge_adapter.weight'}
    assert all(torch.equal(value, packet['model'][name]) for name, value in seq.state_dict().items())
    assert all(torch.count_nonzero(packet['model'][name]) for name in extras)
    for precision in ('fp32', 'bf16'):
        restored, _ = common.build_model('tiled', precision, checkpoint=packet, device='cpu')
        assert common.state_digest(restored.state_dict()) == packet['initialization']['full_initialization_sha256']
