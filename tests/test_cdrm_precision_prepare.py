"""Prospective-generation guards; no native cases or seed7502 model are drawn."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cdrm_precision_prepare as prepare
from cdrm_precision_confirm import ROLE_SPECS


@pytest.fixture(autouse=True)
def prohibit_scientific_generation(monkeypatch):
    assert not torch.cuda.is_available(), 'Use explicitly GPU-disabled CPU container'
    def forbidden(*args,**kwargs):
        raise AssertionError('Tests must not draw fresh cases or the seed7502 initialization')
    monkeypatch.setattr(prepare,'generate_dataset',forbidden)
    monkeypatch.setattr(prepare,'cpu_initial_model',forbidden)


def roles():
    return json.loads((prepare.LINEAGE/'decisions/future-confirmation-roles-v1.json').read_text())


def test_output_guard_rejects_existing_escaping_or_overlapping_paths(tmp_path):
    base=tmp_path/'lineage';base.mkdir()
    prepare.validate_outputs(base/'prepare',base/'data/confirmation',lineage=base)
    with pytest.raises(ValueError,match='new numerical-resolution'):
        prepare.validate_outputs(tmp_path/'old',base/'data',lineage=base)
    with pytest.raises(ValueError,match='nonoverlapping'):
        prepare.validate_outputs(base/'prepare',base/'prepare/data',lineage=base)
    existing=base/'closed';existing.mkdir();(existing/'evidence').write_text('unchanged')
    with pytest.raises(FileExistsError):prepare.validate_outputs(existing,base/'data',lineage=base)
    assert (existing/'evidence').read_text()=='unchanged'


def test_wrong_decision_anchor_stops_before_generation(tmp_path):
    decision=tmp_path/'decision.json';decision.write_text('{"status":"draft"}')
    args=SimpleNamespace(decision=decision,decision_sha256='wrong')
    with pytest.raises(ValueError,match='Immutable JSON SHA'):prepare.preflight(args)
    assert list(tmp_path.iterdir())==[decision]


def test_unfrozen_decision_stops_before_generation(tmp_path):
    rolefile=tmp_path/'roles.json';rolefile.write_text(json.dumps(roles()))
    decision=tmp_path/'decision.json';decision.write_text(json.dumps({
        'schema':'cdrm-precision-candidate-freeze-v1','status':'draft',
        'roles':{'path':str(rolefile),'sha256':prepare.file_digest(rolefile)}}))
    args=SimpleNamespace(decision=decision,decision_sha256=prepare.file_digest(decision))
    with pytest.raises(ValueError,match='frozen candidate'):prepare.preflight(args)


def test_source_freeze_cannot_omit_or_change_producer(monkeypatch):
    monkeypatch.setattr(prepare,'required_sources',lambda:{'producer.py':'frozen'})
    monkeypatch.setattr(prepare,'verify_sources',lambda _:None)
    for tracked in ({},{'producer.py':'changed'}):
        with pytest.raises(ValueError,match='producer, tests'):
            prepare.validate_frozen_sources({'source_sha256':tracked})
    assert prepare.validate_frozen_sources({'source_sha256':{'producer.py':'frozen'}})=={'producer.py':'frozen'}


def test_manifest_preserves_exact_three_role_offsets_and_checkpoint_bindings():
    class SyntheticDataset:
        # Descriptive constants; this never invokes native data generation.
        sha256='synthetic-corpus'
        manifest={'seed':925903,'manifest_sha256':'synthetic-manifest'}
        def __len__(self):return 192
        def take(self,selection):return SimpleNamespace(sha256=f'synthetic-{selection.start}-{selection.stop}')
    checkpoints={name:{'path':name+'.pt','sha256':spec[1] or 'synthetic-initial'} for name,spec in ROLE_SPECS.items()}
    result=prepare.fixture_manifest('candidate','roles',roles(),SyntheticDataset(),'synthetic-root',checkpoints)
    assert [(r['example_offset'],r['example_stop']) for r in result['roles']]==[(0,64),(64,128),(128,192)]
    assert [r['batch_sha256'] for r in result['roles']]==['synthetic-0-64','synthetic-64-128','synthetic-128-192']
    swapped=copy.deepcopy(checkpoints);swapped['fp32_trained_u1000']=checkpoints['bf16_trained_u1000']
    with pytest.raises(ValueError,match='wrong trained checkpoint'):
        prepare.fixture_manifest('candidate','roles',roles(),SyntheticDataset(),'synthetic-root',swapped)


def test_failed_preflight_does_not_write_into_existing_directory(tmp_path,monkeypatch):
    output=tmp_path/'existing';output.mkdir();(output/'evidence').write_bytes(b'closed')
    def reject(*args):raise ValueError('preflight rejects existing path')
    monkeypatch.setattr(prepare,'prepare',reject)
    with pytest.raises(ValueError,match='preflight rejects'):
        prepare.main(['--decision','unavailable','--decision-sha256','invalid','--output-dir',str(output)])
    assert {p.name for p in output.iterdir()}=={'evidence'}
    assert (output/'evidence').read_bytes()==b'closed'
