"""SSD assets are streamed separately from persistent authority evidence."""
import json
from pathlib import Path
import pytest
from scripts import olmo_campaign_ssd_restore as restore
from test_campaign_execution_restore import publication


@pytest.fixture
def environment(publication,monkeypatch):
    p=publication;p.evidence=p.root/'persistent'/'recovery';p.checkpoint=p.root/'ssd'/'recovery'
    def validate(checkpoint,evidence):
        for path in (Path(checkpoint),Path(evidence)):
            if path.is_symlink() or any(x.is_symlink() for x in path.parents):raise ValueError('symlink')
        assert Path(checkpoint).is_relative_to(p.root/'ssd')
        assert Path(evidence).is_relative_to(p.root/'persistent')
    monkeypatch.setattr(restore,'validate_storage_paths',validate)
    return p


def run(p,fetch=None):
    return restore.restore(p.path,restore.old.sha(p.path),p.evidence,p.checkpoint,fetch=fetch or p.fetch)


def test_evidence_persists_separately_with_pinned_generations_and_no_tensor_load(environment):
    p=environment;r=run(p)
    assert p.checkpoint.joinpath('state.pt').read_bytes()==p.state
    assert json.loads(p.checkpoint.joinpath('manifest.json').read_text())==p.manifest
    assert set(x.name for x in p.checkpoint.iterdir())=={'state.pt','manifest.json'}
    assert (p.evidence/'authorities/publication-original.json').read_bytes()==p.path.read_bytes()
    assert [g for _,g in p.calls]==['456','123']
    assert r['model_tensors_loaded'] is r['gpu_used'] is r['cloud_writes'] is False
    for name,pin in r['sources'].items():assert restore.old.sha(p.evidence/'source-snapshot'/name)==pin


@pytest.mark.parametrize('failure',['interrupt','corrupt','overflow','source_mutation'])
def test_failure_leaves_prior_cloud_authority_and_no_ssd_commit(environment,failure):
    p=environment
    def fetch(record,sink):
        raw=p.storage[(record['uri'],record['generation'])]
        assert not (p.checkpoint/'manifest.json').exists()
        if record['uri'].endswith('/state.pt'):
            if failure=='interrupt':sink.write(raw[:5]);raise OSError('interruption')
            if failure=='corrupt':sink.write(b'x'*len(raw));return
            if failure=='overflow':sink.write(raw+b'x');return
            if failure=='source_mutation':p.path.write_text('{}')
        sink.write(raw)
    with pytest.raises((ValueError,OSError)):run(p,fetch)
    assert not (p.checkpoint/'manifest.json').exists()
    assert json.loads((p.evidence/'failure.json').read_text())['checkpoint_manifest_published'] is False
    assert (p.evidence/'authorities/publication-original.json').is_file()


def test_manifest_publication_is_last(environment,monkeypatch):
    p=environment;real=restore.old.durable_publish
    def publish(partial,destination):
        if destination.name=='manifest.json':
            assert (p.checkpoint/'state.pt').read_bytes()==p.state
            raise OSError('interrupted commit')
        return real(partial,destination)
    monkeypatch.setattr(restore.old,'durable_publish',publish)
    with pytest.raises(OSError):run(p)
    assert not (p.checkpoint/'manifest.json').exists()
    assert (p.evidence/'failure.json').exists()


@pytest.mark.parametrize('where',['ssd','persistent'])
def test_existing_destination_is_never_replaced(environment,where):
    p=environment
    (p.checkpoint if where=='ssd' else p.evidence).mkdir(parents=True)
    with pytest.raises(ValueError):run(p)
    assert not p.calls
