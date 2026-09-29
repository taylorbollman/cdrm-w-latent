"""Ordered identity restore; immutable generation-pinned CPU byte handling."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import olmo_pilot_execution_restore as restore
from test_campaign_execution_restore import publication, encoded, pins


def republish(p):
    """Rebuild this synthetic fixture's manifest/object bytes after an explicit edit."""
    p.manifest={key:deepcopy(p.receipt[key]) for key in restore.MANIFEST_FIELDS}
    raw=encoded(p.manifest)
    obj=next(o for o in p.receipt['retention']['objects'] if o['uri'].endswith('/manifest.json'))
    obj.update(pins(raw));p.receipt['manifest_sha256']=obj['sha256']
    p.storage[(obj['uri'],obj['generation'])]=raw
    p.path.write_bytes(encoded(p.receipt))


@pytest.fixture
def ordered_publication(publication):
    """New synthetic ordered authority; never converts a real checkpoint."""
    p=publication
    identity=p.receipt['metadata']['configuration']['execution_identity']
    identity['schema']=restore.IDENTITY_SCHEMA
    identity['payload']['scope']='ordered-tiny-acceptance-not-native'
    identity['payload']['data']={'index_manifest_sha256':'a'*64,
        'suite_manifest_sha256':'b'*64,'policy':{'order':'fixture-ordered-chunks'}}
    identity['sha256']=restore.digest(identity['payload'])
    p.receipt['metadata']['source_fingerprint']['execution_identity_sha256']=identity['sha256']
    republish(p)
    return p


@pytest.fixture
def environment(ordered_publication,monkeypatch):
    p=ordered_publication;p.evidence=p.root/'persistent'/'recovery';p.checkpoint=p.root/'ssd'/'recovery'
    def validate(checkpoint,evidence):
        for path in (Path(checkpoint),Path(evidence)):
            if path.is_symlink() or any(x.is_symlink() for x in path.parents):raise ValueError('symlink')
        assert Path(checkpoint).is_relative_to(p.root/'ssd')
        assert Path(evidence).is_relative_to(p.root/'persistent')
    monkeypatch.setattr(restore,'validate_storage_paths',validate)
    return p


def run(p,fetch=None):
    return restore.restore(p.path,restore.old.sha(p.path),p.evidence,p.checkpoint,fetch=fetch or p.fetch)


def test_new_identity_is_accepted_and_old_identity_is_rejected_without_fetch(environment):
    p=environment
    manifest,objects,identity=restore.publication_metadata(p.receipt)
    assert identity['schema']==restore.IDENTITY_SCHEMA and manifest==p.manifest
    with pytest.raises(ValueError,match='Execution identity'):
        restore.old.publication_metadata(p.receipt)
    p.receipt['metadata']['configuration']['execution_identity']['schema']=restore.old.IDENTITY_SCHEMA
    republish(p)
    with pytest.raises(ValueError,match='Execution identity'):run(p)
    assert not p.calls and not p.evidence.exists() and not p.checkpoint.exists()


def test_pinned_stream_restores_only_assets_and_keeps_persistent_evidence(environment):
    p=environment;r=run(p)
    assert r['schema']==restore.SCHEMA and r['status']=='checkpoint_assets_verified_launch_pending'
    assert p.checkpoint.joinpath('state.pt').read_bytes()==p.state
    assert json.loads(p.checkpoint.joinpath('manifest.json').read_text())==p.manifest
    assert set(x.name for x in p.checkpoint.iterdir())=={'state.pt','manifest.json'}
    assert (p.evidence/'authorities/publication-original.json').read_bytes()==p.path.read_bytes()
    assert [g for _,g in p.calls]==['456','123']
    assert r['model_tensors_loaded'] is r['gpu_used'] is r['cloud_writes'] is False
    assert r['original_receipt_local_paths_used'] is False
    for name,pin in r['sources'].items():assert restore.old.sha(p.evidence/'source-snapshot'/name)==pin


@pytest.mark.parametrize('failure',['interrupt','corrupt','overflow','receipt_mutation','wrong_manifest'])
def test_failure_retains_partial_bytes_and_never_commits_ssd_manifest(environment,failure):
    p=environment
    if failure=='wrong_manifest':
        changed=deepcopy(p.manifest);changed['counters']['input_tokens']+=1;raw=encoded(changed)
        obj=next(o for o in p.receipt['retention']['objects'] if o['uri'].endswith('/manifest.json'))
        obj.update(pins(raw));p.receipt['manifest_sha256']=obj['sha256']
        p.storage[(obj['uri'],obj['generation'])]=raw;p.path.write_bytes(encoded(p.receipt))
    def fetch(record,sink):
        raw=p.storage[(record['uri'],record['generation'])]
        assert not (p.checkpoint/'manifest.json').exists()
        if record['uri'].endswith('/state.pt'):
            if failure=='interrupt':sink.write(raw[:5]);raise OSError('interruption')
            if failure=='corrupt':sink.write(b'x'*len(raw));return
            if failure=='overflow':sink.write(raw+b'x');return
            if failure=='receipt_mutation':p.path.write_text('{}')
        sink.write(raw)
    with pytest.raises((ValueError,OSError)):run(p,fetch)
    assert not (p.checkpoint/'manifest.json').exists()
    assert json.loads((p.evidence/'failure.json').read_text())['checkpoint_manifest_published'] is False
    assert (p.evidence/'authorities/publication-original.json').is_file()


def test_manifest_is_durably_published_after_state_and_interruption_is_not_committed(environment,monkeypatch):
    p=environment;real=restore.old.durable_publish;order=[]
    def publish(partial,destination):
        order.append(destination.name)
        if destination.name=='manifest.json':
            assert (p.checkpoint/'state.pt').read_bytes()==p.state
            raise OSError('interrupted commit')
        return real(partial,destination)
    monkeypatch.setattr(restore.old,'durable_publish',publish)
    with pytest.raises(OSError):run(p)
    assert order==['state.pt','manifest.json'] and not (p.checkpoint/'manifest.json').exists()
    assert (p.evidence/'failure.json').exists()


@pytest.mark.parametrize('mutation',['generation','identity','fingerprint','cursor','unverified','different_parent'])
def test_invalid_ordered_receipt_is_rejected_before_destination(environment,mutation):
    p=environment;r=p.receipt;obj=r['retention']['objects'][0]
    if mutation=='generation':obj['generation']='latest'
    elif mutation=='identity':r['metadata']['configuration']['execution_identity']['payload']['arm']='B'
    elif mutation=='fingerprint':r['metadata']['source_fingerprint']['execution_identity_sha256']='0'*64
    elif mutation=='cursor':r['rank_cursors'][1]['cursor']['next_update']=2
    elif mutation=='unverified':obj['verification']['download_sha256']=False
    else:obj['uri']=obj['uri'].replace('update-000001','update-000002')
    p.path.write_bytes(encoded(r))
    with pytest.raises(ValueError):run(p)
    assert not p.calls and not p.evidence.exists() and not p.checkpoint.exists()


@pytest.mark.parametrize('where',['ssd','persistent'])
def test_existing_destination_is_never_replaced(environment,where):
    p=environment;(p.checkpoint if where=='ssd' else p.evidence).mkdir(parents=True)
    with pytest.raises(ValueError):run(p)
    assert not p.calls


def test_standalone_import_is_cpu_only_without_tensor_or_cloud_clients():
    command='import sys; import scripts.olmo_pilot_execution_restore; assert "torch" not in sys.modules; assert "google.cloud.storage" not in sys.modules'
    subprocess.run([sys.executable,'-c',command],cwd=restore.ROOT,check=True,capture_output=True,text=True)
