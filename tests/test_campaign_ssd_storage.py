"""Real local byte/file lifecycle; cloud verification receipts are synthetic."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import olmo_campaign_ssd_storage as storage
from test_campaign_execution_restore import publication, encoded, pins


@pytest.fixture
def manager(publication,tmp_path):
    mount=tmp_path/'ssd';mount.mkdir()
    persistent=tmp_path/'project';persistent.mkdir()
    evidence=persistent/'stage';evidence.mkdir()
    base=mount/'cdrm-checkpoints';root=base/'new-namespace'/'segment'
    identity=publication.receipt['metadata']['configuration']['execution_identity']['sha256']
    prefix=storage.authority.REMOTE_ROOT+'campaign-storage-tests/new-namespace/segment'
    kwargs=dict(ssd_base=base,persistent_root=persistent,mount_root=mount,mount_check=lambda value:value==mount)
    obj=storage.SSDCheckpointStorage.create(root,evidence,execution_identity_sha256=identity,
        storage_prefix=prefix,**kwargs)
    return SimpleNamespace(storage=obj,publication=publication,kwargs=kwargs,root=root,
        evidence=evidence,mount=mount,persistent=persistent,prefix=prefix,identity=identity)


def checkpoint(fixture,update):
    obj=fixture.storage;directory=obj.destination(update);directory.mkdir()
    receipt=deepcopy(fixture.publication.receipt)
    receipt['directory']=str(directory);receipt['counters']['optimizer_updates']=update
    for row in receipt['rank_cursors']:row['cursor']['next_update']=update
    raw_state=fixture.publication.state+str(update).encode()
    receipt['state'].update(sha256=pins(raw_state)['sha256'],size_bytes=len(raw_state))
    manifest={key:receipt[key] for key in storage.authority.MANIFEST_FIELDS};raw_manifest=encoded(manifest)
    receipt['manifest_sha256']=pins(raw_manifest)['sha256']
    for name,data in [('state.pt',raw_state),('manifest.json',raw_manifest)]:
        (directory/name).write_bytes(data)
        row=next(row for row in receipt['retention']['objects'] if row['uri'].endswith('/'+name))
        row.update(pins(data));row['uri']=fixture.prefix+f'/update-{update:06d}/'+name
        row['generation']=str(1000+update*2+(name=='manifest.json'))
    return receipt


def latest(fixture):return json.loads((fixture.evidence/'latest-checkpoint.json').read_text())
def journal(fixture):return json.loads((fixture.evidence/storage.JOURNAL).read_text())


def test_real_file_lifecycle_prunes_only_verified_owned_old_states_and_preserves_receipts(manager):
    reports=[]
    for step in range(4):
        receipt=checkpoint(manager,step)
        reports.append(manager.storage.publish_retained(receipt))
        assert latest(manager)==receipt
    assert reports[-1]['kept_updates']==[2,3] and reports[-1]['pruned_updates']==[0,1]
    assert sorted(path.name for path in manager.root.iterdir())==[storage.MARKER,'update-000002','update-000003']
    saved=journal(manager)
    assert [row['status'] for row in saved['prune_operations']]==['completed','completed']
    for entry in saved['published']:
        path=manager.evidence/entry['receipt_path']
        assert storage.authority.sha(path)==entry['receipt_sha256']
        storage.authority.publication_metadata(json.loads(path.read_text()))
    assert len(list(manager.storage.receipts.iterdir()))==4
    manager.storage.validate()


def test_prune_observes_durable_new_authority_and_planned_journal_before_deletion(manager,monkeypatch):
    first=checkpoint(manager,0);manager.storage.publish_retained(first)
    manager.storage.publish_retained(checkpoint(manager,1))
    original=manager.storage._prune_one;seen=[]
    def inspect(entry):
        assert latest(manager)['counters']['optimizer_updates']==2
        saved=journal(manager)
        assert saved['published'][-1]['update']==2
        assert saved['prune_operations'][-1]['status']=='planned'
        assert (manager.evidence/saved['published'][-1]['receipt_path']).is_file()
        seen.append(entry['update']);original(entry)
    monkeypatch.setattr(manager.storage,'_prune_one',inspect)
    manager.storage.publish_retained(checkpoint(manager,2))
    assert seen==[0]


def test_unknown_incomplete_and_historical_directories_are_never_candidates(manager):
    unknown=manager.root/'update-000077';unknown.mkdir();(unknown/'state.pt').write_bytes(b'unknown')
    foreign=manager.root.parent/'historical';foreign.mkdir();(foreign/'important').write_bytes(b'keep')
    for step in range(3):manager.storage.publish_retained(checkpoint(manager,step))
    assert (unknown/'state.pt').read_bytes()==b'unknown'
    assert (foreign/'important').read_bytes()==b'keep'
    assert not (manager.root/'update-000000').exists()


@pytest.mark.parametrize('fault',['root_symlink','root_replaced','marker','journal','receipt_symlink','latest'])
def test_changed_authority_fails_before_deletion(manager,fault):
    manager.storage.publish_retained(checkpoint(manager,0))
    if fault=='root_symlink':
        moved=manager.root.with_name('moved');manager.root.rename(moved);manager.root.symlink_to(moved,target_is_directory=True)
    elif fault=='root_replaced':
        moved=manager.root.with_name('moved');manager.root.rename(moved);manager.root.mkdir()
        (manager.root/storage.MARKER).write_bytes((moved/storage.MARKER).read_bytes())
    elif fault=='marker':(manager.root/storage.MARKER).write_text('{}')
    elif fault=='journal':(manager.evidence/storage.JOURNAL).write_text('{}')
    elif fault=='latest':(manager.evidence/'latest-checkpoint.json').write_text('{}')
    else:
        directory=manager.storage.receipts;new=directory.with_name('moved-receipts');directory.rename(new);directory.symlink_to(new,target_is_directory=True)
    with pytest.raises(ValueError):manager.storage.validate()


@pytest.mark.parametrize('fault',['unverified','wrong_generation','wrong_prefix','identity','local_bytes','manifest_bytes',
    'extra_file','symlink_state','hardlink_state','unregistered'])
def test_bad_receipt_or_local_state_never_publishes_or_prunes(manager,fault):
    old=checkpoint(manager,0);manager.storage.publish_retained(old)
    receipt=checkpoint(manager,1);directory=Path(receipt['directory'])
    if fault=='unverified':receipt['retention']['download_sha256_verified']=False
    elif fault=='wrong_generation':receipt['retention']['objects'][0]['generation']='latest'
    elif fault=='wrong_prefix':receipt['retention']['objects'][0]['uri']=receipt['retention']['objects'][0]['uri'].replace('new-namespace','other-namespace')
    elif fault=='identity':receipt['metadata']['configuration']['execution_identity']['payload']['arm']='B'
    elif fault=='local_bytes':(directory/'state.pt').write_bytes(b'changed')
    elif fault=='manifest_bytes':(directory/'manifest.json').write_text('{}')
    elif fault=='extra_file':(directory/'unrelated').write_bytes(b'keep')
    elif fault=='symlink_state':
        state=directory/'state.pt';dest=directory.parent/'outside';state.rename(dest);state.symlink_to(dest)
    elif fault=='hardlink_state':os.link(directory/'state.pt',directory.parent/'hardlinked')
    else:
        receipt['directory']=str(manager.root/'update-000009');directory.rename(receipt['directory'])
    with pytest.raises((ValueError,FileNotFoundError)):manager.storage.publish_retained(receipt)
    assert latest(manager)==old
    assert (manager.root/'update-000000'/'state.pt').is_file()
    assert len(journal(manager)['published'])==1


@pytest.mark.parametrize('phase',['receipt','journal','latest'])
def test_publication_failure_never_starts_pruning(manager,monkeypatch,phase):
    for step in range(2):manager.storage.publish_retained(checkpoint(manager,step))
    original=storage._write_durable
    def fail(path,value,**kwargs):
        path=Path(path)
        if ((phase=='receipt' and path.parent==manager.storage.receipts)
                or (phase=='journal' and path.name==storage.JOURNAL)
                or (phase=='latest' and path.name=='latest-checkpoint.json')):
            raise OSError('simulated durable publication failure')
        return original(path,value,**kwargs)
    receipt=checkpoint(manager,2);monkeypatch.setattr(storage,'_write_durable',fail)
    with pytest.raises(OSError):manager.storage.publish_retained(receipt)
    assert all((manager.root/f'update-{step:06d}'/'state.pt').is_file() for step in range(3))
    assert latest(manager)['counters']['optimizer_updates']==1


@pytest.mark.parametrize('failure',['before_delete','after_manifest_delete'])
def test_failed_prune_leaves_new_gcs_authority_and_durable_failure(manager,monkeypatch,failure):
    for step in range(2):manager.storage.publish_retained(checkpoint(manager,step))
    original=os.unlink
    def unlink(path,*args,**kwargs):
        if kwargs.get('dir_fd') is not None and (failure=='before_delete' or path=='state.pt'):
            raise OSError('simulated deletion interruption')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(os,'unlink',unlink)
    receipt=checkpoint(manager,2)
    with pytest.raises(storage.StoragePruneError):manager.storage.publish_retained(receipt)
    assert latest(manager)==receipt
    storage.authority.publication_metadata(latest(manager))
    saved=journal(manager)
    assert saved['prune_operations'][-1]['status']=='failed'
    assert saved['published'][0]['local_status']=='prune_failed'
    assert (manager.root/'update-000002'/'state.pt').is_file()
    assert (manager.root/'update-000000'/'manifest.json').exists()==(failure=='before_delete')


def test_changed_old_content_blocks_prune_without_invalidating_new_publication(manager):
    for step in range(2):manager.storage.publish_retained(checkpoint(manager,step))
    (manager.root/'update-000000'/'state.pt').write_bytes(b'changed between save and prune')
    receipt=checkpoint(manager,2)
    with pytest.raises(storage.StoragePruneError):manager.storage.publish_retained(receipt)
    assert latest(manager)==receipt
    assert (manager.root/'update-000000'/'state.pt').is_file()


@pytest.mark.parametrize('change',['no_mount','outside','traversal','depth','symlink','ssd_evidence','outside_evidence'])
def test_readonly_path_preflight_rejects_unsafe_or_nonssd_roots(manager,change):
    root=manager.root.parent/'new-segment';evidence=manager.persistent/'new-stage';kwargs=dict(manager.kwargs)
    if change=='no_mount':kwargs['mount_check']=lambda _:False
    elif change=='outside':root=manager.persistent/'checkpoint'
    elif change=='traversal':root=manager.root.parent/'..'/'namespace'/'new'
    elif change=='depth':root=root/'extra'
    elif change=='symlink':
        link=manager.mount/'alias';link.symlink_to(manager.root.parent,target_is_directory=True);root=link/'new'
    elif change=='ssd_evidence':evidence=manager.mount/'evidence'
    else:evidence=manager.persistent.parent/'elsewhere'
    with pytest.raises(ValueError):storage.validate_storage_paths(root,evidence,**kwargs)
    assert not evidence.exists()


def test_existing_segment_and_protected_resume_source_cannot_be_reused(manager):
    with pytest.raises(FileExistsError):storage.SSDCheckpointStorage.create(manager.root,manager.evidence,
        execution_identity_sha256=manager.identity,storage_prefix=manager.prefix,**manager.kwargs)
    manager.storage.resume_source=str(manager.root/'update-000001')
    with pytest.raises(ValueError,match='Resume'):manager.storage.destination(1)
    assert '1' not in journal(manager)['destinations']


def test_production_preflight_rejects_bare_directory_without_mount(tmp_path):
    ssd=tmp_path/'bare';ssd.mkdir();project=tmp_path/'project';project.mkdir()
    with pytest.raises(ValueError,match='mount'):
        storage.validate_storage_paths(ssd/'cdrm-checkpoints'/'namespace'/'segment',project/'evidence',
            ssd_base=ssd/'cdrm-checkpoints',persistent_root=project,mount_root=ssd)
