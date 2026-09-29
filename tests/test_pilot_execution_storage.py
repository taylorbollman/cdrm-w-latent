"""Ordered identity storage uses the accepted owned-file lifecycle unchanged."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from scripts import olmo_campaign_ssd_storage as old
from scripts import olmo_campaign_execution_restore as base_authority
from scripts import olmo_pilot_execution_storage as new
from test_campaign_execution_restore import publication as original_publication
from test_campaign_ssd_storage import checkpoint, latest, journal


@pytest.fixture
def manager(original_publication,tmp_path):
    publication=original_publication
    identity=publication.receipt['metadata']['configuration']['execution_identity']
    identity['schema']='olmo-pilot-execution-identity-v1'
    identity['payload']['scope']='ordered-tiny-acceptance-not-native'
    identity['sha256']=base_authority.digest(identity['payload'])
    publication.receipt['metadata']['source_fingerprint']['execution_identity_sha256']=identity['sha256']
    mount=tmp_path/'ssd';mount.mkdir();persistent=tmp_path/'project';persistent.mkdir()
    evidence=persistent/'stage';evidence.mkdir();base=mount/'cdrm-checkpoints';root=base/'pilot'/'segment'
    prefix=base_authority.REMOTE_ROOT+'pilot-storage-tests/pilot/segment'
    kwargs=dict(ssd_base=base,persistent_root=persistent,mount_root=mount,mount_check=lambda x:x==mount)
    obj=new.SSDCheckpointStorage.create(root,evidence,execution_identity_sha256=identity['sha256'],storage_prefix=prefix,**kwargs)
    return SimpleNamespace(storage=obj,publication=publication,kwargs=kwargs,root=root,evidence=evidence,
        mount=mount,persistent=persistent,prefix=prefix,identity=identity['sha256'])


def test_ordered_identity_publishes_then_prunes_only_owned_verified_boundaries(manager):
    foreign=manager.root.parent/'historical';foreign.mkdir();(foreign/'keep').write_bytes(b'keep')
    for step in range(4):
        receipt=checkpoint(manager,step);result=manager.storage.publish_retained(receipt)
        assert latest(manager)==receipt
    assert result['kept_updates']==[2,3] and result['pruned_updates']==[0,1]
    assert (foreign/'keep').read_bytes()==b'keep'
    assert [x['status'] for x in journal(manager)['prune_operations']]==['completed','completed']
    manager.storage.validate()


def test_storage_critical_lifecycle_methods_are_inherited():
    for name in ('create','validate','destination','_prune_one','publish_retained','_commit'):
        actual=getattr(new.SSDCheckpointStorage,name);expected=getattr(old.SSDCheckpointStorage,name)
        assert getattr(actual,'__func__',actual) is getattr(expected,'__func__',expected)


@pytest.mark.parametrize('fault',['old_identity','unretained','wrong_prefix','corrupt_bytes'])
def test_changed_receipt_or_bytes_fail_before_publication_or_prune(manager,fault):
    manager.storage.publish_retained(checkpoint(manager,0))
    receipt=checkpoint(manager,1)
    if fault=='old_identity':receipt['metadata']['configuration']['execution_identity']['schema']='olmo-campaign-execution-identity-v1'
    elif fault=='unretained':receipt['retention']['download_sha256_verified']=False
    elif fault=='wrong_prefix':
        for row in receipt['retention']['objects']:row['uri']=row['uri'].replace('/pilot/segment/','/different/segment/')
    else:(manager.root/'update-000001'/'state.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):manager.storage.publish_retained(receipt)
    assert latest(manager)['counters']['optimizer_updates']==0
    assert (manager.root/'update-000000').exists()
    assert journal(manager)['prune_operations']==[]
