"""Literal JSON accounting/pruning oracles; no model, CUDA or cloud calls."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts import olmo_campaign_ssd_audit as audit
from test_campaign_eval_audit import make_pair, reidentify, resumed


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()


def pin(raw):
    return {'size_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'md5_base64': base64.b64encode(hashlib.md5(raw).digest()).decode()}


def pair():
    _, reference = make_pair()
    reference['scale'] = 'tiny'
    rp = reference['configuration']['execution_identity']['payload']
    rp['declaration'] = {'schema': audit.evaluation.NEW_TINY_SCHEMA,
        'evaluation': deepcopy(reference['evaluation_policy']['plan']['declaration']),
        'storage_prefix': 'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/pr47', 'seed': 20260929}
    rp['resolved_contract_sha256'] = audit.digest(rp['declaration'])
    reidentify(reference)
    actual = deepcopy(reference)
    actual['schema'] = audit.SSD_REPORT_SCHEMA
    actual['configuration']['schema'] = audit.SSD_CONFIG_SCHEMA
    actual['sources'].update({name: '9'*64 for name in audit.NEW_SOURCES})
    ap = actual['configuration']['execution_identity']['payload']
    ap['storage_policy'] = deepcopy(audit.STORAGE_POLICY)
    ap['declaration']['storage_prefix'] = 'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/ssd'
    ap['resolved_contract_sha256'] = audit.digest(ap['declaration'])
    reidentify(actual)
    actual['storage'] = {'checkpoint_root': '/mnt/localssd/cdrm-checkpoints/example/reference',
        'evidence_dir': '/workspace/cdrm-w-latent/.runtime/ssd-reference', 'policy': deepcopy(audit.STORAGE_POLICY)}
    return reference, actual


def storage(report):
    report['local_checkpoints'] = []; report['published_checkpoints'] = []; report['storage_publications'] = []
    identity = report['configuration']['execution_identity']; payload = identity['payload']
    root, evidence = Path(report['storage']['checkpoint_root']), Path(report['storage']['evidence_dir'])
    owned = {'schema': 'olmo-campaign-ssd-ownership-v1', **{k: report['storage'][k] for k in ('checkpoint_root','evidence_dir')},
        'ssd_mount': '/mnt/localssd', 'namespace': root.parts[-2], 'segment': root.name,
        'durability': 'volatile SSD; persistent verified-GCS receipts', 'execution_identity_sha256': identity['sha256'],
        'storage_prefix': payload['declaration']['storage_prefix']+'/'+report['arm']+'/'+evidence.name,
        'keep_local_completed': 2, 'resume_source': '/mnt/localssd/cdrm-checkpoints/example/restored/checkpoint' if 'resume' in report else None}
    journal = {'schema': 'olmo-campaign-ssd-journal-v1', 'ownership': owned, 'destinations': {}, 'published': [], 'prune_operations': []}
    receipts = {}
    start = report.get('resume', {}).get('completed_update', 0)
    numbers = list(range(start+1, report['final_counters']['optimizer_updates']+1))
    if 'resume' not in report: numbers.insert(0, 0)
    for number in numbers:
        boundary = audit.legacy.boundary_at(report, number)
        directory = str(root/f'update-{number:06d}')
        state = ('opaque-state-'+str(number)).encode()
        manifest = {'schema': 'olmo-replicated-ddp-checkpoint-v1', 'world_size': 2,
            'metadata': {'schema': 'olmo-replicated-ddp-checkpoint-v1', 'world_size': 2,
                'model_type': 'test.Model', 'configuration': report['configuration'],
                'source_fingerprint': report['source_fingerprint']},
            'counters': deepcopy(boundary[0]['state']['counters']),
            'rank_cursors': [deepcopy(row['cursor']) for row in boundary],
            'state': {'filename': 'state.pt', **{k: pin(state)[k] for k in ('size_bytes', 'sha256')}}}
        manifest_bytes = encoded(manifest)
        receipt = {**manifest, 'directory': directory, 'manifest_sha256': pin(manifest_bytes)['sha256']}
        local_receipt = deepcopy(receipt)
        receipt['retention'] = {'create_only': True, 'download_sha256_verified': True, 'objects': [
            {'uri': owned['storage_prefix']+f'/update-{number:06d}/'+name, 'generation': str(1000+number*2+i),
             **pin(raw), 'verification': dict.fromkeys(('download_sha256','server_md5','server_size','sha256_metadata'), True)}
            for i,(name,raw) in enumerate([('manifest.json',manifest_bytes), ('state.pt',state)])]}
        relative = f'checkpoint-publications/update-{number:06d}.json'
        receipts[relative] = deepcopy(receipt)
        entry = {'update': number, 'directory': directory, 'receipt_path': relative,
                 'receipt_sha256': pin(encoded(receipt))['sha256'], 'local_status': 'retained'}
        journal['destinations'][str(number)] = directory
        journal['published'].append(entry)
        if len(journal['published']) > 2:
            old = journal['published'][-3]; old['local_status'] = 'pruned'
            journal['prune_operations'].append({'update': old['update'], 'directory': old['directory'],
                'receipt_sha256': old['receipt_sha256'], 'replaced_by_update': number, 'status': 'completed'})
        report['local_checkpoints'].append({'optimizer_update': number, 'reason': 'scheduled',
                                           'receipt': local_receipt, 'boundary_by_rank': deepcopy(boundary)})
        report['published_checkpoints'].append(deepcopy(receipt))
        report['storage_publications'].append({'status': 'published_and_local_retention_applied', 'update': number,
            'receipt_path': str(evidence/relative), 'receipt_sha256': entry['receipt_sha256'],
            'journal_path': str(evidence/'ssd-journal.json'), 'journal_sha256': pin(encoded(journal))['sha256'],
            'kept_updates': [r['update'] for r in journal['published'] if r['local_status']=='retained'],
            'pruned_updates': [r['update'] for r in journal['published'] if r['local_status']=='pruned']})
    return {'journal': journal, 'receipts': receipts,
            'latest': deepcopy(report['published_checkpoints'][-1]) if numbers else None,
            'journal_sha256': pin(encoded(journal))['sha256']}


def test_cross_version_exact_training_and_literal_keep_two_journal():
    reference, actual = pair(); evidence = storage(actual)
    before = deepcopy((reference, actual, evidence))
    result = audit.compare(reference, actual, kind='storage_transition', storage_evidence=evidence)
    assert result['passed'], result['failures']
    assert (reference, actual, evidence) == before
    assert actual['schema'] != reference['schema']
    assert [r['update'] for r in evidence['journal']['prune_operations']] == [0, 1]
    assert actual['storage_publications'][-1]['kept_updates'] == [2, 3]


@pytest.mark.parametrize('mutation', ['backbone','rng','gradient','inputs','rank_accounting','recipe','startup',
    'runtime','precision','unknown_payload','old_source','unknown_source','unknown_policy','keep','evaluation',
    'declaration_seed','same_schema','local_receipt','cloud_hash','cloud_generation','cloud_prefix',
    'cloud_verification','publication_order','prune_latest','prune_before_replacement','failed_prune',
    'unknown_destination','wrong_identity','wrong_cursor','local_status','missing_latest','journal_hash'])
def test_rejects_scientific_or_storage_authority_drift(mutation):
    reference, actual = pair(); evidence = storage(actual)
    payload = actual['configuration']['execution_identity']['payload']
    journal = evidence['journal']
    if mutation=='backbone': actual['updates']['3'][0]['boundary']['state']['model']['weight']['sha256']='0'*64
    elif mutation=='rng': actual['final_boundary_by_rank'][0]['rng']['python']=[999]
    elif mutation=='gradient': actual['updates']['3'][0]['raw_gradients']['weight']['sha256']='0'*64
    elif mutation=='inputs': actual['updates']['3'][0]['input']['batches']['sha256']='0'*64
    elif mutation=='rank_accounting': actual['observations']['3']['rank_data'][0]['physical_rows']+=1
    elif mutation=='recipe': payload['recipe']['fbt_passes']=3
    elif mutation=='startup': payload['startup']['kind']='other'
    elif mutation=='runtime': payload['runtime']['torch']='different'
    elif mutation=='precision': payload['execution']['precision']='different'
    elif mutation=='unknown_payload': payload['unapproved']=True
    elif mutation=='old_source': actual['sources']['scripts/example.py']='0'*64
    elif mutation=='unknown_source': actual['sources']['scripts/unapproved.py']='0'*64
    elif mutation=='unknown_policy': payload['storage_policy']['hidden_fallback']=True
    elif mutation=='keep': payload['storage_policy']['keep_local_completed']=1
    elif mutation=='evaluation': actual['evaluations'][0]['result']['passes'][0]['means']['ce']+=1
    elif mutation=='declaration_seed': payload['declaration']['seed']+=1; payload['resolved_contract_sha256']=audit.digest(payload['declaration'])
    elif mutation=='same_schema': actual['schema']=reference['schema']
    elif mutation=='local_receipt': actual['local_checkpoints'][0]['receipt']['state']['sha256']='0'*64
    elif mutation.startswith('cloud_'):
        obj=evidence['receipts']['checkpoint-publications/update-000000.json']['retention']['objects'][0]
        if mutation=='cloud_hash': obj['sha256']='0'*64
        elif mutation=='cloud_generation': obj['generation']='latest'
        elif mutation=='cloud_prefix': obj['uri']=obj['uri'].replace('/ssd/','/other/')
        else: obj['verification']['download_sha256']=False
    elif mutation=='publication_order': journal['published'].reverse()
    elif mutation=='prune_latest': journal['prune_operations'][0]['update']=3
    elif mutation=='prune_before_replacement': journal['prune_operations'][0]['replaced_by_update']=1
    elif mutation=='failed_prune': journal['prune_operations'][0]['status']='failed'
    elif mutation=='unknown_destination': journal['destinations']['99']='/mnt/localssd/foreign'
    elif mutation=='wrong_identity': journal['ownership']['execution_identity_sha256']='0'*64
    elif mutation=='wrong_cursor': actual['published_checkpoints'][0]['rank_cursors'][0]['cursor']['next_update']=999
    elif mutation=='local_status': journal['published'][-1]['local_status']='pruned'
    elif mutation=='missing_latest': evidence['latest']=None
    else: evidence['journal_sha256']='0'*64
    reidentify(actual)
    result=audit.compare(reference,actual,kind='storage_transition',storage_evidence=evidence)
    assert not result['passed'], mutation


def test_same_identity_resume_matches_update_three_and_repeated_eval():
    _, reference = pair(); storage(reference)
    actual = resumed(reference)
    publication = deepcopy(reference['published_checkpoints'][2])
    actual['resume']['manifest_sha256'] = publication['manifest_sha256']
    actual['storage']['checkpoint_root']='/mnt/localssd/cdrm-checkpoints/example/resume'
    actual['storage']['evidence_dir']='/workspace/cdrm-w-latent/.runtime/ssd-resume'
    evidence = storage(actual)
    result = audit.compare(reference,actual,kind='resume',storage_evidence=evidence,resume_publication=publication)
    assert result['passed'],result['failures']
    assert not evidence['journal']['prune_operations']


def test_resume_rejects_storage_identity_migration():
    _, reference = pair(); storage(reference); actual=resumed(reference)
    actual['configuration']['execution_identity']['payload']['declaration']['storage_prefix']+='/other'
    reidentify(actual)
    assert not audit.compare(reference,actual,kind='resume')['passed']


def test_persistent_reader_verifies_receipt_bytes_and_never_needs_old_ssd_paths(tmp_path):
    reference, actual=pair(); evidence=storage(actual)
    (tmp_path/'ssd-journal.json').write_bytes(encoded(evidence['journal']))
    for relative, receipt in evidence['receipts'].items():
        path=tmp_path/relative;path.parent.mkdir(exist_ok=True);path.write_bytes(encoded(receipt))
    (tmp_path/'latest-checkpoint.json').write_bytes(encoded(evidence['latest']))
    loaded,pins=audit.load_storage_evidence(tmp_path)
    assert loaded==evidence and len(pins)==6
    assert audit.compare(reference,actual,kind='storage_transition',storage_evidence=loaded)['passed']
    (tmp_path/'checkpoint-publications/update-000000.json').write_text('{}')
    with pytest.raises(ValueError,match='bytes differ'):audit.load_storage_evidence(tmp_path)


def test_persistent_reader_rejects_receipt_path_traversal(tmp_path):
    _,actual=pair(); evidence=storage(actual)
    evidence['journal']['published'][0]['receipt_path']='../outside.json'
    (tmp_path/'ssd-journal.json').write_bytes(encoded(evidence['journal']))
    with pytest.raises(ValueError,match='Receipt path'):audit.load_storage_evidence(tmp_path)


def test_no_training_or_cloud_imports():
    command='import sys; import scripts.olmo_campaign_ssd_audit; assert "torch" not in sys.modules; assert "google.cloud.storage" not in sys.modules'
    subprocess.run([sys.executable,'-c',command],cwd=audit.ROOT,check=True,capture_output=True,text=True)

def test_terminal_resume_requires_no_graph_update_or_new_publication():
    _,reference=pair();storage(reference);actual=resumed(reference)
    publication=deepcopy(reference['published_checkpoints'][2])
    actual['resume']['manifest_sha256']=publication['manifest_sha256']
    actual['final_boundary_by_rank']=deepcopy(actual['origin_boundary_by_rank'])
    actual['final_clocks']=deepcopy(actual['origin_clocks'])
    actual['final_counters']=deepcopy(actual['origin_boundary_by_rank'][0]['state']['counters'])
    actual['loop']['completed_update']=2;actual['plan_completed']=False;actual['status']='stopped_at_boundary'
    actual['graph_prepared']=False;actual['runner_by_rank']=[None,None]
    for key in ('updates','observations','preparation_boundary_exact'):actual.pop(key,None)
    actual['storage']['checkpoint_root']='/mnt/localssd/cdrm-checkpoints/example/terminal'
    actual['storage']['evidence_dir']='/workspace/cdrm-w-latent/.runtime/ssd-terminal'
    evidence=storage(actual)
    result=audit.compare(reference,actual,kind='resume',storage_evidence=evidence,resume_publication=publication)
    assert result['passed'],result['failures']
    assert evidence['latest'] is None and not evidence['receipts']


def test_cli_rechecks_journal_authority_before_output_publication(tmp_path,monkeypatch):
    reference,actual=pair();evidence=storage(actual)
    root=tmp_path/'records';root.mkdir()
    (root/'ssd-journal.json').write_bytes(encoded(evidence['journal']))
    for relative,receipt in evidence['receipts'].items():
        path=root/relative;path.parent.mkdir(exist_ok=True);path.write_bytes(encoded(receipt))
    (root/'latest-checkpoint.json').write_bytes(encoded(evidence['latest']))
    paths={}
    for name,value in [('reference',reference),('actual',actual)]:
        path=tmp_path/(name+'.json');path.write_bytes(encoded(value));paths[name]=path
    original=audit.compare
    def mutate(*args,**kwargs):
        result=original(*args,**kwargs);(root/'ssd-journal.json').write_text('{}');return result
    monkeypatch.setattr(audit,'compare',mutate)
    args=['--kind','storage_transition','--storage-evidence-root',str(root),'--output-dir',str(tmp_path/'out')]
    for name,path in paths.items():args+=['--'+name,str(path),'--'+name+'-sha256',audit.file_sha(path)]
    with pytest.raises(ValueError,match='changed during verification'):audit.main(args)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('mutation',['missing','manifest','configuration','cursor'])
def test_resume_binds_exact_published_source_not_merely_equal_boundary(mutation):
    _,reference=pair();storage(reference);actual=resumed(reference)
    publication=deepcopy(reference['published_checkpoints'][2])
    actual['resume']['manifest_sha256']=publication['manifest_sha256']
    if mutation=='missing':publication=None
    elif mutation=='manifest':actual['resume']['manifest_sha256']='0'*64
    elif mutation=='configuration':publication['metadata']['configuration']['model']={'different':True}
    else:publication['rank_cursors'][0]['cursor']['next_update']=1
    result=audit.compare(reference,actual,kind='resume',resume_publication=publication)
    assert not result['passed']


def test_native_report_is_explicitly_outside_tiny_acceptance_auditor_scope():
    reference,actual=pair();actual['scale']='native'
    result=audit.compare(reference,actual,kind='storage_transition')
    assert not result['passed'] and 'tiny acceptance only' in result['failures'][0]
