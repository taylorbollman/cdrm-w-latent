"""CPU/file-only recovery tests, no checkpoint deserialization or cloud access."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from scripts import olmo_campaign_execution_restore as restore


def encoded(value):
    return (json.dumps(value, sort_keys=True)+'\n').encode()


def pins(payload):
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload),
            'md5_base64': base64.b64encode(hashlib.md5(payload).digest()).decode('ascii')}


@pytest.fixture
def publication(tmp_path):
    state = b'opaque serialized checkpoint bytes, deliberately never deserialized'
    payload = {'scope':'tiny-acceptance-not-native', 'arm':'NFR',
        'partition':{'world_size':2,'physical_batch_per_rank':2},
        'cursor_schema':'olmo-campaign-execution-cursor-v1'}
    identity = {'schema':restore.IDENTITY_SCHEMA, 'payload':payload, 'sha256':restore.digest(payload)}
    cursor = {'schema':'olmo-campaign-execution-cursor-v1','world_size':2,'physical_batch_per_rank':2,
        'cursor':{'manifest_sha256':'a'*64,'split':'train','next_chunk':5,'next_update':1}}
    manifest = {'schema':restore.CHECKPOINT_SCHEMA, 'world_size':2,
        'metadata':{'schema':restore.CHECKPOINT_SCHEMA, 'world_size':2,'model_type':'actual.runner.Model',
            'configuration':{'execution_identity':identity},
            'source_fingerprint':{'execution_identity_sha256':identity['sha256']}},
        'counters':{'optimizer_updates':1,'input_tokens':80,'documents':5,'microbatches':4,
            'ce_positions':75,'latent_pairs':70,'kl_triples':65},
        'rank_cursors':[{**copy.deepcopy(cursor),'rank':rank} for rank in range(2)],
        'state':{'filename':'state.pt','sha256':pins(state)['sha256'],'size_bytes':len(state)}}
    manifest_bytes=encoded(manifest)
    prefix=restore.REMOTE_ROOT+'fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/test/update-000001/'
    objects=[{'uri':prefix+name, 'generation':generation, **pins(data),
              'verification':{key:True for key in restore.VERIFICATION}}
             for name,generation,data in [('state.pt','123',state),('manifest.json','456',manifest_bytes)]]
    receipt={**manifest, 'manifest_sha256':pins(manifest_bytes)['sha256'],
        'directory':'/missing/old/container/location',
        'retention':{'objects':objects,'create_only':True,'download_sha256_verified':True}}
    path=tmp_path/'latest-checkpoint.json';path.write_bytes(encoded(receipt))
    storage={(objects[0]['uri'],'123'):state,(objects[1]['uri'],'456'):manifest_bytes}
    calls=[]
    def fetch(record,sink):
        calls.append((record['uri'],record['generation']))
        data=storage[calls[-1]]
        for offset in range(0,len(data),7):sink.write(data[offset:offset+7])
        return {'generation':record['generation']}
    return SimpleNamespace(path=path,receipt=receipt,storage=storage,fetch=fetch,calls=calls,
                           state=state,manifest=manifest,root=tmp_path)


def run(fixture, name='restore', fetch=None):
    return restore.restore(fixture.path,restore.sha(fixture.path),fixture.root/name,
                           fetch=fetch or fixture.fetch,persistent_root=fixture.root)


def test_restores_distinct_generations_without_success_report_or_old_paths(publication):
    result=run(publication)
    checkpoint=Path(result['checkpoint_directory'])
    assert checkpoint.joinpath('state.pt').read_bytes()==publication.state
    assert json.loads(checkpoint.joinpath('manifest.json').read_text())==publication.manifest
    assert result['status']=='checkpoint_assets_verified_launch_pending'
    assert [generation for _,generation in publication.calls]==['456','123']
    assert result['model_tensors_loaded'] is result['gpu_used'] is result['original_receipt_local_paths_used'] is False
    assert result['cloud_writes'] is False
    assert publication.path.read_bytes()==(checkpoint.parent/'authorities/latest-checkpoint.json').read_bytes()
    assert not list(checkpoint.glob('*.partial'))
    for name,digest in result['sources'].items():assert restore.sha(checkpoint.parent/'source-snapshot'/name)==digest


def test_completed_boundary_is_a_valid_asset_restore(publication):
    receipt=publication.receipt
    receipt['counters']['optimizer_updates']=3
    for row in receipt['rank_cursors']:row['cursor']['next_update']=3
    manifest={k:receipt[k] for k in restore.MANIFEST_FIELDS};raw=encoded(manifest)
    obj=receipt['retention']['objects'][1];obj.update(pins(raw))
    publication.storage[(obj['uri'],obj['generation'])]=raw
    receipt['manifest_sha256']=obj['sha256'];publication.path.write_bytes(encoded(receipt))
    assert run(publication)['completed_optimizer_updates']==3


@pytest.mark.parametrize('mutation',['unretained','extra_object','duplicate','generation','wrong_bucket','traversal',
    'encoded_path','query','other_directory','manifest_pin','state_pin','identity','fingerprint','cursor','oversize',
    'float_size','oversize_manifest','zero_size','md5','verification','unknown_schema'])
def test_invalid_receipts_rejected_before_fetch_or_destination(publication,mutation):
    row=publication.receipt;obj=row['retention']['objects'][0]
    if mutation=='unretained':row['retention']['create_only']=False
    elif mutation=='extra_object':row['retention']['objects'].append(copy.deepcopy(obj))
    elif mutation=='duplicate':row['retention']['objects'][1]=copy.deepcopy(obj)
    elif mutation=='generation':obj['generation']='latest'
    elif mutation=='wrong_bucket':obj['uri']=obj['uri'].replace('fast-chunks','other')
    elif mutation=='traversal':obj['uri']=obj['uri'].replace('/state.pt','/../state.pt')
    elif mutation=='encoded_path':obj['uri']=obj['uri'].replace('/state.pt','/%2e%2e/state.pt')
    elif mutation=='query':obj['uri']+='?generation=123'
    elif mutation=='other_directory':obj['uri']=obj['uri'].replace('update-000001','update-000002')
    elif mutation=='manifest_pin':row['manifest_sha256']='b'*64
    elif mutation=='state_pin':row['state']['sha256']='b'*64
    elif mutation=='identity':row['metadata']['configuration']['execution_identity']['payload']['arm']='B'
    elif mutation=='fingerprint':row['metadata']['source_fingerprint']['execution_identity_sha256']='b'*64
    elif mutation=='cursor':row['rank_cursors'][1]['cursor']['next_update']=2
    elif mutation=='oversize':row['state']['size_bytes']=obj['size_bytes']=restore.MAX_STATE_BYTES+1
    elif mutation=='float_size':obj['size_bytes']=float(obj['size_bytes'])
    elif mutation=='oversize_manifest':row['retention']['objects'][1]['size_bytes']=restore.MAX_MANIFEST_BYTES+1
    elif mutation=='zero_size':row['state']['size_bytes']=obj['size_bytes']=0
    elif mutation=='md5':obj['md5_base64']='not-md5'
    elif mutation=='verification':obj['verification']['download_sha256']=False
    else:row['schema']='old-checkpoint'
    publication.path.write_bytes(encoded(row))
    with pytest.raises(ValueError):run(publication)
    assert not publication.calls and not (publication.root/'restore').exists()


@pytest.mark.parametrize('failure',['manifest_corrupt','manifest_semantic','state_corrupt','interrupted','overflow'])
def test_failed_download_keeps_partial_and_never_commits_manifest(publication,failure):
    if failure=='manifest_semantic':
        changed=copy.deepcopy(publication.manifest);changed['counters']['input_tokens']+=1
        obj=publication.receipt['retention']['objects'][1];raw=encoded(changed);obj.update(pins(raw))
        publication.receipt['manifest_sha256']=obj['sha256'];publication.path.write_bytes(encoded(publication.receipt))
        publication.storage[(obj['uri'],obj['generation'])]=raw
    def fetch(record,sink):
        name=record['uri'].rsplit('/',1)[-1]
        raw=publication.storage[(record['uri'],record['generation'])]
        assert not (publication.root/'restore/checkpoint/manifest.json').exists()
        if (failure=='manifest_corrupt' and name=='manifest.json') or (failure=='state_corrupt' and name=='state.pt'):
            sink.write(b'x'*len(raw));return
        if name=='state.pt' and failure=='interrupted':
            sink.write(raw[:5]);raise OSError('synthetic interrupted transfer')
        if name=='state.pt' and failure=='overflow':sink.write(raw+b'x');return
        sink.write(raw)
    with pytest.raises((ValueError,OSError)):run(publication,fetch=fetch)
    checkpoint=publication.root/'restore/checkpoint'
    assert not (checkpoint/'manifest.json').exists() and not (checkpoint/'state.pt').exists()
    assert list(checkpoint.glob('*.partial'))
    assert json.loads((checkpoint.parent/'failure.json').read_text())['checkpoint_manifest_published'] is False


def test_input_mutation_during_transfer_prevents_manifest_commit(publication):
    def fetch(record,sink):
        publication.fetch(record,sink)
        if record['uri'].endswith('/state.pt'):publication.path.write_text('{}')
    with pytest.raises(ValueError,match='authority changed'):run(publication,fetch=fetch)
    assert not (publication.root/'restore/checkpoint/manifest.json').exists()


def test_manifest_is_last_and_interrupted_publication_is_not_committed(publication,monkeypatch):
    original=restore.durable_publish;order=[]
    def publish(partial,destination):
        order.append(destination.name)
        if destination.name=='manifest.json':
            assert destination.with_name('state.pt').read_bytes()==publication.state
            raise OSError('synthetic manifest publication interruption')
        return original(partial,destination)
    monkeypatch.setattr(restore,'durable_publish',publish)
    with pytest.raises(OSError):run(publication)
    assert order==['state.pt','manifest.json']
    assert (publication.root/'restore/checkpoint/state.pt').is_file()
    assert not (publication.root/'restore/checkpoint/manifest.json').exists()


def test_destination_pin_and_disk_guards(publication,monkeypatch):
    with pytest.raises(ValueError):restore.restore(publication.path,'b'*64,publication.root/'bad',fetch=publication.fetch,persistent_root=publication.root)
    existing=publication.root/'existing';existing.mkdir()
    with pytest.raises(ValueError):run(publication,'existing')
    linked=publication.root/'linked';linked.symlink_to(existing,target_is_directory=True)
    with pytest.raises(ValueError):run(publication,'linked/new')
    with pytest.raises(ValueError):restore.restore(publication.path,restore.sha(publication.path),publication.root.parent/'escape',fetch=publication.fetch,persistent_root=publication.root)
    monkeypatch.setattr(restore.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='disk'):run(publication)
    assert not publication.calls


def test_append_only_streaming_disk_sink_and_exact_limit(tmp_path):
    payload=b'abc123'
    with (tmp_path/'stream').open('xb') as stream:
        sink=restore.HashingWriter(stream,pins(payload))
        sink.write(memoryview(payload[:3]));sink.write(payload[3:]);assert sink.tell()==len(payload)
        with pytest.raises(ValueError,match='rewind'):sink.seek(0)
        with pytest.raises(ValueError,match='bound'):sink.write(b'x')
        assert sink.finish()==pins(payload)
    assert (tmp_path/'stream').read_bytes()==payload


@pytest.mark.parametrize('raw_download',[False,True])
def test_installed_sdk_retry_appends_to_disk_without_rewind(tmp_path,monkeypatch,raw_download):
    from google.cloud.storage._media.requests import download as sdk
    from google.cloud.storage._media.requests import _request_helpers
    from requests.exceptions import ConnectionError
    payload=b'checkpoint-bytes';calls=[]
    class Response:
        def __init__(self,status,parts,fail=False):
            self.status_code,self.parts,self.fail=status,parts,fail
            self.headers={'content-length':str(sum(map(len,parts))),'x-goog-generation':'123',
                'x-goog-hash':'md5='+pins(payload)['md5_base64']}
            if status==206:self.headers['content-range']='bytes 5-15/16'
            def raw_stream(_chunk_size, *, decode_content):
                assert decode_content is False
                return self.iter_content()
            self.raw=SimpleNamespace(_decoder=None,headers={},stream=raw_stream)
        def __enter__(self):return self
        def __exit__(self,*_):return False
        def iter_content(self,**_):
            yield from self.parts
            if self.fail:raise ConnectionError('synthetic interruption')
    class Transport:
        def request(self,method,url,**kwargs):
            calls.append({'url':url,'headers':dict(kwargs['headers'])})
            return Response(200,[payload[:5]],True) if len(calls)==1 else Response(206,[payload[5:]])
    def retry(action,_):
        try:return action()
        except ConnectionError:return action()
    monkeypatch.setattr(_request_helpers,'wait_and_retry',retry)
    with (tmp_path/'retry').open('xb') as stream:
        sink=restore.HashingWriter(stream,pins(payload))
        download_type=sdk.RawDownload if raw_download else sdk.Download
        download_type('https://unused.invalid/object?generation=123',stream=sink,checksum='md5').consume(Transport())
        assert sink.finish()==pins(payload)
    assert calls[1]['headers']['range']=='bytes=5-' and all('generation=123' in c['url'] for c in calls)
    assert (tmp_path/'retry').read_bytes()==payload


def test_standalone_import_does_not_import_torch_or_google_client():
    command='import sys; import scripts.olmo_campaign_execution_restore; assert "torch" not in sys.modules; assert "google.cloud.storage" not in sys.modules'
    subprocess.run([sys.executable,'-c',command],cwd=restore.ROOT,check=True,capture_output=True,text=True)


@pytest.mark.parametrize('mismatch',[None,'generation','size','md5','sha','encoding'])
def test_google_fetch_preserves_generation_and_checks_metadata(publication,tmp_path,monkeypatch,mismatch):
    from google.cloud import storage
    record=publication.receipt['retention']['objects'][0]
    calls=[]
    class Blob:
        generation=record['generation'];size=record['size_bytes'];md5_hash=record['md5_base64']
        metadata={'sha256':record['sha256']};content_encoding=None
        def reload(self,**kwargs):
            assert kwargs['if_generation_match']==123;calls.append('reload')
        def download_to_file(self,sink,**kwargs):
            assert kwargs=={'if_generation_match':123,'checksum':'auto','single_shot_download':False,
                            'raw_download':True,'timeout':(15,60)}
            calls.append('download');sink.write(publication.state)
    blob=Blob()
    if mismatch=='generation':blob.generation='124'
    elif mismatch=='size':blob.size+=1
    elif mismatch=='md5':blob.md5_hash='wrong'
    elif mismatch=='sha':blob.metadata={'sha256':'0'*64}
    elif mismatch=='encoding':blob.content_encoding='gzip'
    class Bucket:
        def blob(self,key,*,generation):
            assert generation==123 and key=='cdrm-w-latent/'+record['uri'].split('/cdrm-w-latent/',1)[1]
            return blob
    class Client:
        def bucket(self,name):assert name=='fast-chunks';return Bucket()
    monkeypatch.setattr(storage,'Client',Client)
    with (tmp_path/'metadata-stream').open('xb') as stream:
        sink=restore.HashingWriter(stream,record)
        if mismatch:
            with pytest.raises(ValueError,match='metadata'):restore.google_fetch(record,sink)
            assert calls==['reload'] and sink.tell()==0
        else:
            assert restore.google_fetch(record,sink)['generation']=='123'
            assert sink.finish()==pins(publication.state) and calls==['reload','download']
