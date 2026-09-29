"""CPU/fake-cloud byte authority and atomic marker tests; no live transfers."""
import base64
from copy import deepcopy
import hashlib
import io
import os
import subprocess
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cdrm.pretrained import document_shards as shards
from scripts import olmo_pilot_index_retain as module
from test_pilot_ordered_data import fixture as ordered_fixture

PREFIX='gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/pilot-test/ordered-index'


def digest(data):
    return {'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
        'md5_base64':base64.b64encode(hashlib.md5(data).digest()).decode()}


class Blob:
    def __init__(self,bucket,key):
        self.bucket,self.key=bucket,key;self.metadata={};self.content_encoding=None;self.generation='0'
    def upload_from_filename(self,path,*,if_generation_match,checksum):
        assert if_generation_match==0 and checksum=='md5'
        self.payload=Path(path).read_bytes();self.size=len(self.payload)
        self.md5_hash=digest(self.payload)['md5_base64'];self.generation=str(100+len(self.bucket.objects))
        self.bucket.objects[self.key]=self;self.bucket.writes.append(self.key)
    def reload(self,**kwargs):pass
    def download_to_file(self,sink,*,if_generation_match,raw_download,checksum,**kwargs):
        assert str(if_generation_match)==self.generation and raw_download is True
        self.bucket.reads.append((self.key,self.generation))
        if self.bucket.fail_key==self.key:raise OSError('Interrupted cloud readback')
        value=self.payload if self.bucket.corrupt_key!=self.key else b'x'*len(self.payload)
        for offset in range(0,len(value),113):sink.write(value[offset:offset+113])


class Bucket:
    name='fast-chunks'
    def __init__(self):self.objects={};self.writes=[];self.reads=[];self.fail_key=None;self.corrupt_key=None
    def get_blob(self,key):return self.objects.get(key)
    def blob(self,key,**kwargs):return self.objects.get(key) or Blob(self,key)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    class Tokenizer:
        def encode(self,text,*,add_special_tokens):return SimpleNamespace(ids=list(map(int,text.split())))
    monkeypatch.setattr(shards,'_load_tokenizer',lambda path:Tokenizer())
    f=ordered_fixture(tmp_path);f.bucket=Bucket();f.receipt=tmp_path/'receipt.json'
    f.pin=module.file_digest(f.output/'manifest.json')['sha256']
    return f


def retain(f):return module.retain(f.output,f.pin,PREFIX,f.receipt,bucket=f.bucket)


def fetcher(f,*,bad=None):
    def fetch(row,sink):
        key=row['uri'][5:].split('/',1)[1];blob=f.bucket.objects[key]
        assert row['generation']==blob.generation
        assert not (f.output.parent/'restored/manifest.json').exists()
        if bad is not None:return bad(row,sink,blob)
        return blob.download_to_file(sink,if_generation_match=int(row['generation']),raw_download=True,checksum=None)
    return fetch


def restore(f,fetch=None):
    return module.restore(f.receipt,module.file_digest(f.receipt)['sha256'],f.output.parent/'restored',
                          f.output.parent/'restore-evidence',fetch=fetch or fetcher(f))


def test_actual_suite_roundtrip_manifest_last_and_exact_generation(fixture):
    f=fixture;r=retain(f)
    assert len(r['objects'])==44 and r['publication_marker']==PREFIX+'/manifest.json'
    assert [x['path'] for x in r['objects']]==module.ordered_names()
    assert all(key.endswith('.sqlite') for key in f.bucket.writes[:22])
    assert f.bucket.writes[-1].endswith('/ordered-index/manifest.json')
    result=restore(f)
    assert result['objects']==44 and result['status']=='verified'
    for path in module.FILES:assert (f.output/path).read_bytes()==(f.output.parent/'restored'/path).read_bytes()
    assert not list((f.output.parent/'restored').rglob('*.partial'))
    writes=list(f.bucket.writes);assert retain(f)==r and f.bucket.writes==writes


@pytest.mark.parametrize('change',['extra_file','extra_dir','missing','symlink','hardlink','catalog','panel','suite','root_pin'])
def test_malformed_suite_rejected_before_cloud(fixture,change):
    f=fixture
    if change=='extra_file':(f.output/'extra.sqlite').write_bytes(b'bad')
    elif change=='extra_dir':(f.output/'unused').mkdir()
    elif change=='missing':(f.output/'panels/train/documents.sqlite').unlink()
    elif change=='symlink':
        p=f.output/'catalog.sqlite';p.rename(f.output.parent/'other.sqlite');p.symlink_to(f.output.parent/'other.sqlite')
    elif change=='hardlink':os.link(f.output/'catalog.sqlite',f.output.parent/'linked.sqlite')
    elif change=='catalog':
        p=f.output/'catalog.sqlite';p.write_bytes(p.read_bytes()+b'x')
    elif change=='panel':
        p=f.output/'panels/train/manifest.json';value=json.loads(p.read_bytes());value['total_tokens']+=1;p.write_text(json.dumps(value))
    elif change=='suite':
        p=f.output/'manifest.json';value=json.loads(p.read_bytes());value['round_id']=1;p.write_text(json.dumps(value));f.pin=module.file_digest(p)['sha256']
    else:f.pin='a'*64
    with pytest.raises(ValueError):retain(f)
    assert not f.bucket.writes and not f.receipt.exists()


def test_interrupted_retention_retries_exact_existing_objects_without_commit(fixture):
    f=fixture;key=PREFIX[5:].split('/',1)[1]+'/panels/train/documents.sqlite';f.bucket.fail_key=key
    with pytest.raises(OSError):retain(f)
    assert not f.receipt.exists() and PREFIX[5:].split('/',1)[1]+'/manifest.json' not in f.bucket.objects
    prior=set(f.bucket.objects);f.bucket.fail_key=None;retain(f)
    assert all(f.bucket.writes.count(k)==1 for k in prior)


@pytest.mark.parametrize('change',['payload','encoding','schema','md5','generation','corrupt_readback'])
def test_existing_or_uploaded_cloud_conflict_fails_before_suite_marker(fixture,change):
    f=fixture;key=PREFIX[5:].split('/',1)[1]+'/catalog.sqlite';blob=f.bucket.blob(key)
    expected=module.file_digest(f.output/'catalog.sqlite');blob.metadata={'sha256':expected['sha256'],'artifact_schema':module.SCHEMA}
    blob.upload_from_filename(f.output/'catalog.sqlite',if_generation_match=0,checksum='md5')
    if change=='payload':blob.metadata['sha256']='a'*64
    elif change=='encoding':blob.content_encoding='gzip'
    elif change=='schema':blob.metadata['artifact_schema']='other'
    elif change=='md5':blob.md5_hash='bad'
    elif change=='generation':blob.generation='0'
    else:f.bucket.corrupt_key=key
    with pytest.raises(ValueError):retain(f)
    assert not f.receipt.exists() and PREFIX[5:].split('/',1)[1]+'/manifest.json' not in f.bucket.objects


@pytest.mark.parametrize('change',['escape','traversal','missing','duplicate','generation','size','verification','md5','marker','suite_pin'])
def test_bad_receipt_rejected_before_restore(fixture,change):
    f=fixture;r=retain(f);o=r['objects'][0]
    if change=='escape':o['uri']='gs://other/catalog.sqlite'
    elif change=='traversal':o['path']='../catalog.sqlite'
    elif change=='missing':r['objects'].pop()
    elif change=='duplicate':r['objects'].append(deepcopy(o))
    elif change=='generation':o['generation']='latest'
    elif change=='size':o['size_bytes']=module.MAX_BYTES+1
    elif change=='verification':o['verification']['download_sha256']=False
    elif change=='md5':o['md5_base64']='invalid'
    elif change=='marker':r['publication_marker']+='x'
    else:r['suite_manifest_sha256']='a'*64
    f.receipt.write_text(json.dumps(r))
    with pytest.raises(ValueError):restore(f,lambda *_:pytest.fail('network before validation'))
    assert not (f.output.parent/'restored').exists()


@pytest.mark.parametrize('change',['corrupt','overflow','interrupt','receipt_mutation','payload_mutation','extra_member','extra_directory'])
def test_partial_restore_never_commits_suite(fixture,change):
    f=fixture;retain(f)
    def bad(row,sink,blob):
        if row['path']=='catalog.sqlite':
            if change=='corrupt':sink.write(b'x'*blob.size);return
            if change=='overflow':sink.write(blob.payload+b'x');return
            if change=='interrupt':sink.write(blob.payload[:3]);raise OSError('Interrupted')
            if change=='receipt_mutation':f.receipt.write_text('{}')
        if row['path']=='manifest.json':
            if change=='payload_mutation':(f.output.parent/'restored/catalog.sqlite').write_bytes(b'bad')
            elif change=='extra_member':(f.output.parent/'restored/unexpected.sqlite').write_bytes(b'bad')
            elif change=='extra_directory':(f.output.parent/'restored/unexpected').mkdir()
        sink.write(blob.payload)
    with pytest.raises((ValueError,OSError)):restore(f,fetcher(f,bad=bad))
    assert not (f.output.parent/'restored/manifest.json').exists()
    failure=json.loads((f.output.parent/'restore-evidence/failure.json').read_bytes())
    assert failure['manifest_published'] is False


def test_oversized_index_metadata_and_stream_above_old128mib_limit(fixture):
    f=fixture;r=retain(f);row=next(o for o in r['objects'] if o['path']=='catalog.sqlite')
    old=row['size_bytes'];size=129*1024**2;row['size_bytes']=size;r['total_bytes']+=size-old
    f.receipt.write_text(json.dumps(r))
    parsed,records=module.load_receipt(f.receipt,module.file_digest(f.receipt)['sha256'])
    assert records['catalog.sqlite']['size_bytes']==size
    # Exercise actual streaming hashes above128MiB while allocating one1MiB
    # block; this is not a claim to have uploaded a real129MiB SQLite fixture.
    block=b'a'*1024**2;sha=hashlib.sha256();md5=hashlib.md5()
    for _ in range(129):sha.update(block);md5.update(block)
    expected={'size_bytes':size,'sha256':sha.hexdigest(),'md5_base64':base64.b64encode(md5.digest()).decode()}
    sink=module.DigestSink(expected)
    for _ in range(129):sink.write(block)
    sink.finish()
    assert sink.tell()==size and sink.stream is None
    with pytest.raises(io.UnsupportedOperation):sink.seek(0)


def test_fresh_restore_paths_and_safe_uploader_namespace(fixture):
    f=fixture;retain(f);(f.output.parent/'restored').mkdir()
    with pytest.raises(ValueError):restore(f)
    with pytest.raises(ValueError):module.upload_verified(f.bucket,'other/catalog.sqlite',f.output/'catalog.sqlite',module.file_digest(f.output/'catalog.sqlite'))


def test_concurrent_identical_create_only_publication_is_verified(fixture,monkeypatch):
    from google.api_core.exceptions import PreconditionFailed
    f=fixture;original=Blob.upload_from_filename
    def race(blob,*args,**kwargs):
        original(blob,*args,**kwargs)
        raise PreconditionFailed('Equivalent concurrent publisher won')
    monkeypatch.setattr(Blob,'upload_from_filename',race)
    result=retain(f)
    assert result['status']=='verified' and len(f.bucket.objects)==44
    assert len(f.bucket.reads)==44


def test_direct_script_cli_help_bootstraps_project_imports():
    env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
    result=subprocess.run([sys.executable,str(module.ROOT/'scripts/olmo_pilot_index_retain.py'),'--help'],
        cwd='/',env=env,capture_output=True,text=True,check=True,timeout=20)
    assert '{retain,restore}' in result.stdout
