"""Generation-pinned data recovery, including corruption and interruption."""
import base64
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import pytest
from scripts import olmo_pilot_data_restore as module


def pin(data):
    return {'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'md5_base64':base64.b64encode(hashlib.md5(data).digest()).decode()}


@pytest.fixture
def fixture(tmp_path):
    payload={'tokens.bin':b'\1\0\2\0\3\0','documents.jsonl':b'{"source_line":1}\n','manifest.json':b'{"schema":"olmo-fixture"}\n'}
    prefix='gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/pilot-20260929-v1/shard-000000'
    objects=[{'uri':prefix+'/'+n,'generation':str(10+i),**pin(v),
        'verification':{k:True for k in ('server_size','server_md5','sha256_metadata','download_sha256')}} for i,(n,v) in enumerate(payload.items())]
    receipt={'schema':'olmo-document-retention-v1','status':'verified','prefix':prefix,'objects':objects,
        'publication_marker':prefix+'/manifest.json','local_files_deleted':False}
    p=tmp_path/'receipt.json';p.write_text(json.dumps(receipt))
    calls=[]
    def fetch(record,sink):
        n=record['uri'].split('/')[-1];calls.append((n,record['generation']))
        assert not (tmp_path/'data/manifest.json').exists()
        data=payload[n]
        for i in range(0,len(data),3):sink.write(data[i:i+3])
    return SimpleNamespace(root=tmp_path,path=p,receipt=receipt,payload=payload,fetch=fetch,calls=calls)


def run(f,fetch=None):
    return module.restore(f.path,module.sha(f.path),f.root/'data',f.root/'evidence',fetch=fetch or f.fetch)


def test_exact_bytes_generation_and_manifest_last(fixture):
    f=fixture;r=run(f)
    assert r['status']=='verified' and f.calls[-1][0]=='manifest.json'
    assert r['bytes']==sum(len(x) for x in f.payload.values())
    assert (f.root/'evidence/receipt.json').read_bytes()==f.path.read_bytes()
    for name,data in f.payload.items():assert (f.root/'data'/name).read_bytes()==data
    assert not list((f.root/'data').glob('*.partial'))


@pytest.mark.parametrize('change',['generation','sha','verification','size','prefix','traversal','duplicate','marker','md5'])
def test_bad_authorities_rejected_before_network_or_files(fixture,change):
    f=fixture;r=f.receipt;o=r['objects'][0]
    if change=='generation':o['generation']='latest'
    elif change=='sha':o['sha256']='BAD'
    elif change=='verification':o['verification']['server_md5']=False
    elif change=='size':o['size_bytes']=module.MAX_BYTES+1
    elif change=='prefix':r['prefix']='gs://other/private'
    elif change=='traversal':o['uri']=r['prefix']+'/../tokens.bin'
    elif change=='duplicate':r['objects'].append(deepcopy(o))
    elif change=='marker':r['publication_marker']+='-wrong'
    else:o['md5_base64']='invalid'
    f.path.write_text(json.dumps(r))
    with pytest.raises(ValueError):run(f)
    assert not f.calls and not (f.root/'data').exists()


@pytest.mark.parametrize('mode',['corrupt','overflow','interrupt','receipt_mutation'])
def test_incomplete_restore_is_never_committed(fixture,mode):
    f=fixture
    def fetch(record,sink):
        name=record['uri'].split('/')[-1];data=f.payload[name]
        if name=='tokens.bin':
            if mode=='corrupt':data=b'x'*len(data)
            elif mode=='overflow':data+=b'x'
            elif mode=='interrupt':sink.write(data[:1]);raise OSError('interrupted')
            else:f.path.write_text('{}')
        sink.write(data)
    with pytest.raises((ValueError,OSError)):run(f,fetch)
    assert not (f.root/'data/manifest.json').exists()
    assert (f.root/'evidence/intent.json').exists()
    assert json.loads((f.root/'evidence/failure.json').read_text())['manifest_published'] is False


def test_existing_destination_and_symlink_never_replaced(fixture):
    f=fixture;(f.root/'data').mkdir()
    with pytest.raises(ValueError):run(f)
    (f.root/'data').rmdir();(f.root/'elsewhere').mkdir();(f.root/'data').symlink_to(f.root/'elsewhere',target_is_directory=True)
    with pytest.raises(ValueError):run(f)
    assert not f.calls


def test_sink_rejects_wrong_md5_and_oversized_writes(tmp_path):
    p=tmp_path/'part';record=pin(b'abc');record['md5_base64']=pin(b'xyz')['md5_base64']
    with p.open('wb') as f:
        s=module.Sink(f,record);s.write(b'abc')
        with pytest.raises(ValueError):s.finish()
        with pytest.raises(ValueError):s.write(b'd')
