import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import olmo_checkpoint_readback_probe as probe


def expected(payload=b'checkpoint-bytes'):
    return {'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
            'md5_base64': base64.b64encode(hashlib.md5(payload).digest()).decode(), 'generation': '123'}


@pytest.mark.parametrize('sizes', [(16,), (0,1,2,3,10), (4,4,4,4), (1,)*16])
def test_sink_is_chunk_boundary_independent(sizes):
    payload = b'checkpoint-bytes'
    sink = probe.HashSink(expected(payload))
    offset = 0
    for size in sizes:
        chunk = memoryview(payload)[offset:offset+size]
        assert sink.write(chunk) == len(chunk)
        offset += size
    assert sink.finish() == {k: expected(payload)[k] for k in ('size_bytes','sha256','md5_base64')}
    assert sink.tell() == len(payload)
    assert sink.write_calls == len(sizes)


@pytest.mark.parametrize('mutation', ['truncated', 'wrong_sha', 'wrong_md5', 'overflow', 'seek'])
def test_sink_failures_do_not_claim_success(mutation):
    payload = b'checkpoint-bytes'
    pin = expected(payload)
    if mutation == 'wrong_sha': pin['sha256'] = '0'*64
    if mutation == 'wrong_md5': pin['md5_base64'] = 'not-correct'
    sink = probe.HashSink(pin)
    if mutation == 'overflow':
        with pytest.raises(ValueError, match='overflow'): sink.write(payload+b'!')
        assert sink.tell() == 0
    elif mutation == 'seek':
        sink.write(payload[:4]); before = (sink.tell(), sink.sha256.hexdigest(), sink.md5.hexdigest())
        with pytest.raises(ValueError, match='seek'): sink.seek(0)
        assert before == (sink.tell(), sink.sha256.hexdigest(), sink.md5.hexdigest())
    else:
        sink.write(payload[:-1] if mutation == 'truncated' else payload)
        with pytest.raises(ValueError, match='digest'): sink.finish()


def remote(pin):
    return SimpleNamespace(generation=int(pin['generation']), size=pin['size_bytes'],
        md5_hash=pin['md5_base64'], metadata={'sha256': pin['sha256']}, content_encoding=None, chunk_size=None)


@pytest.mark.parametrize('field,value', [('generation',124), ('size',99), ('md5_hash','bad'),
                                      ('metadata',{}), ('content_encoding','gzip')])
def test_remote_requires_exact_metadata_and_no_encoding(field,value):
    blob = remote(expected())
    assert probe.validate_remote(blob,expected())['content_encoding'] is None
    setattr(blob,field,value)
    with pytest.raises(ValueError, match='metadata or encoding'): probe.validate_remote(blob,expected())


def authority_report():
    manifest_sha='a'*64
    receipt={'manifest_sha256':manifest_sha, 'counters':{'optimizer_updates':1},
        'state':{'sha256':probe.STATE['sha256'],'size_bytes':probe.STATE['size_bytes']},
        'retention':{'create_only':True,'download_sha256_verified':True,'objects':[
            {**probe.STATE,'verification':{'download_sha256':True}},
            {'uri':probe.STATE['uri'].rsplit('/',1)[0]+'/manifest.json','sha256':manifest_sha,
             'verification':{'download_sha256':True}}]}}
    return {'phase':'reference','status':'passed','local_checkpoints':[{'optimizer_update':1,'receipt':receipt}],
            'published_checkpoints':[receipt]}


@pytest.mark.parametrize('mutation', ['none','report_sha','state_pin','manifest_pin','unverified','duplicate'])
def test_authority_uses_completed_receipt_chain(tmp_path,mutation):
    report=authority_report()
    if mutation=='state_pin': report['published_checkpoints'][0]['retention']['objects'][0]['generation']='999'
    if mutation=='manifest_pin': report['published_checkpoints'][0]['retention']['objects'][1]['sha256']='0'*64
    if mutation=='unverified': report['published_checkpoints'][0]['retention']['objects'][0]['verification']['download_sha256']=False
    if mutation=='duplicate': report['local_checkpoints']*=2
    path=tmp_path/'reference.json';path.write_text(json.dumps(report))
    digest='0'*64 if mutation=='report_sha' else probe.sha(path)
    if mutation=='none': assert probe.endpoint_authority(path,digest)['state']==probe.STATE
    else:
        with pytest.raises(ValueError): probe.endpoint_authority(path,digest)


def test_measurements_share_bytes_and_generation_but_stream_never_returns_payload(monkeypatch):
    payload=b'checkpoint-bytes'
    seen=[]
    class Blob:
        chunk_size=None
        def download_as_bytes(self,**kwargs): seen.append(('bytes',kwargs));return payload
        def download_to_file(self,sink,**kwargs):
            seen.append(('stream',kwargs))
            for i in range(0,len(payload),3):sink.write(payload[i:i+3])
    counter=iter(range(100))
    monkeypatch.setattr(probe,'current_rss_kib',lambda:123)
    results=[probe.measure(method,Blob(),expected(payload),clock=lambda:next(counter)) for method in ('bytes','stream')]
    assert results[0]['verified']==results[1]['verified']
    assert results[0]['max_write_bytes']==len(payload) and results[1]['max_write_bytes']==3
    assert results[0]['write_calls']==1 and results[1]['write_calls']==6
    assert seen[0][1]==seen[1][1]
    assert seen[0][1]['if_generation_match']==123 and not seen[0][1]['single_shot_download']
    assert all(r['total_seconds']==2 for r in results)


def test_installed_sdk_retry_appends_after_accepted_bytes_without_seek(monkeypatch):
    """Actual SDK consume logic with one interrupted small response, no network."""
    from google.cloud.storage._media.requests import download as sdk
    from google.cloud.storage._media.requests import _request_helpers
    from requests.exceptions import ConnectionError
    payload=b'checkpoint-bytes';sink=probe.HashSink(expected(payload))
    calls=[]
    class Response:
        def __init__(self,status,parts,fail=False):
            self.status_code=status;self.parts=parts;self.fail=fail
            self.headers={'content-length':str(sum(map(len,parts))), 'x-goog-generation':'123',
                          'x-goog-hash':'md5='+expected(payload)['md5_base64']}
            if status==206:self.headers['content-range']='bytes 5-15/16'
            self.raw=SimpleNamespace(_decoder=None,headers={})
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
    download=sdk.Download('https://unused.invalid/object?generation=123',stream=sink,checksum='md5')
    download.consume(Transport())
    assert sink.finish()['sha256']==expected(payload)['sha256']
    assert len(calls)==2 and calls[1]['headers']['range']=='bytes=5-'
    assert sink.write_calls==2 and download._bytes_downloaded==len(payload)


def test_sdk_response_transcoding_seek_is_rejected(monkeypatch):
    from google.cloud.storage._media.requests import download as sdk
    from google.cloud.storage._media.requests import _request_helpers
    from google.cloud.storage._media import _helpers
    sink=probe.HashSink(expected())
    monkeypatch.setattr(_request_helpers,'wait_and_retry',lambda action,_:action())
    monkeypatch.setattr(_helpers,'_is_decompressive_transcoding',lambda *_:True)
    response=SimpleNamespace(status_code=200,headers={'content-length':'16'})
    transport=SimpleNamespace(request=lambda *_args,**_kwargs:response)
    with pytest.raises(Exception,match='Error writing to stream') as failure:
        sdk.Download('https://unused.invalid/object?generation=123',stream=sink).consume(transport)
    assert isinstance(failure.value.__cause__,ValueError)
    assert 'seek' in str(failure.value.__cause__)
    assert sink.tell()==0


@pytest.mark.parametrize('mutation',['none','pid','previous_pid','sdk','torch','gpu'])
def test_child_contract_excludes_pid_from_platform_match_but_requires_fresh_process(mutation):
    parent={'python':'py','platform':'linux','google_cloud_storage':'3.12.0','sdk_source_sha256':{'a':'hash'},
            'rss_units':'Linux KiB','pid':1,'torch_loaded':False,'cuda_device_nodes_present':False}
    child={**parent,'pid':3}
    if mutation=='pid':child['pid']=1
    if mutation=='previous_pid':child['pid']=2
    if mutation=='sdk':child['sdk_source_sha256']={'a':'changed'}
    if mutation=='torch':child['torch_loaded']=True
    if mutation=='gpu':child['cuda_device_nodes_present']=True
    if mutation=='none':probe.validate_child_environment(child,parent,[2])
    else:
        with pytest.raises(ValueError):probe.validate_child_environment(child,parent,[2])
