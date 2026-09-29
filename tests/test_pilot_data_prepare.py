"""Bounded preparation with real gzip/JSON/shard files and in-memory cloud I/O."""
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import resource
from types import SimpleNamespace

import pytest

from scripts import olmo_pilot_data_prepare as prep
from test_olmo_document_retain import Bucket


class Tokenizer:
    eos_token_id=50279
    def encode(self,text,**kwargs):
        values=[int(word) for word in text.split()]
        return SimpleNamespace(ids=values) if kwargs else values


class Response(io.BytesIO):
    status=200
    def __init__(self,body,url,etag='"pinned"'):
        super().__init__(body);self.url=url;self.headers={'ETag':etag}
    def geturl(self):return self.url


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    sources=[]
    for stratum,count in prep.STRATUM_COUNTS.items():
        for i in range(count):
            url=f'{prep.URL_ROOT}{stratum}/{len(sources)}.json.gz'
            sources.append({'name':f'{stratum}-{i:02d}','url':url,'etag':'\"pinned\"','upstream_size_bytes':1000,
                'stratum':stratum,'family':'common_crawl' if stratum.startswith('cc_') else stratum,
                'selection_sha256':hashlib.sha256(('cdrm-dolma-pilot-v1-url\0'+url).encode()).hexdigest()})
    tokenizer=tmp_path/'tokenizer.json';tokenizer.write_bytes(b'fake tokenizer graph')
    token_sha=prep.extractor.digest(tokenizer)
    monkeypatch.setitem(prep.extractor.FILE_SPECS,'tokenizer.json',(len(b'fake tokenizer graph'),token_sha))
    monkeypatch.setattr(prep.extractor,'OLMoNativeTokenizer',lambda path:Tokenizer())
    monkeypatch.setattr(prep.shards,'_load_tokenizer',lambda path:Tokenizer())
    monkeypatch.setitem(prep.POLICY,'target_tokens_per_source',8)
    monkeypatch.setitem(prep.POLICY,'max_documents_per_shard',10)
    monkeypatch.setitem(prep.POLICY,'target_tokens_per_shard',32)
    monkeypatch.setitem(prep.POLICY,'max_new_shards',2)
    recipe={'release':'v1_5','hf_repo':'allenai/dolma','hf_revision':'7f48140530a023e9ea4c5cfb141160922727d4d3','bounds':{'candidate_tokens':prep.POLICY['max_total_tokens'],
        'compressed_bytes':prep.POLICY['max_total_compressed_bytes'],'retained_raw_bytes':prep.POLICY['max_total_raw_bytes'],
        'max_line_bytes':prep.POLICY['max_line_bytes'],'max_documents_per_object':prep.POLICY['max_documents_per_source']},
        'object_limits':{'tokens':prep.POLICY['max_tokens_per_source'],'compressed_bytes':prep.POLICY['max_compressed_bytes_per_source'],
            'retained_raw_bytes':prep.POLICY['max_raw_bytes_per_source'],'timeout_seconds':prep.POLICY['extraction_timeout_seconds']},
        'selection':{'candidate_tokens_per_object':8,'objects_per_stratum':dict(prep.STRATUM_COUNTS),
            'url_namespace':'cdrm-dolma-pilot-v1-url','policy':'hash_rank_urls_then_complete_record_prefix'},'split':{'seed':20260929,'weights':[['train',90],['dev',5],['confirmation',5]]},
        'tokenizer':{'repo':prep.extractor.REPO_ID,'revision':prep.extractor.REVISION,'sha256':token_sha,
            'eos_id':50279,'pad_id':1,'vocab_size':50280,'token_dtype':'uint16_le'}}
    recipe_sha=hashlib.sha256((json.dumps(recipe,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
    plan={'schema':'olmo-pilot-source-plan-test-v1','recipe':recipe,'recipe_sha256':recipe_sha,'sources':sources}
    path=tmp_path/'plan.json';path.write_bytes(prep.canonical(plan));sha=prep.extractor.digest(path)
    calls=[]
    def opener(request,timeout):
        assert timeout==60
        url=request.full_url;calls.append(url);i=int(url.rsplit('/',1)[1].split('.')[0])
        rows=[{'id':f'{i}-shared','text':'3 4 5'},{'id':f'{i}-own','text':f'{100+i} 6 7'}]
        body=b''.join((json.dumps(row)+'\n').encode() for row in rows)
        return Response(gzip.compress(body,mtime=0),url)
    monkeypatch.setattr(prep.extractor.urllib.request,'urlopen',opener)
    return SimpleNamespace(root=tmp_path,plan=plan,path=path,sha=sha,tokenizer=tokenizer,
        raw=tmp_path/'raw',acquisition=tmp_path/'acquisition',prepared=tmp_path/'tokenized',
        evidence=tmp_path/'preparation',bucket=Bucket(),calls=calls)


def acquire(f,i=0,**kwargs):
    return prep.acquire(f.path,f.sha,f.raw,f.acquisition,f.plan['sources'][i]['name'],f.tokenizer,
        worker=prep.extract_worker,bucket=f.bucket,**kwargs)


def prepare(f):
    return prep.prepare(f.path,f.sha,f.raw,f.prepared,f.evidence,f.acquisition,f.tokenizer,bucket=f.bucket)


def all_raw(f):
    # Avoid rechecking every prior source while building this local test fixture;
    # the production CLI's sequential acquisition path is tested separately.
    sources=prep.source_hashes();prep.bind_evidence(f.acquisition,f.path,f.sha,sources)
    for row in f.plan['sources']:
        prep.extract_worker(f.path,f.sha,f.raw,row['name'],f.tokenizer)
        prep.verified_receipt(f.raw/row['name'],f.acquisition,'raw-'+row['name'],bucket=f.bucket)


def test_actual_extract_retention_is_atomic_pinned_and_idempotent(fixture):
    f=fixture;result=acquire(f)
    assert result['status']=='retained' and result['raw']['totals']['candidate_tokens']==8
    assert result['raw']['complete'] is False
    assert f.bucket.upload_order[-1].endswith('raw-books-00/manifest.json')
    assert len(f.calls)==1
    before=(f.raw/'books-00/raw.jsonl').read_bytes();again=acquire(f)
    assert len(f.calls)==1 and again==result and (f.raw/'books-00/raw.jsonl').read_bytes()==before
    attempt=json.loads((f.acquisition/'attempts/books-00-0001.json').read_text())
    assert attempt['status']=='committed' and attempt['failed_attempt_traffic_counted_in_success_caps'] is False


@pytest.mark.parametrize('change',['pin','missing_source','duplicate_name','duplicate_url','etag','size','url','name'])
def test_invalid_plan_authority_never_reaches_network(fixture,change):
    f=fixture;plan=deepcopy(f.plan);sha=f.sha
    if change=='pin':sha='a'*64
    elif change=='missing_source':plan['sources'].pop()
    elif change=='duplicate_name':plan['sources'][1]['name']=plan['sources'][0]['name']
    elif change=='duplicate_url':plan['sources'][1]['url']=plan['sources'][0]['url']
    elif change=='etag':plan['sources'][0]['etag']=None
    elif change=='size':plan['sources'][0]['upstream_size_bytes']=True
    elif change=='url':plan['sources'][0]['url']='https://foreign.invalid/data'
    else:plan['sources'][0]['name']='../escape'
    if change!='pin':f.path.write_bytes(prep.canonical(plan));sha=prep.extractor.digest(f.path)
    with pytest.raises(ValueError):prep.load_plan(f.path,sha)
    assert not f.calls


def test_source_order_and_missing_retention_fail_before_download(fixture):
    f=fixture
    with pytest.raises((ValueError,FileNotFoundError)):acquire(f,1)
    assert not f.calls
    prep.extract_worker(f.path,f.sha,f.raw,'books-00',f.tokenizer)
    with pytest.raises(ValueError,match='retained'):acquire(f,1)
    assert len(f.calls)==1


def test_token_budget_rejects_whole_document_without_truncation():
    wrapped=prep.TokenBudget(Tokenizer(),5)
    assert wrapped.encode('1 2')==[1,2] and wrapped.count==3
    with pytest.raises(ValueError,match='no truncation'):wrapped.encode('3 4')
    assert wrapped.count==3
    assert wrapped.encode('50279')==[50279] and wrapped.count==4


def test_file_size_scope_restores_soft_limit_on_success_and_error():
    before=resource.getrlimit(resource.RLIMIT_FSIZE)
    with prep.file_size_limit(16*1024**2):assert resource.getrlimit(resource.RLIMIT_FSIZE)[0]==16*1024**2
    assert resource.getrlimit(resource.RLIMIT_FSIZE)==before
    with pytest.raises(RuntimeError):
        with prep.file_size_limit(16*1024**2):raise RuntimeError('fixture')
    assert resource.getrlimit(resource.RLIMIT_FSIZE)==before


def test_failed_source_attempt_is_recorded_uncommitted_and_not_retried(fixture,monkeypatch):
    f=fixture;monkeypatch.setitem(prep.POLICY,'max_tokens_per_source',5)
    f.plan['recipe']['object_limits']['tokens']=5
    f.plan['recipe_sha256']=hashlib.sha256((json.dumps(f.plan['recipe'],sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
    f.path.write_bytes(prep.canonical(f.plan));f.sha=prep.extractor.digest(f.path)
    with pytest.raises(ValueError,match='candidate-token'):acquire(f)
    assert len(f.calls)==1 and not (f.raw/'books-00').exists()
    assert (f.raw/'books-00.partial').is_dir()
    attempt=json.loads((f.acquisition/'attempts/books-00-0001.json').read_text())
    assert attempt['status']=='failed' and 'candidate-token' in attempt['error']
    assert not f.bucket.objects


def test_changed_etag_and_failed_remote_verification_do_not_publish_progress(fixture,monkeypatch):
    f=fixture;original=prep.extractor.urllib.request.urlopen
    def changed(*args,**kwargs):
        response=original(*args,**kwargs);response.headers={'ETag':'changed'};return response
    monkeypatch.setattr(prep.extractor.urllib.request,'urlopen',changed)
    with pytest.raises(ValueError,match='ETag'):acquire(f)
    assert not (f.raw/'books-00').exists() and not (f.acquisition/'acquisition-progress.json').exists()


def test_corrupt_existing_raw_and_receipt_are_rejected_without_new_fetch(fixture):
    f=fixture;acquire(f);raw=f.raw/'books-00/raw.jsonl';original=raw.read_bytes();raw.write_bytes(b'changed')
    with pytest.raises(ValueError):acquire(f)
    assert len(f.calls)==1
    raw.write_bytes(original)
    receipt=f.acquisition/'receipts/raw-books-00.json';value=json.loads(receipt.read_text())
    value['objects'][0]['generation']='999999';receipt.write_bytes(prep.canonical(value))
    with pytest.raises(ValueError,match='authority'):acquire(f)
    assert len(f.calls)==1


def test_all_sources_global_dedup_split_and_byte_identical_resume(fixture):
    f=fixture;all_raw(f)
    results=[]
    while True:
        result=prepare(f);results.append(result)
        if result['status']=='complete':break
        assert len(results)<20
    final=results[-1];summary=final['tokenized']
    assert summary['rows']==74 and summary['documents']==38 and summary['duplicate_rows']==36
    assert summary['tokens']==38*4 and len(results)>1
    config=json.loads((f.prepared/'config.json').read_text())
    assert config['split_policy']=={'seed':20260929,'weights':[['train',90],['dev',5],['confirmation',5]]}
    assert len(config['sources'])==37 and config['deduplication']=='first_occurrence_of_split_identity_global_to_this_artifact'
    assert all(row['pin']==final['source_authorities'][row['pin']['name']]['source_pin'] for row in config['sources'])
    assert all(row['pin']['uri'].startswith(prep.CLOUD_RUN+'/raw-') for row in config['sources'])
    for name in ('config.json','manifest.json'):
        retained_name='tokenized-manifest.json' if name=='manifest.json' else name
        assert (f.prepared/name).read_bytes()==(f.evidence/'root-summary'/retained_name).read_bytes()
    assert (f.evidence/'root-summary/source-plan.json').read_bytes()==f.path.read_bytes()
    assert final['final_retention']['receipt']['status']=='verified'
    assert all(row['receipt']['status']=='verified' for row in final['shard_retention'].values())
    before={str(p.relative_to(f.prepared)):p.read_bytes() for p in f.prepared.rglob('*') if p.is_file() and not p.name.startswith('.')}
    repeated=prepare(f)
    assert repeated==final
    assert before=={str(p.relative_to(f.prepared)):p.read_bytes() for p in f.prepared.rglob('*') if p.is_file() and not p.name.startswith('.')}
    # Every duplicate retains its original split rather than creating leakage.
    rows=[json.loads(line) for path in f.prepared.glob('shard-*/documents.jsonl') for line in path.read_text().splitlines()]
    split_by_key={row['document_key']:row['split'] for row in rows if row['kind']=='document'}
    assert all(row['split']==split_by_key[row['duplicate_of']] for row in rows if row['kind']=='duplicate')


def test_prepare_requires_all_raw_and_never_uses_per_source_dedup_artifacts(fixture):
    f=fixture;acquire(f)
    with pytest.raises(ValueError,match='All 37'):prepare(f)
    assert not (f.prepared/'config.json').exists()


def test_retains_prior_committed_shards_before_advancing_after_interruption(fixture,monkeypatch):
    f=fixture;all_raw(f);real=prep.verified_receipt
    def fail_shard(directory,evidence,stage,**kwargs):
        if stage=='shard-000000':raise OSError('upload interrupted')
        return real(directory,evidence,stage,**kwargs)
    monkeypatch.setattr(prep,'verified_receipt',fail_shard)
    with pytest.raises(OSError):prepare(f)
    assert len(list(f.prepared.glob('shard-*')))==2
    monkeypatch.setattr(prep,'verified_receipt',real)
    original=prep.shards.prepare_document_shards
    def inspect(*args,**kwargs):
        for index in range(2):assert (f.evidence/f'receipts/shard-{index:06d}.json').is_file()
        return original(*args,**kwargs)
    monkeypatch.setattr(prep.shards,'prepare_document_shards',inspect)
    assert prepare(f)['status']=='retained_partial'


def test_locked_acquisition_rejects_concurrent_writer(fixture):
    with prep.exclusive(fixture.raw,'.pilot-acquire.lock'):
        with pytest.raises(BlockingIOError):acquire(fixture)
    assert not fixture.calls


@pytest.mark.parametrize('field',['split','object_limits','bounds'])
def test_recipe_policy_drift_is_rejected(fixture,field):
    f=fixture;f.plan['recipe'][field]={}
    f.plan['recipe_sha256']=hashlib.sha256((json.dumps(f.plan['recipe'],sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
    f.path.write_bytes(prep.canonical(f.plan))
    with pytest.raises(ValueError,match='recipe'):prep.load_plan(f.path,prep.extractor.digest(f.path))
    assert not f.calls


@pytest.mark.parametrize('mutation',['release','revision','upstream_release','stratum','selection_hash','native_eos','stratum_count'])
def test_wrong_release_or_source_selection_authority_cannot_be_mislabeled(fixture,mutation):
    f=fixture;row=f.plan['sources'][0]
    if mutation=='release':f.plan['recipe']['release']='v1_7'
    elif mutation=='revision':f.plan['recipe']['hf_revision']='different'
    elif mutation=='upstream_release':row['url']=row['url'].replace('v1_5r1','v1_7')
    elif mutation=='stratum':row['stratum']='c4'
    elif mutation=='selection_hash':row['selection_sha256']='0'*64
    elif mutation=='native_eos':f.plan['recipe']['tokenizer']['eos_id']=1
    else:f.plan['recipe']['selection']['objects_per_stratum']['books']=4
    f.plan['recipe_sha256']=hashlib.sha256((json.dumps(f.plan['recipe'],sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
    f.path.write_bytes(prep.canonical(f.plan))
    with pytest.raises(ValueError):prep.load_plan(f.path,prep.extractor.digest(f.path))
    assert not f.calls


def test_plan_parsing_uses_the_exact_bytes_that_were_pinned(fixture,monkeypatch):
    f=fixture;original=Path.read_bytes;reads=[]
    def read(path):
        raw=original(path)
        if path==f.path:
            reads.append(path);path.write_bytes(b'changed after exact byte read')
        return raw
    monkeypatch.setattr(Path,'read_bytes',read)
    assert prep.load_plan(f.path,f.sha)==f.plan
    assert reads==[f.path]
