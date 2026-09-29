"""Independent raw/token audit oracles; no network, model or GPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cdrm.pretrained import document_shards as shards
from scripts import olmo_pilot_data_audit as audit
from scripts import olmo_pilot_data_plan as plan
from test_pilot_ordered_data import fixture as ordered_fixture


def encode(text):return list(map(int,text.split()))


def row_for(tokens,text='7 50279 8',document_id='doc'):
    stored=audit.u16(tokens)
    return {'document_id':document_id,'token_count':len(tokens),'token_sha256':audit.raw_sha(stored),
        'content_token_sha256':audit.raw_sha(stored[:-2]),'text_sha256':audit.raw_sha(text.encode()),
        'had_terminal_eos':text.endswith('50279'),'embedded_eos_count':tokens[:-1].count(50279)},stored


def test_complete_document_oracle_preserves_literal_and_terminal_eos():
    row,stored=row_for([7,50279,8,50279])
    seen=[]
    result=audit.compare_document(row,{'id':'doc','text':'7 50279 8'},stored,lambda text:seen.append(text) or encode(text))
    assert seen==['7 50279 8'] and result['tokens']==4 and result['embedded_eos_count']==1
    text='7 50279 8 50279';row,stored=row_for([7,50279,8,50279],text)
    assert audit.compare_document(row,{'id':'doc','text':text},stored,encode)['had_terminal_eos'] is True


@pytest.mark.parametrize('change',['id','text','payload','content_hash','token_count','literal_eos','truncated_encoding'])
def test_literal_retokenization_detects_meaningful_mutations(change):
    row,stored=row_for([7,50279,8,50279]);raw={'id':'doc','text':'7 50279 8'};fn=encode
    if change=='id':raw['id']='wrong'
    elif change=='text':raw['text']='7 8'
    elif change=='payload':stored=stored[:-2]
    elif change=='content_hash':row['content_token_sha256']='a'*64
    elif change=='token_count':row['token_count']-=1
    elif change=='literal_eos':row['embedded_eos_count']=0
    else:fn=lambda text:encode(text)[:1]
    with pytest.raises(ValueError):audit.compare_document(row,raw,stored,fn)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    class Tokenizer:
        def encode(self,text,*,add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(ids=encode(text))
    monkeypatch.setattr(shards,'_load_tokenizer',lambda path:Tokenizer())
    f=ordered_fixture(tmp_path,build=False)
    raw_root=tmp_path/'raw';raw_root.mkdir()
    for source in f.acquisition['sources']:
        name=source['name'];directory=raw_root/name;directory.mkdir()
        raw=(tmp_path/(name+'.jsonl')).read_bytes();(directory/'raw.jsonl').write_bytes(raw)
        records=[json.loads(line) for line in raw.splitlines()]
        mapping=b''.join(plan.canonical({'source_line':i,'output_line':i,'id':row['id'],'status':'retained'})
            for i,row in enumerate(records,1))
        (directory/'source-lines.jsonl').write_bytes(mapping)
        tokens=sum(len(encode(row['text'])) for row in records)
        manifest={'schema':'olmo-dolma-source-extract-v2','status':'complete','full_upstream_gzip_sha256_verified':False,
            'config':{'source':source,'tokenizer_sha256':f.recipe['tokenizer']['sha256'],
            'upstream_manifest':{'source_plan_sha256':f.args['acquisition_plan_sha256']}},
            'remote':{'etag':source['etag'],'requested_url':source['url']},'source_rows':[1,len(records)],
            'documents':len(records),'tokens_including_terminal_eos':tokens,'compressed_bytes_read':100,
            'files':{name:{'size_bytes':len(value),'sha256':audit.raw_sha(value)} for name,value in
                (('raw.jsonl',raw),('source-lines.jsonl',mapping))}}
        value=plan.canonical(manifest);(directory/'manifest.json').write_bytes(value)
        f.authorities[name]['extraction_manifest_sha256']=audit.raw_sha(value)
    f.raw=raw_root
    return f


def run(f,tmp_path,**overrides):
    scratch=tmp_path/'scratch';scratch.mkdir()
    kwargs=dict(recipe=f.recipe,acquisition=f.acquisition,acquisition_sha256=f.args['acquisition_plan_sha256'],
        source_authorities=f.authorities,excluded_content_hashes=[f.excluded],encode=encode,
        scratch_dir=scratch,samples_per_source=2)
    kwargs.update(overrides)
    return audit.audit_documents(f.corpus,f.raw,**kwargs)


def test_actual_shards_independent_split_coverage_and_complete_raw_samples(fixture,tmp_path):
    f=fixture;progress=[];result=run(f,tmp_path,progress=progress.append)
    assert result['status']=='passed' and result['source_count']==9 and result['sampled_documents']==18
    assert result['tokens']==json.loads((f.corpus/'manifest.json').read_text())['tokens']
    assert result['excluded_documents_present_in_candidate_corpus']==1
    assert result['exact_content_split_intersections']==0 and result['no_payload_truncation']
    assert len(progress)==10 and progress[-1]['sampled_documents']==18
    assert all(r['raw_line']==r['upstream_source_line'] for r in result['sample_rows'])
    # Independent complete metadata collection, followed by a literal sort.
    candidates={i:[] for i in range(9)}
    for path in f.corpus.glob('shard-*/documents.jsonl'):
        for raw in path.read_text().splitlines():
            row=json.loads(raw)
            if row['kind']=='document':
                key=hashlib.sha256(('cdrm-dolma-pilot-v1-raw-audit\0'+row['content_token_sha256']).encode()).hexdigest()
                candidates[row['source_index']].append((key,row['document_key']))
    for i,rows in candidates.items():
        actual=[(r['selection_hash'],r['document_key']) for r in result['sample_rows'] if r['source_index']==i]
        assert actual==sorted(rows)[:2]


@pytest.mark.parametrize('change',['source_order','authority','generation','mapping','raw_bytes','sample_count','encode','split_seed'])
def test_source_raw_mapping_and_integrity_failures_are_detected(fixture,tmp_path,change):
    f=fixture;overrides={};first=f.acquisition['sources'][0]['name']
    if change=='source_order':f.acquisition['sources'].reverse()
    elif change=='authority':f.authorities[first]['source_pin']['sha256']='a'*64
    elif change=='generation':f.authorities[first]['raw_object']['generation']='latest'
    elif change=='mapping':
        path=f.raw/first/'source-lines.jsonl';rows=path.read_text().splitlines();row=json.loads(rows[0]);row['source_line']=2;rows[0]=json.dumps(row)
        path.write_text('\n'.join(rows)+'\n')
    elif change=='raw_bytes':
        path=f.raw/first/'raw.jsonl';path.write_bytes(path.read_bytes()+b' ')
    elif change=='sample_count':overrides['samples_per_source']=17
    elif change=='encode':overrides['encode']=lambda text:[1]
    else:f.recipe['split']['seed']+=1
    with pytest.raises((ValueError,KeyError)):run(f,tmp_path,**overrides)


def test_content_split_oracle_matches_declared_hash_rule_on_large_integers():
    recipe=plan.load_recipe()
    for n in range(1000):
        identity=hashlib.sha256(str(n).encode()).hexdigest()
        assert audit.split_oracle(recipe,identity)==plan.split_for_content_hash(recipe,identity)


@pytest.mark.parametrize('metadata,raw,declared',[(99,100,100),(101,100,100),(100,100,99)])
def test_source_eof_coverage_catches_nonsampled_trailing_record_omission(metadata,raw,declared):
    # All sampled rows can match when the omitted row is outside the sample;
    # the complete per-source row counts are a separate mandatory oracle.
    audit.source_row_coverage(100,100,100)
    with pytest.raises(ValueError,match='coverage differs'):audit.source_row_coverage(metadata,raw,declared)


def test_line_reader_rejects_oversized_record_without_silent_truncation(tmp_path):
    path=tmp_path/'raw.jsonl';path.write_bytes(b'0123456789\n')
    with pytest.raises(ValueError,match='complete-record'):
        audit._line_stream(path,5,audit.sha(path),consume=lambda *_:None)


def test_line_reader_detects_mutation_during_sample_scan(tmp_path):
    path=tmp_path/'raw.jsonl';path.write_bytes(b'{}\n')
    expected=audit.sha(path)
    def mutate(*_):path.write_bytes(b'{ }\n')
    with pytest.raises(ValueError,match='changed'):
        audit._line_stream(path,10,expected,consume=mutate)
