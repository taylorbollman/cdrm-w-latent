"""CPU literal document/ordering/mask oracles over newly prepared tiny corpora."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

import pytest
import torch
from cdrm.pretrained import document_shards as shards
from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained.nextlat import build_nextlat_masks
from cdrm.pretrained.packed_campaign_data import PackedCampaignData
from scripts import olmo_pilot_data_plan as plan
from scripts import olmo_pilot_ordered_data as ordered


@pytest.fixture(autouse=True)
def cpu_tokenizer(monkeypatch):
    class Tokenizer:
        def encode(self,text,*,add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(ids=list(map(int,text.split())) if text else [])
    monkeypatch.setattr(shards,'_load_tokenizer',lambda path:Tokenizer())
    torch.set_num_threads(1)


def raw_json(path,value):
    path.write_bytes((json.dumps(value,sort_keys=True)+'\n').encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path,*,length=4,build=True):
    recipe=deepcopy(plan.load_recipe())
    recipe['inventory']['directory_counts']=dict.fromkeys(plan.STRATA,1)
    recipe['inventory']['url_count']=len(plan.STRATA)
    inventory=''.join(plan.URL_ROOT+s+'/part.json.gz\n' for s in plan.STRATA).encode()
    recipe['inventory']['sha256']=hashlib.sha256(inventory).hexdigest()
    recipe['selection']['objects_per_stratum']=dict.fromkeys(plan.STRATA,1)
    recipe['panels'].update(length=length,train_tokens=length*27,heldout_main_tokens=length*18,
        heldout_source_tokens=length*2,minimum_reserve_tokens_per_stratum=length*3,
        minimum_documents_per_stratum=dict.fromkeys(plan.STRATA,2))
    selected=plan.source_selection(recipe,inventory)
    documents=[];excluded=None
    for source_index,source in enumerate(selected):
        counts=dict.fromkeys(('train','dev','confirmation'),0);rows=[];trial=0
        while counts['train']<30 or counts['dev']<8 or counts['confirmation']<8:
            unique=2+source_index*3000+trial*2;trial+=1
            assert unique+1<50279
            values=[unique,50279,unique+1]+[unique]*(length+1)+[50279]
            content=hashlib.sha256(shards._u16(values[:-1])).hexdigest()
            split=plan.split_for_content_hash(recipe,content)
            target=30 if split=='train' else 8
            if counts[split]>=target:continue
            counts[split]+=1
            rows.append({'id':f'{source_index}-{trial}','text':' '.join(map(str,values))})
            if source_index==0 and split=='train' and excluded is None:excluded=content
        documents.append(rows)
    recipe['exclusions'].update(unique_documents=1,
        content_ids_sha256=hashlib.sha256(plan.exclusion_bytes([excluded])).hexdigest())
    acquisition={'recipe':recipe,'sources':[{**r,'etag':'"fixture-etag"','upstream_size_bytes':123456} for r in selected]}
    acquisition_path=tmp_path/'acquisition.json';acquisition_sha=raw_json(acquisition_path,acquisition)
    sources=[];authorities={}
    for source,rows in zip(acquisition['sources'],documents):
        path=tmp_path/(source['name']+'.jsonl')
        path.write_bytes(b''.join((json.dumps(row)+'\n').encode() for row in rows))
        raw_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        pin=SourcePin(source['name'],'gs://fast-chunks/cdrm-w-latent/data/test/raw-'+source['name']+'/raw.jsonl',
                      recipe['hf_revision'],raw_sha)
        sources.append(LocalJSONLSource(pin,path))
        authorities[source['name']]={'source_pin':asdict(pin),'upstream_source':source,
            'extraction_manifest_sha256':'a'*64,'acquisition_plan_sha256':acquisition_sha,
            'raw_object':{'uri':pin.uri,'sha256':raw_sha,'size_bytes':path.stat().st_size,'generation':'123',
                'verification':dict.fromkeys(('server_size','server_md5','sha256_metadata','download_sha256'),True)}}
    authority_path=tmp_path/'authorities.json';authority_sha=raw_json(authority_path,authorities)
    corpus=tmp_path/'corpus'
    shards.prepare_document_shards(sources,corpus,tokenizer_path='fixture-only',
        split_policy=SplitPolicy(recipe['split']['seed'],recipe['split']['weights']),max_documents_per_shard=23)
    args=dict(recipe=recipe,inventory_bytes=inventory,excluded_content_hashes=[excluded],
        acquisition_plan_path=acquisition_path,acquisition_plan_sha256=acquisition_sha,
        source_authorities_path=authority_path,source_authorities_sha256=authority_sha)
    output=tmp_path/'ordered'
    manifest=ordered.build_ordered_data(corpus,output,**args) if build else None
    return SimpleNamespace(corpus=corpus,output=output,manifest=manifest,args=args,recipe=recipe,excluded=excluded,
                           acquisition=acquisition,authorities=authorities)


def panel(f,name='train'):
    return ordered.OrderedCampaignData(f.corpus,f.output/'panels'/name)


def test_literal_tokens_true_intervals_and_masks_after_weighted_chunk_shuffle(tmp_path):
    f=fixture(tmp_path)
    original=list(shards.iter_documents(f.corpus))
    with panel(f) as data:
        chunks=[data.read_chunk(i) for i in range(data.total_chunks)]
        for chunk in chunks:
            tokens=[];ids=[]
            for segment in chunk.segments:
                source=original[segment.document_index].tokens
                tokens.extend(source[segment.document_token_offset:segment.document_token_offset+segment.length])
                ids.extend([segment.document_index]*segment.length)
            assert tuple(tokens)==chunk.tokens and tuple(ids)==chunk.document_ids
            assert len(chunk.tokens)==4
        expected_order=list(plan.weighted_stratum_order(plan.panel_quotas(f.recipe,'train')))
        assert [data.source_chunk(i)['stratum'] for i in range(data.total_chunks)]==expected_order
        assert [data.source_chunk(i)['source_chunk_index'] for i in range(data.total_chunks)]!=list(range(data.total_chunks))
        update=data.peek_update(data.cursor(),10**6)
        counts=dict.fromkeys(('ce','latent','kl'),0)
        for rank in range(3):
            batched=data.rank_batches(update,rank=rank,world_size=3,physical_batch_size=4)
            for batch in batched.batches:
                for term,mask in build_nextlat_masks(batch,document_policy='continuous-stream-v1').items():
                    counts[term]+=int(mask.sum())
        assert counts==update.counts.objective_counts
        assert asdict(update.counts)==data.manifest['counts']
        assert counts['ce']==27*3
        assert any(50279 in c.tokens[:-1] and len(set(c.document_ids))==1 for c in chunks)


def test_split_exclusion_and_panel_overlap_membership_are_explicit(tmp_path):
    f=fixture(tmp_path)
    memberships={s:set() for s in ('train','dev','confirmation')}
    for name in ordered.panel_names():
        with panel(f,name) as data:
            ids={row[0] for row in data._connection.execute('SELECT content_token_sha256 FROM documents')}
            memberships[data.split]|=ids
            assert f.excluded not in ids
            assert data.manifest['selected_documents']<=data.manifest['reserved_documents']
            assert sum(v['selected_tokens'] for v in data.manifest['membership_by_source'].values())==data.total_tokens
    assert not (memberships['train']&memberships['dev'] or memberships['train']&memberships['confirmation'] or memberships['dev']&memberships['confirmation'])
    with panel(f,'dev-main') as main,panel(f,'dev-source/cc_en_tail') as source:
        a={main.source_chunk(i)['chunk_key'] for i in range(main.total_chunks)}
        b={source.source_chunk(i)['chunk_key'] for i in range(source.total_chunks)}
        assert len(a&b)==min(plan.panel_quotas(f.recipe,'dev-main')['cc_en_tail'],2)


def test_rebuild_is_byte_identical_relocation_and_cursor_resume_preserve_membership(tmp_path):
    f=fixture(tmp_path);other=tmp_path/'rebuild'
    ordered.build_ordered_data(f.corpus,other,**f.args)
    files=lambda root:{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    assert files(other)==files(f.output)
    relocated=tmp_path/'relocated';shutil.copytree(f.corpus,relocated)
    with panel(f) as data:
        first=data.peek_update(data.cursor(),17);saved=data.commit(data.cursor(),first)
        future=data.peek_update(saved,17)
        with ordered.OrderedCampaignData(relocated,other/'panels/train') as restored:
            restored.restore_cursor(asdict(saved))
            assert restored.peek_update(restored.cursor(),17)==future
            assert restored.read_chunk(future.rows[0].index)==data.read_chunk(future.rows[0].index)
            with pytest.raises(ValueError,match='stale|out-of-order'):data.commit(first.start_cursor,first)


def test_finite_exhaustion_dummy_rows_and_pure_peek_read_no_tokens(tmp_path,monkeypatch):
    f=fixture(tmp_path)
    with panel(f) as data:
        original=data._token_slice
        monkeypatch.setattr(data,'_token_slice',lambda *_:(_ for _ in ()).throw(AssertionError('token read during planning')))
        update=data.peek_update(data.cursor(),10**6)
        assert update.counts.valid_tokens==108 and not update.reaches_target
        assert data.cursor()==update.start_cursor
        monkeypatch.setattr(data,'_token_slice',original)
        slots=data.partition(update,world_size=2,physical_batch_size=8)
        assert tuple(r for slot in slots for rank in slot for r in rank)==update.rows
        rank=data.rank_batches(update,rank=1,world_size=2,physical_batch_size=8)
        assert rank.empty_rows==5
        assert not rank.batches[-1].valid_mask[3:].any()
        data.commit(data.cursor(),update)
        assert data.peek_update(data.cursor(),1) is None


@pytest.mark.parametrize('mutation',['inventory','exclusion','raw_mapping','upstream','split','shortfall'])
def test_rejects_authority_or_coverage_failure_without_partial_publication(tmp_path,mutation):
    f=fixture(tmp_path,build=False);args=deepcopy(f.args)
    if mutation=='inventory':args['inventory_bytes']+=b'x'
    elif mutation=='exclusion':args['excluded_content_hashes']=['b'*64]
    elif mutation in ('raw_mapping','upstream'):
        authorities=deepcopy(f.authorities);first=next(iter(authorities))
        if mutation=='raw_mapping':authorities[first]['source_pin']['sha256']='b'*64
        else:authorities[first]['upstream_source']['url']='https://invalid.example/object'
        args['source_authorities_sha256']=raw_json(args['source_authorities_path'],authorities)
    elif mutation=='split':args['recipe']['split']['seed']+=1
    else:
        args['recipe']['panels']['train_tokens']*=1_000_000
        acquisition=deepcopy(f.acquisition);acquisition['recipe']=args['recipe']
        args['acquisition_plan_sha256']=raw_json(args['acquisition_plan_path'],acquisition)
        authorities=deepcopy(f.authorities)
        for row in authorities.values():row['acquisition_plan_sha256']=args['acquisition_plan_sha256']
        args['source_authorities_sha256']=raw_json(args['source_authorities_path'],authorities)
    with pytest.raises(ValueError,match='Insufficient' if mutation=='shortfall' else None):
        ordered.build_ordered_data(f.corpus,f.output,**args)
    assert not f.output.exists() and not list(tmp_path.glob('.pending-ordered-pilot-*'))


def test_old_reader_rejects_new_schema_and_new_reader_rejects_mutations(tmp_path):
    f=fixture(tmp_path)
    with pytest.raises(ValueError,match='schema'):PackedCampaignData(f.corpus,f.output/'panels/train')
    with panel(f) as data:
        row=data.descriptor(0)
        with pytest.raises(ValueError,match='provenance'):data.batch([replace(row,key='bad')],physical_batch_size=1)
        with pytest.raises(ValueError):data.restore_cursor({**asdict(data.cursor()),'manifest_sha256':'a'*64})
        data.length=8
        with pytest.raises(ValueError,match='runtime contract'):data.cursor()
        data.length=4
        path=f.output/'panels/train/documents.sqlite';path.write_bytes(path.read_bytes()+b'x')
        with pytest.raises(ValueError,match='changed'):data.cursor()
    with pytest.raises(ValueError,match='index bytes'):panel(f)


def test_t1024_literal_masks_and_selected_token_accounting(tmp_path):
    f=fixture(tmp_path,length=1024)
    with panel(f,'dev-source/books') as data:
        update=data.peek_update(data.cursor(),2048)
        batch=data.batch(update.rows,physical_batch_size=3)
        masks=build_nextlat_masks(batch,document_policy='continuous-stream-v1')
        assert {k:int(v.sum()) for k,v in masks.items()}==update.counts.objective_counts
        assert update.counts.valid_tokens==2048 and update.counts.ce_targets==2046
        assert not batch.valid_mask[2].any()


@pytest.mark.parametrize('changed',['config','acquisition','selection'])
def test_source_order_binds_first_occurrence_deduplication(tmp_path,changed):
    f=fixture(tmp_path,build=False)
    selected=plan.source_selection(f.recipe,f.args['inventory_bytes'])
    config={'sources':[{'pin':deepcopy(f.authorities[row['name']]['source_pin'])} for row in selected]}
    acquisition=deepcopy(f.acquisition)
    target={'config':config['sources'],'acquisition':acquisition['sources'],'selection':selected}[changed]
    target[0],target[1]=target[1],target[0]
    with pytest.raises(ValueError,match='source order differs'):
        ordered._source_contract(config,selected,f.recipe,acquisition,f.args['acquisition_plan_sha256'],f.authorities)


def test_late_chunk_lookup_uses_bounded_document_range(tmp_path):
    connection=ordered._db(tmp_path/'scale.sqlite')
    try:
        connection.execute('CREATE TABLE documents (document_index INTEGER,document_key TEXT,document_id TEXT,'
            'source_index INTEGER,source_name TEXT,source_line INTEGER,shard TEXT,token_offset INTEGER,'
            'stratum TEXT,stream_start INTEGER,stream_end INTEGER)')
        connection.execute('CREATE INDEX doc_stream_start ON documents(stratum,stream_start)')
        connection.executemany('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            ((i,str(i),str(i),0,'books-00',i,'shard',i*10,'books',i*10,i*10+10) for i in range(20000)))
        callbacks=0
        def budget():
            nonlocal callbacks
            callbacks+=1
            return int(callbacks>20)
        connection.set_progress_handler(budget,100)
        segments=ordered._segments(connection,{'stratum':'books','source_stream_start':199985,'length':12})
        assert [(s.document_index,s.document_token_offset,s.chunk_offset,s.length) for s in segments]==[
            (19998,5,0,5),(19999,0,5,7)]
        assert callbacks<=20
    finally:connection.close()
