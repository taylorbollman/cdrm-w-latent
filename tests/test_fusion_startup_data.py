"""CPU-only exact-target scheduling, provenance and fresh-fixture contracts."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained import document_shards as shards
from cdrm.pretrained.campaign_data import SourcePin, TokenizedDocument
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts import olmo_fusion_startup_data as data_module
from scripts.olmo_fusion_startup_data import StartupData, load_fresh_fixture


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def documents():
    source = SourcePin('tiny', 'https://example.invalid/data', 'pinned', 'a'*64)
    values = [tuple(range(10,50))+(50279,70,71,50279), tuple(range(100,131))+(50279,)]
    values += [tuple(range(200+30*i,220+30*i))+(50279,) for i in range(4)]
    return [TokenizedDocument(source,f'doc-{i}','train' if i<2 else 'dev',
                              hashlib.sha256(str(i).encode()).hexdigest(),tokens)
            for i,tokens in enumerate(values)]


def tiny(**kwargs):
    return StartupData._from_documents(documents(),provenance={'manifest_sha256':'b'*64},
                                      length=8,ce_per_update=11,physical_batch_size=3,**kwargs)


def tensor_pins(fixtures):
    return [({name: data_module._tensor_pin(value) for name,value in vars(batch).items()},
             data_module._noise_pins(noise))
            for batches,noises in fixtures for batch,noise in zip(batches,noises)]


def test_exact_target_coverage_matches_independent_document_positions():
    data = tiny()
    seen=[]
    for update in range(data.total_updates):
        rows=data.selections(update)
        assert sum(r.target_stop-r.target_start for r in rows)==11
        seen.extend((r.window.document_index,r.window.start+p)
                    for r in rows for p in range(r.target_start,r.target_stop))
    assert len(seen)==len(set(seen))==data.total_updates*11
    # Unsorted input documents are intentionally shuffled; within each selected
    # document every target stays sequential, including embedded and terminal EOS.
    expected=[]
    ordered_indices=[]
    for row in data._windows:
        if row.document_index not in ordered_indices: ordered_indices.append(row.document_index)
    for index in ordered_indices:
        matching=[r for r in data._windows if r.document_index==index]
        final=max(r.start+len(r.tokens) for r in matching)
        expected.extend((index,p) for p in range(1,final))
    assert seen==expected[:len(seen)]
    assert data.manifest['unused_final_ce_targets']==len(expected)-len(seen)


def test_update_boundary_reuses_full_context_with_complementary_masks():
    data=tiny(); left=data.selections(0)[-1]; right=data.selections(1)[0]
    assert left.window==right.window
    assert left.target_stop==right.target_start
    assert set(range(left.target_start,left.target_stop)).isdisjoint(range(right.target_start,right.target_stop))
    assert left.window.tokens==right.window.tokens and len(left.window.tokens)==8


def test_batches_counts_padding_and_internal_eos_are_exact():
    data=tiny(); recipe=CampaignRecipe('NF')
    embedded_seen=False
    for update in range(data.total_updates):
        batches,noises=data.update_batches(update,recipe,4)
        meta=data.update_metadata(update)
        counts=dict.fromkeys(('ce','latent','kl'),0)
        inputs=rows=0
        for batch,noise in zip(batches,noises):
            assert batch.input_ids.shape==(3,8)
            assert not (batch.valid_mask[:,1:] & ~batch.valid_mask[:,:-1]).any()
            assert (batch.document_ids[~batch.valid_mask]==-1).all()
            assert (batch.input_ids[~batch.valid_mask]==1).all()
            for name,mask in build_nextlat_masks(batch).items(): counts[name]+=int(mask.sum())
            inputs+=int(batch.valid_mask.sum());rows+=int(batch.valid_mask.any(-1).sum())
            for values,valid in zip(batch.input_ids,batch.valid_mask):
                tokens=values[valid].tolist()
                if 50279 in tokens[:-1]: embedded_seen=True
            assert len(noise)==3 and all(t.shape==(3,7,4) for t in noise)
            assert all(not t[~batch.valid_mask.any(-1)].any() for t in noise)
        assert counts==meta['counts'] and counts['ce']==11
        assert (inputs,rows,len(batches))==(meta['input_tokens'],meta['documents'],meta['microbatches'])
        assert meta['padding_tokens']==len(batches)*3*8-inputs
    assert embedded_seen


def test_cursor_is_explicit_pure_and_schedule_bound():
    data=tiny(); cursor=data.cursor(1)
    before=data.update_metadata(1)
    data.selections(2);data.update_batches(2,CampaignRecipe('NF'),4)
    assert data.restore_cursor(cursor)==1 and data.update_metadata(1)==before
    assert data.restore_cursor(data.cursor(data.total_updates))==data.total_updates
    for bad in [dict(cursor,next_update=True),dict(cursor,next_update=-1),
                dict(cursor,next_update=data.total_updates+1),dict(cursor,manifest_sha256='c'*64),
                dict(cursor,extra=1)]:
        with pytest.raises(ValueError):data.restore_cursor(bad)
    with pytest.raises(ValueError):data.selections(data.total_updates)
    with pytest.raises(ValueError):tiny(seed=7).restore_cursor(cursor)


def test_noise_is_partition_independent_and_does_not_consume_global_rng():
    small=tiny();large=StartupData._from_documents(documents(),provenance={'manifest_sha256':'b'*64},
        length=8,ce_per_update=11,physical_batch_size=8)
    recipe=CampaignRecipe('NF'); before=torch.get_rng_state().clone()
    def keyed(data):
        batches,noises=data.update_batches(0,recipe,4)
        rows=data.selections(0); result={};offset=0
        for batch,noise in zip(batches,noises):
            for row in range(int(batch.valid_mask.any(-1).sum())):
                result[rows[offset].key]=tuple(t[row] for t in noise);offset+=1
        return result
    a,b=keyed(small),keyed(large)
    assert set(a)==set(b) and all(torch.equal(x,y) for k in a for x,y in zip(a[k],b[k]))
    assert torch.equal(before,torch.get_rng_state())


def test_fresh_fixture_has_disjoint_documents_real_prefixes_and_all_objectives():
    data=tiny(); fixtures=data.fresh_fixture(CampaignRecipe('NF'),4)
    assert len(fixtures)==2
    assert not {r.document_key for r in data._fresh_rows}&{r.document_key for r in data._windows}
    counts=dict.fromkeys(('ce','latent','kl'),0)
    for (batch,),_ in fixtures:
        for term,mask in build_nextlat_masks(batch).items():counts[term]+=int(mask.sum())
    assert counts=={'ce':25,'latent':25,'kl':21}
    assert [int(row.sum()) for (b,),_ in fixtures for row in b.valid_mask]==[16,5,6,2]
    # None of these real dev prefixes terminates at the source document's EOS.
    assert all(r.tokens[-1]!=50279 for r in data._fresh_rows)


def test_standalone_fresh_fixture_roundtrip_and_noise_pins(tmp_path):
    data=tiny();recipe=CampaignRecipe('NF');path=tmp_path/'fresh.json'
    record=data.export_fresh_fixture(path,recipe=recipe,width=4)
    actual,meta=load_fresh_fixture(path,expected_sha256=record['sha256'],recipe=recipe,width=4)
    assert tensor_pins(actual)==tensor_pins(data.fresh_fixture(recipe,4))
    assert meta['training_manifest_sha256']==data.manifest_sha256
    assert all(d['split']=='dev' and d['source']['name']=='tiny' for d in meta['documents'])
    assert record==data.export_fresh_fixture(path,recipe=recipe,width=4)
    with pytest.raises(ValueError,match='contract'):
        load_fresh_fixture(path,expected_sha256=record['sha256'],recipe=replace(recipe,jitter_seed=999),width=4)


@pytest.mark.parametrize('mutation',['split','noise','mask','raw'])
def test_fresh_fixture_rejects_corruption_even_if_rehashed_metadata(tmp_path,mutation):
    data=tiny();recipe=CampaignRecipe('NF');p=tmp_path/'fresh.json'
    record=data.export_fresh_fixture(p,recipe=recipe,width=4);x=json.loads(p.read_text())
    if mutation=='split':x['documents'][0]['split']='train'
    elif mutation=='noise':x['records'][0]['noise_pins'][0]['sha256']='0'*64
    elif mutation=='mask':x['records'][0]['batch']['ce_mask'][0][0]=True
    else:x['documents'][0]['document_id']='changed'
    p.write_text(json.dumps(x));digest=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_fresh_fixture(p,expected_sha256=record['sha256'] if mutation=='raw' else digest,recipe=recipe,width=4)


def prepared(tmp_path,monkeypatch):
    class Tokenizer:
        def encode(self,text,*,add_special_tokens):return SimpleNamespace(ids=list(map(int,text.split())))
    monkeypatch.setattr(shards,'_load_tokenizer',lambda _:Tokenizer())
    raw=''.join(json.dumps({'id':str(i),'text':' '.join(str(10+32*i+j) for j in range(20))})+'\n'
                for i in range(40)).encode()
    source=tmp_path/'input.jsonl';source.write_bytes(raw)
    descriptor=LocalJSONLSource(SourcePin('fixture','https://example.invalid','fixed',hashlib.sha256(raw).hexdigest()),source)
    root=tmp_path/'prepared'
    shards.prepare_document_shards([descriptor],root,tokenizer_path='unused',
        split_policy=SplitPolicy(7,(('train',1),('dev',1))),max_documents_per_shard=8)
    return root,hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()


def test_prepared_corpus_pin_validation_and_relocation(tmp_path,monkeypatch):
    root,pin=prepared(tmp_path,monkeypatch)
    data=StartupData.from_prepared(root,expected_manifest_sha256=pin,length=8,ce_per_update=11)
    copied=tmp_path/'relocated';shutil.copytree(root,copied)
    relocated=StartupData.from_prepared(copied,expected_manifest_sha256=pin,length=8,ce_per_update=11)
    assert data.manifest==relocated.manifest
    with pytest.raises(ValueError,match='manifest'):
        StartupData.from_prepared(root,expected_manifest_sha256='0'*64)
    tokens=next(copied.glob('shard-*/tokens.bin'));raw=bytearray(tokens.read_bytes());raw[0]^=1;tokens.write_bytes(raw)
    with pytest.raises(ValueError,match='bytes'):
        StartupData.from_prepared(copied,expected_manifest_sha256=pin,length=8,ce_per_update=11)


def test_prepared_mutation_during_initialization_rejected(tmp_path,monkeypatch):
    root,pin=prepared(tmp_path,monkeypatch);original=data_module.iter_documents
    def changed(*args,**kwargs):
        yield from original(*args,**kwargs)
        path=root/'config.json';path.write_bytes(path.read_bytes()+b' ')
    monkeypatch.setattr(data_module,'iter_documents',changed)
    with pytest.raises(ValueError,match='changed during'):
        StartupData.from_prepared(root,expected_manifest_sha256=pin,length=8,ce_per_update=11)


def test_cross_split_duplicate_content_and_insufficient_holdout_rejected():
    docs=documents();docs[-1]=replace(docs[-1],tokens=docs[0].tokens)
    with pytest.raises(ValueError,match='Duplicate token'):
        StartupData._from_documents(docs,provenance={})
    with pytest.raises(ValueError,match='four dev'):
        StartupData._from_documents(documents()[:3],provenance={})
