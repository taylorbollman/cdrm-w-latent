#!/usr/bin/env python3
"""Independent interval/count and bounded token audit of a completed pilot suite."""
from __future__ import annotations
import argparse
from array import array
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.olmo_pilot_ordered_data import OrderedCampaignData
from scripts.olmo_pilot_data_plan import panel_quotas
from cdrm.pretrained.nextlat import build_nextlat_masks

SCHEMA='olmo-pilot-ordered-independent-audit-v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def open_db(path):
    c=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    c.row_factory=sqlite3.Row
    return c


class Stream:
    """Plain sorted intervals; does not use the reader's SQL segment/count code."""
    def __init__(self,documents):
        self.docs=sorted((dict(d) for d in documents),key=lambda d:d['stream_start'])
        self.starts=[d['stream_start'] for d in self.docs]
        cursor=0
        for d in self.docs:
            require(d['stream_start']==cursor and d['stream_end']-cursor==d['token_count']>0,'Noncontiguous document intervals')
            cursor=d['stream_end']
        self.total=cursor

    def doc_at(self,p):
        return None if p<0 or p>=self.total else self.docs[bisect_right(self.starts,p)-1]['document_index']

    def intersect(self,start,length):
        require(0<=start<start+length<=self.total,'Chunk outside source stream')
        i=bisect_right(self.starts,start)-1;end=start+length;out=[]
        while i<len(self.docs) and self.docs[i]['stream_start']<end:
            d=self.docs[i];left=max(start,d['stream_start']);right=min(end,d['stream_end'])
            out.append((d,left-d['stream_start'],right-left,right==d['stream_end']));i+=1
        require(sum(x[2] for x in out)==length,'Interval coverage differs')
        return out

    def counts(self,start,length):
        parts=self.intersect(start,length);end=start+length
        ce=length-1;latent=sum(max(0,p[2]-1) for p in parts);kl=sum(max(0,p[2]-2) for p in parts)
        omitted=int(end<self.total)
        triples=0
        for a in (end-2,end-1):
            if omitted and a>=0 and a+2<self.total:
                triples+=int(self.doc_at(a)==self.doc_at(a+1)==self.doc_at(a+2))
        return {'packed_rows':1,'valid_tokens':length,'ce_targets':ce,'latent_pairs':latent,'kl_triples':kl,
            'cross_document_ce_targets':len(parts)-1,'excluded_boundary_latent_pairs':ce-latent,
            'excluded_boundary_kl_triples':length-2-kl,'omitted_cross_chunk_ce_targets':omitted,
            'omitted_cross_chunk_latent_pairs':int(bool(omitted) and self.doc_at(end-1)==self.doc_at(end)),
            'omitted_cross_chunk_kl_triples':triples,'tail_padding_tokens':0,'document_segments':len(parts),
            'document_completions':sum(p[3] for p in parts)}


def literal(corpus,parts):
    tokens=[];ids=[]
    for d,offset,n,_ in parts:
        with (Path(corpus)/d['shard']/'tokens.bin').open('rb') as f:
            f.seek(2*(d['token_offset']+offset));raw=f.read(2*n)
        require(len(raw)==2*n,'Truncated token payload')
        values=array('H');values.frombytes(raw)
        if sys.byteorder!='little':values.byteswap()
        tokens.extend(values);ids.extend([d['document_index']]*n)
    return tuple(tokens),tuple(ids)


def verify_originals(corpus,manifest,originals):
    """Link generated catalog records back to the independently pinned corpus."""
    corpus=Path(corpus);config=json.loads((corpus/'config.json').read_bytes())
    root=json.loads((corpus/'manifest.json').read_bytes());seen=set();index=0
    for relative,pin in manifest['corpus_files'].items():
        path=Path(relative)
        require(not path.is_absolute() and '..' not in path.parts,'Unsafe corpus file')
        require(sha(corpus/path)==pin['sha256'] and (corpus/path).stat().st_size==pin['size_bytes'],'Original corpus bytes differ')
    for shard in root['shards']:
        with (corpus/shard['path']/'documents.jsonl').open('rb') as f:
            for line in f:
                record=json.loads(line)
                if record['kind']!='document':continue
                if index in originals:
                    selected=originals[index]
                    keys=('token_offset','token_count','source_index','source_line','split','document_id',
                        'document_key','token_sha256','content_token_sha256')
                    require(all(record[k]==selected[k] for k in keys)
                        and selected['shard']==shard['path']
                        and selected['source_name']==config['sources'][record['source_index']]['pin']['name'],
                        'Generated catalog differs from original corpus document')
                    authority=manifest['acquisition_authority']['source_authorities'][selected['source_name']]
                    url=authority['upstream_source']['url'];prefix='https://olmo-data.org/dolma-v1_5r1/'
                    require(url.startswith(prefix) and selected['stratum']==url[len(prefix):].split('/')[0]
                        and authority['source_pin']==config['sources'][record['source_index']]['pin'],
                        'Generated catalog source stratum differs from declared upstream')
                    seen.add(index)
                index+=1
    require(seen==set(originals),'Catalog references absent original document')


def audit(corpus,suite,suite_sha):
    corpus,suite=Path(corpus),Path(suite)
    require(sha(suite/'manifest.json')==suite_sha,'Independent suite pin differs')
    manifest=json.loads((suite/'manifest.json').read_bytes())
    strata=('books','c4','cc_en_head','cc_en_middle','cc_en_tail','pes2o','reddit','stack','wiki')
    expected_panels={'train','dev-main','confirmation-main',*(f'{s}-source/{v}' for s in ('dev','confirmation') for v in strata)}
    require(set(manifest['panels'])==expected_panels,'Exact 21-panel set differs')
    cp=manifest['selection_authority']
    require(cp['path']=='catalog.sqlite' and sha(suite/cp['path'])==cp['sha256'],'Catalog pin differs')
    with open_db(suite/cp['path']) as catalog:
        originals={d['document_index']:dict(d) for d in catalog.execute('SELECT * FROM documents')}
    verify_originals(corpus,manifest,originals)
    panels={};memberships={};selected_by_split=defaultdict(set);counts_checked=0;sampled=0
    for name,entry in manifest['panels'].items():
        path=Path(entry['path'])
        require(not path.is_absolute() and '..' not in path.parts,'Unsafe panel path')
        directory=suite/path
        require(sha(directory/'manifest.json')==entry['manifest_sha256'],'Panel manifest pin differs')
        m=json.loads((directory/'manifest.json').read_bytes())
        require(m['stratum_quotas']==panel_quotas(manifest['recipe'],name),'Panel quota differs from declared recipe')
        require(sha(directory/'documents.sqlite')==m['index']['sha256'],'Panel index pin differs')
        streams={};all_docs=defaultdict(list)
        with open_db(directory/'documents.sqlite') as db:
            for row in db.execute('SELECT * FROM documents'):
                d=dict(row);require(d==originals[d['document_index']],'Panel document differs from reserved catalog')
                require(not d['excluded'] and d['split']==m['split'],'Excluded or wrong split document')
                all_docs[d['stratum']].append(d)
            streams={s:Stream(ds) for s,ds in all_docs.items()}
            total=Counter();by_stratum=defaultdict(Counter);by_doc=Counter();keys=set();quotas=Counter();sample_rows=[]
            for ordinal,row in enumerate(db.execute('SELECT * FROM chunks ORDER BY ordinal')):
                r=dict(row);require(r['ordinal']==ordinal and r['length']==m['length'],'Chunk dimensions/order differ')
                require(r['source_stream_start']==r['source_chunk_index']*m['length'],'Source chunk offset differs')
                require(r['chunk_key'] not in keys,'Repeated selected chunk');keys.add(r['chunk_key'])
                stream=streams[r['stratum']];c=stream.counts(r['source_stream_start'],r['length'])
                total.update(c);by_stratum[r['stratum']].update(c);quotas[r['stratum']]+=1;counts_checked+=1
                for d,_,n,_ in stream.intersect(r['source_stream_start'],r['length']):by_doc[d['document_index']]+=n
                rank=hashlib.sha256(('pilot-independent-chunk-v1\0'+name+'\0'+r['chunk_key']).encode()).hexdigest()
                sample_rows.append((rank,r));sample_rows.sort(key=lambda x:x[0]);sample_rows=sample_rows[:16]
            require(dict(total)==m['counts']==entry['counts'],'Independent aggregate counts differ')
            require({s:dict(c) for s,c in by_stratum.items()}==m['counts_by_stratum'],'Per-stratum counts differ')
            require(dict(quotas)=={s:n for s,n in m['stratum_quotas'].items() if n},'Mixture quotas differ')
            expected_docs=dict(db.execute('SELECT document_index,selected_tokens FROM panel_documents').fetchall())
            require(dict(by_doc)==expected_docs and len(by_doc)==m['selected_documents'],'Actual selected document coverage differs')
            source_membership=defaultdict(lambda:{'documents':0,'selected_tokens':0})
            for i,n in by_doc.items():
                value=source_membership[originals[i]['source_name']];value['documents']+=1;value['selected_tokens']+=n
            require(dict(source_membership)==m['membership_by_source'],'Per-source selected membership differs')
            selected_hashes={originals[i]['content_token_sha256'] for i in by_doc}
            selected_by_split[m['split']].update(selected_hashes)
            members={k for k in keys};memberships[name]=members
        with OrderedCampaignData(corpus,directory) as reader:
            for _,r in sample_rows:
                expected=literal(corpus,streams[r['stratum']].intersect(r['source_stream_start'],r['length']))
                actual=reader.read_chunk(r['ordinal'])
                require(expected==(actual.tokens,actual.document_ids),'Independent literal chunk differs');sampled+=1
            # Uneven thirteen-row allocation checks two ranks, dummy rows and all loss masks.
            n=min(13,reader.total_chunks);u=reader.peek_update(reader.cursor(),n*reader.length)
            objective=Counter();rank_keys=[]
            for rank in range(2):
                b=reader.rank_batches(u,rank=rank,world_size=2,physical_batch_size=4)
                rank_keys.extend(k for slot in b.keys for k in slot)
                for batch in b.batches:
                    for term,mask in build_nextlat_masks(batch,document_policy='continuous-stream-v1').items():objective[term]+=int(mask.sum())
            require(len(rank_keys)==len(set(rank_keys))==n and set(rank_keys)=={r.key for r in u.rows},'Rank membership differs')
            require(dict(objective)==u.counts.objective_counts,'Rank objective masks differ')
            committed=asdict(reader.commit(reader.cursor(),u))
            with OrderedCampaignData(corpus,directory) as restored:
                restored.restore_cursor(committed)
                require(restored.peek_update(restored.cursor(),3*reader.length)==reader.peek_update(reader.cursor(),3*reader.length),'Restored cursor continuation differs')
            terminal={'manifest_sha256':reader.manifest_sha256,'split':reader.split,'next_chunk':reader.total_chunks,'next_update':reader.total_chunks}
            with OrderedCampaignData(corpus,directory) as finished:
                finished.restore_cursor(terminal);require(finished.peek_update(finished.cursor(),1) is None,'Finite stream silently cycles')
        panels[name]={'counts':dict(total),'selected_documents':len(by_doc),'sampled_chunks':len(sample_rows),
            'rank_mask_and_cursor_checks':True,'main_source_independent_replication':False}
    split_names=sorted(selected_by_split)
    for i,s in enumerate(split_names):
        for other in split_names[i+1:]:require(not selected_by_split[s]&selected_by_split[other],'Selected split intersection')
    overlaps={}
    for split in ('dev','confirmation'):
        overlaps[split]={name:len(keys&memberships[split+'-main']) for name,keys in memberships.items() if name.startswith(split+'-source/')}
    require(sha(suite/'manifest.json')==suite_sha and sha(suite/cp['path'])==cp['sha256'],'Suite changed during audit')
    return {'schema':SCHEMA,'status':'passed','suite_sha256':suite_sha,'all_selected_chunks_counted':counts_checked,
        'literal_sampled_chunks':sampled,'panels':panels,'main_source_chunk_overlap':overlaps,
        'actual_selected_unique_documents_by_split':{s:len(v) for s,v in selected_by_split.items()},
        'scope':'All interval/count metadata; bounded literal token/mask reads. No GPU, training, model outcomes or near-duplicate guarantee.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','suite','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--suite-sha256',required=True);a=p.parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent') or list(Path('/dev').glob('nvidia[0-9]*')):raise RuntimeError('Require project CPU-only container')
    a.output.mkdir(parents=True,exist_ok=False)
    names=['scripts/olmo_pilot_ordered_audit.py','tests/test_pilot_ordered_audit.py','scripts/olmo_pilot_ordered_data.py',
        'scripts/olmo_pilot_data_plan.py','cdrm/pretrained/packed_campaign_data.py','cdrm/pretrained/nextlat.py']
    sources={n:sha(ROOT/n) for n in names}
    for n in names:
        target=a.output/'source-snapshot'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
    start=time.monotonic()
    result=audit(a.corpus,a.suite,a.suite_sha256)
    require(sources=={n:sha(ROOT/n) for n in names},'Audit sources changed')
    result.update(sources=sources,seconds=time.monotonic()-start)
    with (a.output/'report.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result[k] for k in ('status','all_selected_chunks_counted','literal_sampled_chunks','seconds')}))


if __name__=='__main__':main()
