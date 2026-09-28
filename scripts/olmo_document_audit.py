#!/usr/bin/env python3
"""Bounded CPU audit of real document preparation, resume parity and length mix."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.document_shards import verify_document_shards
from cdrm.pretrained.olmo_artifacts import OLMoNativeTokenizer
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_document_extract import digest
from scripts.olmo_prepare_documents import load_sources


def inventory(root):
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not any(part.startswith('.') for part in p.relative_to(root).parts)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--source-plan',type=Path,default=ROOT/'configs/data/dolma-v1_5-readiness-coverage.json')
    parser.add_argument('--cloud-root',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    summary=verify_document_shards(args.prepared)
    reference=verify_document_shards(args.reference)
    if not summary['completed'] or summary != reference or inventory(args.prepared)!=inventory(args.reference):
        raise AssertionError('Resumed corpus differs from uninterrupted preparation')
    if summary['rows']>50_000 or summary['tokens']>20_000_000:
        raise ValueError('Readiness audit exceeds bounded slice scope')
    sources,provenance=load_sources(args.source_plan,args.raw_dir,args.cloud_root)
    tokenizer=OLMoNativeTokenizer(ROOT/'.runtime/olmo1b-step60000/artifacts/native/tokenizer.json')
    records={}
    source_stats={source.pin.name:Counter() for source in sources}
    split_stats=defaultdict(Counter)
    lengths=[]
    intervals=(32,128,512,1024,4096,16384)
    histogram=Counter()
    handles={}
    try:
        for shard in summary['shards']:
            directory=args.prepared/shard['path']
            handles[shard['path']]=(directory/'tokens.bin').open('rb')
            for line in (directory/'documents.jsonl').read_text().splitlines():
                row=json.loads(line); key=(row['source_index'],row['source_line'])
                if key in records: raise AssertionError('Duplicate source line')
                records[key]=(row,shard['path'])
        audited=0
        for source_index,source in enumerate(sources):
            stats=source_stats[source.pin.name]
            with source.path.open() as raw:
                for line_number,line in enumerate(raw,1):
                    raw_row=json.loads(line); row,shard=records[(source_index,line_number)]
                    ids=tokenizer.encode(raw_row['text'])
                    if not ids or ids[-1]!=tokenizer.eos_token_id: ids.append(tokenizer.eos_token_id)
                    token_bytes=np.asarray(ids,dtype='<u2').tobytes()
                    if (hashlib.sha256(token_bytes).hexdigest()!=row['token_sha256']
                            or hashlib.sha256(raw_row['text'].encode()).hexdigest()!=row['text_sha256']
                            or raw_row['id']!=row['document_id']):
                        raise AssertionError('Independent raw-document retokenization differs')
                    stats['raw_rows']+=1; audited+=1
                    if row['kind']=='duplicate':
                        stats['duplicate_rows']+=1; continue
                    handle=handles[shard];handle.seek(2*row['token_offset'])
                    if handle.read(len(token_bytes))!=token_bytes:
                        raise AssertionError('Stored uint16 document differs from native tokenizer')
                    length=len(ids)
                    stats['documents']+=1;stats['tokens']+=length
                    stats['embedded_eos']+=ids[:-1].count(tokenizer.eos_token_id)
                    stats['over_1024']+=int(length>1024)
                    windows=max(1,(max(length-1,1)+1022)//1023)
                    stats['isolated_windows_1024']+=windows
                    stats['isolated_presented_tokens_1024']+=length+windows-1
                    stats['isolated_capacity_1024']+=1024*windows
                    split_stats[row['split']]['documents']+=1
                    split_stats[row['split']]['tokens']+=length
                    lengths.append(length)
                    histogram[sum(length>=bound for bound in intervals)]+=1
        if audited!=summary['rows'] or len(records)!=audited:
            raise AssertionError('Preparation omitted/added source rows')
    finally:
        for handle in handles.values():handle.close()
    for stats in source_stats.values():
        stats['isolated_valid_utilization_1024']=stats['isolated_presented_tokens_1024']/stats['isolated_capacity_1024']
    report={'schema':'olmo-document-readiness-audit-v1','status':'passed',
            'scope':'Seven source prefixes; no production mixture, model execution or packed-model qualification',
            'summary':summary,'raw_extractions':provenance,
            'resume_all_files_exact':True,'resume_file_count':len(inventory(args.prepared)),
            'independently_retokenized_rows':audited,'sources':source_stats,'splits':split_stats,
            'lengths':{'min':min(lengths),'max':max(lengths),
                       'percentiles':{str(p):float(np.percentile(lengths,p)) for p in (50,90,95,99)},
                       'histogram_edges':list(intervals),'histogram_counts':[histogram[i] for i in range(7)]},
            'manifest_sha256':digest(args.prepared/'manifest.json'),
            'source_hashes':{p:digest(ROOT/p) for p in (
                'cdrm/pretrained/document_shards.py','scripts/olmo_document_extract.py',
                'scripts/olmo_document_retain.py','scripts/olmo_prepare_documents.py',
                'scripts/olmo_document_audit.py')}}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/'report.json',report)
    tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',output_dir=args.output_dir,
        group='olmo-document-readiness',name='dolma-v1_5-seven-source-document-preparation')
    tracker.start({'scope':report['scope'],'manifest_sha256':report['manifest_sha256'],
                   'sources':[s.pin.name for s in sources]})
    try:
        for step,(name,stats) in enumerate(source_stats.items()):
            tracker.log({f'data/{name}/{k}':v for k,v in stats.items()},step=step)
        tracker.summary({'data/documents':summary['documents'],'data/tokens':summary['tokens'],
                         'data/duplicates':summary['duplicate_rows'],'data/resume_exact':True,
                         'data/independent_tokenization_exact':True})
    finally:
        tracker.finish(succeeded=True)
        report['wandb']=tracker.record
        write_json(args.output_dir/'report.json',report)
    print(json.dumps({'status':report['status'],'documents':summary['documents'],'tokens':summary['tokens'],
                      'resume_file_count':report['resume_file_count'],'wandb':tracker.record['run_url']}),flush=True)


if __name__=='__main__':main()
