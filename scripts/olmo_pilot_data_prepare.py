#!/usr/bin/env python3
"""Bounded CPU acquisition and resumable global preparation of a pinned pilot.

Successful extracts obey the declared corpus caps. Failed attempts and SDK
retries are recorded separately; these caps do not bound lifetime network traffic.
There is no automatic extraction retry, source substitution, or GPU execution.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import olmo_document_extract as extractor
from scripts import olmo_document_retain as retainer
from cdrm.pretrained import document_shards as shards
from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource,SplitPolicy

SCHEMA='olmo-pilot-data-preparation-v1'
COUNT=37
URL_ROOT='https://olmo-data.org/dolma-v1_5r1/'
STRATUM_COUNTS={'books':3,'c4':4,'cc_en_head':4,'cc_en_middle':4,'cc_en_tail':8,'pes2o':4,'reddit':4,'stack':4,'wiki':2}
MIB=1024**2
POLICY={'target_tokens_per_source':8*MIB,'max_tokens_per_source':10*MIB,
    'max_compressed_bytes_per_source':110*MIB,'max_raw_bytes_per_source':221*MIB,
    'max_line_bytes':64*MIB,'max_documents_per_source':1_000_000,'extraction_timeout_seconds':900,
    'max_total_tokens':384*MIB,'max_total_compressed_bytes':4*1024**3,'max_total_raw_bytes':8*1024**3,
    'max_documents_per_shard':65536,'target_tokens_per_shard':8*MIB,'max_new_shards':4,
    'split_seed':20260929,'split_weights':[['train',90],['dev',5],['confirmation',5]],
    'budget_scope':'complete successful extracts; failed attempts/retries are separate, not lifetime traffic limits'}
CLOUD_RUN=retainer.PREFIX_ROOT+'pilot-20260929-v1'
TOKENIZER=ROOT/'.runtime/olmo1b-step60000/artifacts/native/tokenizer.json'


def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def regular_path(path):
    path=Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Reject traversal and symlink data/evidence paths')
    return path


def durable(path,value,*,once=False):
    return durable_bytes(path,canonical(value),once=once)


def durable_bytes(path,raw,*,once=False):
    path=regular_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and once:
        if path.read_bytes()!=raw:raise ValueError('Existing immutable authority differs')
        return
    temporary=path.with_name('.'+path.name+'.partial')
    with temporary.open('wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    if once:os.link(temporary,path);temporary.unlink()
    else:os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def source_hashes():
    names=('scripts/olmo_pilot_data_prepare.py','tests/test_pilot_data_prepare.py',
        'scripts/olmo_document_extract.py','scripts/olmo_document_retain.py',
        'cdrm/pretrained/document_shards.py','cdrm/pretrained/campaign_ingest.py',
        'cdrm/pretrained/campaign_data.py','cdrm/pretrained/olmo_artifacts.py')
    return {name:extractor.digest(ROOT/name) for name in names}


def load_plan(path,expected_sha256):
    path=regular_path(path)
    raw=path.read_bytes()
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256) or hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Source-plan bytes differ from independent pin')
    plan=json.loads(raw);sources=plan.get('sources');recipe=plan.get('recipe')
    if not isinstance(recipe,dict):raise ValueError('Require the selected pilot recipe inside its source plan')
    revision=recipe.get('hf_revision')
    if (recipe.get('release'),recipe.get('hf_repo'),revision)!=('v1_5','allenai/dolma','7f48140530a023e9ea4c5cfb141160922727d4d3'):
        raise ValueError('Plan release/repository/revision authority differs')
    recipe_raw=(json.dumps(recipe,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
    if plan.get('recipe_sha256')!=hashlib.sha256(recipe_raw).hexdigest():raise ValueError('Recipe semantic SHA256 differs')
    if (recipe.get('selection',{}).get('objects_per_stratum')!=STRATUM_COUNTS
            or recipe['selection'].get('url_namespace')!='cdrm-dolma-pilot-v1-url'
            or recipe['selection'].get('policy')!='hash_rank_urls_then_complete_record_prefix'):
        raise ValueError('Selected source-stratum/hash-prefix policy differs')
    expected_tokenizer={'repo':extractor.REPO_ID,'revision':extractor.REVISION,
        'sha256':extractor.FILE_SPECS['tokenizer.json'][1],'eos_id':50279,'pad_id':1,
        'vocab_size':50280,'token_dtype':'uint16_le'}
    if recipe.get('tokenizer')!=expected_tokenizer:raise ValueError('Native tokenizer identity/ID contract differs')
    expected_bounds={'candidate_tokens':POLICY['max_total_tokens'],'compressed_bytes':POLICY['max_total_compressed_bytes'],
        'retained_raw_bytes':POLICY['max_total_raw_bytes'],'max_line_bytes':POLICY['max_line_bytes'],
        'max_documents_per_object':POLICY['max_documents_per_source']}
    expected_object={'tokens':POLICY['max_tokens_per_source'],'compressed_bytes':POLICY['max_compressed_bytes_per_source'],
        'retained_raw_bytes':POLICY['max_raw_bytes_per_source'],'timeout_seconds':POLICY['extraction_timeout_seconds']}
    if (recipe.get('bounds')!=expected_bounds or recipe.get('object_limits')!=expected_object
            or recipe.get('selection',{}).get('candidate_tokens_per_object')!=POLICY['target_tokens_per_source']
            or recipe.get('split',{}).get('weights')!=POLICY['split_weights']
            or recipe.get('split',{}).get('seed')!=POLICY['split_seed']
            or recipe.get('tokenizer',{}).get('sha256')!=extractor.FILE_SPECS['tokenizer.json'][1]):
        raise ValueError('Pinned recipe bounds/split/tokenizer differ from executable preparation policy')
    if not isinstance(sources,list) or len(sources)!=COUNT:raise ValueError('Require the complete 37-object pilot plan')
    if any(not isinstance(row,dict) for row in sources):raise ValueError('Source records must be mappings')
    if (len({row.get('name') for row in sources})!=COUNT or len({row.get('url') for row in sources})!=COUNT):
        raise ValueError('Source names/URLs must be distinct')
    counts=dict.fromkeys(STRATUM_COUNTS,0)
    for row in sources:
        if (not re.fullmatch('[a-z0-9_-]+',row.get('name',''))
                or not row.get('url','').startswith(URL_ROOT)
                or not isinstance(row.get('etag'),str) or not row['etag']
                or type(row.get('upstream_size_bytes')) is not int or row['upstream_size_bytes']<=0):
            raise ValueError('Every source requires official URL, name, pinned ETag and size')
        parts=row['url'][len(URL_ROOT):].split('/');stratum=row.get('stratum')
        if (len(parts)!=2 or stratum not in counts or parts[0]!=stratum
                or not re.fullmatch(r'[A-Za-z0-9_.-]+\.json\.gz',parts[1])
                or row.get('family')!=('common_crawl' if stratum.startswith('cc_') else stratum)
                or row['name']!=f'{stratum}-{counts[stratum]:02d}'
                or row.get('selection_sha256')!=hashlib.sha256(('cdrm-dolma-pilot-v1-url\0'+row['url']).encode()).hexdigest()):
            raise ValueError('Source stratum/path/name/selection identity differs')
        counts[stratum]+=1
    if counts!=STRATUM_COUNTS:raise ValueError('Selected source-stratum counts differ')
    for per,total in [('max_tokens_per_source','max_total_tokens'),
                      ('max_compressed_bytes_per_source','max_total_compressed_bytes'),
                      ('max_raw_bytes_per_source','max_total_raw_bytes')]:
        if COUNT*POLICY[per]>POLICY[total]:raise ValueError('Per-object caps exceed complete pilot bound')
    return plan


def upstream(plan_sha,sources):
    return {'source_plan_sha256':plan_sha,'release':'v1_5','tokenizer_repo':extractor.REPO_ID,
        'tokenizer_revision':extractor.REVISION,'preparation_policy_sha256':digest(POLICY),
        'acquisition_implementation_sha256':digest(sources)}


def extraction_options(plan_sha,sources):
    return {'token_target':POLICY['target_tokens_per_source'],'max_documents':POLICY['max_documents_per_source'],
        'max_compressed_bytes':POLICY['max_compressed_bytes_per_source'],'max_line_bytes':POLICY['max_line_bytes'],
        'upstream_manifest':upstream(plan_sha,sources),'tokenizer_sha':extractor.FILE_SPECS['tokenizer.json'][1]}


class TokenBudget:
    def __init__(self,tokenizer,maximum):
        self.tokenizer,self.maximum=tokenizer,maximum;self.count=0;self.eos_token_id=tokenizer.eos_token_id
    def encode(self,text):
        values=self.tokenizer.encode(text)
        count=len(values)+int(not values or values[-1]!=self.eos_token_id)
        if self.count+count>self.maximum:raise ValueError('Complete document would exceed candidate-token cap; no truncation')
        self.count+=count
        return values


@contextmanager
def file_size_limit(maximum):
    before=resource.getrlimit(resource.RLIMIT_FSIZE)
    if before[0]!=resource.RLIM_INFINITY and before[0]<maximum:
        raise ValueError('Existing process file-size limit is below acquisition policy')
    resource.setrlimit(resource.RLIMIT_FSIZE,(maximum,before[1]))
    try:yield
    finally:resource.setrlimit(resource.RLIMIT_FSIZE,before)


def extract_worker(plan_path,plan_sha,raw_root,source_name,tokenizer_path):
    plan=load_plan(plan_path,plan_sha);sources=source_hashes()
    tokenizer_path=regular_path(tokenizer_path)
    if extractor.digest(tokenizer_path)!=extractor.FILE_SPECS['tokenizer.json'][1]:raise ValueError('Native tokenizer changed')
    selected=[row for row in plan['sources'] if row['name']==source_name]
    if len(selected)!=1:raise ValueError('Source is not in the frozen plan')
    tokenizer=TokenBudget(extractor.OLMoNativeTokenizer(tokenizer_path),POLICY['max_tokens_per_source'])
    with file_size_limit(POLICY['max_raw_bytes_per_source']):
        report=extractor.extract_source(selected[0],regular_path(raw_root),tokenizer=tokenizer,
            **extraction_options(plan_sha,sources))
    if source_hashes()!=sources or extractor.digest(plan_path)!=plan_sha:raise ValueError('Acquisition authority changed')
    return report


def validate_raw(source,raw_root,plan_sha,sources):
    directory=regular_path(Path(raw_root)/source['name'])
    report=json.loads((directory/'manifest.json').read_text())
    options=extraction_options(plan_sha,sources)
    config={'source':source,'upstream_manifest':options['upstream_manifest'],
        'tokenizer_sha256':options['tokenizer_sha'],'target_tokens':options['token_target'],
        'max_documents':options['max_documents'],'max_compressed_bytes':options['max_compressed_bytes'],
        'max_line_bytes':options['max_line_bytes'],
        'policy':'complete-document source prefix; no truncation; stop after token target',
        'empty_text_policy':'skip empty/whitespace-only string with audited source line; reject nonstring'}
    if (report.get('schema')!='olmo-dolma-source-extract-v2' or report.get('status')!='complete'
            or report.get('config')!=config or report.get('remote',{}).get('etag')!=source['etag']
            or report['remote'].get('requested_url')!=source['url']
            or report.get('full_upstream_gzip_sha256_verified') is not False):
        raise ValueError('Committed raw extraction authority/configuration differs')
    retainer.collect_committed(directory)
    for value,maximum in [(report['tokens_including_terminal_eos'],POLICY['max_tokens_per_source']),
        (report['compressed_bytes_read'],POLICY['max_compressed_bytes_per_source']),
        (report['files']['raw.jsonl']['size_bytes'],POLICY['max_raw_bytes_per_source'])]:
        if type(value) is not int or not 0<value<=maximum:raise ValueError('Committed raw extraction exceeds bound')
    if set(report['files'])!={'raw.jsonl','source-lines.jsonl'}:raise ValueError('Raw extraction inventory differs')
    return report


def bind_evidence(evidence,plan_path,plan_sha,sources):
    evidence=regular_path(evidence);evidence.mkdir(parents=True,exist_ok=True)
    durable(evidence/'authority.json',{'schema':SCHEMA,'plan_sha256':plan_sha,'sources':sources,
        'policy':POLICY,'cloud_run':CLOUD_RUN},once=True)
    copy=evidence/'source-plan.json'
    if copy.exists():
        if extractor.digest(copy)!=plan_sha:raise ValueError('Evidence source-plan changed')
    else:
        shutil.copyfile(plan_path,copy)
        if extractor.digest(copy)!=plan_sha:raise ValueError('Source-plan changed while copying')
    for name,pin in sources.items():
        dest=evidence/'source-snapshot'/name
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        if extractor.digest(dest)!=pin:raise ValueError('Preparation source snapshot differs')
    return evidence


def verified_receipt(directory,evidence,stage,*,bucket=None,require_existing=False):
    directory=regular_path(directory);evidence=regular_path(evidence)
    members=retainer.collect_committed(directory);prefix=CLOUD_RUN+'/'+stage
    receipt_path=evidence/'receipts'/(stage+'.json');seal=receipt_path.with_suffix('.sha256.json')
    if not receipt_path.exists():
        if require_existing:raise ValueError('Raw/shard progress must be retained before continuing')
        retainer.retain(SimpleNamespace(input_dir=directory,prefix=prefix,receipt=receipt_path,dry_run=False),bucket=bucket)
    receipt=json.loads(receipt_path.read_text())
    if (receipt.get('schema')!=retainer.SCHEMA or receipt.get('status')!='verified'
            or receipt.get('prefix')!=prefix or receipt.get('publication_marker')!=prefix+'/manifest.json'
            or receipt.get('local_files_deleted') is not False):raise ValueError('Invalid document retention receipt')
    expected={prefix+'/'+name:pin for _,name,pin in members}
    objects=receipt.get('objects',[])
    if len(objects)!=len(expected) or {row['uri'] for row in objects}!=set(expected):raise ValueError('Receipt inventory differs')
    for row in objects:
        if (any(row.get(k)!=v for k,v in expected[row['uri']].items()) or not str(row.get('generation','')).isdigit()
                or int(row['generation'])<1 or row.get('verification')!=dict.fromkeys(
                    ('server_size','server_md5','sha256_metadata','download_sha256'),True)):
            raise ValueError('Retained generation or byte verification differs')
    durable(seal,{'receipt_sha256':extractor.digest(receipt_path)},once=True)
    return {'receipt_path':str(receipt_path),'receipt_sha256':extractor.digest(receipt_path),'receipt':receipt}


def raw_inventory(plan,raw_root,plan_sha,sources,*,require_all=False):
    rows=[]
    for source in plan['sources']:
        if not (Path(raw_root)/source['name']).exists():
            if require_all:raise ValueError('All 37 raw objects are required for global deduplication')
            continue
        report=validate_raw(source,raw_root,plan_sha,sources)
        rows.append({'source':source['name'],'manifest_sha256':extractor.digest(Path(raw_root)/source['name']/'manifest.json'),
            'candidate_tokens':report['tokens_including_terminal_eos'],'compressed_bytes':report['compressed_bytes_read'],
            'raw_bytes':report['files']['raw.jsonl']['size_bytes'],'documents':report['documents']})
    totals={key:sum(row[key] for row in rows) for key in ('candidate_tokens','compressed_bytes','raw_bytes','documents')}
    for key,limit in [('candidate_tokens','max_total_tokens'),('compressed_bytes','max_total_compressed_bytes'),('raw_bytes','max_total_raw_bytes')]:
        if totals[key]>POLICY[limit]:raise ValueError('Committed corpus exceeds global acquisition bound')
    return {'sources':rows,'totals':totals,'complete':len(rows)==COUNT}


@contextmanager
def exclusive(root,name):
    root=regular_path(root);root.mkdir(parents=True,exist_ok=True)
    path=regular_path(root/name)
    with path.open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def acquire(plan_path,plan_sha,raw_root,evidence,source_name,tokenizer_path=TOKENIZER,*,worker=None,bucket=None):
    plan=load_plan(plan_path,plan_sha);sources=source_hashes()
    evidence=bind_evidence(evidence,plan_path,plan_sha,sources)
    raw_root=regular_path(raw_root)
    with exclusive(raw_root,'.pilot-acquire.lock'):
        selected=[row for row in plan['sources'] if row['name']==source_name]
        if len(selected)!=1:raise ValueError('Select exactly one frozen source')
        index=plan['sources'].index(selected[0])
        for prior in plan['sources'][:index]:
            validate_raw(prior,raw_root,plan_sha,sources)
            verified_receipt(raw_root/prior['name'],evidence,'raw-'+prior['name'],require_existing=True)
        inventory=raw_inventory(plan,raw_root,plan_sha,sources)
        if not (raw_root/source_name).exists():
            attempts=evidence/'attempts';attempts.mkdir(exist_ok=True)
            number=len(list(attempts.glob(source_name+'-*.json')))+1
            attempt_path=attempts/f'{source_name}-{number:04d}.json'
            attempt={'source':source_name,'attempt':number,'status':'running','plan_sha256':plan_sha,
                'started_utc':datetime.now(timezone.utc).isoformat(),'timeout_seconds':POLICY['extraction_timeout_seconds'],
                'failed_attempt_traffic_counted_in_success_caps':False}
            durable(attempt_path,attempt,once=True);started=time.monotonic()
            try:
                if worker is not None:worker(plan_path,plan_sha,raw_root,source_name,tokenizer_path)
                else:
                    command=[sys.executable,str(Path(__file__)),'extract-worker','--plan',str(plan_path),
                        '--plan-sha256',plan_sha,'--raw-root',str(raw_root),'--source',source_name,'--tokenizer',str(tokenizer_path)]
                    with (attempts/f'{source_name}-{number:04d}.log').open('xb') as log:
                        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=POLICY['extraction_timeout_seconds'])
                validate_raw(selected[0],raw_root,plan_sha,sources)
                attempt['status']='committed'
            except BaseException as error:
                attempt.update(status='failed',error=f'{type(error).__name__}: {error}')
                raise
            finally:
                attempt['elapsed_seconds']=time.monotonic()-started;durable(attempt_path,attempt)
        retained=verified_receipt(raw_root/source_name,evidence,'raw-'+source_name,bucket=bucket)
        inventory=raw_inventory(plan,raw_root,plan_sha,sources)
        if source_hashes()!=sources or extractor.digest(plan_path)!=plan_sha:raise ValueError('Acquisition authority changed')
        result={'schema':SCHEMA,'stage':'acquire','status':'retained','source':source_name,
            'raw':inventory,'retention':retained,'sources':sources,'plan_sha256':plan_sha,'policy':POLICY}
        durable(evidence/'acquisition-progress.json',result)
        return result


def metadata_stage(directory,payloads):
    directory=regular_path(directory);directory.mkdir(parents=True,exist_ok=True)
    files={}
    for name,value in payloads.items():
        if isinstance(value,bytes):durable_bytes(directory/name,value,once=True)
        else:durable(directory/name,value,once=True)
        pin=retainer.file_digest(directory/name)
        files[name]={'sha256':pin['sha256'],'size_bytes':pin['size_bytes']}
    durable(directory/'manifest.json',{'schema':'olmo-pilot-data-metadata-v1','status':'complete','files':files},once=True)
    return directory


def portable_receipt(value):
    # Local evidence relocation must not change immutable prepared-data identity.
    return {key:value[key] for key in ('receipt_sha256','receipt')}


def prepare(plan_path,plan_sha,raw_root,output_dir,evidence,acquisition_evidence,tokenizer_path=TOKENIZER,*,bucket=None):
    plan=load_plan(plan_path,plan_sha);sources=source_hashes()
    evidence=bind_evidence(evidence,plan_path,plan_sha,sources)
    raw_root,output_dir=regular_path(raw_root),regular_path(output_dir)
    with exclusive(output_dir,'.pilot-prepare.lock'):
        raw=raw_inventory(plan,raw_root,plan_sha,sources,require_all=True)
        raw_receipts=[verified_receipt(raw_root/row['name'],acquisition_evidence,'raw-'+row['name'],require_existing=True)
                      for row in plan['sources']]
        revision=plan['recipe']['hf_revision']
        source_authorities={}
        for row,retained_raw in zip(plan['sources'],raw_receipts):
            objects=retained_raw['receipt']['objects']
            raw_object=next(obj for obj in objects if obj['uri'].endswith('/raw.jsonl'))
            manifest_object=next(obj for obj in objects if obj['uri'].endswith('/manifest.json'))
            source_authorities[row['name']]={'source_pin':{'name':row['name'],'uri':raw_object['uri'],
                'revision':revision,'sha256':raw_object['sha256']},'upstream_source':row,
                'extraction_manifest_sha256':manifest_object['sha256'],'raw_object':raw_object,
                'acquisition_plan_sha256':plan_sha,'scope':'retained complete-document raw prefix, not full upstream gzip'}
        local=[LocalJSONLSource(SourcePin(**source_authorities[row['name']]['source_pin']),
                    raw_root/row['name']/'raw.jsonl') for row in plan['sources']]
        if extractor.digest(regular_path(tokenizer_path))!=extractor.FILE_SPECS['tokenizer.json'][1]:raise ValueError('Native tokenizer changed')
        retained={}
        def retain_shards(summary):
            for shard in summary['shards']:
                retained[shard['path']]=verified_receipt(output_dir/shard['path'],evidence,shard['path'],bucket=bucket)
        if (output_dir/'config.json').exists():retain_shards(shards.verify_document_shards(output_dir))
        summary=shards.prepare_document_shards(local,output_dir,tokenizer_path=tokenizer_path,
            split_policy=SplitPolicy(POLICY['split_seed'],tuple(map(tuple,POLICY['split_weights']))),
            max_documents_per_shard=POLICY['max_documents_per_shard'],target_tokens_per_shard=POLICY['target_tokens_per_shard'],
            max_record_bytes=POLICY['max_line_bytes'],max_new_shards=POLICY['max_new_shards'])
        config_bytes=(output_dir/'config.json').read_bytes()
        config_stage=metadata_stage(evidence/'config',{'config.json':config_bytes,'source-plan.json':Path(plan_path).read_bytes(),
            'raw-receipts.json':[portable_receipt(row) for row in raw_receipts],
            'source-authorities.json':source_authorities,'policy.json':POLICY})
        config_receipt=verified_receipt(config_stage,evidence,'config',bucket=bucket)
        retain_shards(summary)
        final=None
        if summary['completed']:
            stage=metadata_stage(evidence/'root-summary',{'tokenized-manifest.json':(output_dir/'manifest.json').read_bytes(),'config.json':config_bytes,
                'source-plan.json':Path(plan_path).read_bytes(),'raw-receipts.json':[portable_receipt(row) for row in raw_receipts],
                'shard-receipts.json':{key:portable_receipt(row) for key,row in retained.items()},
                'source-authorities.json':source_authorities,'config-receipt.json':portable_receipt(config_receipt)})
            final=verified_receipt(stage,evidence,'root-summary',bucket=bucket)
        if source_hashes()!=sources or extractor.digest(plan_path)!=plan_sha:raise ValueError('Preparation authority changed')
        result={'schema':SCHEMA,'stage':'prepare','status':'complete' if summary['completed'] else 'retained_partial',
            'sources':sources,'plan_sha256':plan_sha,'policy':POLICY,'raw':raw,'tokenized':summary,
            'source_authorities':source_authorities,
            'config_retention':config_receipt,'shard_retention':retained,'final_retention':final,
            'resume':'Repeat same full-source configuration; committed shards retained before further preparation'}
        durable(evidence/'preparation-progress.json',result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('acquire','prepare','extract-worker'))
    for name in ('plan','raw-root','tokenizer','evidence-dir','output-dir','acquisition-evidence'):
        parser.add_argument('--'+name,type=Path,required=name in ('plan','raw-root'),default=TOKENIZER if name=='tokenizer' else None)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--source')
    args=parser.parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent') or list(Path('/dev').glob('nvidia[0-9]*')):
        raise RuntimeError('Use the project GPU-disabled CPU container')
    if args.stage=='extract-worker':
        if args.source is None:parser.error('extract-worker requires --source')
        result=extract_worker(args.plan,args.plan_sha256,args.raw_root,args.source,args.tokenizer)
    else:
        if args.evidence_dir is None or not regular_path(args.evidence_dir).is_relative_to(ROOT):
            parser.error('Persistent project --evidence-dir is required')
        if args.stage=='acquire':
            if args.source is None:parser.error('acquire requires --source')
            result=acquire(args.plan,args.plan_sha256,args.raw_root,args.evidence_dir,args.source,args.tokenizer)
        else:
            if args.output_dir is None or args.acquisition_evidence is None:parser.error('prepare requires --output-dir and --acquisition-evidence')
            result=prepare(args.plan,args.plan_sha256,args.raw_root,args.output_dir,args.evidence_dir,args.acquisition_evidence,args.tokenizer)
    print(json.dumps({'stage':args.stage,'status':result.get('status'),'source':args.source}),flush=True)


if __name__=='__main__':main()
