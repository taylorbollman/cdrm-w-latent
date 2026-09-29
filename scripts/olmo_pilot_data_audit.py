#!/usr/bin/env python3
"""Independent bounded raw→token, source, split and readiness-exclusion audit.

All metadata is counted; only sixteen hash-selected unique documents per source
are independently retokenized in the production CLI. No network/model/GPU work.
Ordered-panel membership/counts are a separate audit, not claimed here.
"""
from __future__ import annotations

import argparse
from array import array
from bisect import insort
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cdrm.pretrained import document_shards
from cdrm.pretrained.olmo_artifacts import OLMoNativeTokenizer
from scripts import olmo_pilot_data_plan as plan

SCHEMA='olmo-pilot-raw-token-audit-v1'
SAMPLE_NAMESPACE='cdrm-dolma-pilot-v1-raw-audit'
SAMPLES_PER_SOURCE=16


def raw_sha(value):return hashlib.sha256(value).hexdigest()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def signature(path):
    path=Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file():
        raise ValueError('Require a regular non-symlink audit input')
    stat=path.stat()
    return tuple(getattr(stat,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))


def pinned_json(path,expected):
    before=signature(path)
    if before[2]>64*1024**2:raise ValueError('Audit JSON authority exceeds64MiB')
    raw=Path(path).read_bytes()
    if raw_sha(raw)!=plan._pin(expected) or signature(path)!=before:
        raise ValueError('Audit authority bytes changed or differ')
    return json.loads(raw),before


def split_oracle(recipe,identity):
    value=int.from_bytes(hashlib.sha256(f'{recipe["split"]["seed"]}:{identity}'.encode()).digest(),'big')%100
    return 'train' if value<90 else 'dev' if value<95 else 'confirmation'


def sample_key(identity,document_key):
    return hashlib.sha256((SAMPLE_NAMESPACE+'\0'+identity).encode()).hexdigest(),document_key


def u16(tokens):
    if any(type(v) is not int or not 0<=v<50280 for v in tokens):raise ValueError('Tokenizer returned invalid native IDs')
    values=array('H',tokens)
    if sys.byteorder!='little':values.byteswap()
    return values.tobytes()


def compare_document(row,raw,stored,encode,*,eos_id=50279):
    """Literal complete-document oracle: never truncate/token-window raw text."""
    if not isinstance(raw,dict) or not isinstance(raw.get('text'),str) or raw.get('id')!=row['document_id']:
        raise ValueError('Raw document identity/text differs')
    tokens=list(encode(raw['text']));had_eos=bool(tokens and tokens[-1]==eos_id)
    if not had_eos:tokens.append(eos_id)
    encoded=u16(tokens)
    if (len(tokens)!=row['token_count'] or encoded!=stored or raw_sha(encoded)!=row['token_sha256']
            or raw_sha(encoded[:-2])!=row['content_token_sha256']
            or raw_sha(raw['text'].encode())!=row['text_sha256']
            or had_eos!=row['had_terminal_eos'] or tokens[:-1].count(eos_id)!=row['embedded_eos_count']):
        raise ValueError('Complete native retokenization/stored token metadata differs')
    return {'tokens':len(tokens),'text_sha256':row['text_sha256'],'content_token_sha256':row['content_token_sha256'],
            'token_sha256':row['token_sha256'],'embedded_eos_count':row['embedded_eos_count'],
            'had_terminal_eos':had_eos}


def _line_stream(path,maximum,expected,*,consume):
    before=signature(path);digest=hashlib.sha256();rows=0
    with Path(path).open('rb') as stream:
        while True:
            line=stream.readline(maximum+1)
            if not line:break
            if len(line)>maximum:raise ValueError('Raw/metadata line exceeds complete-record audit bound')
            rows+=1;digest.update(line);consume(rows,line)
    if digest.hexdigest()!=expected or signature(path)!=before:
        raise ValueError('Raw/metadata bytes differ or changed during audit')
    return rows


def source_row_coverage(metadata_rows,raw_rows,declared_rows):
    """Bind prepared metadata's EOF to actual retained raw EOF, including duplicates."""
    if (any(type(n) is not int or n<=0 for n in (metadata_rows,raw_rows,declared_rows))
            or metadata_rows!=raw_rows or raw_rows!=declared_rows):
        raise ValueError('Per-source prepared metadata/raw/extraction row coverage differs')


def audit_documents(corpus_dir,raw_root,*,recipe,acquisition,acquisition_sha256,source_authorities,
                    excluded_content_hashes,encode,scratch_dir,samples_per_source=SAMPLES_PER_SOURCE,
                    progress=None):
    """The production caller pins the native tokenizer before supplying encode.

    A literal callable permits small CPU oracles in tests; this function alone
    makes no tokenizer-file or ordered-data qualification claim.
    """
    plan.validate_recipe(recipe)
    if type(samples_per_source) is not int or not 1<=samples_per_source<=SAMPLES_PER_SOURCE:
        raise ValueError('Sample size must be in1..16 per source')
    excluded=plan.verify_exclusions(recipe,excluded_content_hashes)
    corpus,raw_root=Path(corpus_dir),Path(raw_root)
    manifest_raw=(corpus/'manifest.json').read_bytes();summary_unverified=json.loads(manifest_raw)
    if summary_unverified.get('completed') is not True or summary_unverified['tokens']>recipe['bounds']['candidate_tokens']:
        raise ValueError('Incomplete or over-budget candidate corpus')
    sources=acquisition['sources'];names=[r['name'] for r in sources]
    if len(names)!=len(set(names)) or set(names)!=set(source_authorities):raise ValueError('Source authority membership differs')
    if summary_unverified['rows']>len(names)*recipe['bounds']['max_documents_per_object']:
        raise ValueError('Candidate metadata exceeds declared row bound')
    if progress is not None:progress({'phase':'verify_complete_document_corpus','sources_complete':0})
    summary=document_shards.verify_document_shards(corpus)
    if summary!=summary_unverified:raise ValueError('Verified corpus summary changed')
    config=json.loads((corpus/'config.json').read_bytes())
    if [s['pin']['name'] for s in config['sources']]!=names or acquisition['recipe']!=recipe:
        raise ValueError('Source order/recipe differs from prepared corpus')
    if config['split_policy']!={'seed':recipe['split']['seed'],'weights':recipe['split']['weights']}:
        raise ValueError('Prepared split policy differs')
    if (any(config['tokenizer'][k]!=recipe['tokenizer'][k] for k in ('repo','revision','sha256'))
            or any(config[k]!=recipe['tokenizer'][k] for k in ('eos_id','vocab_size','token_dtype'))):
        raise ValueError('Prepared tokenizer differs')
    snapshots={}
    for path in corpus.rglob('*'):
        if path.is_file() and not path.name.startswith('.'):
            snapshots[path]=signature(path)
    count=defaultdict(Counter);samples={i:[] for i in range(len(sources))}
    database=sqlite3.connect(Path(scratch_dir)/'membership.sqlite')
    database.execute('PRAGMA cache_size=-4096');database.execute('PRAGMA temp_store=FILE')
    database.execute('CREATE TABLE seen(content TEXT PRIMARY KEY,doc_key TEXT UNIQUE,split TEXT,source_index INT)')
    try:
        for shard in summary['shards']:
            with (corpus/shard['path']/'documents.jsonl').open('rb') as stream:
                for raw in stream:
                    row=json.loads(raw);source=row['source_index'];identity=row['content_token_sha256']
                    if not 0<=source<len(sources) or row['split']!=split_oracle(recipe,identity):
                        raise ValueError('Independent metadata source/split mismatch')
                    stats=count[(names[source],row['split'])];stats['raw_rows']+=1
                    old=database.execute('SELECT doc_key,split FROM seen WHERE content=?',(identity,)).fetchone()
                    if row['kind']=='duplicate':
                        if old!=(row['duplicate_of'],row['split']):raise ValueError('Duplicate identity/split/first occurrence differs')
                        stats['duplicate_rows']+=1;continue
                    if row['kind']!='document' or old is not None:raise ValueError('Unique content stored twice or unknown row type')
                    database.execute('INSERT INTO seen VALUES(?,?,?,?)',(identity,row['document_key'],row['split'],source))
                    stats['documents']+=1;stats['tokens']+=row['token_count']
                    if identity in excluded:stats['excluded_documents']+=1;stats['excluded_tokens']+=row['token_count']
                    item=(sample_key(identity,row['document_key']),{**row,'shard':shard['path']})
                    insort(samples[source],item)
                    if len(samples[source])>samples_per_source:samples[source].pop()
        database.commit()
        if any(len(rows)!=samples_per_source for rows in samples.values()):
            raise ValueError('Too few unique documents for every declared source sample')
        if (sum(v['documents'] for v in count.values())!=summary['documents']
                or sum(v['tokens'] for v in count.values())!=summary['tokens']
                or sum(v['raw_rows'] for v in count.values())!=summary['rows']):
            raise ValueError('Independent metadata totals differ')
        sampled=[];input_pins={};aggregate=Counter()
        for index,source in enumerate(sources):
            name=source['name'];authority=source_authorities[name];pin=config['sources'][index]['pin']
            directory=raw_root/name
            raw_manifest,raw_sig=pinned_json(directory/'manifest.json',authority['extraction_manifest_sha256'])
            input_pins[str(directory/'manifest.json')]={'sha256':authority['extraction_manifest_sha256'],'size_bytes':raw_sig[2]}
            if (authority['source_pin']!=pin or authority['upstream_source']!=source
                    or authority['acquisition_plan_sha256']!=acquisition_sha256
                    or raw_manifest.get('schema')!='olmo-dolma-source-extract-v2'
                    or raw_manifest['config']['source']!=source
                    or raw_manifest['config']['upstream_manifest']['source_plan_sha256']!=acquisition_sha256
                    or raw_manifest['config']['tokenizer_sha256']!=recipe['tokenizer']['sha256']
                    or raw_manifest['remote']['etag']!=source['etag']
                    or raw_manifest['remote']['requested_url']!=source['url']
                    or raw_manifest['status']!='complete'
                    or raw_manifest.get('full_upstream_gzip_sha256_verified') is not False
                    or set(raw_manifest['files'])!={'raw.jsonl','source-lines.jsonl'}):
                raise ValueError('Retained raw source/acquisition authority differs')
            remote=authority['raw_object'];raw_pin=raw_manifest['files']['raw.jsonl']
            if (pin['sha256']!=raw_pin['sha256'] or remote['sha256']!=pin['sha256'] or remote['uri']!=pin['uri']
                    or remote['size_bytes']!=raw_pin['size_bytes'] or pin['revision']!=recipe['hf_revision']
                    or remote.get('verification')!=dict.fromkeys(('server_size','server_md5','sha256_metadata','download_sha256'),True)
                    or not isinstance(remote.get('generation'),str) or not remote['generation'].isdigit() or int(remote['generation'])<=0):
                raise ValueError('Raw retained byte/generation metadata differs')
            for key,maximum in (('tokens_including_terminal_eos',recipe['object_limits']['tokens']),
                                ('compressed_bytes_read',recipe['object_limits']['compressed_bytes'])):
                if type(raw_manifest[key]) is not int or not 0<raw_manifest[key]<=maximum:
                    raise ValueError('Raw source exceeds absolute bound')
            if raw_pin['size_bytes']>recipe['object_limits']['retained_raw_bytes']:
                raise ValueError('Retained raw bytes exceed source bound')
            aggregate['candidate_tokens']+=raw_manifest['tokens_including_terminal_eos']
            aggregate['compressed_bytes']+=raw_manifest['compressed_bytes_read'];aggregate['retained_raw_bytes']+=raw_pin['size_bytes']
            wanted={row['source_line']:row for _,row in samples[index]};mappings={};source_result=[]
            def mapping_line(ordinal,raw):
                item=json.loads(raw)
                if item['source_line']!=ordinal:raise ValueError('Original source-line mapping is not contiguous')
                if item.get('status')=='retained' and item.get('output_line') in wanted:
                    line=item['output_line']
                    if line in mappings or item['id']!=wanted[line]['document_id']:raise ValueError('Selected raw/source line identity differs')
                    mappings[line]=item['source_line']
            mp=raw_manifest['files']['source-lines.jsonl'];mapping_path=directory/'source-lines.jsonl'
            mapped_rows=_line_stream(mapping_path,recipe['bounds']['max_line_bytes'],mp['sha256'],consume=mapping_line)
            if set(mappings)!=set(wanted) or mapped_rows!=raw_manifest['source_rows'][1]:raise ValueError('Selected original source lines missing')
            def raw_line(ordinal,raw):
                if ordinal not in wanted:return
                row=wanted[ordinal];record=json.loads(raw)
                key=raw_sha(plan.canonical({'source':name,'line':ordinal,'id':row['document_id']}))
                if key!=row['document_key']:raise ValueError('Independent source document key differs')
                if len(raw)>recipe['bounds']['max_line_bytes']:raise ValueError('Whole document exceeds raw audit bound')
                with (corpus/row['shard']/'tokens.bin').open('rb') as tokens:
                    tokens.seek(2*row['token_offset']);stored=tokens.read(2*row['token_count'])
                match=compare_document(row,record,stored,encode)
                source_result.append({'source_name':name,'source_index':index,'raw_line':ordinal,
                    'upstream_source_line':mappings[ordinal],'document_id':row['document_id'],
                    'document_key':row['document_key'],'split':row['split'],'excluded_from_pilot':row['content_token_sha256'] in excluded,
                    'selection_hash':sample_key(row['content_token_sha256'],row['document_key'])[0],**match})
            raw_path=directory/'raw.jsonl'
            raw_rows=_line_stream(raw_path,recipe['bounds']['max_line_bytes'],raw_pin['sha256'],consume=raw_line)
            source_row_coverage(sum(v['raw_rows'] for (source_name,_),v in count.items() if source_name==name),
                                raw_rows,raw_manifest['documents'])
            if len(source_result)!=samples_per_source:
                raise ValueError('Raw row count or sampled comparison count differs')
            for filename,pin_row in raw_manifest['files'].items():
                path=directory/filename
                if signature(path)[2]!=pin_row['size_bytes']:raise ValueError('Raw file size differs')
                input_pins[str(path)]=pin_row
            if signature(directory/'manifest.json')!=raw_sig:raise ValueError('Raw source manifest changed during audit')
            sampled.extend(sorted(source_result,key=lambda r:r['selection_hash']))
            if progress is not None:progress({'phase':'retokenization','sources_complete':index+1,'sampled_documents':len(sampled),'last_source':name})
        for key,total in aggregate.items():
            if total>recipe['bounds'][key]:raise ValueError('Actual raw corpus exceeds global bound')
        if aggregate['candidate_tokens']<summary['tokens']:raise ValueError('Stored tokens exceed pre-dedup candidate tokens')
        if any(signature(path)!=before for path,before in snapshots.items()):raise ValueError('Corpus changed during sample audit')
        return {'schema':SCHEMA,'status':'passed','scope':'All metadata plus bounded independent whole-document retokenization; ordered suite is not audited here',
            'corpus_manifest_sha256':raw_sha(manifest_raw),'recipe_sha256':plan.recipe_sha256(recipe),
            'sample_namespace':SAMPLE_NAMESPACE,'samples_per_source':samples_per_source,'sampled_documents':len(sampled),
            'source_count':len(names),'documents':summary['documents'],'raw_rows':summary['rows'],'tokens':summary['tokens'],
            'counts_by_source_split':{name+'/'+split:dict(value) for (name,split),value in sorted(count.items())},
            'exact_content_split_intersections':0,'excluded_documents_present_in_candidate_corpus':sum(v['excluded_documents'] for v in count.values()),
            'exclusion_authority':recipe['exclusions'],'acquisition_totals':dict(aggregate),'raw_input_pins':input_pins,
            'sample_rows':sampled,'no_payload_truncation':True,'near_duplicate_policy':'not_run','original_pretraining_exposure':'unknown'}
    finally:database.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','raw-root','acquisition-plan','source-authorities','inventory','readiness-corpus','tokenizer','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('corpus-manifest','acquisition-plan','source-authorities'):
        parser.add_argument('--'+name+'-sha256',required=True)
    args=parser.parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent') or list(Path('/dev').glob('nvidia[0-9]*')):
        raise RuntimeError('Use the project GPU-disabled CPU container')
    if args.output_dir.exists():raise FileExistsError('Use a fresh audit output directory')
    acquisition,ap_sig=pinned_json(args.acquisition_plan,args.acquisition_plan_sha256)
    authorities,auth_sig=pinned_json(args.source_authorities,args.source_authorities_sha256)
    _,corpus_sig=pinned_json(args.corpus/'manifest.json',args.corpus_manifest_sha256)
    recipe=plan.validate_recipe(acquisition['recipe']);selected=plan.source_selection(recipe,args.inventory.read_bytes())
    if acquisition.get('recipe_sha256')!=plan.recipe_sha256(recipe):raise ValueError('Acquisition recipe digest differs')
    if (len(selected)!=37 or len(acquisition['sources'])!=37
            or any(any(actual.get(k)!=v for k,v in expected.items()) for expected,actual in zip(selected,acquisition['sources']))):
        raise ValueError('Actual acquisition differs from independently selected37 sources')
    excluded=plan.load_exclusions(args.readiness_corpus,recipe)
    tok_sig=signature(args.tokenizer)
    if sha(args.tokenizer)!=recipe['tokenizer']['sha256']:raise ValueError('Native tokenizer bytes differ')
    tokenizer=OLMoNativeTokenizer(args.tokenizer)
    sources={name:sha(ROOT/name) for name in ('scripts/olmo_pilot_data_audit.py','tests/test_pilot_data_audit.py',
        'scripts/olmo_pilot_data_plan.py','tests/test_pilot_data_plan.py',
        'scripts/olmo_pilot_ordered_data.py','tests/test_pilot_ordered_data.py',
        'cdrm/pretrained/document_shards.py','cdrm/pretrained/olmo_artifacts.py',
        'cdrm/pretrained/campaign_data.py','cdrm/pretrained/campaign_ingest.py','cdrm/pretrained/artifacts.py')}
    args.output_dir.mkdir(parents=True)
    def progress(row):
        value={'schema':SCHEMA,'status':'running','sources':sources,**row}
        temporary=args.output_dir/'.progress.partial'
        with temporary.open('wb') as stream:stream.write(plan.canonical(value));stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,args.output_dir/'progress.json')
    with tempfile.TemporaryDirectory(prefix='.audit-',dir=args.output_dir) as scratch:
        result=audit_documents(args.corpus,args.raw_root,recipe=recipe,acquisition=acquisition,
            acquisition_sha256=args.acquisition_plan_sha256,source_authorities=authorities,
            excluded_content_hashes=excluded,encode=tokenizer.encode,scratch_dir=scratch,progress=progress)
    for path,before in ((args.acquisition_plan,ap_sig),(args.source_authorities,auth_sig),
                        (args.corpus/'manifest.json',corpus_sig),(args.tokenizer,tok_sig)):
        if signature(path)!=before:raise ValueError('Pinned authority changed during audit')
    if any(sha(ROOT/name)!=pin for name,pin in sources.items()):raise ValueError('Audit sources changed')
    result.update(sources=sources,tokenizer_file_sha256=recipe['tokenizer']['sha256'],
        tokenizer_mode='pinned_native_graph_complete_documents',acquisition_plan_sha256=args.acquisition_plan_sha256,
        source_authorities_sha256=args.source_authorities_sha256)
    for name,pin in sources.items():
        destination=args.output_dir/'source-snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
        if sha(destination)!=pin:raise ValueError('Audit snapshot bytes differ')
    temporary=args.output_dir/'.report.partial'
    with temporary.open('xb') as stream:stream.write(plan.canonical(result));stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,args.output_dir/'report.json')
    print(json.dumps({k:result[k] for k in ('status','source_count','sampled_documents','documents','tokens')}))


if __name__=='__main__':main()
