#!/usr/bin/env python3
"""Stream one complete ordered pilot index suite to/from exact GCS generations.

Only index bytes are retained here: no corpus, model, tensor, SQL execution,
training, deletion, archives, or transfer before full local receipt validation.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.olmo_document_retain import file_digest, parse_prefix, _write_once, PREFIX_ROOT

SCHEMA='olmo-pilot-index-retention-v1'
RESTORE_SCHEMA='olmo-pilot-index-restore-v1'
MAX_BYTES=8*1024**3
MAX_FILES=1000
CHUNK_BYTES=8*1024**2
STRATA=('books','c4','cc_en_head','cc_en_middle','cc_en_tail','pes2o','reddit','stack','wiki')
PANELS=('train','dev-main','confirmation-main',*(f'{split}-source/{s}' for split in ('dev','confirmation') for s in STRATA))
FILES=frozenset(('manifest.json','catalog.sqlite',*(f'panels/{p}/{name}' for p in PANELS for name in ('manifest.json','documents.sqlite'))))
VERIFIED=dict.fromkeys(('server_size','server_md5','sha256_metadata','download_sha256'),True)


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def pin(value):
    if not isinstance(value,str) or re.fullmatch('[a-f0-9]{64}',value) is None:raise ValueError('Invalid SHA256')
    return value


def safe_path(path):
    path=Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('No traversal or symlink paths')
    return path


def signature(path):
    path=safe_path(path)
    if not path.is_file():raise ValueError('Require regular index file')
    s=path.stat()
    if s.st_nlink!=1:raise ValueError('Hard-linked index authorities are unsupported')
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def read_json(path,expected=None):
    before=signature(path)
    if before[2]>32*1024**2:raise ValueError('Index JSON exceeds32MiB')
    raw=Path(path).read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest()!=pin(expected):raise ValueError('JSON byte pin differs')
    if signature(path)!=before:raise ValueError('Index JSON changed')
    return json.loads(raw)


def member(name):
    p=PurePosixPath(name) if isinstance(name,str) else None
    if p is None or p.as_posix()!=name or name not in FILES:raise ValueError('Unexpected ordered-suite member')
    return name


def precommit_paths(root):
    expected=(FILES-{'manifest.json'})|{'manifest.json.partial'}
    directories={str(p) for n in expected for p in PurePosixPath(n).parents if str(p)!='.'}
    observed=set()
    for path in root.rglob('*'):
        path=safe_path(path);name=path.relative_to(root).as_posix()
        if path.is_dir():
            if name not in directories:raise ValueError('Unexpected recovery directory before commit')
        else:
            signature(path);observed.add(name)
    if observed!=expected:raise ValueError('Unexpected/missing recovery member before commit')


def ordered_names():
    return sorted(FILES,key=lambda n:(n=='manifest.json',n.endswith('/manifest.json'),n))


def validate_suite_manifests(suite,panels,inventory,expected):
    if (suite.get('schema')!='olmo-pilot-ordered-suite-v1' or suite.get('round_id')!=0
            or set(suite.get('panels',{}))!=set(PANELS) or set(inventory)!=FILES
            or inventory['manifest.json']['sha256']!=pin(expected)
            or suite.get('identity_sha256')!=digest({k:v for k,v in suite.items() if k!='identity_sha256'})):
        raise ValueError('Ordered suite schema/identity/member authority differs')
    catalog=suite['selection_authority']
    if catalog!={'path':'catalog.sqlite',**{k:inventory['catalog.sqlite'][k] for k in ('size_bytes','sha256')}}:
        raise ValueError('Catalog bytes differ from suite authority')
    common=('recipe','recipe_sha256','round_id','selection_authority','source_selection','acquisition_authority',
        'exclusion_authority','corpus_manifest_sha256','corpus_config_sha256','corpus_files','tokenizer','pad_id',
        'eos_id','vocab_size','token_dtype')
    for name in PANELS:
        item=suite['panels'][name];panel=panels[name];path='panels/'+name
        if (item['path']!=path or item['manifest_sha256']!=inventory[path+'/manifest.json']['sha256']
                or panel.get('schema')!='olmo-pilot-ordered-data-v1' or panel.get('panel')!=name
                or panel.get('split')!=('train' if name=='train' else name.split('-',1)[0])
                or any(panel.get(k)!=suite.get(k) for k in common)
                or panel.get('catalog_authority')!=catalog
                or panel.get('identity_sha256')!=digest({k:v for k,v in panel.items() if k not in ('identity_sha256','index')})
                or panel.get('index')!={'path':'documents.sqlite',**{k:inventory[path+'/documents.sqlite'][k] for k in ('size_bytes','sha256')}}
                or any(item.get(a)!=panel.get(b) for a,b in (('identity_sha256','identity_sha256'),
                    ('tokens','total_tokens'),('chunks','total_chunks'),('counts','counts'),('stratum_quotas','stratum_quotas')))):
            raise ValueError('Ordered panel identity/member authority differs: '+name)


def collect_suite(directory,expected):
    root=safe_path(directory)
    if not root.is_dir():raise ValueError('Require complete ordered-suite directory')
    paths={};signatures={};total=0
    allowed_dirs={str(PurePosixPath(n).parent) for n in FILES}
    allowed_dirs|={str(p) for n in FILES for p in PurePosixPath(n).parents if str(p)!='.'}
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs:
            path=safe_path(Path(directory)/name)
            if path.relative_to(root).as_posix() not in allowed_dirs:raise ValueError('Unexpected index directory')
        for name in files:
            path=Path(directory)/name;relative=member(path.relative_to(root).as_posix())
            before=signature(path);total+=before[2]
            if total>MAX_BYTES or len(paths)>=MAX_FILES:raise ValueError('Index suite exceeds bounded storage size')
            paths[relative]=path;signatures[relative]=before
    if set(paths)!=FILES:raise ValueError('Missing ordered-suite files')
    inventory={n:file_digest(p) for n,p in paths.items()}
    suite=read_json(root/'manifest.json',expected)
    panels={n:read_json(root/'panels'/n/'manifest.json',suite['panels'][n]['manifest_sha256']) for n in PANELS}
    validate_suite_manifests(suite,panels,inventory,expected)
    if any(signature(paths[n])!=v for n,v in signatures.items()):raise ValueError('Index suite changed during inventory')
    return suite,inventory,signatures


def validate_record(row):
    size=row.get('size_bytes');generation=row.get('generation')
    if type(size) is not int or not 0<size<=MAX_BYTES:raise ValueError('Invalid retained size')
    pin(row.get('sha256'))
    if not isinstance(generation,str) or not generation.isascii() or not generation.isdigit() or int(generation)<1 or str(int(generation))!=generation:
        raise ValueError('Require exact positive generation')
    try:md5=base64.b64decode(row.get('md5_base64',''),validate=True)
    except (TypeError,ValueError):raise ValueError('Invalid MD5') from None
    if len(md5)!=16 or row.get('verification')!=VERIFIED:raise ValueError('Missing retained MD5/verification')


class DigestSink(io.RawIOBase):
    """Append-only bounded streaming hash; optional file receives the same bytes."""
    def __init__(self,expected,stream=None):
        self.expected,self.stream=expected,stream;self.size=0;self.sha=hashlib.sha256();self.md5=hashlib.md5()
    def writable(self):return True
    def tell(self):return self.size
    def write(self,data):
        if self.size+len(data)>self.expected['size_bytes']:raise ValueError('Transfer exceeds declared bytes')
        if self.stream is not None and self.stream.write(data)!=len(data):raise OSError('Short local write')
        self.sha.update(data);self.md5.update(data);self.size+=len(data);return len(data)
    def finish(self):
        result={'size_bytes':self.size,'sha256':self.sha.hexdigest(),'md5_base64':base64.b64encode(self.md5.digest()).decode()}
        if any(result[k]!=self.expected[k] for k in result):raise ValueError('Streamed bytes differ')
        if self.stream is not None:self.stream.flush();os.fsync(self.stream.fileno())


def verify_blob(blob,expected):
    if (blob.size!=expected['size_bytes'] or blob.md5_hash!=expected['md5_base64']
            or (blob.metadata or {}).get('sha256')!=expected['sha256']
            or (blob.metadata or {}).get('artifact_schema')!=SCHEMA or blob.content_encoding is not None
            or not str(blob.generation).isdigit() or int(blob.generation)<1):
        raise ValueError('Exact cloud object metadata differs')


def upload_verified(bucket,key,path,expected):
    from google.api_core.exceptions import PreconditionFailed
    if bucket.name!='fast-chunks':raise ValueError('Wrong index bucket')
    root_key=PREFIX_ROOT[5:].split('/',1)[1]
    if not key.startswith(root_key):raise ValueError('Wrong index namespace')
    parts=key[len(root_key):].split('/',2)
    if len(parts)!=3:raise ValueError('Require run/stage/member key')
    parse_prefix(PREFIX_ROOT+'/'.join(parts[:2]));member(parts[2])
    if file_digest(path)!=expected:raise ValueError('Local index changed before upload')
    blob=bucket.get_blob(key)
    if blob is None:
        blob=bucket.blob(key);blob.metadata={'sha256':expected['sha256'],'artifact_schema':SCHEMA}
        try:blob.upload_from_filename(str(path),if_generation_match=0,checksum='md5')
        except PreconditionFailed:
            blob=bucket.get_blob(key)
            if blob is None:raise
        blob.reload()
    verify_blob(blob,expected);generation=int(blob.generation);blob.chunk_size=CHUNK_BYTES
    sink=DigestSink(expected)
    blob.download_to_file(sink,if_generation_match=generation,raw_download=True,checksum=None)
    sink.finish()
    if file_digest(path)!=expected:raise ValueError('Local index changed during upload')
    return {'uri':f'gs://{bucket.name}/{key}','generation':str(generation),**expected,'verification':dict(VERIFIED)}


def retain(suite_dir,suite_manifest_sha256,prefix,receipt_path,*,bucket=None):
    bucket_name,key=parse_prefix(prefix);prefix=prefix.rstrip('/')
    root=safe_path(suite_dir);receipt_path=safe_path(receipt_path)
    if receipt_path.is_relative_to(root):raise ValueError('Receipt must be outside index suite')
    suite,inventory,signatures=collect_suite(root,suite_manifest_sha256)
    if bucket is None:
        from google.cloud import storage
        bucket=storage.Client().bucket(bucket_name)
    if bucket.name!=bucket_name:raise ValueError('Provided bucket differs')
    objects=[];started=time.monotonic()
    for name in ordered_names():
        if any(signature(root/n)!=s for n,s in signatures.items()):raise ValueError('Index suite changed before publication')
        objects.append({'path':name,**upload_verified(bucket,key+'/'+name,root/name,inventory[name])})
    result={'schema':SCHEMA,'status':'verified','prefix':prefix,'suite_manifest_sha256':suite_manifest_sha256,
        'suite_identity_sha256':suite['identity_sha256'],'objects':objects,'publication_marker':prefix+'/manifest.json',
        'total_bytes':sum(r['size_bytes'] for r in objects),'local_files_deleted':False,
        'scope':'Complete ordered index bytes only; corpus objects and semantic model/executor qualification are separate.'}
    _write_once(receipt_path,result)
    return result


def load_receipt(path,expected):
    r=read_json(path,expected);parse_prefix(r.get('prefix',''))
    if (r.get('prefix')!=r.get('prefix','').rstrip('/') or r.get('schema')!=SCHEMA or r.get('status')!='verified' or r.get('local_files_deleted') is not False
            or r.get('publication_marker')!=r['prefix']+'/manifest.json'):
        raise ValueError('Require verified ordered-suite receipt')
    pin(r.get('suite_manifest_sha256'));pin(r.get('suite_identity_sha256'))
    records={}
    for obj in r.get('objects',[]):
        name=member(obj.get('path'))
        if name in records or obj.get('uri')!=r['prefix']+'/'+name:raise ValueError('Duplicate/escaped object')
        validate_record(obj);records[name]=obj
    if (set(records)!=FILES or len(records)>MAX_FILES or sum(o['size_bytes'] for o in records.values())>MAX_BYTES
            or sum(o['size_bytes'] for o in records.values())!=r.get('total_bytes')
            or records['manifest.json']['sha256']!=r['suite_manifest_sha256']):raise ValueError('Receipt suite inventory differs')
    return r,records


def google_fetch(record,sink):
    from google.cloud import storage
    bucket,key=record['uri'][5:].split('/',1);generation=int(record['generation'])
    blob=storage.Client().bucket(bucket).blob(key,generation=generation)
    blob.reload(if_generation_match=generation,timeout=(15,60));verify_blob(blob,record)
    if str(blob.generation)!=record['generation']:raise ValueError('Generation differs')
    blob.chunk_size=CHUNK_BYTES
    blob.download_to_file(sink,if_generation_match=generation,raw_download=True,checksum=None,timeout=(15,60))


def restore(receipt_path,receipt_sha256,destination,evidence_dir,*,fetch=google_fetch):
    receipt_path=safe_path(receipt_path);receipt,records=load_receipt(receipt_path,receipt_sha256)
    dest,evidence=safe_path(destination),safe_path(evidence_dir)
    if (dest==evidence or dest.is_relative_to(evidence) or evidence.is_relative_to(dest)
            or dest.exists() or evidence.exists()):raise ValueError('Use separate fresh data/evidence destinations')
    parent=dest.parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<receipt['total_bytes']+64*1024**2:raise ValueError('Insufficient restore disk')
    dest.mkdir(parents=True);evidence.mkdir(parents=True)
    shutil.copyfile(receipt_path,evidence/'receipt.json')
    if file_digest(evidence/'receipt.json')['sha256']!=receipt_sha256:raise ValueError('Receipt changed')
    _write_once(evidence/'intent.json',{'schema':RESTORE_SCHEMA,'receipt_sha256':receipt_sha256,'destination':str(dest)})
    completed=[];started=time.monotonic()
    try:
        for name in ordered_names():
            target=safe_path(dest/name);target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.partial')
            with partial.open('xb') as stream:
                sink=DigestSink(records[name],stream);fetch(records[name],sink);sink.finish()
            if file_digest(receipt_path)['sha256']!=receipt_sha256:raise ValueError('Receipt changed during recovery')
            if name=='manifest.json':
                precommit_paths(dest)
                suite=read_json(partial,receipt['suite_manifest_sha256'])
                panels={p:read_json(dest/'panels'/p/'manifest.json',records[f'panels/{p}/manifest.json']['sha256']) for p in PANELS}
                validate_suite_manifests(suite,panels,records,receipt['suite_manifest_sha256'])
                if suite['identity_sha256']!=receipt['suite_identity_sha256']:raise ValueError('Recovered suite identity differs')
                for relative in FILES-{'manifest.json'}:
                    actual=file_digest(safe_path(dest/relative))
                    if any(actual[k]!=records[relative][k] for k in actual):raise ValueError('Recovered payload changed before suite commit')
            os.link(partial,target);partial.unlink();sync_dir(target.parent);completed.append(name)
        # Fresh destination contains precisely the verified inventory. Rehash the
        # completed bytes as a final integrity guard, without SQL/token loading.
        collect_suite(dest,receipt['suite_manifest_sha256'])
        result={'schema':RESTORE_SCHEMA,'status':'verified','receipt_sha256':receipt_sha256,
            'suite_manifest_sha256':receipt['suite_manifest_sha256'],'suite_identity_sha256':receipt['suite_identity_sha256'],
            'objects':len(completed),'bytes':receipt['total_bytes'],'seconds':time.monotonic()-started,
            'destination':str(dest),'qualification':'Exact index bytes recovered; corpus availability and executor semantics remain separate.'}
        _write_once(evidence/'report.json',result);return result
    except BaseException as error:
        _write_once(evidence/'failure.json',{'schema':RESTORE_SCHEMA,'status':'failed','completed':completed,
            'error_type':type(error).__name__,'manifest_published':(dest/'manifest.json').exists()});raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    a=commands.add_parser('retain');a.add_argument('--suite',type=Path,required=True);a.add_argument('--suite-manifest-sha256',required=True)
    a.add_argument('--prefix',required=True);a.add_argument('--receipt',type=Path,required=True)
    a=commands.add_parser('restore')
    for name in ('receipt','destination','evidence-dir'):a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--receipt-sha256',required=True);args=p.parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent') or list(Path('/dev').glob('nvidia[0-9]*')):
        raise RuntimeError('Use required GPU-disabled project container')
    if args.command=='retain':result=retain(args.suite,args.suite_manifest_sha256,args.prefix,args.receipt)
    else:result=restore(args.receipt,args.receipt_sha256,args.destination,args.evidence_dir)
    print(json.dumps({k:result[k] for k in ('schema','status')}))

if __name__=='__main__':main()
