#!/usr/bin/env python3
"""Restore one retained pilot data stage from exact GCS object generations.

No model/tensor imports, no cloud listing, and no receipt-local paths are used.
The caller reconstructs the corpus using its explicitly retained stage catalog.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.olmo_document_retain import parse_prefix, _relative

SCHEMA='olmo-pilot-data-restore-v1'
MAX_BYTES=8*1024**3


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def safe_path(path):
    path=Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('No traversal or symlink paths')
    return path


def sync_dir(path):
    fd=os.open(path,os.O_DIRECTORY|os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_once(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    sync_dir(Path(path).parent)


def load_receipt(path,pin):
    path=safe_path(path)
    if not path.is_file() or path.stat().st_size>16*1024**2 or sha(path)!=pin:
        raise ValueError('Receipt SHA or size differs')
    r=json.loads(path.read_text());parse_prefix(r.get('prefix',''))
    if r.get('schema')!='olmo-document-retention-v1' or r.get('status')!='verified' or r.get('local_files_deleted') is not False:
        raise ValueError('Require complete retained document-stage receipt')
    if r.get('publication_marker')!=r['prefix']+'/manifest.json':raise ValueError('Wrong stage commit marker')
    records=[];names=set();total=0
    for obj in r.get('objects',[]):
        prefix=r['prefix']+'/'
        if not isinstance(obj.get('uri'),str) or not obj['uri'].startswith(prefix):raise ValueError('Object escapes stage')
        name=obj['uri'][len(prefix):];_relative(name)
        if name in names:raise ValueError('Duplicate retained object')
        names.add(name)
        size=obj.get('size_bytes');generation=obj.get('generation');digest=obj.get('sha256')
        if type(size) is not int or not 0<=size<=MAX_BYTES:raise ValueError('Invalid object size')
        if not isinstance(generation,str) or not generation.isascii() or not generation.isdigit() or int(generation)<=0:raise ValueError('Require exact generation')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid object SHA')
        try:md5=base64.b64decode(obj.get('md5_base64',''),validate=True)
        except (ValueError,TypeError):raise ValueError('Invalid MD5') from None
        if len(md5)!=16 or obj.get('verification')!={k:True for k in ('server_size','server_md5','sha256_metadata','download_sha256')}:
            raise ValueError('Missing producer verification')
        total+=size;records.append((name,obj))
    if not records or len(records)>10000 or total>MAX_BYTES or 'manifest.json' not in names:raise ValueError('Invalid bounded stage inventory')
    if sha(path)!=pin:raise ValueError('Receipt changed during preflight')
    return r,sorted(records,key=lambda x:(x[0]=='manifest.json',Path(x[0]).name=='manifest.json',x[0]))


class Sink:
    def __init__(self,f,record):self.f,self.record=f,record;self.size=0;self.sha=hashlib.sha256();self.md5=hashlib.md5()
    def write(self,value):
        if self.size+len(value)>self.record['size_bytes']:raise ValueError('Download exceeds declared bound')
        written=self.f.write(value)
        if written!=len(value):raise OSError('Short local write')
        self.size+=written;self.sha.update(value);self.md5.update(value);return written
    def tell(self):return self.size
    def finish(self):
        if (self.size,self.sha.hexdigest(),base64.b64encode(self.md5.digest()).decode())!=(self.record['size_bytes'],self.record['sha256'],self.record['md5_base64']):raise ValueError('Downloaded bytes differ')
        self.f.flush();os.fsync(self.f.fileno())


def google_fetch(record,sink):
    from google.cloud import storage
    bucket,key=record['uri'][5:].split('/',1);generation=int(record['generation'])
    blob=storage.Client().bucket(bucket).blob(key,generation=generation)
    blob.reload(if_generation_match=generation,timeout=(15,60))
    if (str(blob.generation)!=record['generation'] or blob.size!=record['size_bytes']
            or blob.md5_hash!=record['md5_base64'] or (blob.metadata or {}).get('sha256')!=record['sha256']
            or blob.content_encoding is not None):raise ValueError('Exact generation metadata differs')
    blob.download_to_file(sink,if_generation_match=generation,checksum='auto',raw_download=True,
                          single_shot_download=False,timeout=(15,60))


def restore(receipt_path,pin,destination,evidence_dir,*,fetch=google_fetch):
    receipt_path=safe_path(receipt_path);r,records=load_receipt(receipt_path,pin)
    dest,evidence=safe_path(destination),safe_path(evidence_dir)
    if dest==evidence or dest.is_relative_to(evidence) or evidence.is_relative_to(dest):raise ValueError('Keep data and evidence separate')
    if dest.exists() or evidence.exists():raise ValueError('Use fresh restore destinations')
    parent=dest.parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<sum(o['size_bytes'] for _,o in records)+64*1024**2:raise ValueError('Insufficient disk')
    evidence.mkdir(parents=True);dest.mkdir(parents=True)
    shutil.copyfile(receipt_path,evidence/'receipt.json')
    if sha(evidence/'receipt.json')!=pin:raise ValueError('Receipt changed before restore')
    write_once(evidence/'intent.json',{'schema':SCHEMA,'receipt_sha256':pin,'prefix':r['prefix'],'destination':str(dest)})
    completed=[];started=time.monotonic()
    try:
        for name,record in records:
            safe_path(dest);target=safe_path(dest/name);target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.partial')
            with partial.open('xb') as f:
                sink=Sink(f,record);fetch(record,sink);sink.finish()
            if sha(receipt_path)!=pin:raise ValueError('Receipt changed during transfer')
            os.link(partial,target);partial.unlink();sync_dir(target.parent)
            completed.append({'name':name,'generation':record['generation'],'sha256':record['sha256'],'size_bytes':record['size_bytes']})
        result={'schema':SCHEMA,'status':'verified','receipt_sha256':pin,'prefix':r['prefix'],'objects':completed,
            'bytes':sum(o['size_bytes'] for _,o in records),'seconds':time.monotonic()-started,
            'destination':str(dest),'qualification':'Byte recovery only; corpus/index semantic verification and matching preparation recipe remain mandatory.'}
        write_once(evidence/'report.json',result);return result
    except BaseException as e:
        write_once(evidence/'failure.json',{'schema':SCHEMA,'status':'failed','completed':completed,
            'error_type':type(e).__name__,'manifest_published':(dest/'manifest.json').exists()});raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--evidence-dir',type=Path,required=True)
    a=p.parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent') or list(Path('/dev').glob('nvidia[0-9]*')):
        raise RuntimeError('Use required CPU-only project container')
    r=restore(a.receipt,a.receipt_sha256,a.destination,a.evidence_dir);print(json.dumps({'status':r['status'],'bytes':r['bytes']}))


if __name__=='__main__':main()
