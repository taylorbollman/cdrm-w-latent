#!/usr/bin/env python3
"""CPU streaming restore to SSD, with small evidence on persistent storage."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import olmo_campaign_execution_restore as old
from scripts.olmo_campaign_ssd_storage import validate_storage_paths

SCHEMA='olmo-campaign-ssd-restore-v1'
PROTOCOL='docs/reports/olmo-campaign-storage/protocol.md'


def source_hashes():
    return {**old.source_hashes(),**{name:old.sha(ROOT/name) for name in (
        'scripts/olmo_campaign_ssd_restore.py','scripts/olmo_campaign_ssd_storage.py',
        'tests/test_campaign_ssd_restore.py','tests/test_campaign_ssd_storage.py',PROTOCOL)}}


def restore(publication_path,publication_sha256,output_dir,checkpoint_dir,*,fetch=old.google_fetch):
    publication_path=old.regular(publication_path)
    if publication_path.stat().st_size>old.MAX_PUBLICATION_BYTES:
        raise ValueError('Publication receipt exceeds metadata bound')
    receipt=old.pinned_json(publication_path,old.pin(publication_sha256))
    manifest,objects,identity=old.publication_metadata(receipt)
    output,checkpoint=Path(output_dir).absolute(),Path(checkpoint_dir).absolute()
    validate_storage_paths(checkpoint,output)
    if output.exists() or checkpoint.exists():raise ValueError('Use fresh evidence and SSD restore directories')
    ancestor=checkpoint.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<sum(o['size_bytes'] for o in objects.values())+64*1024**2:
        raise ValueError('Insufficient SSD disk for the complete pinned checkpoint')
    sources=source_hashes();started=time.monotonic()
    output.mkdir(parents=True,exist_ok=False)
    authority=output/'authorities';authority.mkdir()
    old.write_manifest_once(authority/'latest-checkpoint.json',receipt)
    # Preserve the original byte pin as well as the decoded authority.
    with (authority/'publication-original.json').open('xb') as f:
        f.write(publication_path.read_bytes());f.flush();os.fsync(f.fileno())
    if old.sha(authority/'publication-original.json')!=publication_sha256:
        raise ValueError('Publication receipt changed before recovery')
    for name,pin in sources.items():
        destination=output/'source-snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
        if old.sha(destination)!=pin:raise ValueError('Recovery source snapshot changed')
    downloads=[]
    old.write_manifest_once(output/'intent.json',{'schema':SCHEMA,'checkpoint_directory':str(checkpoint),
        'publication_sha256':publication_sha256,'sources':sources,'storage':'volatile SSD, restore only'})
    try:
        validate_storage_paths(checkpoint,output)
        checkpoint.mkdir(parents=True,exist_ok=False)
        partial_manifest,partial_state=checkpoint/'manifest.json.partial',checkpoint/'state.pt.partial'
        downloads.append(old.download(objects['manifest.json'],partial_manifest,fetch))
        decoded=old.pinned_json(partial_manifest,receipt['manifest_sha256'])
        old.committed_metadata(decoded)
        if decoded!=manifest:raise ValueError('Downloaded manifest differs from published boundary')
        downloads.append(old.download(objects['state.pt'],partial_state,fetch))
        if source_hashes()!=sources or old.sha(publication_path)!=publication_sha256:
            raise ValueError('Source or publication authority changed during download')
        validate_storage_paths(checkpoint,output)
        old.durable_publish(partial_state,checkpoint/'state.pt')
        old.durable_publish(partial_manifest,checkpoint/'manifest.json')
        result={'schema':SCHEMA,'status':'checkpoint_assets_verified_launch_pending','sources':sources,
            'publication_sha256':publication_sha256,'checkpoint_manifest_sha256':receipt['manifest_sha256'],
            'execution_identity_sha256':identity['sha256'],'completed_optimizer_updates':manifest['counters']['optimizer_updates'],
            'checkpoint_directory':str(checkpoint),'evidence_directory':str(output),'downloads':downloads,
            'total_downloaded_bytes':sum(o['size_bytes'] for o in objects.values()),
            'elapsed_seconds':time.monotonic()-started,'model_tensors_loaded':False,'gpu_used':False,
            'cloud_writes':False,'original_receipt_local_paths_used':False,
            'qualification':'Verified assets only; runner must separately enforce exact execution identity and restore complete state. No cross-version or cross-topology resume clearance.'}
        old.write_manifest_once(output/'report.json',result)
        return result
    except Exception as error:
        old.write_manifest_once(output/'failure.json',{'schema':SCHEMA,'status':'failed_assets_not_completed',
            'error':f'{type(error).__name__}: {error}','completed_downloads':downloads,
            'checkpoint_manifest_published':(checkpoint/'manifest.json').exists(),
            'checkpoint_directory':str(checkpoint),'partial_bytes_retained':True,
            'publication_sha256':publication_sha256})
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication',type=Path,required=True);p.add_argument('--publication-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--checkpoint-dir',type=Path,required=True)
    a=p.parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent') or list(Path('/dev').glob('nvidia[0-9]*')) or 'torch' in sys.modules:
        raise RuntimeError('Use the project CPU-only container; no torch/model/GPU process')
    r=restore(a.publication,a.publication_sha256,a.output_dir,a.checkpoint_dir)
    print(json.dumps({k:r[k] for k in ('status','checkpoint_manifest_sha256','total_downloaded_bytes')}))


if __name__=='__main__':main()
