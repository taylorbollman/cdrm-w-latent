#!/usr/bin/env python3
"""CPU-only exact-generation restoration of an ordered pilot checkpoint to SSD.

The new identity schema is checked directly. Historical packed checkpoint
metadata is never relabeled. Byte streaming, cloud object bounds and durable
manifest-last publication reuse the accepted implementation unchanged.
"""
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

SCHEMA='olmo-pilot-execution-restore-v1'
IDENTITY_SCHEMA='olmo-pilot-execution-identity-v1'
PROTOCOL='docs/reports/olmo-pilot-execution/protocol.md'
CHECKPOINT_SCHEMA=old.CHECKPOINT_SCHEMA
MANIFEST_FIELDS=old.MANIFEST_FIELDS
MAX_STATE_BYTES=old.MAX_STATE_BYTES
positive=old.positive
pin=old.pin
digest=old.digest
object_record=old.object_record


def source_hashes():
    return {**old.source_hashes(),**{name:old.sha(ROOT/name) for name in (
        'scripts/olmo_pilot_execution_restore.py','scripts/olmo_campaign_ssd_storage.py',
        'tests/test_pilot_execution_restore.py','tests/test_campaign_ssd_storage.py',PROTOCOL)}}


def committed_metadata(manifest):
    """Basic JSON self-consistency only; strict execution/tensor load remains separate."""
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS or manifest['schema'] != CHECKPOINT_SCHEMA:
        raise ValueError('Require the complete committed distributed manifest')
    world = positive(manifest['world_size'], maximum=64)
    meta = manifest['metadata']
    if (not isinstance(meta, dict) or meta.get('schema') != CHECKPOINT_SCHEMA or meta.get('world_size') != world
            or not isinstance(meta.get('model_type'), str) or not meta['model_type']):
        raise ValueError('Committed checkpoint topology/model metadata differs')
    identity = meta.get('configuration', {}).get('execution_identity', {})
    if (set(identity) != {'schema', 'payload', 'sha256'} or identity.get('schema') != IDENTITY_SCHEMA
            or not isinstance(identity.get('payload'), dict) or identity['sha256'] != digest(identity['payload'])
            or meta.get('source_fingerprint', {}).get('execution_identity_sha256') != identity['sha256']):
        raise ValueError('Execution identity hash/source metadata differ')
    payload = identity['payload']
    if (payload.get('arm') not in ('B', 'N', 'F', 'R', 'NF', 'NR', 'FR', 'NFR')
            or payload.get('partition', {}).get('world_size') != world
            or payload.get('cursor_schema') != 'olmo-campaign-execution-cursor-v1'):
        raise ValueError('Execution identity arm/topology/cursor differs')
    batch = positive(payload['partition'].get('physical_batch_per_rank'), maximum=65536)
    counters = manifest['counters']
    if (not isinstance(counters, dict) or set(counters) != {'optimizer_updates', 'input_tokens', 'documents',
            'microbatches', 'ce_positions', 'latent_pairs', 'kl_triples'}
            or any(type(v) is not int or v < 0 for v in counters.values())):
        raise ValueError('Committed counters must be complete nonnegative integer counts')
    ranks = manifest['rank_cursors']
    if not isinstance(ranks, list) or len(ranks) != world:
        raise ValueError('Committed rank cursor count differs')
    first = None
    for rank, row in enumerate(ranks):
        if (not isinstance(row, dict) or set(row) != {'schema', 'rank', 'world_size', 'physical_batch_per_rank', 'cursor'}
                or row['schema'] != payload['cursor_schema'] or type(row['rank']) is not int or row['rank'] != rank
                or row['world_size'] != world or row['physical_batch_per_rank'] != batch
                or not isinstance(row['cursor'], dict) or row['cursor'].get('next_update') != counters['optimizer_updates']):
            raise ValueError('Committed cursor and optimizer boundary disagree')
        if first is None:
            first = row['cursor']
        elif row['cursor'] != first:
            raise ValueError('Ranks disagree on committed logical cursor')
    state = manifest['state']
    if not isinstance(state, dict) or set(state) != {'filename', 'size_bytes', 'sha256'} or state['filename'] != 'state.pt':
        raise ValueError('Committed state filename differs')
    positive(state['size_bytes'], maximum=MAX_STATE_BYTES); pin(state['sha256'])
    return identity


def publication_metadata(receipt):
    if not isinstance(receipt, dict) or set(receipt) != MANIFEST_FIELDS|{'directory', 'manifest_sha256', 'retention'}:
        raise ValueError('Require a published latest-checkpoint receipt')
    manifest = {key: receipt[key] for key in MANIFEST_FIELDS}
    identity = committed_metadata(manifest)
    pin(receipt['manifest_sha256'])
    retention = receipt['retention']
    if (not isinstance(retention, dict) or set(retention) != {'objects', 'create_only', 'download_sha256_verified'}
            or retention['create_only'] is not True or retention['download_sha256_verified'] is not True
            or not isinstance(retention['objects'], list) or len(retention['objects']) != 2):
        raise ValueError('Receipt lacks a completely retained object pair')
    objects = {}
    for row in retention['objects']:
        filename = row.get('uri', '').rsplit('/', 1)[-1] if isinstance(row, dict) else None
        object_record(row, filename)
        if filename in objects:
            raise ValueError('Duplicate checkpoint object')
        objects[filename] = row
    if (set(objects) != {'manifest.json', 'state.pt'}
            or objects['manifest.json']['uri'].rsplit('/', 1)[0] != objects['state.pt']['uri'].rsplit('/', 1)[0]
            or objects['manifest.json']['sha256'] != receipt['manifest_sha256']
            or any(objects['state.pt'][k] != manifest['state'][k] for k in ('sha256', 'size_bytes'))):
        raise ValueError('Retained objects do not bind the same checkpoint boundary')
    return manifest, objects, identity


def restore(publication_path,publication_sha256,output_dir,checkpoint_dir,*,fetch=old.google_fetch):
    publication_path=old.regular(publication_path)
    if publication_path.stat().st_size>old.MAX_PUBLICATION_BYTES:
        raise ValueError('Publication receipt exceeds metadata bound')
    receipt=old.pinned_json(publication_path,old.pin(publication_sha256))
    manifest,objects,identity=publication_metadata(receipt)
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
        committed_metadata(decoded)
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
