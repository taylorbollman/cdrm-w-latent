#!/usr/bin/env python3
"""Prepare reusable complete-document shards from retained Dolma coverage extracts.

This CPU preparation command chooses no training windows or model boundary policy.
The explicit split allocation is for the readiness slice, not a final corpus mix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained.document_shards import prepare_document_shards
from cdrm.pretrained.olmo_artifacts import FILE_SPECS
from scripts.olmo_document_extract import digest


def load_sources(plan_path, raw_dir, cloud_root):
    plan_path, raw_dir = Path(plan_path), Path(raw_dir)
    plan = json.loads(plan_path.read_text())
    sources, provenance = [], []
    for entry in plan['sources']:
        name = entry['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
            raise ValueError('Invalid source name')
        directory = raw_dir/name
        manifest_path = directory/'manifest.json'
        if directory.is_symlink() or manifest_path.is_symlink():
            raise ValueError('Source extraction must be a regular directory')
        record = json.loads(manifest_path.read_text())
        if (record['schema'] != 'olmo-dolma-source-extract-v2' or record['status'] != 'complete'
                or record['config']['source'] != entry
                or record['config']['upstream_manifest']['source_plan_sha256'] != digest(plan_path)
                or record['config']['tokenizer_sha256'] != FILE_SPECS['tokenizer.json'][1]):
            raise ValueError('Extraction provenance differs from selected source plan')
        if set(record['files']) != {'raw.jsonl', 'source-lines.jsonl'}:
            raise ValueError('Extraction lacks raw file/source line inventory')
        for file, expected in record['files'].items():
            path = directory/file
            if (path.is_symlink() or path.stat().st_size != expected['size_bytes']
                    or digest(path) != expected['sha256']):
                raise ValueError('Raw extraction bytes differ from committed manifest')
        pin = SourcePin(name, f'{cloud_root}/raw-{name}/raw.jsonl',
                        plan['hf_revision'], record['files']['raw.jsonl']['sha256'])
        sources.append(LocalJSONLSource(pin, directory/'raw.jsonl'))
        provenance.append({'name': name, 'extraction_manifest_sha256': digest(manifest_path),
                           'uri': f'{cloud_root}/raw-{name}/manifest.json'})
    return sources, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', type=Path, default=ROOT/'configs/data/dolma-v1_5-readiness-coverage.json')
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cloud-root', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--max-new-shards', type=int)
    parser.add_argument('--tokenizer', type=Path, default=ROOT/'.runtime/olmo1b-step60000/artifacts/native/tokenizer.json')
    args = parser.parse_args(argv)
    if not args.cloud_root.startswith('gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/'):
        raise ValueError('Expected the authorized durable corpus namespace')
    started = time.monotonic()
    sources, provenance = load_sources(args.source_plan, args.raw_dir, args.cloud_root)
    summary = prepare_document_shards(sources, args.output_dir, tokenizer_path=args.tokenizer,
        split_policy=SplitPolicy(20260928, (('train', 98), ('dev', 1), ('confirmation', 1))),
        max_documents_per_shard=1024, target_tokens_per_shard=250_000,
        max_new_shards=args.max_new_shards)
    report = {'schema': 'olmo-document-preparation-invocation-v1', 'summary': summary,
              'raw_extractions': provenance, 'source_plan_sha256': digest(args.source_plan),
              'elapsed_seconds': time.monotonic()-started,
              'source_hashes': {p: digest(ROOT/p) for p in (
                  'cdrm/pretrained/document_shards.py', 'cdrm/pretrained/campaign_ingest.py',
                  'cdrm/pretrained/campaign_data.py', 'cdrm/pretrained/olmo_artifacts.py',
                  'scripts/olmo_document_extract.py', 'scripts/olmo_prepare_documents.py')}}
    write_json(args.report, report)
    print(json.dumps({k: v for k, v in summary.items() if k != 'shards'}
                     | {'shards':len(summary['shards']), 'elapsed_seconds':report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
