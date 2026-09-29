#!/usr/bin/env python3
"""Create a tiny, explicitly synthetic ordered-data execution fixture on CPU.

Uses the exact native tokenizer and real document/index writers. Source/cloud
metadata is simulated for contract testing, not evidence of Dolma acquisition.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained import document_shards as shards
from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from scripts import olmo_pilot_data_plan as plan
from scripts import olmo_pilot_ordered_data as ordered

SCHEMA = 'olmo-pilot-execution-fixture-v1'
DEFAULT_TOKENIZER = ROOT/'.runtime/olmo1b-step60000/artifacts/native/tokenizer.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.write_bytes(plan.canonical(value))
    return sha(path)


def build_fixture(output_dir, *, tokenizer_path=DEFAULT_TOKENIZER):
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Fixture needs a fresh output directory')
    # Validate before creating output. There is no custom-tokenizer escape.
    tokenizer = shards._load_tokenizer(Path(tokenizer_path))
    output.mkdir(parents=True)
    shutil.copyfile(tokenizer_path, output/'tokenizer.json')
    recipe = deepcopy(plan.load_recipe())
    recipe['inventory']['directory_counts'] = dict.fromkeys(plan.STRATA, 1)
    recipe['inventory']['url_count'] = len(plan.STRATA)
    inventory = ''.join(plan.URL_ROOT+s+'/synthetic-fixture.json.gz\n' for s in plan.STRATA).encode()
    recipe['inventory']['sha256'] = hashlib.sha256(inventory).hexdigest()
    recipe['selection']['objects_per_stratum'] = dict.fromkeys(plan.STRATA, 1)
    recipe['panels'].update(length=16, train_tokens=16*27, heldout_main_tokens=16*18,
        heldout_source_tokens=16*2, minimum_reserve_tokens_per_stratum=16*3,
        minimum_documents_per_stratum=dict.fromkeys(plan.STRATA, 2))
    selected = plan.source_selection(recipe, inventory)
    documents = []; excluded = None
    for source_index, source in enumerate(selected):
        counts = dict.fromkeys(('train', 'dev', 'confirmation'), 0)
        rows = []
        for trial in range(10000):
            # True internal EOS is retained as content, not mistaken for a
            # document boundary. Two decimal identifiers keep tokenized docs
            # short enough to force cross-document T16 chunks.
            text = f'{source_index} {trial}<|endoftext|> tail'
            values = list(tokenizer.encode(text, add_special_tokens=False).ids)
            if not values or values[-1] != shards.EOS_ID:
                values.append(shards.EOS_ID)
            content = hashlib.sha256(shards._u16(values[:-1])).hexdigest()
            split = plan.split_for_content_hash(recipe, content)
            target = 64 if split == 'train' else 32
            if counts[split] >= target:
                continue
            counts[split] += 1
            rows.append({'id':f'synthetic-{source_index}-{trial}', 'text':text})
            if source_index == 0 and split == 'train' and excluded is None:
                excluded = content
            if all(counts[s] >= (64 if s == 'train' else 32) for s in counts):
                break
        else:
            raise ValueError('Finite synthetic split search exhausted')
        documents.append(rows)
    exclusion_bytes = plan.exclusion_bytes([excluded])
    recipe['exclusions'].update(unique_documents=1,
        content_ids_sha256=hashlib.sha256(exclusion_bytes).hexdigest())
    # The otherwise inherited authority hash is explicitly a synthetic fixture
    # identity and does not claim the production readiness exclusion corpus.
    recipe['exclusions']['corpus_manifest_sha256'] = hashlib.sha256(b'synthetic-excluded-fixture-v1').hexdigest()
    plan.validate_recipe(recipe)
    acquisition = {'recipe':recipe, 'sources':[
        {**row, 'etag':'"synthetic-fixture-not-a-cloud-object"', 'upstream_size_bytes':123456}
        for row in selected]}
    acquisition_path = output/'acquisition.json'
    acquisition_sha = write(acquisition_path, acquisition)
    sources = []; authorities = {}
    for source, rows in zip(acquisition['sources'], documents):
        path = output/(source['name']+'.jsonl')
        path.write_bytes(b''.join(plan.canonical(row) for row in rows))
        raw_sha = sha(path)
        pin = SourcePin(source['name'], 'gs://fast-chunks/cdrm-w-latent/data/synthetic-fixture/raw-'+source['name']+'/raw.jsonl',
                        recipe['hf_revision'], raw_sha)
        sources.append(LocalJSONLSource(pin, path))
        authorities[source['name']] = {'source_pin':asdict(pin), 'upstream_source':source,
            'extraction_manifest_sha256':hashlib.sha256(('synthetic-'+source['name']).encode()).hexdigest(),
            'acquisition_plan_sha256':acquisition_sha,
            'raw_object':{'uri':pin.uri, 'sha256':raw_sha, 'size_bytes':path.stat().st_size,
                'generation':'123', 'verification':dict.fromkeys(
                    ('server_size','server_md5','sha256_metadata','download_sha256'), True)}}
    authorities_path = output/'authorities.json'
    authorities_sha = write(authorities_path, authorities)
    (output/'inventory.txt').write_bytes(inventory)
    (output/'exclusions.txt').write_bytes(exclusion_bytes)
    write(output/'recipe.json', recipe)
    corpus = output/'corpus'; suite = output/'ordered'
    summary = shards.prepare_document_shards(sources, corpus, tokenizer_path=output/'tokenizer.json',
        split_policy=SplitPolicy(recipe['split']['seed'], recipe['split']['weights']),
        max_documents_per_shard=60)
    manifest = ordered.build_ordered_data(corpus, suite, recipe=recipe, inventory_bytes=inventory,
        excluded_content_hashes=[excluded], acquisition_plan_path=acquisition_path,
        acquisition_plan_sha256=acquisition_sha, source_authorities_path=authorities_path,
        source_authorities_sha256=authorities_sha)
    with ordered.OrderedCampaignData(corpus, suite/'panels/train') as data:
        cursor = data.cursor(); updates = []; cross_document = 0
        for _ in range(3):
            update = data.peek_update(cursor, 80)
            assert len(update.rows) == 5
            cross_document += update.counts.cross_document_ce_targets
            rank_batches = [data.rank_batches(update, rank=r, world_size=2, physical_batch_size=2) for r in range(2)]
            updates.append({'counts':asdict(update.counts), 'rank_microbatches':[len(x.batches) for x in rank_batches]})
            cursor = update.next_cursor
        if cross_document <= 0:
            raise ValueError('Synthetic fixture unexpectedly lacks cross-document CE')
    report = {'schema':SCHEMA, 'passed':True,
        'scope':'Synthetic CPU fixture; cloud generations and verification flags are simulated; no Dolma acquisition or GPU execution',
        'corpus_summary':summary, 'suite_identity_sha256':manifest['identity_sha256'],
        'updates':updates, 'first_three_updates_cross_document_ce_targets':cross_document,
        'data_spec':{'corpus':str(corpus), 'corpus_manifest_sha256':sha(corpus/'manifest.json'),
            'suite':str(suite), 'suite_manifest_sha256':sha(suite/'manifest.json'),
            'index':str(suite/'panels/train'), 'index_manifest_sha256':sha(suite/'panels/train/manifest.json'),
            'source_plan':str(acquisition_path), 'source_plan_sha256':acquisition_sha,
            'source_authorities':str(authorities_path), 'source_authorities_sha256':authorities_sha,
            'inventory':str(output/'inventory.txt'), 'inventory_sha256':sha(output/'inventory.txt'),
            'exclusions':str(output/'exclusions.txt'), 'exclusions_sha256':sha(output/'exclusions.txt'),
            'split':'train', 'policy':ordered.POLICY},
        'pins':{name:sha(output/name) for name in ('tokenizer.json','recipe.json','inventory.txt','exclusions.txt','acquisition.json','authorities.json','corpus/manifest.json','ordered/manifest.json')}}
    write(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args(argv)
    report = build_fixture(args.output_dir, tokenizer_path=args.tokenizer)
    print(json.dumps({'passed':report['passed'], 'output_dir':str(args.output_dir), 'pins':report['pins']}, sort_keys=True))


if __name__ == '__main__':
    main()
