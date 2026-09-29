#!/usr/bin/env python3
"""Pure, versioned Dolma pilot planning. No downloads, torch or token payloads.

Exact content membership is independent of order and acquisition round. A later
round appends a separately frozen chunk stream; it never re-sorts prior rounds.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT/'configs/data/dolma-v1_5-pilot-v1.json'
SCHEMA = 'olmo-dolma-pilot-recipe-v1'
PLAN_SCHEMA = 'olmo-dolma-pilot-plan-v1'
STRATA = ('books', 'c4', 'cc_en_head', 'cc_en_middle', 'cc_en_tail', 'pes2o', 'reddit', 'stack', 'wiki')
CC = ('cc_en_head', 'cc_en_middle', 'cc_en_tail')
FAMILIES = ('books', 'c4', 'common_crawl', 'pes2o', 'reddit', 'stack', 'wiki')
URL_ROOT = 'https://olmo-data.org/dolma-v1_5r1/'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pin(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Expected lowercase SHA256')
    return value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{name} fields differ')


def validate_recipe(recipe):
    _keys(recipe, ('schema', 'release', 'hf_repo', 'hf_revision', 'inventory', 'tokenizer',
                  'mixture', 'selection', 'split', 'exclusions', 'panels', 'bounds', 'object_limits',
                  'ordering', 'qualifications'), 'recipe')
    if (recipe['schema'], recipe['release'], recipe['hf_repo'], recipe['hf_revision']) != (
            SCHEMA, 'v1_5', 'allenai/dolma', '7f48140530a023e9ea4c5cfb141160922727d4d3'):
        raise ValueError('Pilot release/repository authority differs')
    inventory = recipe['inventory']
    _keys(inventory, ('sha256', 'url_count', 'directory_counts'), 'inventory')
    _pin(inventory['sha256']); _integer(inventory['url_count'], 'url_count')
    _keys(inventory['directory_counts'], STRATA, 'inventory directories')
    for n in inventory['directory_counts'].values(): _integer(n, 'directory count')
    if sum(inventory['directory_counts'].values()) != inventory['url_count']:
        raise ValueError('Inventory totals differ')
    tokenizer = recipe['tokenizer']
    _keys(tokenizer, ('repo', 'revision', 'sha256', 'eos_id', 'pad_id', 'vocab_size', 'token_dtype'), 'tokenizer')
    if tokenizer != {'repo':'allenai/OLMo-1B', 'revision':'81b71efbce6f4dada57c94860301af4298bcd351',
            'sha256':'9ad33b4b39a9f83973c3f8c42a01948dd5b877a28ac9a5356956c4ff4ed0b714',
            'eos_id':50279, 'pad_id':1, 'vocab_size':50280, 'token_dtype':'uint16_le'}:
        raise ValueError('Native tokenizer contract differs')
    mix = recipe['mixture']
    _keys(mix, ('family_weights', 'cc_stratum_weights'), 'mixture')
    _keys(mix['family_weights'], FAMILIES, 'family weights')
    _keys(mix['cc_stratum_weights'], CC, 'CC weights')
    for weights in mix.values():
        for n in weights.values(): _integer(n, 'mixture weight')
    selection = recipe['selection']
    _keys(selection, ('objects_per_stratum', 'url_namespace', 'candidate_tokens_per_object', 'policy'), 'selection')
    _keys(selection['objects_per_stratum'], STRATA, 'object quotas')
    for name, n in selection['objects_per_stratum'].items():
        _integer(n, 'selected objects')
        if n > inventory['directory_counts'][name]: raise ValueError('Object selection exceeds inventory')
    _integer(selection['candidate_tokens_per_object'], 'candidate_tokens_per_object')
    if selection['policy'] != 'hash_rank_urls_then_complete_record_prefix':
        raise ValueError('Unsupported acquisition policy')
    split = recipe['split']
    _keys(split, ('seed', 'weights', 'identity'), 'split')
    _integer(split['seed'], 'split seed', 0)
    if split['weights'] != [['train',90],['dev',5],['confirmation',5]] or split['identity'] != 'content_token_sha256_without_one_terminal_eos':
        raise ValueError('Pilot split policy differs')
    exclusion = recipe['exclusions']
    _keys(exclusion, ('corpus_manifest_sha256', 'content_ids_sha256', 'unique_documents', 'encoding'), 'exclusions')
    _pin(exclusion['corpus_manifest_sha256']); _pin(exclusion['content_ids_sha256'])
    _integer(exclusion['unique_documents'], 'excluded documents')
    if exclusion['encoding'] != 'sorted_unique_lowercase_sha256_ascii_newline':
        raise ValueError('Exclusion encoding differs')
    panels = recipe['panels']
    _keys(panels, ('length', 'train_tokens', 'heldout_main_tokens', 'heldout_source_tokens',
                  'minimum_reserve_tokens_per_stratum', 'minimum_documents_per_stratum'), 'panels')
    length = _integer(panels['length'], 'context length', 3)
    if length > 2048: raise ValueError('Unsupported context length')
    for key in ('train_tokens', 'heldout_main_tokens', 'heldout_source_tokens', 'minimum_reserve_tokens_per_stratum'):
        if _integer(panels[key], key) % length: raise ValueError('Panel tokens must be full chunks')
    _keys(panels['minimum_documents_per_stratum'], STRATA, 'heldout document minima')
    for n in panels['minimum_documents_per_stratum'].values(): _integer(n, 'heldout document minimum')
    _keys(recipe['bounds'], ('candidate_tokens', 'compressed_bytes', 'retained_raw_bytes',
                            'max_line_bytes', 'max_documents_per_object'), 'bounds')
    for key, n in recipe['bounds'].items(): _integer(n, key)
    limits = recipe['object_limits']
    _keys(limits, ('tokens', 'compressed_bytes', 'retained_raw_bytes', 'timeout_seconds'), 'object limits')
    for key, n in limits.items(): _integer(n,key)
    objects = sum(selection['objects_per_stratum'].values())
    if selection['candidate_tokens_per_object'] > limits['tokens']:
        raise ValueError('Object hard token limit precedes target')
    for key, bound in (('tokens','candidate_tokens'),('compressed_bytes','compressed_bytes'),
                       ('retained_raw_bytes','retained_raw_bytes')):
        if objects*limits[key] > recipe['bounds'][bound]:
            raise ValueError('Aggregate successful-object caps exceed global bound')
    nominal = sum(selection['objects_per_stratum'].values()) * selection['candidate_tokens_per_object']
    if recipe['bounds']['candidate_tokens'] < nominal: raise ValueError('Candidate cap precedes nominal acquisition')
    ordering = recipe['ordering']
    _keys(ordering, ('document_namespace', 'chunk_namespace', 'interleave', 'round_policy', 'document_policy'), 'ordering')
    if (ordering['interleave'], ordering['round_policy'], ordering['document_policy']) != (
            'integer_deficit_lexical_tie_v1', 'append_frozen_rounds_without_resorting', 'continuous-stream-v1'):
        raise ValueError('Unsupported ordering policy')
    namespaces = (selection['url_namespace'], ordering['document_namespace'], ordering['chunk_namespace'])
    if any(not isinstance(n, str) or not n or '\0' in n for n in namespaces) or len(set(namespaces)) != 3:
        raise ValueError('Require three distinct nonempty hash namespaces')
    if recipe['qualifications'] != {'near_duplicate_policy':'not_run', 'original_pretraining_exposure':'unknown',
            'sampling':'conditional_on_selected_object_prefixes', 'cc_weights':'object_count_proxy_not_token_mass'}:
        raise ValueError('Required sampling qualifications differ')
    return recipe


def load_recipe(path=DEFAULT_RECIPE):
    return validate_recipe(json.loads(Path(path).read_bytes()))


def recipe_sha256(recipe):
    return digest(validate_recipe(recipe))


def _hash(namespace, *parts):
    if any(not isinstance(s, str) or '\0' in s for s in (namespace, *parts)):
        raise ValueError('Hash fields must be unambiguous strings')
    return hashlib.sha256('\0'.join((namespace, *parts)).encode()).hexdigest()


def stratum_weights(recipe):
    mix = recipe['mixture']; total = sum(mix['family_weights'].values()); cc_total = sum(mix['cc_stratum_weights'].values())
    return {name: Fraction(mix['family_weights']['common_crawl'], total)*Fraction(mix['cc_stratum_weights'][name], cc_total)
            if name in CC else Fraction(mix['family_weights'][name], total) for name in STRATA}


def largest_remainder(total_chunks, weights):
    _integer(total_chunks, 'total chunks', 0)
    if not weights or any(not isinstance(name, str) or not name for name in weights): raise ValueError('Named weights required')
    if any(type(v) not in (int, Fraction) or v <= 0 for v in weights.values()): raise ValueError('Positive exact weights required')
    denominator = sum(weights.values()); exact = {s: total_chunks*Fraction(w, denominator) for s,w in weights.items()}
    result = {s: v.numerator//v.denominator for s,v in exact.items()}
    ranked = sorted(exact, key=lambda s: (-(exact[s]-result[s]), s))
    for name in ranked[:total_chunks-sum(result.values())]: result[name] += 1
    return dict(sorted(result.items()))


def panel_quotas(recipe, panel):
    panels = recipe['panels']; length = panels['length']
    if panel in ('train', 'dev-main', 'confirmation-main'):
        tokens = panels['train_tokens' if panel == 'train' else 'heldout_main_tokens']
        return largest_remainder(tokens//length, stratum_weights(recipe))
    for split in ('dev', 'confirmation'):
        if panel.startswith(split+'-source/') and panel.split('/',1)[1] in STRATA:
            selected = panel.split('/',1)[1]
            return {s: panels['heldout_source_tokens']//length if s == selected else 0 for s in STRATA}
    raise ValueError('Unknown pilot panel')


def heldout_requirements(recipe):
    q = panel_quotas(recipe, 'dev-main'); p = recipe['panels']
    return {s:{'tokens':max(p['minimum_reserve_tokens_per_stratum'], p['heldout_source_tokens'], q[s]*p['length']),
               'documents':p['minimum_documents_per_stratum'][s]} for s in STRATA}


def source_selection(recipe, inventory_bytes):
    validate_recipe(recipe)
    if not isinstance(inventory_bytes, bytes) or hashlib.sha256(inventory_bytes).hexdigest() != recipe['inventory']['sha256']:
        raise ValueError('URL inventory bytes differ from recipe pin')
    urls = inventory_bytes.decode('utf-8').splitlines()
    if len(urls) != len(set(urls)) or len(urls) != recipe['inventory']['url_count']:
        raise ValueError('URL inventory duplicate/count mismatch')
    groups = {s:[] for s in STRATA}
    for url in urls:
        if not url.startswith(URL_ROOT): raise ValueError('Non-pinned Dolma source host/release')
        parts = url[len(URL_ROOT):].split('/')
        if len(parts) != 2 or parts[0] not in groups or not re.fullmatch(r'[A-Za-z0-9_.-]+\.json\.gz', parts[1]):
            raise ValueError('Unsafe or unknown Dolma object')
        groups[parts[0]].append(url)
    if {s:len(rows) for s,rows in groups.items()} != recipe['inventory']['directory_counts']:
        raise ValueError('URL directory inventory differs')
    result = []
    for stratum in STRATA:
        ranked = sorted(groups[stratum], key=lambda u:(_hash(recipe['selection']['url_namespace'],u),u))
        for i,url in enumerate(ranked[:recipe['selection']['objects_per_stratum'][stratum]]):
            result.append({'name':f'{stratum}-{i:02d}', 'stratum':stratum,
                          'family':'common_crawl' if stratum in CC else stratum, 'url':url,
                          'selection_sha256':_hash(recipe['selection']['url_namespace'],url)})
    return result


def split_for_content_hash(recipe, content_hash):
    """Literal existing SplitPolicy algorithm, without importing model modules."""
    _pin(content_hash)
    split = recipe['split']; draw = int(hashlib.sha256(f'{split["seed"]}:{content_hash}'.encode()).hexdigest(),16)
    draw %= sum(weight for _,weight in split['weights'])
    for name,weight in split['weights']:
        if draw < weight: return name
        draw -= weight
    raise AssertionError('Unreachable split')


def document_order_key(recipe, content_hash, *, round_id=0):
    _pin(content_hash); _integer(round_id,'round_id',0)
    return _hash(recipe['ordering']['document_namespace'],str(round_id),content_hash), content_hash


def chunk_order_key(recipe, chunk_key, *, round_id=0):
    if not isinstance(chunk_key,str) or not chunk_key: raise ValueError('Chunk key must be nonempty')
    _integer(round_id,'round_id',0)
    return _hash(recipe['ordering']['chunk_namespace'],str(round_id),chunk_key), chunk_key


def weighted_stratum_order(quotas):
    """Smooth weighted round-robin using integers; exact quotas, lexical ties."""
    if not isinstance(quotas,dict) or not quotas: raise ValueError('Require named quotas')
    for name,value in quotas.items():
        if not isinstance(name,str) or not name: raise ValueError('Invalid stratum name')
        _integer(value,'chunk quota',0)
    total = sum(quotas.values()); deficit = dict.fromkeys(quotas,0); used = dict.fromkeys(quotas,0)
    for _ in range(total):
        for name in quotas: deficit[name] += quotas[name]
        selected = min((s for s in quotas if used[s] < quotas[s]),key=lambda s:(-deficit[s],s))
        deficit[selected] -= total; used[selected] += 1
        yield selected


def exclusion_bytes(identities):
    values = list(identities)
    for value in values: _pin(value)
    if len(values) != len(set(values)): raise ValueError('Duplicate exclusion identity')
    return ''.join(value+'\n' for value in sorted(values)).encode('ascii')


def verify_exclusions(recipe, identities):
    raw = exclusion_bytes(identities); contract = recipe['exclusions']
    if len(raw)//65 != contract['unique_documents'] or hashlib.sha256(raw).hexdigest() != contract['content_ids_sha256']:
        raise ValueError('Readiness exclusion membership differs')
    return frozenset(raw.decode().splitlines())


def _read_pinned(path, pin):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Require regular metadata file')
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('Metadata changed while reading')
    if hashlib.sha256(raw).hexdigest() != _pin(pin): raise ValueError('Metadata bytes differ from authority')
    return raw


def load_exclusions(corpus_dir, recipe):
    """Verify old manifest→shard manifest→metadata chain, never read token arrays."""
    root = Path(corpus_dir)
    if root.is_symlink(): raise ValueError('Exclusion root must not be a symlink')
    summary = json.loads(_read_pinned(root/'manifest.json',recipe['exclusions']['corpus_manifest_sha256']))
    if summary.get('completed') is not True: raise ValueError('Incomplete exclusion corpus')
    identities = []
    for shard in summary['shards']:
        if not re.fullmatch('shard-[0-9]{6}',shard['path']): raise ValueError('Unsafe exclusion shard path')
        directory = root/shard['path']
        if directory.is_symlink(): raise ValueError('Symlink exclusion shard')
        manifest = json.loads(_read_pinned(directory/'manifest.json',shard['manifest_sha256']))
        metadata = _read_pinned(directory/'documents.jsonl',manifest['files']['documents.jsonl']['sha256'])
        for line in metadata.splitlines():
            row = json.loads(line)
            if row['kind']=='document': identities.append(row['content_token_sha256'])
    return verify_exclusions(recipe,identities)


def append_round(previous, *, round_id, recipe_sha, order_sha256, chunks):
    """Append a ledger without editing prior rounds; do not assert token disjointness.

    An extension builder must separately exclude all prior document identities.
    This metadata helper cannot infer chunk/content overlap from opaque hashes.
    """
    _integer(round_id,'round_id',0); _integer(chunks,'chunks'); _pin(recipe_sha); _pin(order_sha256)
    if not isinstance(previous,list): raise ValueError('Round ledger must be a list')
    for i,row in enumerate(previous):
        _keys(row,('round_id','recipe_sha256','order_sha256','chunks','previous_ledger_sha256'),'round record')
        if row['round_id'] != i or row['previous_ledger_sha256'] != digest(previous[:i]):
            raise ValueError('Round ledger prefix identity differs')
        _pin(row['recipe_sha256']); _pin(row['order_sha256']); _integer(row['chunks'],'round chunks')
    if round_id != len(previous): raise ValueError('Rounds must append without replacement or gaps')
    return json.loads(canonical(previous))+[{'round_id':round_id,'recipe_sha256':recipe_sha,
        'order_sha256':order_sha256,'chunks':chunks,'previous_ledger_sha256':digest(previous)}]


def build_plan(recipe, inventory_bytes):
    validate_recipe(recipe)
    return {'schema':PLAN_SCHEMA,'recipe':recipe,'recipe_sha256':recipe_sha256(recipe),
            'sources':source_selection(recipe,inventory_bytes),
            'stratum_weights':{s:{'numerator':v.numerator,'denominator':v.denominator} for s,v in stratum_weights(recipe).items()},
            'quotas':{p:panel_quotas(recipe,p) for p in ('train','dev-main','confirmation-main')},
            'heldout_requirements':heldout_requirements(recipe),
            'nominal_candidate_tokens':sum(recipe['selection']['objects_per_stratum'].values())*recipe['selection']['candidate_tokens_per_object'],
            'status':'plan_only_no_acquisition_or_training'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,default=DEFAULT_RECIPE)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    value=build_plan(load_recipe(args.recipe),args.inventory.read_bytes())
    with args.output.open('xb') as handle: handle.write(canonical(value))
    print(json.dumps({'schema':PLAN_SCHEMA,'sources':len(value['sources']),'recipe_sha256':value['recipe_sha256']}))


if __name__=='__main__': main()
