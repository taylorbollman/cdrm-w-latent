"""Literal deterministic selection/round/membership oracles, no network/GPU."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import olmo_pilot_data_plan as plan


@pytest.fixture
def recipe():
    return plan.load_recipe()


def fake_inventory(recipe):
    urls = [f'{plan.URL_ROOT}{s}/sample-{i:04d}.json.gz'
            for s,n in recipe['inventory']['directory_counts'].items() for i in range(n)]
    raw = ('\n'.join(urls)+'\n').encode()
    recipe['inventory']['sha256'] = hashlib.sha256(raw).hexdigest()
    return raw


def test_default_budgets_and_exact_rational_family_mass(recipe):
    w = plan.stratum_weights(recipe)
    assert sum(w.values()) == 1
    for name, weight in recipe['mixture']['family_weights'].items():
        got = sum(w[s] for s in plan.CC) if name=='common_crawl' else w[name]
        assert got == Fraction(weight,26679)
    assert sum(plan.panel_quotas(recipe,'train').values()) == 131072
    assert sum(plan.panel_quotas(recipe,'dev-main').values()) == 1024
    assert plan.panel_quotas(recipe,'dev-main') == plan.panel_quotas(recipe,'confirmation-main')
    assert plan.panel_quotas(recipe,'dev-source/books')['books'] == 64
    assert sum(plan.panel_quotas(recipe,'dev-source/books').values()) == 64
    objects=sum(recipe['selection']['objects_per_stratum'].values())
    assert objects==37 and objects*recipe['selection']['candidate_tokens_per_object']==310378496
    assert objects*recipe['object_limits']['tokens'] == 387973120
    assert objects*recipe['object_limits']['compressed_bytes'] < recipe['bounds']['compressed_bytes']
    assert objects*recipe['object_limits']['retained_raw_bytes'] < recipe['bounds']['retained_raw_bytes']


def test_largest_remainder_uses_literal_exact_arithmetic_and_lexical_ties():
    assert plan.largest_remainder(2,{'c':1,'b':1,'a':1}) == {'a':1,'b':1,'c':0}
    assert plan.largest_remainder(0,{'a':1}) == {'a':0}
    n=2**60+1
    assert plan.largest_remainder(n,{'z':1,'a':1}) == {'a':(n+1)//2,'z':n//2}
    assert plan.largest_remainder(7,{'a':Fraction(1,7),'b':Fraction(2,7),'c':Fraction(4,7)}) == {'a':1,'b':2,'c':4}


@pytest.mark.parametrize('total,weights',[(True,{'a':1}),(-1,{'a':1}),(1,{}),(1,{'a':0}),(1,{'a':.5}),(1,{'a':True})])
def test_reject_inexact_or_invalid_apportionment(total,weights):
    with pytest.raises(ValueError): plan.largest_remainder(total,weights)


def test_selection_literal_hash_oracle_unique_names_and_inventory_order_independence(recipe):
    raw=fake_inventory(recipe); result=plan.source_selection(recipe,raw)
    assert len(result)==len({r['name'] for r in result})==len({r['url'] for r in result})==37
    assert Counter(r['stratum'] for r in result)==recipe['selection']['objects_per_stratum']
    for s,n in recipe['selection']['objects_per_stratum'].items():
        candidates=[u for u in raw.decode().splitlines() if '/'+s+'/' in u]
        expected=sorted(candidates,key=lambda u:(hashlib.sha256(('cdrm-dolma-pilot-v1-url\0'+u).encode()).hexdigest(),u))[:n]
        assert [r['url'] for r in result if r['stratum']==s]==expected
    changed=b'\n'.join(reversed(raw.splitlines()))+b'\n'
    recipe['inventory']['sha256']=hashlib.sha256(changed).hexdigest()
    assert plan.source_selection(recipe,changed)==result


@pytest.mark.parametrize('change',['byte_pin','duplicate','host','directory','query'])
def test_inventory_drift_or_unsafe_url_is_rejected(recipe,change):
    raw=fake_inventory(recipe)
    if change=='byte_pin': raw+=b'\n'
    else:
        urls=raw.decode().splitlines()
        if change=='duplicate':urls[0]=urls[1]
        elif change=='host':urls[0]=urls[0].replace('olmo-data.org','example.com')
        elif change=='directory':urls[0]=urls[0].replace('/books/','/unknown/')
        else:urls[0]+='?unexpected=1'
        raw=('\n'.join(urls)+'\n').encode();recipe['inventory']['sha256']=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError): plan.source_selection(recipe,raw)


def test_membership_matches_frozen_split_policy_and_is_independent_of_order_round(recipe):
    from cdrm.pretrained.campaign_ingest import SplitPolicy
    oracle=SplitPolicy(20260929,(('train',90),('dev',5),('confirmation',5)))
    hashes=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(500)]
    actual={h:plan.split_for_content_hash(recipe,h) for h in hashes}
    assert actual=={h:oracle.split_for_text_hash(h) for h in reversed(hashes)}
    assert set(actual.values())=={'train','dev','confirmation'}
    assert plan.document_order_key(recipe,hashes[0]) != plan.document_order_key(recipe,hashes[0],round_id=1)
    assert plan.split_for_content_hash(recipe,hashes[0])==actual[hashes[0]]
    assert plan.document_order_key(recipe,hashes[0])[0] != plan.chunk_order_key(recipe,hashes[0])[0]
    with pytest.raises(ValueError):plan.chunk_order_key(recipe,'a\0b')


def test_integer_deficit_interleave_is_exact_and_independent_of_mapping_order(recipe):
    q={'a':3,'b':2,'c':1}
    assert list(plan.weighted_stratum_order(q))==['a','b','a','c','b','a']
    quotas=plan.panel_quotas(recipe,'train')
    actual=list(plan.weighted_stratum_order(quotas))
    assert Counter(actual)==quotas and len(actual)==131072
    assert actual==list(plan.weighted_stratum_order(dict(reversed(list(quotas.items())))))
    assert list(plan.weighted_stratum_order({'a':0}))==[]


def test_reserve_includes_main_panel_and_source_panel_with_document_minima(recipe):
    requirements=plan.heldout_requirements(recipe)
    assert requirements['books']=={'tokens':131072,'documents':8}
    assert requirements['cc_en_tail']['tokens'] >= plan.panel_quotas(recipe,'dev-main')['cc_en_tail']*1024
    assert requirements['cc_en_tail']['tokens'] > 131072


def test_small_declared_fixture_is_valid_but_unknown_fields_fail(recipe):
    recipe['panels'].update(length=4,train_tokens=80,heldout_main_tokens=36,
        heldout_source_tokens=4,minimum_reserve_tokens_per_stratum=8,
        minimum_documents_per_stratum=dict.fromkeys(plan.STRATA,1))
    assert plan.validate_recipe(recipe) is recipe
    recipe['undeclared_fallback']=True
    with pytest.raises(ValueError):plan.validate_recipe(recipe)


@pytest.mark.parametrize('field', ['tokenizer','split','namespace','bounds','fractional_budget','qualification'])
def test_recipe_mutations_rejected(recipe,field):
    if field=='tokenizer':recipe['tokenizer']['eos_id']=0
    elif field=='split':recipe['split']['weights'][0][1]=98
    elif field=='namespace':recipe['ordering']['document_namespace']=recipe['ordering']['chunk_namespace']
    elif field=='bounds':recipe['bounds']['compressed_bytes']=100
    elif field=='fractional_budget':recipe['panels']['train_tokens']=3.5
    else:recipe['qualifications']['near_duplicate_policy']='clean'
    with pytest.raises(ValueError):plan.validate_recipe(recipe)


def exclusion_fixture(tmp_path,recipe):
    ids=[hashlib.sha256(b'a').hexdigest(),hashlib.sha256(b'b').hexdigest()]
    metadata=b''.join(plan.canonical({'kind':'document','content_token_sha256':h}) for h in ids)
    directory=tmp_path/'shard-000000';directory.mkdir()
    (directory/'documents.jsonl').write_bytes(metadata)
    shard={'files':{'documents.jsonl':{'sha256':hashlib.sha256(metadata).hexdigest()}}}
    raw=plan.canonical(shard);(directory/'manifest.json').write_bytes(raw)
    summary={'completed':True,'shards':[{'path':'shard-000000','manifest_sha256':hashlib.sha256(raw).hexdigest()}]}
    raw=plan.canonical(summary);(tmp_path/'manifest.json').write_bytes(raw)
    recipe['exclusions'].update(corpus_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        content_ids_sha256=hashlib.sha256(plan.exclusion_bytes(ids)).hexdigest(),unique_documents=2)
    return ids


def test_exclusion_loader_verifies_metadata_chain_without_any_token_arrays(tmp_path,recipe):
    ids=exclusion_fixture(tmp_path,recipe)
    assert not (tmp_path/'shard-000000/tokens.bin').exists()
    assert plan.load_exclusions(tmp_path,recipe)==frozenset(ids)
    path=tmp_path/'shard-000000/documents.jsonl';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='bytes differ'):plan.load_exclusions(tmp_path,recipe)


def test_exclusion_membership_rejects_duplicate_or_missing_content(tmp_path,recipe):
    ids=exclusion_fixture(tmp_path,recipe)
    with pytest.raises(ValueError,match='Duplicate'):plan.verify_exclusions(recipe,ids+[ids[0]])
    with pytest.raises(ValueError,match='membership differs'):plan.verify_exclusions(recipe,ids[:1])


def test_exclusion_metadata_mutation_during_read_fails(tmp_path,recipe,monkeypatch):
    exclusion_fixture(tmp_path,recipe)
    original=Path.read_bytes
    def mutate(path):
        raw=original(path)
        if path.name=='documents.jsonl':path.write_bytes(raw+b' ')
        return raw
    monkeypatch.setattr(Path,'read_bytes',mutate)
    with pytest.raises(ValueError,match='changed while reading'):plan.load_exclusions(tmp_path,recipe)


def test_append_round_preserves_prior_bytes_and_rejects_replacement_or_tampering(recipe):
    first=plan.append_round([],round_id=0,recipe_sha=plan.recipe_sha256(recipe),order_sha256='a'*64,chunks=7)
    before=plan.canonical(first)
    second=plan.append_round(first,round_id=1,recipe_sha=plan.recipe_sha256(recipe),order_sha256='b'*64,chunks=9)
    assert plan.canonical(second[:1])==before and plan.canonical(first)==before
    with pytest.raises(ValueError,match='append'):plan.append_round(first,round_id=0,recipe_sha='a'*64,order_sha256='b'*64,chunks=2)
    second[0]['order_sha256']='c'*64
    with pytest.raises(ValueError,match='prefix'):plan.append_round(second,round_id=2,recipe_sha='a'*64,order_sha256='b'*64,chunks=2)


def test_plan_contains_full_authority_and_can_only_write_a_new_output(tmp_path,recipe):
    raw=fake_inventory(recipe)
    rp=tmp_path/'recipe.json';ip=tmp_path/'inventory.txt';out=tmp_path/'plan.json'
    rp.write_bytes(plan.canonical(recipe));ip.write_bytes(raw)
    args=['--recipe',str(rp),'--inventory',str(ip),'--output',str(out)]
    plan.main(args)
    result=json.loads(out.read_bytes())
    assert result==plan.build_plan(recipe,raw)
    assert result['recipe_sha256']==plan.recipe_sha256(recipe)
    assert result['status']=='plan_only_no_acquisition_or_training'
    with pytest.raises(FileExistsError):plan.main(args)


def test_plan_import_does_not_load_torch_or_network_clients():
    subprocess.run([sys.executable,'-c','import sys; import scripts.olmo_pilot_data_plan; '
        'assert "torch" not in sys.modules; assert "google.cloud.storage" not in sys.modules; '
        'assert "urllib.request" not in sys.modules'],cwd=plan.ROOT,check=True,capture_output=True)
