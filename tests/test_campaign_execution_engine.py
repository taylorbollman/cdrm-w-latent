"""Independent CPU checks of shared component ownership and real packed clocks."""
from dataclasses import asdict, replace
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained import document_shards as shards
from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained.campaign_recipe import ARMS, CampaignRecipe, build_campaign_model, build_campaign_adamw
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.packed_campaign_data import PackedCampaignData, build_packed_index
from scripts import olmo_campaign_execution as engine
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def tiny(arm, *, policy='continuous-stream-v1'):
    torch.manual_seed(2718)
    recipe = CampaignRecipe(arm, sequence_length=16, rt_layers=(0, 1),
                            document_policy=policy, effective_valid_tokens=80)
    native = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend='math',
        attention_precision='fp32', tile_backend='eager', backward_tile_backend='eager')
    return build_campaign_model(native, recipe).train(), recipe


@pytest.mark.parametrize('arm', ARMS)
def test_all_arm_ownership_matches_literal_modules_and_unique_adam_owners(arm):
    model, recipe = tiny(arm)
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    contract = engine.model_contract(model, recipe, optimizer)
    native = {id(p) for p in model.backbone.backbone.parameters()}
    fusion = {id(p) for p in model.backbone.fusion.parameters()}
    predictor = set() if model.predictor is None else {id(p) for p in model.predictor.parameters()}
    expected = native | (fusion if 'F' in arm else set()) | predictor
    assert bool(predictor) == ('N' in arm)
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected
    owners = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(owners) == len(set(owners)) and set(owners) == expected
    assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
    assert contract['trainable_parameters'] == sum(p.numel() for p in model.parameters() if id(p) in expected)
    assert contract['dormant_fusion_parameters'] == (0 if 'F' in arm else sum(p.numel() for p in model.backbone.fusion.parameters()))
    assert contract['weights'] == {'ce': 1., 'latent': float('N' in arm), 'kl': float('N' in arm)}
    assert recipe.mode().num_passes == (4 if 'F' in arm else 1)
    assert recipe.mode().rt_mode.selected_layers == ((0, 1) if 'R' in arm else ())


@pytest.mark.parametrize('mutation', ['native_frozen', 'dormant_trainable', 'predictor_frozen',
    'fusion_frozen', 'wrong_component', 'wrong_policy'])
def test_ownership_rejects_incompatible_flags_and_optimizer_mapping(mutation):
    arm = 'B' if mutation == 'dormant_trainable' else 'NFR'
    model, recipe = tiny(arm)
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    if mutation == 'native_frozen': next(model.backbone.backbone.parameters()).requires_grad_(False)
    elif mutation == 'dormant_trainable': model.backbone.fusion.requires_grad_(True)
    elif mutation == 'predictor_frozen': next(model.predictor.parameters()).requires_grad_(False)
    elif mutation == 'fusion_frozen': next(model.backbone.fusion.parameters()).requires_grad_(False)
    elif mutation == 'wrong_component': optimizer.param_groups[0]['component'] = 'predictor'
    else: recipe = replace(recipe, document_policy='isolated-v1')
    with pytest.raises(ValueError):
        engine.model_contract(model, recipe, optimizer)


@pytest.fixture
def packed_paths(tmp_path, monkeypatch):
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(ids=[int(v) for v in text.split()])
    monkeypatch.setattr(shards, '_load_tokenizer', lambda _: Tokenizer())
    documents = [[2+i, shards.EOS_ID, *range(3+i, 24+i)] for i in range(12)]
    raw = b''.join((json.dumps({'id':str(i), 'text':' '.join(map(str, tokens))})+'\n').encode()
                   for i, tokens in enumerate(documents))
    source_path = tmp_path/'source.jsonl'; source_path.write_bytes(raw)
    source = LocalJSONLSource(SourcePin('fixture', 'https://example.invalid/fixed', 'fixed',
                                      hashlib.sha256(raw).hexdigest()), source_path)
    corpus, index = tmp_path/'corpus', tmp_path/'index'
    shards.prepare_document_shards([source], corpus, tokenizer_path='unused',
        split_policy=SplitPolicy(19, (('train', 1),)), max_documents_per_shard=3)
    build_packed_index(corpus, index, split='train', length=16)
    return corpus, index


@pytest.mark.parametrize('arm', ARMS)
def test_actual_packed_counts_empty_slot_noise_and_committed_resume(packed_paths, arm):
    model, recipe = tiny(arm)
    oracle = dict(optimizer_updates=0, microbatches=0, documents=0, input_tokens=0,
                  ce_positions=0, latent_pairs=0, kl_triples=0)
    with PackedCampaignData(*packed_paths) as data:
        first = data.peek_update(data.cursor(), 80)
        plans = [first, data.peek_update(first.next_cursor, 80)]
        assert data.cursor() == first.start_cursor
        crossings = 0
        for number, plan in enumerate(plans, 1):
            counts = dict(ce=0, latent=0, kl=0)
            returned = dict(counts)
            keys = []
            for rank in range(2):
                packed, noises = engine.materialize(data, plan, recipe, rank=rank, batch_size=2, width=model.config.model_dim)
                assert len(packed.batches) == 2
                if rank == 1:
                    assert not packed.batches[-1].valid_mask.any()
                    if recipe.feedback: assert all(torch.count_nonzero(v) == 0 for v in noises[-1])
                for batch, rowkeys in zip(packed.batches, packed.keys):
                    keys.extend(rowkeys)
                    observed = model.counts(batch)  # Mask-only; no token embeddings/model forward.
                    for term in returned: returned[term] += observed[term]
                    for valid, docs in zip(batch.valid_mask.tolist(), batch.document_ids.tolist()):
                        for i in range(len(valid)-1):
                            if valid[i] and valid[i+1]:
                                counts['ce'] += 1
                                if docs[i] == docs[i+1]: counts['latent'] += 1
                                else: crossings += 1
                        counts['kl'] += sum(valid[i] and valid[i+1] and valid[i+2]
                            and docs[i] == docs[i+1] == docs[i+2] for i in range(len(valid)-2))
            assert sorted(keys) == sorted(row.key for row in plan.rows) and len(set(keys)) == 5
            assert returned == {'ce':counts['ce'], 'latent':counts['latent'] if recipe.nextlat else 0,
                                'kl':counts['kl'] if recipe.nextlat else 0}
            oracle['optimizer_updates'] += 1; oracle['microbatches'] += 4
            oracle['documents'] += 5; oracle['input_tokens'] += 80
            for term, field in [('ce','ce_positions'), ('latent','latent_pairs'), ('kl','kl_triples')]:
                oracle[field] += returned[term]
            counters = engine.expected_counters(plans, number, recipe, 2)
            assert asdict(counters) == oracle
            data.commit(plan.start_cursor, plan)
            record = engine.cursor_record(data.cursor(), rank=1, batch_size=2)
            engine.validate_cursor(data, record, counters, plans, recipe, rank=1, batch_size=2)
        assert crossings > 0
        bad = replace(counters, ce_positions=counters.ce_positions+1)
        with pytest.raises(ValueError):
            engine.validate_cursor(data, record, bad, plans, recipe, rank=1, batch_size=2)
    with PackedCampaignData(*packed_paths) as fresh:
        assert engine.validate_cursor(fresh, record, counters, plans, recipe,
            rank=1, batch_size=2, restore=True) == plans[-1].next_cursor


@pytest.mark.parametrize('target_arm', ['NF', 'NFR'])
def test_imported_policy_transition_preserves_state_rng_and_changes_only_intended_masks(target_arm):
    model, historical = tiny('NF', policy='isolated-v1')
    target = replace(historical, arm=target_arm, document_policy='continuous-stream-v1')
    state = tree_digests(model.state_dict()); rng = torch.get_rng_state().clone()
    ownership = {n:(id(p),p.requires_grad) for n,p in model.named_parameters()}
    model.predictor.eval()  # Check heterogeneous modes are preserved too.
    modes = {n:m.training for n,m in model.named_modules()}
    ids = torch.tensor([[2, 3, 60, 4, 5, 6, 60, 7]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    docs = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2]])
    batch = NextLatBatch(ids, valid, docs)
    old_masks = build_nextlat_masks(batch, document_policy='isolated-v1')
    result = engine.transition_imported_model(model, historical, target)
    new_masks = build_nextlat_masks(batch, document_policy=model.config.document_policy)
    assert all(result['checks'].values()) and not result['loaded_optimizer']
    assert tree_digests(model.state_dict()) == state and torch.equal(rng, torch.get_rng_state())
    assert ownership == {n:(id(p),p.requires_grad) for n,p in model.named_parameters()}
    assert modes == {n:m.training for n,m in model.named_modules()}
    assert int(old_masks['ce'].sum()) == 5 and int(new_masks['ce'].sum()) == 7
    assert torch.equal(old_masks['latent'], new_masks['latent']) and torch.equal(old_masks['kl'], new_masks['kl'])
    assert target.mode().rt_mode.selected_layers == ((0, 1) if target_arm == 'NFR' else ())
    fresh = build_campaign_adamw(model, target, fused=False)
    assert not fresh.state
    engine.model_contract(model, target, fresh)


@pytest.mark.parametrize('mutation', ['frozen_backbone', 'wrong_arm', 'already_packed', 'wrong_seed', 'wrong_jitter'])
def test_imported_transition_rejects_unsupported_flags_before_state_changes(mutation):
    model, historical = tiny('NF', policy='isolated-v1')
    target = replace(historical, arm='NFR', document_policy='continuous-stream-v1')
    if mutation == 'frozen_backbone': next(model.backbone.backbone.parameters()).requires_grad_(False)
    elif mutation == 'wrong_arm': target = replace(target, arm='FR')
    elif mutation == 'already_packed': model.config = replace(model.config, document_policy='continuous-stream-v1')
    elif mutation == 'wrong_seed': target = replace(target, predictor_seed=target.predictor_seed+1)
    else: target = replace(target, feedback_jitter=.03)
    before = tree_digests(model.state_dict()); config = copy.deepcopy(model.config)
    with pytest.raises(ValueError): engine.transition_imported_model(model, historical, target)
    assert model.config == config and tree_digests(model.state_dict()) == before
