"""Isolated exact-online/finite semantics, deterministic selection and strict imports."""
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from cdrm.pretrained.lm_training import TrainingCounters, parameter_layout
from cdrm.pretrained.nextlat import NextLatBatch
from scripts import olmo_fbt_stability_online as online
from scripts.olmo_campaign_evaluation import evaluation_runtime
from scripts.olmo_lm_common import tree_digests
from test_fbt_stability_probe import setup
from test_pilot_ordered_data import fixture, cpu_tokenizer
from test_pilot_eval_control import spec as data_spec
from scripts.olmo_fbt_stability_probe import resolve_probe_plan


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(25)


def isolated():
    model, recipe, source = setup()
    ids = source.input_ids[:1].clone()
    return model, recipe, NextLatBatch(ids, torch.ones_like(ids, dtype=torch.bool), torch.zeros_like(ids))


def test_metadata_selection_uses_earliest_qualifying_span_without_token_content():
    rows = [{'descriptor': {'index': r}, 'segments': [
        {'chunk_offset': 0, 'length': 4, 'document_token_offset': 100},
        {'chunk_offset': 4, 'length': 160, 'document_token_offset': 0},
        {'chunk_offset': 164, 'length': 180, 'document_token_offset': 10}]} for r in range(8)]
    got = online.select_spans(rows, crop_length=128)
    assert [(x['panel_row'], x['row_start_offset']) for x in got] == [(0, 4), (0, 164)]
    assert [x['document_start_offset'] for x in got] == [0, 10]
    assert all(x['crop_length'] == 128 for x in got)
    with pytest.raises(ValueError, match='too few'):
        online.select_spans(rows, crop_length=256)


def test_panel_content_matches_independent_document_crops_and_selection_is_reusable(tmp_path):
    data = fixture(tmp_path, length=256)
    source = data_spec(data)
    plan = resolve_probe_plan(source, length=256, updates=128, world_size=2)
    spec = {'manifest': {'data': source}, 'probe_plan': plan}
    batches, authority = online.load_panel(spec)
    repeated, second = online.load_panel(spec)
    assert second == authority
    assert len(batches) == 2 and authority['crop_length'] == 128
    assert authority['packed_membership_sha256'] == plan['panel']['fixed_plan']['updates'][0]['membership_sha256']
    with online.OrderedCampaignData(data.corpus, data.output/'panels/dev-main') as reader:
        for batch, another, record in zip(batches, repeated, authority['records']):
            original = reader.read_chunk(record['descriptor']['index'])
            start = record['row_start_offset']
            assert batch.input_ids.tolist()[0] == list(original.tokens[start:start+128])
            assert batch.document_ids.unique().numel() == 1 and bool(batch.valid_mask.all())
            assert torch.equal(another.input_ids, batch.input_ids)
            assert record['input_ids'] == batch.input_ids.tolist()[0]


def test_online_causal_prefix_agrees_with_sufficient_passes_and_preserves_state():
    model, recipe, batch = isolated()
    before = tree_digests(model.state_dict()); rng = torch.get_rng_state().clone()
    with evaluation_runtime(model) as evidence:
        observed = online.compare_crop(model, recipe, batch, passes=(2, 4, 8))
    assert evidence['integrity_passed'] and tree_digests(model.state_dict()) == before
    assert torch.equal(torch.get_rng_state(), rng) and all(p.grad is None for p in model.parameters())
    assert observed['finite_mode']['document_policy'] == 'isolated-v1'
    assert observed['online_mode']['rt_mode']['selected_layers'] == ()
    for case in observed['passes']:
        k = case['total_pass']
        assert case['regions']['all']['ce_targets'] == 7
        for p in case['positions'][:k]:
            assert p['hidden']['relative_l2'] < 1e-5
            assert p['logits']['relative_l2'] < 1e-5
        assert case['positions'][-1]['finite_ce'] is None
    final = observed['passes'][-1]['regions']['all']
    assert final['hidden']['relative_l2'] < 1e-5
    assert abs(final['ce_gap']) < 1e-5
    assert observed['passes'][0]['regions']['all']['hidden']['relative_l2'] > 1e-3


def test_online_crop_has_no_cache_carry_and_future_tokens_do_not_change_prefix():
    model, recipe, batch = isolated()
    changed = deepcopy(batch); changed.input_ids[0, 5:] = torch.tensor([30, 31, 32])
    with evaluation_runtime(model):
        first = online.compare_crop(model, recipe, batch, passes=(2, 8))
        middle = online.compare_crop(model, recipe, changed, passes=(2, 8))
        last = online.compare_crop(model, recipe, batch, passes=(2, 8))
    assert first['passes'] == last['passes']
    for p, q in zip(first['passes'], middle['passes']):
        # Position4 predicts the changed token5: its state stays causal, but CE
        # need not agree. Positions0..3 have unchanged states and targets.
        assert p['positions'][:4] == q['positions'][:4]


def test_compare_states_literal_relative_norms_ce_and_count_weighted_pooling():
    class Core:
        def project_logits(self, hidden):
            return hidden @ torch.tensor([[1., -2., .5], [0., 1., -1.]])
    reference = torch.tensor([[[1., 2.], [2., 3.], [1., -1.]]])
    actual = reference.clone(); actual[0, 2] += torch.tensor([.3, -.7])
    ids = torch.tensor([[0, 1, 2]])
    result = online.compare_states(Core(), actual, reference, ids, chunk=1)
    want = torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference)
    assert result['regions']['all']['hidden']['relative_l2'] == pytest.approx(float(want))
    assert result['regions']['all']['ce_gap'] == 0.  # Only the no-target final state differs.
    assert result['positions'][2]['hidden']['relative_l2'] > 0
    assert result['positions'][2]['finite_ce'] is None
    pooled = online.summarize_positions([*result['positions'], result['positions'][-1]])
    assert pooled['positions'] == 4 and pooled['ce_targets'] == 2
    assert pooled['ce_gap'] == 0.


@pytest.mark.parametrize('invalid', ['two_documents', 'padding', 'two_rows', 'outside', 'depth'])
def test_online_scope_rejects_unsupported_or_ambiguous_inputs(invalid):
    model, recipe, batch = isolated()
    if invalid == 'outside':
        with pytest.raises(ValueError, match='evaluation runtime'):
            online.compare_crop(model, recipe, batch)
        return
    if invalid == 'two_documents': batch.document_ids[0, 4:] = 1
    elif invalid == 'padding': batch.valid_mask[0, -1] = False
    elif invalid == 'two_rows': batch = NextLatBatch(**{n: v.repeat(2, 1) for n, v in vars(batch).items() if v is not None})
    with evaluation_runtime(model):
        with pytest.raises(ValueError):
            online.compare_crop(model, recipe, batch, passes=(33,) if invalid == 'depth' else (2, 8))


def saved():
    model, recipe, _ = isolated(); update = 32
    cursor = lambda k: {'manifest_sha256': 'a'*64, 'split': 'train', 'next_chunk': k*512, 'next_update': k}
    plan = {'first_cursor': cursor(0), 'updates': [{'next_cursor': cursor(k)} for k in range(1, 193)]}
    spec = {'recipe': recipe, 'declaration': {'fixture': True},
        'resolved': {'contract_sha256': 'b'*64, 'sources': {'native': 'c'*64}}, 'plan': plan}
    identity_payload = {'resolved_contract_sha256': 'b'*64, 'scope': 'clean-F-only-stability',
        'arm': 'F', 'declaration': spec['declaration'], 'sources': spec['resolved']['sources']}
    identity = {'payload': identity_payload, 'sha256': online.legacy.digest(identity_payload)}
    counters = asdict(TrainingCounters(optimizer_updates=update, input_tokens=update*524288))
    metadata = {'parameter_layout': parameter_layout(model), 'world_size': 2,
        'configuration': {'execution_identity': identity, 'recipe': recipe.to_dict(),
            'backbone': model.backbone.backbone.config.to_dict(), 'model': model.config.to_dict()}}
    cursors = [{'rank': rank, 'world_size': 2, 'physical_batch_per_rank': 12,
        'cursor': cursor(update)} for rank in range(2)]
    manifest = {'metadata': metadata, 'counters': counters, 'world_size': 2,
        'rank_cursors': cursors, 'state': {'filename': 'state.pt'}}
    payload = {'metadata': deepcopy(metadata), 'counters': deepcopy(counters),
        'rank_states': [{'rank': rank, 'data_cursor': deepcopy(cursors[rank])} for rank in range(2)],
        'module_training': {name: module.training for name, module in model.named_modules()},
        'model': {name: value.clone() for name, value in model.state_dict().items()},
        'optimizer': {'sentinel_must_not_load': True}, 'scheduler': {'sentinel_must_not_load': True}}
    return model, spec, manifest, payload


def test_exact_saved_lineage_and_weights_import_preserve_aliases_and_do_not_restore_rng(tmp_path, monkeypatch):
    model, spec, manifest, payload = saved()
    for tensor in payload['model'].values(): tensor.add_(.25)
    torch.save(payload, tmp_path/'state.pt')
    observed = []
    def inspect(directory, **kwargs):
        observed.append(kwargs)
        return manifest
    monkeypatch.setattr(online, 'inspect_distributed_checkpoint', inspect)
    ids = {name: id(p) for name, p in model.named_parameters()}; rng = torch.get_rng_state().clone()
    receipt = online.import_saved_weights(model, spec, tmp_path, 'd'*64)
    assert observed == [{'expected_manifest_sha256': 'd'*64, 'verify_state': True}]
    assert receipt['optimizer_updates_performed'] == 0
    assert ids == {name: id(p) for name, p in model.named_parameters()}
    assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
    assert torch.equal(torch.get_rng_state(), rng) and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value, payload['model'][name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize('invalid', ['update', 'tokens', 'rank_cursor', 'rank_count', 'recipe',
    'identity_digest', 'lineage', 'sources', 'declaration', 'arm', 'model', 'world'])
def test_saved_authority_rejects_other_scope_even_with_valid_manifest(invalid):
    model, spec, manifest, _ = saved()
    identity = manifest['metadata']['configuration']['execution_identity']
    if invalid == 'update': manifest['counters']['optimizer_updates'] = 16
    elif invalid == 'tokens': manifest['counters']['input_tokens'] += 1
    elif invalid == 'rank_cursor': manifest['rank_cursors'][0]['cursor']['next_chunk'] += 1
    elif invalid == 'rank_count': manifest['rank_cursors'].pop()
    elif invalid == 'recipe': manifest['metadata']['configuration']['recipe']['feedback_jitter'] = .1
    elif invalid == 'identity_digest': identity['sha256'] = '0'*64
    elif invalid == 'lineage': identity['payload']['resolved_contract_sha256'] = 'e'*64
    elif invalid == 'sources': identity['payload']['sources'] = {'changed': 'f'*64}
    elif invalid == 'declaration': identity['payload']['declaration'] = {'different': True}
    elif invalid == 'arm': identity['payload']['arm'] = 'NF'
    elif invalid == 'model': manifest['metadata']['configuration']['model']['lambda_kl'] = .1
    elif invalid == 'world': manifest['world_size'] = 1
    if invalid != 'identity_digest': identity['sha256'] = online.legacy.digest(identity['payload'])
    with pytest.raises(ValueError): online.validate_saved_authority(manifest, spec, model)
