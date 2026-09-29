"""CPU draft resolution: actual packed masks and independent schedule/accounting."""
import copy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained import document_shards
from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained.campaign_recipe import CampaignRecipe, CampaignTokenSchedule
from cdrm.pretrained.nextlat import build_nextlat_masks
from cdrm.pretrained.packed_campaign_data import PackedCampaignData, build_packed_index
from scripts import olmo_campaign_manifest as resolver


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def dictionary():
    recipe = asdict(CampaignRecipe('B', document_policy=resolver.POLICY['document_policy']))
    recipe.pop('arm'); recipe = json.loads(json.dumps(recipe))
    return {'schema': resolver.SCHEMA, 'purpose': 'review-draft', 'label': 'cpu-test',
        'model': {'artifacts': 'unused', 'manifest_sha256': 'a'*64, 'checkpoint_sha256': resolver.CHECKPOINT_SHA256,
                  'repo': resolver.REPO_ID, 'revision': resolver.REVISION},
        'data': {'corpus': 'unused', 'corpus_manifest_sha256': 'b'*64, 'index': 'unused-index',
                 'index_manifest_sha256': 'c'*64, 'split': 'train', 'policy': copy.deepcopy(resolver.POLICY)},
        'arms': list(resolver.ARMS), 'recipe': recipe,
        'budget': {'updates': 2, 'target_valid_tokens_per_update': 524288, 'whole_chunk_overshoot': 'record', 'insufficient_corpus': 'error'},
        'partition': {'world_size': 2, 'physical_batch_per_rank': {a: 8 for a in resolver.ARMS}},
        'startup': copy.deepcopy(resolver.STARTUP),
        'execution': {**resolver.EXECUTION_COMMON, **resolver.PATHS['bf16_mixed'], 'precision': 'bf16_mixed', 'graph_mode': 'prepared_cuda_graph'},
        'retention': {'storage_prefix': 'gs://fast-chunks/cdrm-w-latent/drafts/test', 'checkpoint_seconds': 600,
            'checkpoint_every_updates': 1, 'keep_local_completed': 2, 'publication': 'verified_generation_before_prune',
            'resume': 'same_topology_exact_configuration'},
        'evaluation': {'kind': 'deferred', 'reason': 'No production evaluation selected.'},
        'tracking': {'entity': 'taylorbollman', 'project': 'pretrained-fbt-rt-nextlat', 'group': 'test'},
        'resource_ledger': {'path': 'unused-ledger', 'sha256': resolver.LEDGER_SHA},
        'implementation_sources': {'pinned.py': 'd'*64}}


def test_all_eight_declared_modes_and_coherent_precision_paths():
    m = dictionary(); recipes = resolver.validate_manifest(m, m['implementation_sources'])
    assert tuple(recipes) == resolver.ARMS
    for name, recipe in recipes.items():
        assert recipe.mode().num_passes == (4 if 'F' in name else 1)
        assert recipe.mode().rt_mode.selected_layers == ((0, 15) if 'R' in name else ())
        assert recipe.nextlat == ('N' in name)
    m['execution'].update(resolver.PATHS['fp32'], precision='fp32', graph_mode='prepared_eager')
    assert len(resolver.validate_manifest(m, m['implementation_sources'])) == 8


@pytest.mark.parametrize('change', ['source', 'unknown_field', 'cycle', 'shuffle', 'tail', 'adapted', 'inherited_adam',
    'bf16_fp32_rt', 'fp32_flash', 'fp32_graph', 'missing_arm_batch', 'boolean_world', 'one_gpu',
    'implicit_eval', 'unsafe_retention', 'foreign_bucket', 'cadence', 'budget_mismatch', 'unbounded', 'recipe_length'])
def test_invalid_drafts_fail_closed_before_opening_artifacts(change):
    m = dictionary(); expected = copy.deepcopy(m['implementation_sources'])
    if change == 'source': m['implementation_sources']['pinned.py'] = 'e'*64
    elif change == 'unknown_field': m['start_training'] = True
    elif change == 'cycle': m['data']['policy']['cycling'] = True
    elif change == 'shuffle': m['data']['policy']['order'] = 'random'
    elif change == 'tail': m['data']['policy']['tail'] = 'drop'
    elif change == 'adapted': m['startup']['kind'] = 'fusion-warmup-128'
    elif change == 'inherited_adam': m['startup']['resume_optimizer'] = True
    elif change == 'bf16_fp32_rt': m['execution']['rt_attention_precision'] = 'fp32'
    elif change == 'fp32_flash': m['execution']['precision'] = 'fp32'
    elif change == 'fp32_graph': m['execution'].update(resolver.PATHS['fp32'], precision='fp32')
    elif change == 'missing_arm_batch': m['partition']['physical_batch_per_rank'].pop('NFR')
    elif change == 'boolean_world': m['partition']['world_size'] = True
    elif change == 'one_gpu': m['partition']['world_size'] = 1
    elif change == 'implicit_eval': m['evaluation'] = {}
    elif change == 'unsafe_retention': m['retention']['storage_prefix'] += '/../unsafe'
    elif change == 'foreign_bucket': m['retention']['storage_prefix'] = 'gs://other-bucket/foo'
    elif change == 'cadence': m['retention']['checkpoint_seconds'] = 601
    elif change == 'budget_mismatch': m['budget']['target_valid_tokens_per_update'] -= 1
    elif change == 'unbounded': m['budget']['updates'] = resolver.MAX_UPDATES+1
    else: m['recipe']['sequence_length'] = 512
    with pytest.raises(ValueError): resolver.validate_manifest(m, expected)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return SimpleNamespace(ids=[100+b for b in text.encode()])
    monkeypatch.setattr(document_shards, '_load_tokenizer', lambda _: Tokenizer())
    raw = b''.join((json.dumps({'id': str(i), 'text': str(i)+chr(65+i%26)*(70+i*53)})+'\n').encode() for i in range(64))
    source_file = tmp_path/'source.jsonl'; source_file.write_bytes(raw)
    source = LocalJSONLSource(SourcePin('test', 'https://example.invalid/pinned', 'fixed', hashlib.sha256(raw).hexdigest()), source_file)
    root = tmp_path/'corpus'
    document_shards.prepare_document_shards([source], root, tokenizer_path='mocked',
        split_policy=SplitPolicy(21, (('train', 3), ('dev', 1))), max_documents_per_shard=10)
    indices = {}
    for split in ('train', 'dev'):
        index = tmp_path/('index-'+split); build_packed_index(root, index, split=split, length=1024); indices[split] = index
    return root, indices


def test_real_metadata_plan_matches_materialized_masks_and_partition(corpus):
    root, indices = corpus
    with PackedCampaignData(root, indices['train']) as data:
        origin = data.cursor(); plan = resolver.plan_updates(data, {'updates': 2, 'target_valid_tokens_per_update': 1025}, {'B': (2, 1), 'NFR': (2, 3)})
        assert data.cursor() == origin and not data._token_fds
        assert plan['valid_token_prefix'] == [0, 2048, 4096]
        assert [r['overshoot_tokens'] for r in plan['updates']] == [1023, 1023]
        for r in plan['updates']:
            update = data.peek_update(type(origin)(**r['start_cursor']), 1025)
            counts = dict.fromkeys(('ce', 'latent', 'kl'), 0)
            for rank in range(2):
                real = data.rank_batches(update, rank=rank, world_size=2, physical_batch_size=3)
                allocation = r['allocation_by_arm']['NFR'][rank]
                assert (allocation['physical_rows'], allocation['dummy_rows'], allocation['padding_tokens']) == (real.physical_rows, real.empty_rows, real.padding_tokens)
                for batch in real.batches:
                    masks = build_nextlat_masks(batch, document_policy='continuous-stream-v1')
                    for term in counts: counts[term] += int(masks[term].sum())
            assert counts == update.counts.objective_counts
        assert data.cursor() == origin


def test_final_tail_recorded_and_insufficient_corpus_never_shortened_or_cycled(corpus):
    root, indices = corpus
    with PackedCampaignData(root, indices['train']) as data:
        plan = resolver.plan_updates(data, {'updates': 1, 'target_valid_tokens_per_update': data.total_tokens}, {'B': (2, 3)})
        assert plan['totals']['valid_tokens'] == data.total_tokens
        assert plan['totals']['tail_padding_tokens'] == (-data.total_tokens)%1024
        for budget in ({'updates': 1, 'target_valid_tokens_per_update': data.total_tokens+1},
                       {'updates': 2, 'target_valid_tokens_per_update': data.total_tokens}):
            with pytest.raises(ValueError, match='Insufficient corpus'): resolver.plan_updates(data, budget, {'B': (2, 3)})


@pytest.fixture
def resolved_fixture(corpus, tmp_path, monkeypatch):
    root, indices = corpus
    m = dictionary(); sources = resolver.source_hashes(); m['implementation_sources'] = sources
    m['data'].update(corpus=str(root), index=str(indices['train']),
        corpus_manifest_sha256=resolver.sha256_file(root/'manifest.json'), index_manifest_sha256=resolver.sha256_file(indices['train']/'manifest.json'))
    m['recipe']['effective_valid_tokens'] = m['budget']['target_valid_tokens_per_update'] = 1025
    m['partition']['physical_batch_per_rank'] = {a: 1 if a == 'B' else 3 for a in resolver.ARMS}
    artifacts = tmp_path/'artifacts'; artifacts.mkdir()
    tokenizer_sha = json.loads((indices['train']/'manifest.json').read_text())['tokenizer']['sha256']
    authority = {'checkpoint': {'sha256': resolver.CHECKPOINT_SHA256},
        'artifacts': {'native/tokenizer.json': {'sha256': tokenizer_sha}},
        'tokenizer': {'vocab_size': 50280, 'eos_token_id': 50279, 'pad_token_id': 1}}
    (artifacts/resolver.MANIFEST_FILENAME).write_text(json.dumps(authority))
    m['model'].update(artifacts=str(artifacts), manifest_sha256=resolver.sha256_file(artifacts/resolver.MANIFEST_FILENAME))
    # Stub only heavyweight immutable file authority, never planner or estimator.
    monkeypatch.setattr(resolver, 'validate_prepared_manifest', lambda _: authority)
    cards = []
    for arm in resolver.ARMS:
        recipe = CampaignRecipe(arm, **m['recipe']); cfg = resolver.OLMoConfig.native_1b()
        architecture = resolver.architecture_parameter_counts(cfg, fbt=recipe.feedback,
            nextlat=resolver.NextLatConfig(cfg.model_dim) if recipe.nextlat else None)
        cards.append({'arm': arm, 'parameters': {'architecture': architecture,
            'checks': {'matched': True}, 'observed_inventory': {'trainable': architecture['training_architecture']}, 'groups': {}}})
    ledger = tmp_path/'ledger.json'; ledger.write_text(json.dumps({'status': 'complete', 'integrity': {'ok': True}, 'sources': sources, 'cards': cards}))
    ledger_sha = resolver.sha256_file(ledger); monkeypatch.setattr(resolver, 'LEDGER_SHA', ledger_sha)
    m['resource_ledger'] = {'path': str(ledger), 'sha256': ledger_sha}
    return m, indices


def test_resolve_no_model_no_optimizer_no_tensor_load_and_schedule_oracle(resolved_fixture, monkeypatch):
    m, _ = resolved_fixture
    def forbidden(*args, **kwargs): raise AssertionError('Resolver attempted model/optimizer/tensor loading')
    with monkeypatch.context() as guard:
        guard.setattr(torch.nn.Module, '__init__', forbidden)
        guard.setattr(torch.optim.AdamW, '__init__', forbidden)
        guard.setattr(torch, 'load', forbidden)
        guard.setattr(torch.cuda, 'init', forbidden)
        result = resolver.resolve(m)
    assert result['status'] == 'cpu_plan_validated_not_authorized'
    assert result['launch_authorized'] is result['numerical_clearance'] is False
    assert all(result['checks'].values()) and len(result['resource_cards']) == 8
    assert result['resource_cards']['B']['supervised_positions']['latent'] == 0
    assert result['resource_cards']['NFR']['dummy_rows'] == 8
    assert result['resource_cards']['B']['dummy_rows'] == 0
    assert result['resource_cards']['NFR']['rt_block_invocations'] == 4*2*2*2
    assert result['resource_cards']['NFR']['parameters']['architecture']['training_architecture'] == 1_267_879_936
    parameter = torch.nn.Parameter(torch.ones(1)); optimizer = torch.optim.SGD([parameter], lr=m['recipe']['plateau_lr'])
    schedule = CampaignTokenSchedule(optimizer, [2048, 2048], warmup_tokens=m['recipe']['warmup_tokens'], start_fraction=m['recipe']['warmup_start_fraction'])
    observed = [optimizer.param_groups[0]['lr']]
    for _ in range(2): optimizer.step(); schedule.step(); observed.append(optimizer.param_groups[0]['lr'])
    assert result['schedule']['lr_at_completed_boundaries'] == observed
    assert result['schedule']['plan_sha256'] == schedule.plan_sha256


def test_explicit_dev_plan_is_disjoint_and_no_eval_is_executed(resolved_fixture):
    m, indices = resolved_fixture
    m['evaluation'] = {'kind': 'finite_pass_teacher_forced', 'index': str(indices['dev']),
        'index_manifest_sha256': resolver.sha256_file(indices['dev']/'manifest.json'), 'split': 'dev',
        'target_valid_tokens': 1024, 'every_updates': 1, 'precision': 'fp32', 'feedback_jitter': 0,
        'report_passes': 'all_trained_passes', 'generation': 'not_implemented'}
    result = resolver.resolve(m)
    assert result['evaluation']['execution'] == 'not_run'
    assert result['evaluation']['fixed_plan']['first_cursor']['split'] == 'dev'
    assert result['evaluation']['scheduled_updates'] == [1, 2]
    m['evaluation'].update(index=str(indices['train']), index_manifest_sha256=resolver.sha256_file(indices['train']/'manifest.json'))
    with pytest.raises(ValueError, match='authority'): resolver.resolve(m)


def test_model_tokenizer_binding_and_source_hash_changes_rejected(resolved_fixture, monkeypatch):
    m, _ = resolved_fixture
    authority = resolver.validate_prepared_manifest(None)
    authority['tokenizer']['eos_token_id'] = 1
    with pytest.raises(ValueError, match='tokenizer authority'): resolver.resolve(m)
    m['implementation_sources']['scripts/olmo_campaign_manifest.py'] = 'f'*64
    with pytest.raises(ValueError, match='source inventory'): resolver.resolve(m)


def test_pinned_input_json_mismatch_and_unknown_uri_rejected(tmp_path):
    path = tmp_path/'plan.json'; path.write_text('{}')
    assert resolver.read_json(path, resolver.sha256_file(path)) == {}
    with pytest.raises(ValueError): resolver.read_json(path, 'f'*64)
    for value in ('gs://bucket/file', '../escape', ''):
        with pytest.raises(ValueError): resolver.local_path(value)
