"""CPU semantic oracles for streamed diagnostics and preserved live updates."""
from copy import deepcopy
from dataclasses import asdict, replace
import math

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.campaign_recipe import (CampaignRecipe, build_campaign_model,
    build_campaign_adamw, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective, CampaignGraphTraining
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_campaign_evaluation as evaluation
from scripts import olmo_fbt_stability_probe as probe
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(23)


def setup():
    recipe = CampaignRecipe('F', sequence_length=8, rt_layers=(0, 1),
                            document_policy='continuous-stream-v1')
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend='sdpa',
        attention_precision='mixed', tile_backend='eager', backward_tile_backend='eager',
        ordinary_activation_checkpointing=True, cast_weights_once=True,
        reuse_rope=True, kv_only_writes=True, backward_memory='recompute')
    model = build_campaign_model(base, recipe).train()
    ids = torch.tensor([[3, 60, 5, 6, 7, 8, 9, 10], [9, 10, 11, 12, 1, 1, 1, 1], [1]*8])
    valid = torch.tensor([[1]*8, [1]*4+[0]*4, [0]*8], dtype=torch.bool)
    docs = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2], [3]*4+[-1]*4, [-1]*8])
    ce = valid.clone(); ce[0, 2] = False
    return model, recipe, NextLatBatch(ids, valid, docs, ce_mask=ce)


@pytest.mark.parametrize('beta', [0., .3, 1.])
def test_stream_matches_canonical_states_including_padding_and_cross_document_feedback(beta):
    model, recipe, batch = setup()
    with evaluation.evaluation_runtime(model) as evidence:
        mode = replace(recipe.mode(), feedback_jitter=0., beta=beta)
        canonical = model.backbone(batch.input_ids, attention_mask=batch.valid_mask,
            document_ids=batch.document_ids, mode=mode, right_padded_causal=True, return_logits=False)
        streamed = list(probe.stream_pass_states(model, batch, recipe, passes=4, beta=beta))
        for (_, actual, scales), expected in zip(streamed, canonical.pass_hidden_states):
            assert torch.equal(actual, expected)
            assert set(scales) == {'pre_norm_square', 'input_square'}
        assert not model.backbone.backbone.norm._forward_pre_hooks
    assert evidence['integrity_passed']


def test_beta_zero_is_ordinary_at_every_pass_and_causal_prefix_settles():
    model, recipe, batch = setup()
    with evaluation.evaluation_runtime(model):
        ordinary = list(probe.stream_pass_states(model, batch, recipe, passes=8, beta=0.))
        assert all(torch.equal(ordinary[0][1], row[1]) for row in ordinary)
        feedback = list(probe.stream_pass_states(model, batch, recipe, passes=10))
        for index in range(1, len(feedback)):
            assert torch.equal(feedback[index][1][:, :index], feedback[index-1][1][:, :index])
        assert torch.equal(feedback[8][1], feedback[7][1])
        assert not torch.equal(feedback[1][1][:, 1:], feedback[0][1][:, 1:])


def test_future_token_changes_do_not_change_prefix_or_earlier_pass_states():
    model, recipe, batch = setup()
    changed = deepcopy(batch); changed.input_ids[0, 5:] = torch.tensor([21, 22, 23])
    with evaluation.evaluation_runtime(model):
        original = list(probe.stream_pass_states(model, batch, recipe, passes=4))
        mutated = list(probe.stream_pass_states(model, changed, recipe, passes=4))
    assert all(torch.equal(a[1][0, :5], b[1][0, :5]) for a, b in zip(original, mutated))


@pytest.mark.parametrize('chunk', [1, 5, 256])
def test_raw_ce_entropy_and_changes_match_dense_literal_oracle(chunk):
    model, recipe, batch = setup()
    ce_mask = build_nextlat_masks(batch, document_policy=recipe.document_policy)['ce']
    with evaluation.evaluation_runtime(model):
        output = probe.probe_batch(model, batch, recipe, passes=4, vocab_position_chunk=chunk)
        states = list(probe.stream_pass_states(model, batch, recipe, passes=4))
        canonical = evaluation.per_pass_sums(model, batch, recipe)
        for index, (_, hidden, scales) in enumerate(states):
            raw = output['passes'][index]['regions']['all']
            logits = model.backbone.project_logits(hidden[:, :-1]).float()
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -log_probs.gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
            entropy = -(log_probs.exp()*log_probs).sum(-1)
            assert raw['ce_sum'] == pytest.approx(float(ce[ce_mask].sum()), rel=2e-6)
            assert raw['ce_sum'] == pytest.approx(canonical['passes'][index]['sums']['ce'], rel=2e-6)
            assert raw['entropy_sum'] == pytest.approx(float(entropy[ce_mask].sum()), rel=2e-6)
            assert raw['ce_targets'] == 9
            if index:
                expected = (hidden-states[index-1][1]).square().mean(-1)[batch.valid_mask].double().sum()
                assert raw['delta_square_sum'] == float(expected)
            else:
                assert raw['delta_square_sum'] == raw['difference_positions'] == 0
    assert output['input_tokens'] == 12


def test_global_reducer_pools_sums_and_counts_with_unequal_rows_and_dummy_contributions():
    model, recipe, batch = setup()
    with evaluation.evaluation_runtime(model):
        split = [probe.probe_batch(model, NextLatBatch(**{
            name: None if value is None else value[i:i+1] for name, value in vars(batch).items()}),
            recipe, passes=4) for i in range(3)]
        together = probe.probe_batch(model, batch, recipe, passes=4)
    actual = probe.summarize(split, expected_tokens=12, expected_ce_targets=9)
    expected = probe.summarize([together], expected_tokens=12, expected_ce_targets=9)
    for a, b in zip(actual['passes'], expected['passes']):
        for name in probe.REGIONS:
            assert a['regions'][name]['metrics'] == pytest.approx(b['regions'][name]['metrics'], rel=2e-5, abs=1e-7)
    assert actual['passes'][0]['regions']['all']['metrics']['relative_delta_rms'] is None
    assert actual['passes'][3]['regions']['unsettled_suffix']['sums']['positions'] == 6
    assert actual['passes'][3]['regions']['quarter_4']['sums']['positions'] == 2


@pytest.mark.parametrize('change', ['nonfinite', 'counts', 'pass_order', 'negative', 'policy', 'empty'])
def test_reducer_rejects_invalid_or_inconsistent_evidence(change):
    model, recipe, batch = setup()
    with evaluation.evaluation_runtime(model):
        row = probe.probe_batch(model, batch, recipe, passes=2)
    if change == 'nonfinite': row['passes'][0]['regions']['all']['ce_sum'] = float('nan')
    elif change == 'counts': row['input_tokens'] += 1
    elif change == 'pass_order': row['passes'][0]['pass'] = 2
    elif change == 'negative': row['passes'][0]['regions']['all']['hidden_square_sum'] = -1.
    elif change == 'policy': row['policy'] = 'bf16'
    with pytest.raises(ValueError):
        probe.summarize([] if change == 'empty' else [row], expected_tokens=12, expected_ce_targets=9)


def test_position_regions_explicitly_exclude_the_settled_causal_prefix():
    valid = torch.ones((1, 1024), dtype=torch.bool)
    regions = probe.region_masks(valid, 32)
    assert [int(regions[f'quarter_{i}'].sum()) for i in range(1, 5)] == [256]*4
    assert int(regions['tail_128'].sum()) == 128
    assert int(regions['unsettled_suffix'].sum()) == 993
    assert not bool(regions['unsettled_suffix'][0, :31].any())


def test_error_restores_flags_rng_modes_gradient_buffers_and_removes_hook(monkeypatch):
    model, recipe, batch = setup()
    for p in model.parameters():
        if p.requires_grad: p.grad = torch.zeros_like(p)
    rng = torch.Generator().manual_seed(4)
    def snapshot():
        return tree_digests({'model': model.state_dict(), 'rng': torch.get_rng_state(),
            'custom_rng': rng.get_state(), 'gradients': {n: p.grad for n, p in model.named_parameters()},
            'flags': {name: getattr(model.backbone.backbone, name) for name in evaluation.RUNTIME_FLAGS}})
    before = snapshot(); owner = {n: p.grad for n, p in model.named_parameters()}
    original = model.backbone._stack
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        result = original(*args, **kwargs)
        if len(calls) == 2:
            torch.rand(3, generator=rng)
            raise RuntimeError('injected')
        return result
    monkeypatch.setattr(model.backbone, '_stack', fail)
    with pytest.raises(RuntimeError, match='injected'):
        with evaluation.evaluation_runtime(model, generators={'probe_test': rng}) as evidence:
            probe.probe_batch(model, batch, recipe, passes=4)
    assert snapshot() == before and evidence['integrity_passed']
    assert not model.backbone.backbone.norm._forward_pre_hooks
    assert all(owner[n] is p.grad for n, p in model.named_parameters())
    assert all(m.training for m in model.modules())


def test_probe_between_real_prepared_updates_leaves_next_update_exact():
    initial, recipe, batch = setup(); results = {}
    for insert in (False, True):
        model = deepcopy(initial)
        optimizer = build_campaign_adamw(model, recipe, fused=False)
        counters = TrainingCounters()
        noise = feedback_noise_for_rows(recipe, ['a', 'b', 'dummy'], logical_update=0,
            sequence_length=8, width=model.config.model_dim)
        with evaluation.sdpa_kernel(evaluation.SDPBackend.MATH):
            objective = CampaignObjective(model, batch, mode=recipe.mode(), global_counts=model.counts(batch),
                feedback_noise=noise, config=LMTrainingConfig(precision='fp32'))
            runner = CampaignGraphTraining(objective)
            first = runner.optimizer_step(optimizer, (batch,), feedback_noises=(noise,), counters=counters)
            runner.zero_grad()
            inputs = tree_digests(objective.owned_inputs())
            pointers = {name: None if p.grad is None else p.grad.data_ptr() for name, p in model.named_parameters()}
            if insert:
                with evaluation.evaluation_runtime(model) as evidence:
                    probe.probe_batch(model, batch, recipe, passes=8)
                assert evidence['integrity_passed']
                assert inputs == tree_digests(objective.owned_inputs())
                assert pointers == {name: None if p.grad is None else p.grad.data_ptr() for name, p in model.named_parameters()}
            second = runner.optimizer_step(optimizer, (batch,), feedback_noises=(noise,), counters=counters)
        results[insert] = (first, second, tree_digests({'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'counters': asdict(counters)}))
    assert results[False] == results[True]


@pytest.mark.parametrize('bad', ['outside', 'other_arm', 'too_deep', 'chunk'])
def test_scope_and_boundedness_fail_closed(bad):
    model, recipe, batch = setup()
    if bad == 'outside':
        with pytest.raises(ValueError, match='evaluation_runtime'):
            probe.probe_batch(model, batch, recipe, passes=4)
        return
    with evaluation.evaluation_runtime(model):
        with pytest.raises(ValueError):
            probe.probe_batch(model, batch, replace(recipe, arm='NF') if bad == 'other_arm' else recipe,
                passes=33 if bad == 'too_deep' else 4,
                vocab_position_chunk=0 if bad == 'chunk' else 256)
