"""CPU checks of fixture/reference contracts; these do not certify NCCL/CUDA."""
from argparse import Namespace
from dataclasses import asdict

import pytest
import torch

from cdrm.pretrained.campaign_recipe import ARMS, CampaignRecipe
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts.olmo_campaign_ddp_probe import (advance_counters, canonical_backward,
    construct, fixture_for_update, global_fixture_metadata, parse_args, prepared_backward)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    torch.set_num_threads(1)


def fixtures(recipe, update, *, batch_size=2, length=8):
    return [fixture_for_update(recipe, 32, rank, update, length=length,
                              token_ids=list(range(2, 59)), eos_id=60, batch_size=batch_size)
            for rank in range(2)]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("length", [8, 16, 32])
def test_fixture_empty_rank_final_sync_slot_distinct_counts_and_right_padding(batch_size, length):
    recipe = CampaignRecipe("NFR", sequence_length=length, rt_layers=(0, 1))
    counts = []
    for update in range(3):
        ranks = fixtures(recipe, update, batch_size=batch_size, length=length)
        assert all(len(batches) == len(noises) == update+1 for batches, noises in ranks)
        totals = dict.fromkeys(("ce", "latent", "kl"), 0)
        for rank, (batches, noises) in enumerate(ranks):
            for batch, noise in zip(batches, noises):
                assert batch.input_ids.shape == (batch_size, length)
                assert not bool((batch.valid_mask[:, 1:] & ~batch.valid_mask[:, :-1]).any())
                assert torch.equal(batch.document_ids == -1, ~batch.valid_mask)
                for term, mask in build_nextlat_masks(batch).items():
                    totals[term] += int(mask.sum())
                assert len(noise) == 3
                for value in noise:
                    assert value.shape == (batch_size, length-1, 32)
                    assert not value[~batch.valid_mask.any(-1)].any()
        assert all(v > 0 for v in totals.values())
        counts.append(totals)
    assert len({tuple(c.values()) for c in counts}) == 3
    assert all(not b.valid_mask.any() for b in fixtures(recipe, 1, batch_size=batch_size, length=length)[1][0])
    assert all(not batches[-1].valid_mask.any() for batches, _ in fixtures(recipe, 2, batch_size=batch_size, length=length))


def test_fixtures_reproduce_keyed_noise_without_consuming_global_rng():
    recipe = CampaignRecipe("NFR", sequence_length=8, rt_layers=(0, 1))
    before = torch.get_rng_state().clone()
    first, second = fixtures(recipe, 2), fixtures(recipe, 2)
    assert torch.equal(before, torch.get_rng_state())
    for (batches, noises), (other_batches, other_noises) in zip(first, second):
        for batch, other, noise, other_noise in zip(batches, other_batches, noises, other_noises):
            assert all(torch.equal(value, vars(other)[name]) for name, value in vars(batch).items())
            assert all(torch.equal(a, b) for a, b in zip(noise, other_noise))
    assert not torch.equal(first[0][1][0][0], first[1][1][0][0])


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("document_policy", ["isolated-v1", "continuous-stream-v1"])
def test_independent_canonical_global_reference_matches_prepared_accumulation(arm, document_policy):
    args = Namespace(scale="tiny", length=8, document_policy=document_policy)
    model, recipe, checkpoint, _, _ = construct(args, arm, "cpu")
    assert "fixture" in checkpoint
    data = fixtures(recipe, 2)
    expected = canonical_backward(model, recipe, data, precision="fp32")
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    actual = prepared_backward(model, recipe, data, precision="fp32")
    for key in ("counts", "microbatches", "documents", "input_tokens"):
        assert actual[key] == expected[key]
    assert actual["objective"] == pytest.approx(expected["objective"], rel=3e-6, abs=1e-6)
    for term in ("ce", "latent", "kl"):
        assert actual["loss_sums"][term] == pytest.approx(expected["loss_sums"][term], rel=3e-6, abs=1e-5)
    active = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert active.keys() == reference.keys()
    for name in active:
        torch.testing.assert_close(active[name], reference[name], atol=3e-5, rtol=3e-4, msg=name)


def test_counter_reference_counts_all_ranks_and_does_not_divide_accumulation():
    args = Namespace(scale="tiny", length=8)
    model, recipe, _, _, _ = construct(args, "NFR", "cpu")
    counters = TrainingCounters()
    records = [global_fixture_metadata(model, fixtures(recipe, u)) for u in range(3)]
    for record in records:
        advance_counters(counters, record)
    assert counters.optimizer_updates == 3
    assert counters.microbatches == 2*(1+2+3)
    assert counters.input_tokens == sum(r["input_tokens"] for r in records)
    assert counters.ce_positions == sum(r["counts"]["ce"] for r in records)
    assert counters.kl_triples == sum(r["counts"]["kl"] for r in records)
    assert all(v >= 0 for v in asdict(counters).values())


def test_cli_is_bounded_and_defaults_to_all_tiny_arms():
    # ROOT differs between host and container; use the live imported script root.
    from scripts.olmo_campaign_ddp_probe import ROOT
    path = str(ROOT / ".runtime/probe-test")
    common = ["--scale", "tiny", "--case", "graph", "--output-dir", path]
    args = parse_args(common)
    assert args.arms == ARMS and args.length == 8 and args.warmup == 11 and args.reference == "canonical"
    assert parse_args(common+["--reference", "prepared"]).reference == "prepared"
    assert args.document_policy == "isolated-v1"
    assert parse_args(common+["--document-policy", "continuous-stream-v1"]).document_policy == "continuous-stream-v1"
    for extra in (["--warmup", "10"], ["--length", "1024"], ["--arms", "B,B"], ["--arms", "bad"]):
        with pytest.raises(SystemExit):
            parse_args(common+extra)


def test_packed_fixture_crosses_true_eos_boundaries_and_moves_layout_without_changing_padding_contract():
    recipe = CampaignRecipe("NFR", sequence_length=8, rt_layers=(0, 1),
                            document_policy="continuous-stream-v1")
    for update in range(3):
        crossings = 0
        for batches, _ in fixtures(recipe, update):
            for batch in batches:
                adjacent = batch.valid_mask[:, :-1] & batch.valid_mask[:, 1:]
                cross = adjacent & (batch.document_ids[:, :-1] != batch.document_ids[:, 1:])
                assert (batch.input_ids[:, :-1][cross] == 60).all()
                masks = build_nextlat_masks(batch, document_policy=recipe.document_policy)
                assert torch.equal(masks["ce"], adjacent & batch.ce_mask[:, 1:])
                assert not masks["latent"][cross].any()
                crossings += int(cross.sum())
        assert crossings > 0


@pytest.mark.parametrize("rank,update,batch_size,length", [(2, 0, 2, 8), (0, 3, 2, 8), (0, 0, 3, 8), (0, 0, 2, 7)])
def test_fixture_rejects_unqualified_layouts(rank, update, batch_size, length):
    with pytest.raises(ValueError):
        fixture_for_update(CampaignRecipe("B"), 32, rank, update, length=length,
                           token_ids=[2, 3], eos_id=60, batch_size=batch_size)
