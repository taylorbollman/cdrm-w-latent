"""Portable accumulation and refill checks, before real CUDA/DDP acceptance."""
import copy

import pytest
import torch

from cdrm.pretrained.campaign_recipe import (ARMS, CampaignRecipe, CampaignTokenSchedule,
    build_campaign_model, build_campaign_adamw, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective, CampaignGraphTraining
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import TrainingCounters, save_training_checkpoint
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM


@pytest.fixture(autouse=True)
def fixed_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(657)


def batches(offset=0):
    result = []
    for lengths in [(6, 4), (2, 0), (0, 0)]:
        ids = (torch.arange(12).reshape(2, 6) + 2 + offset) % 31
        valid = torch.arange(6)[None] < torch.tensor(lengths)[:, None]
        docs = torch.arange(2)[:, None].expand_as(ids).masked_fill(~valid, -1)
        ce, latent, kl = valid.clone(), valid.clone(), valid.clone()
        ce[0, 3] = False
        latent[1, 1] = False
        kl[0, 2] = False
        result.append(NextLatBatch(ids, valid, docs, ce, latent, kl))
    return result


def setup(arm="NFR"):
    recipe = CampaignRecipe(arm, sequence_length=6, rt_layers=(0, 1), warmup_tokens=20)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(),
        attention_backend="math", attention_precision="fp32")
    model = build_campaign_model(base, recipe)
    return recipe, model


def noises(recipe, model, update=0):
    return [feedback_noise_for_rows(recipe, [f"{i}-a", f"{i}-b"], logical_update=update,
                                   sequence_length=6, width=model.config.model_dim) for i in range(3)]


def oracle(model, recipe, data, noise):
    model.zero_grad(set_to_none=True)
    counts = sum_objective_counts([model.counts(batch) for batch in data])
    totals = dict.fromkeys(counts, 0.0)
    for batch, jitter in zip(data, noise):
        result = model.loss_sums(batch, backbone_kwargs={"mode": recipe.mode(),
                                 "feedback_noise": jitter, "right_padded_causal": True})
        objective = sum(result.sums[t] * result.weights[t] / counts[t] for t in counts if counts[t])
        objective.backward()
        for term in totals:
            totals[term] += float(result.sums[term].detach())
    return totals


@pytest.mark.parametrize("arm", ARMS)
def test_all_arms_dynamic_accumulation_matches_canonical_raw_gradients(arm):
    recipe, model = setup(arm)
    reference = copy.deepcopy(model)
    data, noise = batches(), noises(recipe, model)
    counts = sum_objective_counts([model.counts(batch) for batch in data])
    adapter = CampaignObjective(model, data[0], mode=recipe.mode(), global_counts=counts,
                                feedback_noise=noise[0])
    runner = CampaignGraphTraining(adapter)
    before = torch.get_rng_state().clone()
    actual = runner.backward(data, feedback_noises=noise)
    assert torch.equal(before, torch.get_rng_state())
    want = oracle(reference, recipe, data, noise)
    assert actual["counts"] == counts
    assert actual["input_tokens"] == 12 and actual["documents"] == 3
    assert actual["microbatches"] == 3
    for t in counts:
        assert actual["loss_sums"][t] == pytest.approx(want[t], rel=2e-6, abs=2e-6)
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=5e-4, msg=name)


def test_changing_masks_noise_denominators_and_updates_keep_addresses_and_token_clock():
    recipe, model = setup()
    data, noise = batches(), noises(recipe, model)
    counts = sum_objective_counts([model.counts(batch) for batch in data])
    adapter = CampaignObjective(model, data[0], mode=recipe.mode(), global_counts=counts, feedback_noise=noise[0])
    runner = CampaignGraphTraining(adapter)
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    schedule = CampaignTokenSchedule(optimizer, [12, 10], warmup_tokens=20)
    counters = TrainingCounters()
    first = runner.optimizer_step(optimizer, data, feedback_noises=noise, scheduler=schedule, counters=counters)
    addresses = runner.gradient_addresses.copy()
    pointers = tuple(v.data_ptr() for v in adapter.owned_inputs())
    second_data = batches(8)[:1]
    second = runner.optimizer_step(optimizer, second_data, feedback_noises=noises(recipe, model, 1)[:1],
                                   scheduler=schedule, counters=counters)
    assert runner.gradient_addresses == addresses
    assert tuple(v.data_ptr() for v in adapter.owned_inputs()) == pointers
    assert first["lr_used"][0] == pytest.approx(2e-5)
    assert second["lr_used"][0] == pytest.approx(2e-4 * .64)
    assert schedule.completed_tokens == counters.input_tokens == 22
    for name, tensor in second_data[0].__dict__.items():
        torch.testing.assert_close(getattr(adapter.batch, name), tensor, rtol=0, atol=0)
    assert counters.optimizer_updates == 2 and counters.microbatches == 4
    assert all(p.grad is None or not bool(p.grad.any()) for p in model.parameters())


def test_preflight_rejects_bad_later_microbatch_without_gradient_or_input_mutation():
    recipe, model = setup()
    data, noise = batches(), noises(recipe, model)
    counts = sum_objective_counts([model.counts(b) for b in data])
    adapter = CampaignObjective(model, data[0], mode=recipe.mode(), global_counts=counts, feedback_noise=noise[0])
    runner = CampaignGraphTraining(adapter)
    before = adapter.batch.input_ids.clone()
    noise[1][0].fill_(2)
    with pytest.raises(ValueError, match="unit noise"):
        runner.backward(data, feedback_noises=noise)
    assert torch.equal(before, adapter.batch.input_ids)
    assert all(p.grad is None for p in model.parameters())


def test_external_buffer_mutation_and_empty_global_auxiliary_fail_explicitly():
    recipe, model = setup()
    data, noise = batches(), noises(recipe, model)
    counts = sum_objective_counts([model.counts(b) for b in data])
    with pytest.raises(ValueError, match="positive global"):
        CampaignObjective(model, data[2], mode=recipe.mode(), global_counts={"ce": 1,"latent":0,"kl":0},
                          feedback_noise=noise[2])
    adapter = CampaignObjective(model, data[0], mode=recipe.mode(), global_counts=counts, feedback_noise=noise[0])
    adapter.feedback_noise[0].zero_()
    with pytest.raises(ValueError, match="outside validated refill"):
        adapter.validate_execution()


def test_cpu_graph_request_and_world_size_misuse_cannot_silently_fallback():
    recipe, model = setup("B")
    data = batches()
    counts = model.counts(data[0])
    adapter = CampaignObjective(model, data[0], mode=recipe.mode(), global_counts=counts)
    with pytest.raises(ValueError, match="requires CUDA"):
        CampaignGraphTraining(adapter).capture()
    distributed = CampaignObjective(model, data[0], mode=recipe.mode(), global_counts=counts, world_size=2)
    with pytest.raises(ValueError, match="world_size=1"):
        CampaignGraphTraining(distributed)


def test_checkpoint_publication_restores_persistent_grads_and_rejects_pending_update(tmp_path):
    recipe, model = setup()
    data, noise = batches(), noises(recipe, model)
    counts = sum_objective_counts([model.counts(b) for b in data])
    adapter = CampaignObjective(model, data[0], mode=recipe.mode(), global_counts=counts, feedback_noise=noise[0])
    runner = CampaignGraphTraining(adapter)
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    runner.backward(data, feedback_noises=noise)
    with pytest.raises(ValueError, match="boundary"):
        with runner.checkpoint_boundary():
            pass
    runner.discard_backward()
    counters = TrainingCounters()
    runner.optimizer_step(optimizer, data, feedback_noises=noise, counters=counters)
    addresses = runner.gradient_addresses.copy()
    with runner.checkpoint_boundary():
        assert all(p.grad is None for p in model.parameters())
        with pytest.raises(RuntimeError, match="checkpoint publication"):
            runner.validate_execution()
        save_training_checkpoint(tmp_path/'boundary.pt', model, optimizer, counters=counters,
            data_cursor={"update": 1}, configuration=recipe.to_dict(),
            source_fingerprint={"fixture": True, "checkpoint_sha256": "a" * 64})
    runner.validate_execution()
    assert runner._addresses() == addresses
    with pytest.raises(RuntimeError, match="fixture save failure"):
        with runner.checkpoint_boundary():
            raise RuntimeError("fixture save failure")
    assert runner._addresses() == addresses


def test_bad_counters_fail_before_optimizer_or_schedule_mutation():
    recipe, model = setup("B")
    data = batches()
    adapter = CampaignObjective(model, data[0], mode=recipe.mode(),
        global_counts=sum_objective_counts([model.counts(b) for b in data]))
    runner = CampaignGraphTraining(adapter)
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    with pytest.raises(TypeError, match="TrainingCounters"):
        runner.optimizer_step(optimizer, data, counters={})
    assert not optimizer.state
    assert all(p.grad is None for p in model.parameters())


def test_backward_metrics_do_not_retain_completed_autograd_graph():
    recipe, model = setup("NFR")
    data, noise = batches(), noises(recipe, model)
    adapter = CampaignObjective(model, data[0], mode=recipe.mode(),
        global_counts=model.counts(data[0]), feedback_noise=noise[0])
    result = CampaignGraphTraining(adapter)._tensor_backward()
    assert not result["objective"].requires_grad and result["objective"].grad_fn is None
    assert all(not value.requires_grad for value in result["loss_sums"].values())
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
