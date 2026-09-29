"""Real CPU/Gloo reducer tests; CUDA graph/NCCL acceptance is a separate probe."""
import copy
from dataclasses import asdict
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
from cdrm.pretrained.campaign_recipe import (CampaignRecipe, CampaignTokenSchedule,
    build_campaign_adamw, build_campaign_model, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.ddp_training import CoordinatedUpdateError
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM


def _model(arm="NFR"):
    torch.manual_seed(952)
    recipe = CampaignRecipe(arm, sequence_length=6, rt_layers=(0, 1), warmup_tokens=20)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32")
    return recipe, build_campaign_model(base, recipe)


def _batches(rank, update):
    lengths = ((((6, 4),), ((5, 2),)),
               (((6, 3), (2, 0)), ((0, 0), (0, 0))),
               (((5, 4), (2, 0), (0, 0)), ((4, 3), (3, 0), (0, 0))))[update][rank]
    batches = []
    for index, pair in enumerate(lengths):
        ids = (torch.arange(12).reshape(2, 6) + 2 + rank + 2 * update + index) % 31
        valid = torch.arange(6)[None] < torch.tensor(pair)[:, None]
        docs = torch.arange(2)[:, None].expand_as(ids).masked_fill(~valid, -1)
        ce, latent, kl = valid.clone(), valid.clone(), valid.clone()
        ce[0, 2] = False
        latent[1, 1] = False
        kl[0, 3] = False
        batches.append(NextLatBatch(ids, valid, docs, ce, latent, kl))
    return batches


def _noise(recipe, model, rank, update):
    return [feedback_noise_for_rows(recipe, [f"{rank}-{index}-0", f"{rank}-{index}-1"],
            logical_update=update, sequence_length=6, width=model.config.model_dim)
            for index, _ in enumerate(_batches(rank, update))]


def _initialize(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{path}",
                            timeout=timedelta(seconds=120))


def _setup(rank, arm="NFR"):
    recipe, model = _model(arm)
    counts = sum_objective_counts([model.counts(batch) for r in range(2) for batch in _batches(r, 0)])
    adapter = CampaignObjective(model, _batches(rank, 0)[0], mode=recipe.mode(), world_size=2,
            global_counts=counts, feedback_noise=_noise(recipe, model, rank, 0)[0])
    return recipe, model, CampaignDDPGraphTraining(adapter, bucket_cap_mb=.02)


def _oracle(model, recipe, update):
    data = [batch for rank in range(2) for batch in _batches(rank, update)]
    noises = [noise for rank in range(2) for noise in _noise(recipe, model, rank, update)]
    counts = sum_objective_counts([model.counts(batch) for batch in data])
    model.zero_grad(set_to_none=True)
    sums = dict.fromkeys(counts, 0.)
    for batch, noise in zip(data, noises):
        result = model.loss_sums(batch, backbone_kwargs={"mode": recipe.mode(),
            "feedback_noise": noise, "right_padded_causal": True})
        sum(result.sums[t] * result.weights[t] / counts[t] for t in counts if counts[t]).backward()
        for term in sums:
            sums[term] += float(result.sums[term].detach())
    return counts, sums


def _no_tensor(value):
    assert not isinstance(value, torch.Tensor), "Rank-local tensor entered an object collective"
    if isinstance(value, dict):
        for item in value.values():
            _no_tensor(item)
    if isinstance(value, (list, tuple)):
        for item in value:
            _no_tensor(item)


def _success_worker(rank, path, arm):
    _initialize(rank, path)
    try:
        recipe, model, runner = _setup(rank, arm)
        reference = copy.deepcopy(model)
        optimizer = build_campaign_adamw(model, recipe, fused=False)
        other_optimizer = build_campaign_adamw(reference, recipe, fused=False)
        token_plan = [sum(int(b.valid_mask.sum()) for r in range(2) for b in _batches(r, u)) for u in range(3)]
        schedule = CampaignTokenSchedule(optimizer, token_plan, warmup_tokens=20)
        other_schedule = CampaignTokenSchedule(other_optimizer, token_plan, warmup_tokens=20)
        counters = TrainingCounters()
        gather = runner._gather
        def checked_gather(value):
            _no_tensor(value)
            return gather(value)
        runner._gather = checked_gather
        rng = torch.get_rng_state().clone()
        runner.prepare(warmup=11)
        assert torch.equal(torch.get_rng_state(), rng)
        assert runner.warmup_backward_calls == 20 and not optimizer.state
        assert all(p.grad is None or not bool(p.grad.any()) for p in model.parameters())
        assert schedule.completed_tokens == 0 and counters.optimizer_updates == 0
        addresses = runner.gradient_addresses.copy()
        pointers = tuple(v.data_ptr() for v in runner.adapter.owned_inputs())
        sync_flags = []
        hook = runner.ddp.register_forward_pre_hook(lambda module, args: sync_flags.append(module.require_backward_grad_sync))
        for update in range(3):
            counts, sums = _oracle(reference, recipe, update)
            result = runner.backward(_batches(rank, update),
                    feedback_noises=_noise(recipe, model, rank, update), replay=False)
            assert result["counts"] == counts
            assert result["microbatches"] == 2 * (update + 1)
            assert result["local_microbatches"] == update + 1
            assert result["input_tokens"] == token_plan[update]
            for term in sums:
                assert result["loss_sums"][term] == pytest.approx(sums[term], rel=3e-6, abs=3e-6)
            for (name, p), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
                assert (p.grad is None) == (other.grad is None), name
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, other.grad, atol=8e-6, rtol=8e-4, msg=name)
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), runner.adapter.config.max_grad_norm)
            other_optimizer.step(); other_schedule.step()
            actual = runner.step(result, optimizer, scheduler=schedule, counters=counters)
            assert actual["gradient_norm_before_clip"] == pytest.approx(float(expected_norm), rel=3e-5)
            for (name, p), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p, other, atol=5e-7, rtol=5e-5, msg=name)
                for key, value in optimizer.state.get(p, {}).items():
                    torch.testing.assert_close(value, other_optimizer.state[other][key],
                                               atol=5e-7, rtol=1e-4, msg=f"{name}/{key}")
            assert runner._addresses() == addresses
            assert tuple(v.data_ptr() for v in runner.adapter.owned_inputs()) == pointers
            with runner.checkpoint_boundary():
                assert all(p.grad is None for p in model.parameters())
                with pytest.raises(RuntimeError, match="checkpoint publication"):
                    runner.validate_execution()
            assert runner._addresses() == addresses
        assert sync_flags == [True, False, True, False, False, True]
        hook.remove()
        assert counters.input_tokens == sum(token_plan) == schedule.completed_tokens
        assert counters.microbatches == 12 and counters.optimizer_updates == 3
        assert asdict(counters) == actual["counters"]
        with pytest.raises(RuntimeError, match="save failure"):
            with runner.checkpoint_boundary():
                raise RuntimeError("save failure")
        assert runner._addresses() == addresses
        with pytest.raises(CoordinatedUpdateError, match="CUDA/NCCL"):
            runner.capture()
        # Explicit failed capture preflight changes no state and performs no
        # CPU fallback; the already prepared eager diagnostic remains usable.
        runner.validate_execution()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("arm", ["B", "NFR"])
def test_real_ddp_m1_m2_m3_empty_rank_slot_raw_gradients_and_adam(tmp_path, arm):
    mp.spawn(_success_worker, args=(str(tmp_path / f"success-{arm}.init"), arm), nprocs=2, join=True)


def _failure_worker(rank, path):
    _initialize(rank, path)
    try:
        for failure in ("late_noise", "microbatch_count", "global_aux_empty", "mutated_storage"):
            recipe, model, runner = _setup(rank)
            runner.prepare()
            data = _batches(rank, 2)
            noise = _noise(recipe, model, rank, 2)
            before = runner.adapter.batch.input_ids.clone()
            initial = copy.deepcopy(model.state_dict())
            if failure == "global_aux_empty":
                for batch in data:
                    batch.latent_mask.zero_()
            elif rank == 1:
                if failure == "late_noise":
                    noise[-1][0].fill_(1.1)
                elif failure == "microbatch_count":
                    data, noise = data[:-1], noise[:-1]
                else:
                    runner.adapter.batch.input_ids.add_(1)
                    before = runner.adapter.batch.input_ids.clone()
            with pytest.raises(CoordinatedUpdateError):
                runner.backward(data, feedback_noises=noise, replay=False)
            assert torch.equal(runner.adapter.batch.input_ids, before)
            for name, value in model.state_dict().items():
                assert torch.equal(value, initial[name])
            assert all(p.grad is None or not bool(p.grad.any()) for p in model.parameters())
            with pytest.raises(RuntimeError, match="fresh process group"):
                runner.validate_execution()
        recipe, model, runner = _setup(rank)
        runner.prepare()
        result = runner.backward(_batches(rank, 0), feedback_noises=_noise(recipe, model, rank, 0), replay=False)
        with pytest.raises(CoordinatedUpdateError, match="boundary"):
            with runner.checkpoint_boundary():
                pass
        runner.discard_backward()
        with runner.checkpoint_boundary():
            assert all(p.grad is None for p in model.parameters())
        # Consumed/discarded results cannot apply an update, even on one rank.
        optimizer = build_campaign_adamw(model, recipe, fused=False)
        with pytest.raises(CoordinatedUpdateError, match="pending backward"):
            runner.step(result, optimizer)
        assert not optimizer.state
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_all_rank_preflight_rejects_late_inputs_and_global_participation_change(tmp_path):
    mp.spawn(_failure_worker, args=(str(tmp_path / "failure.init"),), nprocs=2, join=True)


def test_runner_requires_initialized_process_group_and_valid_adapter():
    with pytest.raises(TypeError, match="CampaignObjective"):
        CampaignDDPGraphTraining(object())
    recipe, model = _model("B")
    batch = _batches(0, 0)[0]
    adapter = CampaignObjective(model, batch, mode=recipe.mode(), global_counts=model.counts(batch), world_size=2)
    with pytest.raises(RuntimeError, match="Initialize"):
        CampaignDDPGraphTraining(adapter)
    with pytest.raises(ValueError, match="bucket_cap_mb"):
        CampaignDDPGraphTraining(adapter, bucket_cap_mb=False)
