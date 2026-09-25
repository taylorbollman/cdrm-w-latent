"""Real two-process CPU/Gloo checks; these do not establish NCCL readiness."""
import copy
from dataclasses import asdict, replace
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cdrm.pretrained.ddp_training import CoordinatedUpdateError, EagerDDPTrainer
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.lm_training import (LMTrainingConfig, TrainingCounters,
    build_adamw, build_warmup_scheduler, optimizer_step)
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


def make_model(*, enabled=True):
    torch.manual_seed(128)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=True,
        reuse_rope=True, kv_only_writes=True)
    return FBTNextLatLM(OLMoFBT(base), NextLatConfig(model_dim=32, proj_factor=2,
        lambda_latent=.3, lambda_kl=.7, vocab_chunk_size=4), enabled=enabled, gamma=.4).train()


def batches_for(rank, update):
    batches = []
    for micro in range(2):
        ids = torch.tensor([[2, 3, 4, 5, 6, 7]]) + rank + micro
        valid = torch.ones_like(ids, dtype=torch.bool)
        docs = torch.full_like(ids, rank * 2 + micro)
        ce, latent, kl = (valid.clone() for _ in range(3))
        ce[:, 2+rank] = False
        if rank == 1:
            ce[:, 1] = False
        # Critical case: predictor is used under no_sync only by rank 0, and
        # is unused on BOTH ranks during the synchronized final microbatch.
        active = (update != 1 and rank == 0 and micro == 0) or (
            update == 2 and rank == 1 and micro == 1)
        if not active:
            latent.zero_(); kl.zero_()
        else:
            latent[:, 3] = False
            kl[:, 4] = False
        if update == 2 and rank == 1 and micro == 0:
            ce.zero_()  # Entirely empty local objective, globally active.
        batches.append(NextLatBatch(ids, valid, docs, ce, latent, kl))
    return batches


def compare_gradients(model, expected):
    actual = dict(model.named_parameters())
    assert actual.keys() == expected.keys()
    for name, parameter in actual.items():
        gradient = expected[name]
        if gradient is None or parameter.grad is None:
            assert gradient is parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=5e-4, atol=5e-6, msg=name)


def compare_states(model, reference, optimizer, expected_optimizer):
    for (name, parameter), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter, expected, atol=4e-7, rtol=5e-5, msg=name)
        state = optimizer.state.get(parameter, {})
        wanted = expected_optimizer.state.get(expected, {})
        assert state.keys() == wanted.keys(), name
        for key in state:
            torch.testing.assert_close(state[key], wanted[key], atol=5e-7, rtol=8e-5, msg=name+key)


def _initialize(rank, filename):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{filename}",
                            timeout=timedelta(seconds=120))


def _mode_worker(rank, filename):
    _initialize(rank, filename)
    try:
        for fbt in (False, True):
            for rt in (False, True):
                for nextlat in (False, True):
                    model = make_model(enabled=nextlat)
                    trainer = EagerDDPTrainer(model, bucket_cap_mb=.02)
                    reference = copy.deepcopy(model)
                    assert trainer.ddp.find_unused_parameters is True
                    assert trainer.ddp.gradient_as_bucket_view is False
                    assert trainer.ddp.broadcast_buffers is False
                    mode = FBTMode(enabled=fbt, num_passes=2, beta=.35,
                                   rt_mode=RTMode((0, 1) if rt else (), .37))
                    kwargs = {"mode": mode}
                    optimizer = build_adamw(model, lr=1e-4, eps=1e-4)
                    expected_optimizer = build_adamw(reference, lr=1e-4, eps=1e-4)
                    scheduler = build_warmup_scheduler(optimizer, warmup_updates=2)
                    expected_scheduler = build_warmup_scheduler(expected_optimizer, warmup_updates=2)
                    counters, expected_counters = TrainingCounters(), TrainingCounters()
                    sync_states = []
                    def record_forward(module, args):
                        sync_states.append(module.require_backward_grad_sync)
                        assert not torch.is_autocast_cache_enabled()
                    handle = trainer.ddp.register_forward_pre_hook(record_forward)
                    for update in range(3):
                        global_batches = [batch for r in range(2) for batch in batches_for(r, update)]
                        expected_raw = {}
                        real_clip = torch.nn.utils.clip_grad_norm_
                        def record_raw(parameters, *args, **kw):
                            expected_raw.update({name: None if p.grad is None else p.grad.clone()
                                                 for name, p in reference.named_parameters()})
                            return real_clip(parameters, *args, **kw)
                        with patch("torch.nn.utils.clip_grad_norm_", side_effect=record_raw):
                            expected = optimizer_step(reference, expected_optimizer, global_batches,
                                config=LMTrainingConfig(max_grad_norm=.2), backbone_kwargs=kwargs,
                                scheduler=expected_scheduler, counters=expected_counters)
                        result = trainer.backward(batches_for(rank, update),
                            config=LMTrainingConfig(max_grad_norm=.2), backbone_kwargs=kwargs)
                        compare_gradients(model, expected_raw)
                        assert result.global_counts == expected["counts"]
                        assert result.global_microbatches == 4
                        assert result.global_documents == 4 and result.global_input_tokens == 24
                        for term in result.loss_sums:
                            assert result.loss_sums[term] == pytest.approx(expected["loss_sums"][term], rel=3e-6, abs=2e-6)
                        actual = trainer.step(result, optimizer, scheduler=scheduler, counters=counters)
                        assert actual["objective"] == pytest.approx(expected["objective"], rel=3e-6)
                        assert actual["gradient_norm_before_clip"] == pytest.approx(expected["gradient_norm_before_clip"], rel=3e-5)
                        assert asdict(counters) == asdict(expected_counters)
                        assert scheduler.state_dict() == expected_scheduler.state_dict()
                        assert all(p.grad is None for p in model.parameters())
                        compare_states(model, reference, optimizer, expected_optimizer)
                        trainer.assert_update_boundary()
                    assert sync_states == [False, True] * 3
                    handle.remove()
                    del trainer, model, reference, optimizer, expected_optimizer
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_real_two_rank_all_eight_modes_accumulation_and_adam(tmp_path):
    mp.spawn(_mode_worker, args=(str(tmp_path / "modes.init"),), nprocs=2, join=True)


def _failure_worker(rank, filename):
    _initialize(rank, filename)
    try:
        for failure in ("loss", "gradient", "preflight", "contract", "accumulation", "empty", "optimizer", "post_backward"):
            model = make_model()
            trainer = EagerDDPTrainer(model)
            optimizer = build_adamw(model, lr=1e-4, eps=1e-4)
            counters = TrainingCounters()
            snapshot = copy.deepcopy(model.state_dict())
            batches = batches_for(rank, 0)
            config = LMTrainingConfig()
            if failure == "loss" and rank == 1:
                original = model.loss_sums
                def nonfinite_loss(*args, **kwargs):
                    result = original(*args, **kwargs)
                    result.sums["ce"] = result.sums["ce"] * float("nan")
                    return result
                model.loss_sums = nonfinite_loss
            if failure == "gradient" and rank == 1:
                next(model.parameters()).register_hook(lambda gradient: gradient * float("nan"))
            if failure == "preflight" and rank == 1:
                batches = []
            if failure == "contract" and rank == 1:
                config = LMTrainingConfig(max_grad_norm=.25)
            if failure == "accumulation" and rank == 1:
                batches = batches[:1]
            if failure == "empty":
                batches = [replace(batch, ce_mask=torch.zeros_like(batch.valid_mask),
                    latent_mask=torch.zeros_like(batch.valid_mask), kl_mask=torch.zeros_like(batch.valid_mask))
                    for batch in batches]
            if failure == "optimizer" and rank == 1:
                optimizer.param_groups[0]["lr"] *= 2
            with pytest.raises(CoordinatedUpdateError):
                if failure == "post_backward":
                    result = trainer.backward(batches, config=config,
                        backbone_kwargs={"mode": FBTMode(rt_mode=RTMode((0,)))})
                    if rank == 1:
                        next(p.grad for p in model.parameters() if p.grad is not None).view(-1)[0] = float("nan")
                    trainer.step(result, optimizer, counters=counters)
                else:
                    trainer.optimizer_step(optimizer, batches, config=config,
                        backbone_kwargs={"mode": FBTMode(rt_mode=RTMode((0,)))}, counters=counters)
            for name, value in model.state_dict().items():
                assert torch.equal(value, snapshot[name]), (failure, name)
            assert not optimizer.state
            assert counters == TrainingCounters()
            assert all(p.grad is None for p in model.parameters())
            with pytest.raises(RuntimeError, match="reconstructed"):
                trainer.assert_update_boundary()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_real_two_rank_failures_prevent_every_optimizer_step(tmp_path):
    mp.spawn(_failure_worker, args=(str(tmp_path / "failures.init"),), nprocs=2, join=True)


def _discard_worker(rank, filename):
    _initialize(rank, filename)
    try:
        model = make_model()
        trainer = EagerDDPTrainer(model)
        # Repeated backward-only diagnostics must complete the real reducer
        # lifecycle and preserve globally unused optimizer semantics.
        for update in range(3):
            result = trainer.backward(batches_for(rank, update))
            with pytest.raises(RuntimeError, match="boundary"):
                trainer.assert_update_boundary()
            trainer.discard(result)
            trainer.assert_update_boundary()
        result = trainer.backward(batches_for(rank, 1))
        assert all(p.grad is None for p in model.predictor.parameters())
        trainer.discard(result)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_backward_only_discard_allows_repeated_reducer_iterations(tmp_path):
    mp.spawn(_discard_worker, args=(str(tmp_path / "discard.init"),), nprocs=2, join=True)


def test_constructor_requires_initialized_group():
    assert not dist.is_initialized()
    with pytest.raises(RuntimeError, match="initialized"):
        EagerDDPTrainer(make_model())
