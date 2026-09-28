"""Rank-local campaign jitter with real CPU/Gloo DDP, not NCCL qualification."""
import copy
from dataclasses import asdict
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cdrm.pretrained.ddp_training import CoordinatedUpdateError, EagerDDPTrainer
from cdrm.pretrained.distributed_training import (
    ObjectiveForwardAdapter, ddp_normalized_objective, sum_objective_counts,
    validate_microbatch_inputs,
)
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, build_adamw
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


def model_and_mode():
    torch.manual_seed(814)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=True,
        reuse_rope=True, kv_only_writes=True)
    model = FBTNextLatLM(OLMoFBT(base), NextLatConfig(model_dim=32, proj_factor=2,
        lambda_latent=1, lambda_kl=1, vocab_chunk_size=16), pass_loss_policy="campaign_v1").train()
    mode = FBTMode(enabled=True, num_passes=4, rt_mode=RTMode((0,)),
                   first_pass_policy="configured-rt-v1", feedback_jitter=.02)
    return model, mode


def batch_for(rank, micro, update):
    lengths = ((6, 0), (0, 2)) if update == 0 else (((2, 0), (6, 0)) if update == 2 else ((4, 3), (6, 5)))
    length = lengths[rank][micro]
    ids = (torch.arange(12).view(2, 6) + 2 + rank + micro) % 32
    valid = torch.zeros_like(ids, dtype=torch.bool)
    valid[0, :length] = True
    if update == 1:
        valid[1, :length - 1] = True
    docs = torch.full_like(ids, -1)
    docs[valid] = rank * 2 + micro
    if update == 1:
        empty = torch.zeros_like(valid)
        return NextLatBatch(ids, valid, docs, latent_mask=empty, kl_mask=empty)
    return NextLatBatch(ids, valid, docs)


def inputs_for(rank, micro, update):
    generator = torch.Generator().manual_seed(701 + rank * 101 + micro * 17 + update)
    valid = batch_for(rank, micro, update).valid_mask[:, 1:, None]
    return {"feedback_noise": tuple((torch.rand((2, 5, 32), generator=generator) * 2 - 1) * valid
                                    for _ in range(3))}


def assert_no_tensors(value):
    assert not isinstance(value, torch.Tensor), "Tensor payload entered an object collective"
    if isinstance(value, dict):
        for item in value.values():
            assert_no_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_tensors(item)


def test_adapter_keeps_feedback_tensors_outside_shared_configuration():
    torch.set_num_threads(1)
    model, mode = model_and_mode()
    reference = copy.deepcopy(model)
    batch, inputs = batch_for(0, 0, 0), inputs_for(0, 0, 0)
    kwargs = {"mode": mode, "right_padded_causal": True}
    expected = reference.loss_sums(batch, backbone_kwargs={**kwargs, **inputs})
    actual = ObjectiveForwardAdapter(model)(batch, global_counts=model.counts(batch),
                                           backbone_kwargs=kwargs, tensor_inputs=inputs)
    expected.total.backward()
    actual["objective"].backward()
    for name, p in model.named_parameters():
        other = dict(reference.named_parameters())[name]
        if p.grad is None or other.grad is None:
            assert p.grad is other.grad is None
        else:
            torch.testing.assert_close(p.grad, other.grad, rtol=0, atol=0)
    assert set(kwargs) == {"mode", "right_padded_causal"}
    assert all(value.requires_grad is False for value in inputs["feedback_noise"])


@pytest.mark.parametrize("invalid", ["missing", "passes", "shape", "dtype", "device", "nan", "range", "grad", "key", "shared", "disabled"])
def test_bad_tensor_inputs_rejected_before_forward(invalid):
    model, mode = model_and_mode()
    batch, inputs = batch_for(0, 0, 0), inputs_for(0, 0, 0)
    kwargs = {"mode": mode}
    noise = list(inputs["feedback_noise"])
    if invalid == "missing":
        inputs = None
    elif invalid == "passes":
        inputs = {"feedback_noise": tuple(noise[:2])}
    elif invalid == "shape":
        noise[0] = noise[0][:1]
    elif invalid == "dtype":
        noise[0] = noise[0].double()
    elif invalid == "device":
        noise[0] = torch.empty_like(noise[0], device="meta")
    elif invalid == "nan":
        noise[0][0, 0, 0] = float("nan")
    elif invalid == "range":
        noise[0][0, 0, 0] = 1.01
    elif invalid == "grad":
        noise[0].requires_grad_()
    elif invalid == "key":
        inputs = {"unknown_tensor": noise[0]}
    elif invalid == "shared":
        kwargs["feedback_noise"] = tuple(noise)
    elif invalid == "disabled":
        kwargs["mode"] = FBTMode(enabled=False)
    if invalid not in ("missing", "passes", "key"):
        inputs = {"feedback_noise": tuple(noise)}
    with pytest.raises((ValueError, TypeError)):
        validate_microbatch_inputs(model, batch, backbone_kwargs=kwargs, tensor_inputs=inputs)


def _initialize(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{path}",
                            timeout=timedelta(seconds=90))


def _success_worker(rank, path):
    _initialize(rank, path)
    try:
        model, mode = model_and_mode()
        reference = copy.deepcopy(model)
        trainer = EagerDDPTrainer(model, bucket_cap_mb=.02)
        optimizer = build_adamw(model, lr=1e-4, eps=1e-4)
        expected_optimizer = build_adamw(reference, lr=1e-4, eps=1e-4)
        config = LMTrainingConfig(max_grad_norm=.2)
        counters = TrainingCounters()
        kwargs = {"mode": mode, "right_padded_causal": True}
        gather = trainer._gather
        def checked_gather(value):
            assert_no_tensors(value)
            return gather(value)
        trainer._gather = checked_gather
        sync_flags = []
        def before_forward(module, args, kw):
            sync_flags.append(module.require_backward_grad_sync)
            assert "feedback_noise" not in kw["backbone_kwargs"]
            assert "feedback_noise" in kw["tensor_inputs"]
        handle = trainer.ddp.register_forward_pre_hook(before_forward, with_kwargs=True)
        for update in range(3):
            all_batches = [batch_for(r, m, update) for r in range(2) for m in range(2)]
            all_inputs = [inputs_for(r, m, update) for r in range(2) for m in range(2)]
            counts = sum_objective_counts([reference.counts(batch) for batch in all_batches])
            expected_sums = {term: 0. for term in counts}
            expected_optimizer.zero_grad(set_to_none=True)
            for batch, inputs in zip(all_batches, all_inputs):
                result = reference.loss_sums(batch, backbone_kwargs={**kwargs, **inputs})
                ddp_normalized_objective(result, global_counts=counts, world_size=1).backward()
                for term in counts:
                    expected_sums[term] += float(result.sums[term].detach())
            result = trainer.backward([batch_for(rank, m, update) for m in range(2)],
                config=config, backbone_kwargs=kwargs,
                microbatch_inputs=[inputs_for(rank, m, update) for m in range(2)])
            assert result.global_counts == counts
            assert result.global_input_tokens == sum(int(b.valid_mask.sum()) for b in all_batches)
            assert result.global_documents == sum(int(b.valid_mask.any(-1).sum()) for b in all_batches)
            assert result.global_microbatches == 4
            for term in counts:
                assert result.loss_sums[term] == pytest.approx(expected_sums[term], rel=3e-6, abs=2e-6)
            for (name, p), (_, wanted) in zip(model.named_parameters(), reference.named_parameters()):
                if p.grad is None or wanted.grad is None:
                    assert p.grad is wanted.grad is None, name
                else:
                    torch.testing.assert_close(p.grad, wanted.grad, rtol=5e-4, atol=5e-6, msg=name)
            if update == 1:
                assert all(p.grad is None for p in model.predictor.parameters())
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), config.max_grad_norm)
            expected_optimizer.step()
            actual = trainer.step(result, optimizer, counters=counters)
            assert actual["gradient_norm_before_clip"] == pytest.approx(float(expected_norm), rel=3e-5)
            for (name, p), (_, wanted) in zip(model.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p, wanted, rtol=5e-5, atol=4e-7, msg=name)
                assert optimizer.state.get(p, {}).keys() == expected_optimizer.state.get(wanted, {}).keys()
                for key in optimizer.state.get(p, {}):
                    torch.testing.assert_close(optimizer.state[p][key], expected_optimizer.state[wanted][key],
                                               rtol=8e-5, atol=5e-7, msg=f"{name}/{key}")
            trainer.assert_update_boundary()
        assert sync_flags == [False, True] * 3
        assert asdict(counters)["optimizer_updates"] == 3
        handle.remove()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_jitter_accumulation_padding_empty_slots_and_full_adam(tmp_path):
    mp.spawn(_success_worker, args=(str(tmp_path / "inputs.init"),), nprocs=2, join=True)


def _failure_worker(rank, path):
    _initialize(rank, path)
    try:
        for failure in ("late_nan", "entry_count", "shared_tensor", "inactive_noise"):
            model, mode = model_and_mode()
            trainer = EagerDDPTrainer(model)
            optimizer = build_adamw(model, lr=1e-4)
            snapshot = copy.deepcopy(model.state_dict())
            inputs = [inputs_for(rank, m, 0) for m in range(2)]
            kwargs = {"mode": mode, "right_padded_causal": True}
            if rank == 1:
                if failure == "late_nan":
                    inputs[1]["feedback_noise"][2][0, 0, 0] = float("nan")
                elif failure == "entry_count":
                    inputs = inputs[:1]
                elif failure == "shared_tensor":
                    kwargs["feedback_noise"] = inputs[0]["feedback_noise"]
                else:
                    kwargs["mode"] = FBTMode(enabled=False)
            calls = []
            handle = trainer.ddp.register_forward_pre_hook(lambda module, args: calls.append(True))
            counters = TrainingCounters()
            with pytest.raises(CoordinatedUpdateError, match="update preflight"):
                trainer.optimizer_step(optimizer, [batch_for(rank, m, 0) for m in range(2)],
                    backbone_kwargs=kwargs, microbatch_inputs=inputs, counters=counters)
            assert calls == []  # Including malformed noise in the *last* microbatch.
            assert counters == TrainingCounters()
            assert not optimizer.state
            assert all(p.grad is None for p in model.parameters())
            assert all(torch.equal(value, snapshot[name]) for name, value in model.state_dict().items())
            with pytest.raises(RuntimeError, match="reconstructed"):
                trainer.assert_update_boundary()
            handle.remove()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_rank_local_bad_noise_is_coordinated_before_any_forward_or_update(tmp_path):
    mp.spawn(_failure_worker, args=(str(tmp_path / "failures.init"),), nprocs=2, join=True)
