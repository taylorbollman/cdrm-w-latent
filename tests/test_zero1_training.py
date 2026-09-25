"""Real Gloo checks of native ZeRO-1 math, state partition and recovery."""
import copy
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cdrm.pretrained.ddp_training import EagerDDPTrainer
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.lm_training import (LMTrainingConfig, TrainingCounters, build_adamw,
    build_warmup_scheduler, optimizer_state_bytes, optimizer_step)
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTMode
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.zero1_training import (FP32Zero1AdamW, build_zero1_adamw,
    zero1_state_inventory, zero1_checkpoint_configuration,
    save_zero1_checkpoint, load_zero1_checkpoint)


def make_model(combined):
    torch.manual_seed(109)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
                                 attention_precision="fp32", ordinary_activation_checkpointing=True)
    core = OLMoFBT(base)
    if not combined: core.fusion.requires_grad_(False)
    return FBTNextLatLM(core, NextLatConfig(model_dim=32, proj_factor=2, vocab_chunk_size=4),
                        enabled=combined).train()


def batches(rank, update):
    result = []
    for micro in range(2):
        ids = torch.tensor([[2, 3, 4, 5, 6, 7]])+rank+micro+update
        valid = torch.ones_like(ids, dtype=torch.bool)
        docs = torch.full_like(ids, rank*2+micro)
        ce, latent, kl = [valid.clone() for _ in range(3)]
        ce[:, 2+rank] = False
        if rank == 1: ce[:, 1] = False
        if rank or micro or update == 1:
            latent.zero_(); kl.zero_()
        result.append(NextLatBatch(ids, valid, docs, ce, latent, kl))
    return result


def equal_tree(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def compare_local(model, optimizer, reference, reference_optimizer):
    by_name = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        target = by_name[name]
        torch.testing.assert_close(parameter, target, atol=4e-7, rtol=5e-5, msg=name)
        local_state = optimizer.optim.state.get(parameter)
        if local_state is not None:
            expected = reference_optimizer.state[target]
            assert local_state.keys() == expected.keys()
            for key in local_state:
                torch.testing.assert_close(local_state[key], expected[key], atol=5e-7, rtol=8e-5, msg=name+key)


def snapshot(model, optimizer, scheduler, counters):
    return copy.deepcopy({"model": model.state_dict(), "local_optimizer": optimizer.optim.state_dict(),
        "scheduler": scheduler.state_dict(), "counters": asdict(counters)})


def draws():
    return {"python": random.random(), "numpy": np.random.rand(4).tolist(), "torch": torch.rand(4)}


def _worker(rank, filename, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{filename}",
                            timeout=timedelta(seconds=180))
    try:
        for combined in (False, True):
            for fused in (False, True):
                model = make_model(combined)
                reference = copy.deepcopy(model)
                trainer = EagerDDPTrainer(model)
                optimizer = build_zero1_adamw(model, lr=1e-4, eps=1e-4, fused=fused)
                reference_optimizer = build_adamw(reference, lr=1e-4, eps=1e-4, fused=fused)
                scheduler = build_warmup_scheduler(optimizer, warmup_updates=2)
                ref_scheduler = build_warmup_scheduler(reference_optimizer, warmup_updates=2)
                counters, ref_counters = TrainingCounters(), TrainingCounters()
                config = LMTrainingConfig(max_grad_norm=.3)
                kwargs = {"mode": FBTMode(enabled=combined, num_passes=2, rt_mode=RTMode((0, 1)))}
                addresses = [parameter.data_ptr() for parameter in model.parameters()]
                for update in range(2):
                    global_batches = [batch for r in range(2) for batch in batches(r, update)]
                    expected = optimizer_step(reference, reference_optimizer, global_batches,
                        config=config, backbone_kwargs=kwargs, scheduler=ref_scheduler, counters=ref_counters)
                    actual = trainer.optimizer_step(optimizer, batches(rank, update), config=config,
                        backbone_kwargs=kwargs, scheduler=scheduler, counters=counters)
                    assert actual["counts"] == expected["counts"]
                    assert actual["objective"] == pytest.approx(expected["objective"], rel=3e-6)
                    assert actual["gradient_norm_before_clip"] == pytest.approx(expected["gradient_norm_before_clip"], rel=3e-5)
                    assert asdict(counters) == asdict(ref_counters)
                    compare_local(model, optimizer, reference, reference_optimizer)
                    assert addresses == [parameter.data_ptr() for parameter in model.parameters()]
                    inventory = zero1_state_inventory(model, optimizer)
                    gathered = [None, None]
                    dist.all_gather_object(gathered, inventory)
                    names = [name for item in gathered for name in item["local_owned_names"]]
                    assert len(names) == len(set(names))
                    assert set(names) == {name for name, p in model.named_parameters() if p.requires_grad}
                    assert sum(sum(item["local_state_bytes_by_device"].values()) for item in gathered) == sum(optimizer_state_bytes(reference_optimizer).values())
                    assert all(not item["outer_state_bytes_by_device"] for item in gathered)
                    assert all(not item["consolidation_cache_present"] for item in gathered)
                    assert not optimizer.state
                folder = Path(output)/f"combined-{combined}-fused-{fused}"
                source = {"sha256": "a"*64}
                configuration = {"model": "tiny", "combined": combined, "fused": fused}
                receipt = save_zero1_checkpoint(folder, model, optimizer, scheduler=scheduler,
                    counters=counters, data_cursor={"rank": rank, "update": 2}, configuration=configuration,
                    source_fingerprint=source)
                assert not optimizer._all_state_dicts and not optimizer.state
                assert receipt["metadata"]["optimizer_descriptor"]["class"].endswith("FP32Zero1AdamW")
                assert receipt["metadata"]["configuration"]["_zero1"]["checkpoint_layout"] == "consolidated-global-optimizer-rank-zero"
                optimizer.consolidate_state_dict(to=0)
                if rank == 0:
                    consolidated = copy.deepcopy(optimizer.state_dict())
                    untouched = copy.deepcopy(consolidated)
                    optimizer.load_state_dict(consolidated)
                    assert equal_tree(consolidated, untouched)
                    assert not optimizer._all_state_dicts and not optimizer.state
                dist.barrier()
                trainer.optimizer_step(optimizer, batches(rank, 2), config=config, backbone_kwargs=kwargs,
                                       scheduler=scheduler, counters=counters)
                expected = snapshot(model, optimizer, scheduler, counters)
                expected_draws = draws()
                fresh = make_model(combined)
                restored_trainer = EagerDDPTrainer(fresh)
                restored_optimizer = build_zero1_adamw(fresh, lr=1e-4, eps=1e-4, fused=fused)
                restored_scheduler = build_warmup_scheduler(restored_optimizer, warmup_updates=2)
                restored = load_zero1_checkpoint(folder, fresh, restored_optimizer, scheduler=restored_scheduler,
                    configuration=configuration, source_fingerprint=source,
                    expected_manifest_sha256=receipt["manifest_sha256"])
                assert restored["data_cursor"] == {"rank": rank, "update": 2}
                restored_trainer.optimizer_step(restored_optimizer, batches(rank, 2), config=config,
                    backbone_kwargs=kwargs, scheduler=restored_scheduler, counters=restored["counters"])
                assert equal_tree(expected, snapshot(fresh, restored_optimizer, restored_scheduler, restored["counters"]))
                assert equal_tree(expected_draws, draws())
                assert not restored_optimizer.state
                assert not restored_optimizer._all_state_dicts
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_real_zero1_partition_updates_and_consolidated_resume(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path/"group.init"), str(tmp_path/"checkpoints")), nprocs=2, join=True)


def test_builder_requires_initialized_group():
    assert not dist.is_initialized()
    with pytest.raises(RuntimeError, match="initialized"):
        build_zero1_adamw(make_model(False), lr=1e-4)


@pytest.fixture
def one_rank_optimizer(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1,
        init_method=f"file://{tmp_path/'one.init'}", timeout=timedelta(seconds=30))
    try:
        model = torch.nn.Linear(3, 2)
        optimizer = build_zero1_adamw(model, lr=1e-4, fused=True)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        optimizer.consolidate_state_dict(to=0)
        yield model, optimizer, copy.deepcopy(optimizer.state_dict())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", ["schema", "groups", "names", "duplicate", "foreign", "missing_shard"])
def test_loader_rejects_invalid_consolidated_records_before_optimizer_mutation(one_rank_optimizer, failure):
    model, optimizer, state = one_rank_optimizer
    previous = copy.deepcopy(optimizer.optim.state_dict())
    if failure == "schema": state["unexpected"] = 1
    elif failure == "groups": state["param_groups"].pop()
    elif failure == "names": state["param_groups"][0]["param_names"][0] = "wrong"
    elif failure == "duplicate":
        state["param_groups"][1]["params"][0] = state["param_groups"][0]["params"][0]
    elif failure == "foreign": state["state"][1000000] = {}
    elif failure == "missing_shard": state["state"][next(iter(state["state"]))] = None
    with pytest.raises(ValueError): optimizer.load_state_dict(state)
    assert equal_tree(previous, optimizer.optim.state_dict())


def test_policy_records_true_optimizer_and_rejects_reserved_configuration(one_rank_optimizer):
    model, optimizer, _ = one_rank_optimizer
    policy = zero1_checkpoint_configuration(model, optimizer, {"experiment": "unit"})
    assert policy["_zero1"]["optimizer"].endswith("FP32Zero1AdamW")
    assert policy["_zero1"]["parameters_as_bucket_view"] is False
    assert policy["_zero1"]["overlap_with_ddp"] is False
    with pytest.raises(ValueError, match="reserved"):
        zero1_checkpoint_configuration(model, optimizer, {"_zero1": "spoof"})


def test_next_step_invalidates_old_consolidation_and_preserves_storage(one_rank_optimizer):
    model, optimizer, state = one_rank_optimizer
    optimizer.load_state_dict(state)
    optimizer.consolidate_state_dict(to=0)
    addresses = [parameter.data_ptr() for parameter in model.parameters()]
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    assert addresses == [parameter.data_ptr() for parameter in model.parameters()]
    assert not optimizer.state and not optimizer._all_state_dicts
    with pytest.raises(RuntimeError, match="consolidat"):
        optimizer.state_dict()
