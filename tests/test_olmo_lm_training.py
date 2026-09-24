"""Global objective normalization, optimizer ownership, and full boundary resume."""
import copy
from dataclasses import asdict
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cdrm.pretrained.lm_training import (
    LMTrainingConfig, TrainingCounters, build_adamw, build_warmup_scheduler,
    load_training_checkpoint, optimizer_ownership, optimizer_state_bytes,
    optimizer_step, parameter_layout, save_training_checkpoint,
)
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, NextLatLM
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode

FINGERPRINT = {"checkpoint_sha256": "a" * 64, "lineage": "tiny deterministic fixture"}


@pytest.fixture(autouse=True)
def seed_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(1307); random.seed(1307); np.random.seed(1307)


def model(*, recurrent=False, enabled=True, dropout=0.0):
    config = OLMoConfig.tiny()
    backbone = (OLMoTiledRTForCausalLM(config, attention_backend="math", attention_precision="fp32")
                if recurrent else OLMoForCausalLM(config, attention_backend="math"))
    return NextLatLM(backbone, NextLatConfig(config.model_dim, proj_factor=2.0,
                     dropout=dropout, lambda_latent=0.3, lambda_kl=0.7, vocab_chunk_size=3, seed=77), enabled=enabled)


def batch():
    ids = torch.tensor([[2, 3, 5, 8, 13, 21, 34], [4, 7, 10, 17, 1, 1, 1], [6, 9, 1, 1, 1, 1, 1]])
    valid = torch.tensor([[True] * 7, [True] * 4 + [False] * 3, [True] * 2 + [False] * 5])
    docs = torch.arange(3)[:, None].expand_as(ids).clone()
    ce, latent, kl = (torch.ones_like(valid) for _ in range(3))
    ce[0, 1:3] = False
    latent[1, 2] = False
    kl[0, 3] = False
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def subset(value, indices):
    return NextLatBatch(**{name: None if item is None else item[indices] for name, item in value.__dict__.items()})


def equal_tree(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual: equal_tree(actual[name], expected[name])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected): equal_tree(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("recurrent,enabled", [(False, False), (False, True), (True, False), (True, True)])
def test_unequal_microbatches_match_global_objective_and_update(recurrent, enabled):
    whole, accumulated = model(recurrent=recurrent, enabled=enabled), model(recurrent=recurrent, enabled=enabled)
    accumulated.load_state_dict(whole.state_dict())
    optimizers = [build_adamw(m, lr=1e-3, eps=1e-4) for m in (whole, accumulated)]
    kwargs = {"mode": RTMode((0,), 0.37)} if recurrent else {}
    value = batch()
    configuration = LMTrainingConfig(max_grad_norm=None)
    rows = [optimizer_step(whole, optimizers[0], [value], config=configuration, backbone_kwargs=kwargs),
            optimizer_step(accumulated, optimizers[1], [subset(value, slice(0, 1)), subset(value, slice(1, 3))],
                           config=configuration, backbone_kwargs=kwargs)]
    assert rows[0]["counts"] == rows[1]["counts"]
    assert rows[0]["counts"]["ce"] == 8
    if enabled:
        assert rows[0]["counts"] == {"ce": 8, "latent": 9, "kl": 6}
        assert rows[0]["objective_weights"] == {"ce": 1.0, "latent": 0.3, "kl": 0.7}
    assert rows[0]["objective"] == pytest.approx(rows[1]["objective"], rel=2e-6)
    for a, b in zip(whole.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-7)
    assert all(p.grad is None for p in accumulated.parameters())
    assert rows[1]["counters"]["microbatches"] == 2
    assert rows[1]["counters"]["input_tokens"] == 13


def test_adamw_owns_tied_embedding_once_and_accounts_initialized_state():
    module = model()
    optimizer = build_adamw(module, lr=0.0)
    matrix = module.backbone.token_embeddings.weight
    assert module.backbone.readout_weight is matrix
    assert sum(parameter is matrix for group in optimizer.param_groups for parameter in group["params"]) == 1
    assert optimizer_state_bytes(optimizer) == {}
    before = {name: p.detach().clone() for name, p in module.named_parameters()}
    row = optimizer_step(module, optimizer, [batch()])
    assert row["optimizer_state_bytes_by_device"]["cpu"] > 0
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert len(parameter_layout(module)) == len(tuple(module.parameters()))


@pytest.mark.parametrize("fused", [None, False, True])
def test_adamw_fused_choice_preserves_tied_ownership_groups_and_fp32_parameters(fused):
    module = model()
    optimizer = build_adamw(module, lr=1e-5, fused=fused)
    assert optimizer.defaults["fused"] is fused
    assert optimizer.defaults["foreach"] is False
    assert not optimizer.defaults["capturable"]
    assert len(optimizer_ownership(module, optimizer)) == len(optimizer.param_groups)
    matrix = module.backbone.token_embeddings.weight
    assert sum(p is matrix for group in optimizer.param_groups for p in group["params"]) == 1
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert parameter.dtype == torch.float32
            assert group["weight_decay"] == (0.1 if parameter.ndim >= 2 else 0.)
    # Construction is CPU-only; GPU fused update semantics are a separate probe.
    assert not optimizer.state


def test_adamw_default_preserves_legacy_nonfused_descriptor():
    assert build_adamw(model(), lr=1e-5).defaults["fused"] is None


@pytest.mark.parametrize("fused", [1, "yes", object()])
def test_adamw_rejects_nonboolean_fusion_without_mutating_model(fused):
    module = model()
    before = {name: value.clone() for name, value in module.state_dict().items()}
    with pytest.raises(TypeError, match="fused"):
        build_adamw(module, lr=1e-5, fused=fused)
    equal_tree(module.state_dict(), before)


def test_adamw_rejects_conflicting_foreach_and_fused_choices():
    with pytest.raises(ValueError, match="cannot be combined"):
        build_adamw(model(), lr=1e-5, foreach=True, fused=True)


def test_optimizer_rejects_foreign_missing_and_duplicate_ownership():
    module = model()
    optimizer = build_adamw(module, lr=1e-3)
    optimizer.param_groups[0]["params"].append(optimizer.param_groups[0]["params"][0])
    with pytest.raises(ValueError, match="duplicate"):
        optimizer_ownership(module, optimizer)
    missing = torch.optim.AdamW([next(module.parameters())], lr=1e-3)
    with pytest.raises(ValueError, match="missing"):
        optimizer_ownership(module, missing)


def test_nonfinite_gradient_refuses_update_clears_grads_and_keeps_counters():
    module = model(enabled=False)
    optimizer = build_adamw(module, lr=1e-3)
    counters = TrainingCounters()
    before = {name: value.clone() for name, value in module.state_dict().items()}
    hook = next(module.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(RuntimeError, match="non-finite"):
        optimizer_step(module, optimizer, [batch()], counters=counters)
    hook.remove()
    equal_tree(module.state_dict(), before)
    assert not optimizer.state
    assert counters.optimizer_updates == 0
    assert all(parameter.grad is None for parameter in module.parameters())


def test_rejects_packed_documents_before_any_optimizer_state():
    module = model()
    optimizer = build_adamw(module, lr=1e-3)
    value = batch()
    value.document_ids[0, 3:] = 9
    with pytest.raises(ValueError, match="document|Packed"):
        optimizer_step(module, optimizer, [value])
    assert not optimizer.state


def draw_batch(generator):
    ids = torch.randint(2, 60, (2, 5), generator=generator)
    ids = (ids + random.randrange(3) + int(np.random.randint(0, 3))) % 60 + 2
    valid = torch.ones_like(ids, dtype=torch.bool)
    docs = torch.arange(2)[:, None].expand_as(ids)
    return NextLatBatch(ids, valid, docs)


def test_exact_cpu_resume_model_predictor_optimizer_scheduler_rng_and_cursor(tmp_path):
    module = model(dropout=0.2)
    optimizer = build_adamw(module, lr=3e-4)
    scheduler = build_warmup_scheduler(optimizer, warmup_updates=4)
    counters, generator = TrainingCounters(), torch.Generator().manual_seed(281)
    config = {"step": LMTrainingConfig(), "nextlat": module.config, "mode": "ordinary", "data": "fixture"}
    for _ in range(2):
        optimizer_step(module, optimizer, [draw_batch(generator)], counters=counters, scheduler=scheduler)
    record = save_training_checkpoint(tmp_path / "step-2.pt", module, optimizer, scheduler=scheduler,
             counters=counters, data_cursor={"next_example": 4}, configuration=config,
             source_fingerprint=FINGERPRINT, generators={"data": generator})
    third = draw_batch(generator)
    expected_row = optimizer_step(module, optimizer, [third], counters=counters, scheduler=scheduler)
    expected_model = copy.deepcopy(module.state_dict())
    expected_optimizer, expected_scheduler = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
    expected_rng = (torch.get_rng_state().clone(), random.random(), float(np.random.rand()), generator.get_state().clone())
    resumed = model(dropout=0.2)
    new_optimizer = build_adamw(resumed, lr=3e-4)
    new_scheduler = build_warmup_scheduler(new_optimizer, warmup_updates=4)
    new_generator = torch.Generator().manual_seed(999)
    restored = load_training_checkpoint(record["path"], resumed, new_optimizer, scheduler=new_scheduler,
        configuration=config, source_fingerprint=FINGERPRINT, generators={"data": new_generator}, expected_sha256=record["sha256"])
    assert restored["data_cursor"] == {"next_example": 4}
    assert restored["counters"].optimizer_updates == 2
    repeated = draw_batch(new_generator)
    torch.testing.assert_close(repeated.input_ids, third.input_ids, atol=0, rtol=0)
    resumed_row = optimizer_step(resumed, new_optimizer, [repeated], counters=restored["counters"], scheduler=new_scheduler)
    equal_tree(resumed.state_dict(), expected_model)
    equal_tree(new_optimizer.state_dict(), expected_optimizer)
    equal_tree(new_scheduler.state_dict(), expected_scheduler)
    assert resumed_row == expected_row
    torch.testing.assert_close(torch.get_rng_state(), expected_rng[0], atol=0, rtol=0)
    assert random.random() == expected_rng[1]
    assert float(np.random.rand()) == expected_rng[2]
    torch.testing.assert_close(new_generator.get_state(), expected_rng[3], atol=0, rtol=0)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.fixture
def saved_checkpoint(tmp_path):
    module = model()
    optimizer = build_adamw(module, lr=1e-3)
    counters = TrainingCounters()
    optimizer_step(module, optimizer, [batch()], counters=counters)
    config = {"nextlat": module.config.to_dict(), "precision": "fp32", "rt": False}
    record = save_training_checkpoint(tmp_path / "checkpoint.pt", module, optimizer, counters=counters,
                                     data_cursor={"next_example": 3}, configuration=config, source_fingerprint=FINGERPRINT)
    return module, optimizer, counters, config, record


@pytest.mark.parametrize("mismatch", ["configuration", "fingerprint", "sha", "optimizer_names", "moment_shape"])
def test_resume_rejects_mismatch_before_mutating_live_weights(saved_checkpoint, mismatch):
    _, _, _, config, record = saved_checkpoint
    module = model()
    optimizer = build_adamw(module, lr=1e-3)
    before = copy.deepcopy(module.state_dict())
    fingerprint = FINGERPRINT
    digest = None
    if mismatch == "configuration": config = {**config, "rt": True}
    elif mismatch == "fingerprint": fingerprint = {**FINGERPRINT, "checkpoint_sha256": "b" * 64}
    elif mismatch == "sha": digest = "0" * 64
    else:
        payload = torch.load(record["path"], weights_only=True)
        if mismatch == "optimizer_names": payload["optimizer"]["param_groups"][0]["param_names"][0] = "wrong"
        else: next(iter(payload["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(1)
        torch.save(payload, record["path"])
    with pytest.raises(ValueError):
        load_training_checkpoint(record["path"], module, optimizer, configuration=config,
                                  source_fingerprint=fingerprint, expected_sha256=digest)
    equal_tree(module.state_dict(), before)
    assert not optimizer.state


def test_save_refuses_overwrite_and_nonboundary_gradients(saved_checkpoint):
    module, optimizer, counters, config, record = saved_checkpoint
    kwargs = dict(counters=counters, data_cursor={}, configuration=config, source_fingerprint=FINGERPRINT)
    with pytest.raises(FileExistsError):
        save_training_checkpoint(record["path"], module, optimizer, **kwargs)
    next(module.parameters()).grad = torch.zeros_like(next(module.parameters()))
    with pytest.raises(ValueError, match="boundary"):
        save_training_checkpoint(str(record["path"]) + ".new", module, optimizer, **kwargs)


def test_warmup_lr_and_counters_describe_optimizer_updates():
    module = model(enabled=False)
    optimizer = build_adamw(module, lr=3e-4)
    scheduler = build_warmup_scheduler(optimizer, warmup_updates=3)
    counters = TrainingCounters()
    rows = [optimizer_step(module, optimizer, [batch()], scheduler=scheduler, counters=counters) for _ in range(4)]
    assert [row["lr_used"][0] for row in rows] == pytest.approx([1e-4, 2e-4, 3e-4, 3e-4])
    assert counters.optimizer_updates == 4
    assert counters.documents == 12
    assert counters.input_tokens == 52


def test_empty_masked_microbatch_keeps_global_denominators_and_attached_zero_safe():
    baseline, with_empty = model(), model()
    with_empty.load_state_dict(baseline.state_dict())
    optimizers = [build_adamw(m, lr=1e-4, eps=1e-4) for m in (baseline, with_empty)]
    nonempty = subset(batch(), slice(0, 1))
    empty = NextLatBatch(torch.full((1, 4), 1, dtype=torch.long),
                         torch.zeros((1, 4), dtype=torch.bool), torch.full((1, 4), -1, dtype=torch.long))
    first = optimizer_step(baseline, optimizers[0], [nonempty])
    second = optimizer_step(with_empty, optimizers[1], [empty, nonempty])
    assert first["counts"] == second["counts"]
    assert first["objective"] == second["objective"]
    assert second["counters"]["documents"] == 1
    assert second["counters"]["input_tokens"] == 7
    equal_tree(baseline.state_dict(), with_empty.state_dict())
