"""Independent dense FBT/NextLat objective and update-normalization checks."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.fbt_training import FBTNextLatLM, aggregate_pass_losses
from cdrm.pretrained.lm_training import (LMTrainingConfig, TrainingCounters, build_adamw,
    build_warmup_scheduler, optimizer_step, save_training_checkpoint, load_training_checkpoint)
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, NextLatLM, NextLatLosses
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTConfig, FBTMode
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(812)


def model(*, enabled=True, gamma=1.0):
    backbone = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32")
    core = OLMoFBT(backbone, FBTConfig(seed=87))
    return FBTNextLatLM(core, NextLatConfig(backbone.config.model_dim, proj_factor=2.0,
                        lambda_latent=0.3, lambda_kl=0.7, seed=81, vocab_chunk_size=2),
                        enabled=enabled, gamma=gamma)


def batch():
    ids = torch.tensor([[2, 5, 8, 11, 15, 21], [7, 9, 14, 18, 1, 1], [3, 6, 1, 1, 1, 1]])
    valid = torch.tensor([[True]*6, [True]*4+[False]*2, [True]*2+[False]*4])
    docs = torch.arange(3)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    ce, latent, kl = [valid.clone() for _ in range(3)]
    ce[0, 2] = False; latent[1, 1] = False; kl[0, 4] = False
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def sliced(value, row):
    return NextLatBatch(**{name: None if item is None else item[row] for name, item in value.__dict__.items()})


def dense_terms(module, hidden, embeddings, value):
    valid, docs = value.valid_mask, value.document_ids
    adjacent = valid[:, :-1] & valid[:, 1:] & (docs[:, :-1] == docs[:, 1:])
    ce = adjacent & value.ce_mask[:, 1:]
    latent = adjacent & value.latent_mask[:, 1:]
    kl = adjacent[:, :-1] & adjacent[:, 1:] & value.kl_mask[:, 2:]
    weight = module.backbone.readout_weight
    zero = hidden.sum()*0
    ce_sum = F.cross_entropy(F.linear(hidden[:, :-1][ce], weight).float(), value.input_ids[:, 1:][ce], reduction="sum")
    sums, counts = {"ce": ce_sum, "latent": zero, "kl": zero}, {"ce": int(ce.sum()), "latent": 0, "kl": 0}
    if module.enabled:
        predicted = module.predictor(hidden[:, :-1], embeddings[:, 1:])
        sums["latent"] = F.smooth_l1_loss(predicted[latent].float(), hidden[:, 1:][latent].detach().float(), reduction="none").mean(-1).sum()
        student = F.log_softmax(F.linear(predicted[:, :-1][kl], weight.detach()).float(), -1)
        teacher = F.log_softmax(F.linear(hidden[:, 1:-1][kl].detach(), weight.detach()).float(), -1)
        sums["kl"] = F.kl_div(student, teacher, reduction="none", log_target=True).sum()
        counts.update(latent=int(latent.sum()), kl=int(kl.sum()))
    return sums, counts


@pytest.mark.parametrize("fbt,rt,nextlat", [(f, r, n) for f in (False, True) for r in (False, True) for n in (False, True)])
def test_all_independent_switches_match_dense_pass_objectives_and_gradients(fbt, rt, nextlat):
    actual = model(enabled=nextlat, gamma=0.4)
    expected = copy.deepcopy(actual)
    value = batch()
    mode = FBTMode(enabled=fbt, num_passes=3 if fbt else 1, beta=0.35,
                   rt_mode=RTMode((0,), 0.37) if rt else RTMode((), 0.0))
    result = actual.loss_sums(value, backbone_kwargs={"mode": mode})
    embeddings = expected.backbone.token_embeddings(value.input_ids)
    outputs = expected.backbone(inputs_embeds=embeddings, attention_mask=value.valid_mask,
                                document_ids=value.document_ids, mode=mode, return_logits=False)
    terms = [dense_terms(expected, hidden, embeddings, value) for hidden in outputs.pass_hidden_states]
    weights = expected.objective_weights()
    objective = 0.0
    for index, (sums, counts) in enumerate(terms):
        coefficient = 1.0 if index == 0 else 0.4/(len(terms)-1)
        objective = objective + coefficient * sum(weights[name] * sums[name] / max(counts[name], 1) for name in weights)
        for name in weights:
            torch.testing.assert_close(result.pass_losses[index].sums[name], sums[name], atol=2e-5, rtol=2e-5)
    assert len(result.pass_losses) == (3 if fbt else 1)
    assert result.counts == actual.counts(value)
    torch.testing.assert_close(result.total, objective, atol=2e-5, rtol=2e-5)
    result.total.backward(); objective.backward()
    for name, parameter in actual.named_parameters():
        other = dict(expected.named_parameters())[name]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=3e-4, msg=name)


@pytest.mark.parametrize("nextlat", [False, True])
def test_k1_matches_existing_nextlat_ordinary_objective(nextlat):
    wrapped = model(enabled=nextlat, gamma=9.0)
    plain = NextLatLM(copy.deepcopy(wrapped.backbone.backbone), wrapped.config, enabled=nextlat)
    if nextlat:
        plain.predictor.load_state_dict(wrapped.predictor.state_dict())
    value = batch()
    result = wrapped(value, backbone_kwargs={"mode": FBTMode(enabled=True, num_passes=1, beta=1.0, rt_mode=RTMode((0,), 1.0))})
    reference = plain(value, backbone_kwargs={"mode": RTMode((), 0.0)})
    assert result.pass_coefficients == (1.0,)
    for name in reference.sums:
        torch.testing.assert_close(result.sums[name], reference.sums[name], atol=0, rtol=0)


@pytest.mark.parametrize("passes,gamma", [(1, 2.0), (2, 1.0), (3, 0.0), (4, 0.3)])
def test_pass_normalization_is_base_plus_mean_extras_not_mean_all(passes, gamma):
    leaves = [torch.tensor(float(i+1), requires_grad=True) for i in range(passes)]
    records = [NextLatLosses({"ce": x, "latent": x*2, "kl": x*3},
                            {"ce": 5, "latent": 4, "kl": 3}, {"ce": 1.0, "latent": 0.2, "kl": 0.1}) for x in leaves]
    result = aggregate_pass_losses(records, gamma=gamma)
    expected = leaves[0] + (gamma*sum(leaves[1:])/(passes-1) if passes > 1 else 0)
    torch.testing.assert_close(result.sums["ce"], expected)
    assert result.counts == records[0].counts
    result.sums["ce"].backward()
    assert leaves[0].grad.item() == 1.0
    for leaf in leaves[1:]:
        assert leaf.grad.item() == pytest.approx(gamma/(passes-1))


def test_weighted_microbatch_accumulation_preserves_the_full_update():
    whole, accumulated = model(gamma=0.4), model(gamma=0.4)
    accumulated.load_state_dict(whole.state_dict())
    mode = FBTMode(enabled=True, num_passes=2, beta=0.4, rt_mode=RTMode((0,), 0.37))
    value = batch()
    optimizers = [build_adamw(module, lr=1e-4, eps=1e-4) for module in (whole, accumulated)]
    config = LMTrainingConfig(max_grad_norm=None)
    full = optimizer_step(whole, optimizers[0], [value], config=config, backbone_kwargs={"mode": mode})
    parts = optimizer_step(accumulated, optimizers[1], [sliced(value, slice(0,1)), sliced(value, slice(1,3))],
                           config=config, backbone_kwargs={"mode": mode})
    assert full["counts"] == parts["counts"] == whole.counts(value)
    assert full["objective"] == pytest.approx(parts["objective"], rel=2e-6)
    for a, b in zip(whole.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a, b, atol=5e-7, rtol=3e-5)
    assert len(list(whole.parameters())) == len({id(parameter) for parameter in whole.parameters()})
    embedding = whole.backbone.readout_weight
    assert sum(parameter is embedding for group in optimizers[0].param_groups for parameter in group["params"]) == 1


@pytest.mark.parametrize("gamma", [-1, float("nan"), float("inf"), True])
def test_invalid_gamma_rejected(gamma):
    with pytest.raises(ValueError, match="gamma"): model(gamma=gamma)
    with pytest.raises(ValueError, match="gamma"): aggregate_pass_losses([], gamma=gamma)


def test_inconsistent_pass_denominators_rejected():
    x = torch.tensor(1.0, requires_grad=True)
    record = NextLatLosses({"ce": x, "latent": x, "kl": x},
                          {"ce": 2, "latent": 2, "kl": 1}, {"ce": 1., "latent": 1., "kl": 1.})
    other = copy.copy(record); other.counts = {**record.counts, "ce": 3}
    with pytest.raises(ValueError, match="identical"): aggregate_pass_losses([record, other])
    with pytest.raises(ValueError, match="at least one"): aggregate_pass_losses([])


def test_wrapper_rejects_document_override_and_packing():
    module, value = model(), batch()
    with pytest.raises(ValueError, match="owns"):
        module(value, backbone_kwargs={"document_ids": value.document_ids})
    docs = value.document_ids.clone(); docs[0, 3:] = 77
    packed = NextLatBatch(value.input_ids, value.valid_mask, docs)
    with pytest.raises(ValueError, match="Packed"):
        module(packed)


def test_static_fbt_source_snapshot_is_pinned_and_not_executed():
    root = Path(__file__).resolve().parents[1] / "cdrm/pretrained/_fbt_reference"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["repository"] == "xidulu/Full-bandwidth-transformer"
    assert manifest["revision"] == "7037c60924870aca6e30fac95212b0c7caee052d"
    expected = {"gpt.py": "bee2292e7f56fe683dbb0e44413c44ef97b47779ee3dd8130bbae6b59673f1a2",
                "LICENSE": "42bb09f339a8af8b90d3b4969b6f6e4ba4aff3b5c614103b031fccdf71c8f772"}
    assert {record["file"]: record["sha256"] for record in manifest["files"]} == expected
    for record in manifest["files"]:
        payload = (root / record["file"]).read_bytes()
        assert len(payload) == record["bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_full_fbt_nextlat_optimizer_resume_exactly_matches_continuation(tmp_path):
    module = model(enabled=True, gamma=0.4)
    mode = FBTMode(enabled=True, num_passes=2, beta=0.35, rt_mode=RTMode((0,), 0.37))
    training = LMTrainingConfig(max_grad_norm=1.0)
    configuration = {"nextlat": module.config.to_dict(), "nextlat_enabled": module.enabled,
                     "fusion": module.backbone.fusion_config.to_dict(), "gamma": module.gamma,
                     "mode": asdict(mode), "training": asdict(training)}
    fingerprint = {"checkpoint_sha256": "1"*64, "source": "tiny FBT ownership/resume fixture"}
    optimizer = build_adamw(module, lr=1e-4, eps=1e-4)
    scheduler = build_warmup_scheduler(optimizer, warmup_updates=2)
    counters = TrainingCounters()
    value = batch()
    def step(model, optimizer, scheduler, counters):
        return optimizer_step(model, optimizer, [value], config=training,
                              backbone_kwargs={"mode": mode}, scheduler=scheduler, counters=counters)
    step(module, optimizer, scheduler, counters)
    step(module, optimizer, scheduler, counters)
    path = tmp_path / "boundary.pt"
    record = save_training_checkpoint(path, module, optimizer, scheduler=scheduler, counters=counters,
                data_cursor={"next_window": 6}, configuration=configuration, source_fingerprint=fingerprint)
    expected_step = step(module, optimizer, scheduler, counters)
    expected = copy.deepcopy((module.state_dict(), optimizer.state_dict(), scheduler.state_dict(), asdict(counters)))
    expected_rng = torch.rand(7)
    resumed = model(enabled=True, gamma=0.4)
    resumed_optimizer = build_adamw(resumed, lr=1e-4, eps=1e-4)
    resumed_scheduler = build_warmup_scheduler(resumed_optimizer, warmup_updates=2)
    restored = load_training_checkpoint(path, resumed, resumed_optimizer, scheduler=resumed_scheduler,
                configuration=configuration, source_fingerprint=fingerprint, expected_sha256=record["sha256"])
    assert restored["data_cursor"] == {"next_window": 6}
    assert restored["counters"].optimizer_updates == 2
    actual_step = step(resumed, resumed_optimizer, resumed_scheduler, restored["counters"])
    assert actual_step == expected_step
    actual = (resumed.state_dict(), resumed_optimizer.state_dict(), resumed_scheduler.state_dict(), asdict(restored["counters"]))
    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left: equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right): equal(a, b)
        else:
            assert left == right
    equal(actual, expected)
    assert torch.equal(torch.rand(7), expected_rng)
    assert resumed.backbone.readout_weight is resumed.backbone.token_embeddings.weight
