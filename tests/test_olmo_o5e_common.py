"""Ordinary CE, native-only updates, model-only import and exact future recovery."""
import copy
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.lm_training import TrainingCounters, optimizer_ownership, save_training_checkpoint
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTMode
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from scripts import olmo_o5b_common as prior
from scripts import olmo_o5e_common as common


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(7851)


def _model():
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math")
    return prior.ObservedFBTLM(OLMoFBT(base), NextLatConfig(32, vocab_chunk_size=3), enabled=False, gamma=1)


def _batch():
    ids = torch.tensor([[2, 3, 5, 8, 13, 21], [7, 10, 17, 22, 1, 1]])
    valid = torch.tensor([[True]*6, [True]*4+[False]*2])
    docs = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    return NextLatBatch(ids, valid, docs)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left: _equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): _equal(a, b)
    else:
        assert left == right


def test_configuration_preserves_bytes_tying_and_native_only_ownership():
    model = _model(); before = copy.deepcopy(model.state_dict())
    for p in model.parameters(): p.grad = torch.ones_like(p)
    layout = common.configure_trainable(model)
    assert len(layout) == 4*model.backbone.config.num_layers+1
    assert all(row["name"].startswith("backbone.backbone.") for row in layout)
    assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
    assert model.backbone.readout_weight.requires_grad
    assert all(p.grad is None for p in model.parameters())
    _equal(before, model.state_dict())
    assert set(common.frozen_fusion_digests(model)) == set(common.FROZEN_NAMES)
    assert common.make_mode() == FBTMode(enabled=False, num_passes=1, beta=0, rt_mode=RTMode(()))


def test_single_pass_objective_and_all_native_gradients_match_direct_ordinary_ce():
    model = _model(); common.configure_trainable(model); batch = _batch()
    losses = model.loss_sums(batch, backbone_kwargs={"mode": common.make_mode()})
    output = model.backbone.backbone(batch.input_ids, attention_mask=batch.valid_mask, mode=RTMode(()))
    mask = batch.valid_mask[:, :-1] & batch.valid_mask[:, 1:]
    direct = F.cross_entropy(output.logits[:, :-1][mask].float(), batch.input_ids[:, 1:][mask])
    assert losses.pass_coefficients == (1.,) and len(losses.pass_losses) == 1
    assert losses.counts == {"ce": int(mask.sum()), "latent": 0, "kl": 0}
    torch.testing.assert_close(losses.total, direct, rtol=1e-7, atol=1e-7)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    for actual, expected in zip(torch.autograd.grad(losses.total, parameters), torch.autograd.grad(direct, parameters)):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        assert torch.isfinite(actual).all() and actual.norm() > 0
    assert all(p.grad is None for p in model.backbone.fusion.parameters())


def test_native_update_changes_backbone_and_preserves_every_fusion_byte():
    model = _model(); common.configure_trainable(model)
    frozen = common.frozen_fusion_digests(model)
    native_before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = common.build_optimizer(model)
    assert not optimizer.state
    assert optimizer.param_groups[0]["group_name"] == "native"
    assert optimizer.param_groups[0]["lr"] == 1e-5
    scheduler = common.build_scheduler(optimizer)
    result = common.observed_step(model, optimizer, [_batch()], scheduler=scheduler)
    assert len(result["pass_ce_means"]) == 1
    assert result["stack_input_tokens"] == int(_batch().valid_mask.sum())
    assert set(result["group_gradient_norm_after_clip"]) == {"native"}
    assert result["group_gradient_norm_after_clip"]["native"] <= 1.000001
    assert common.frozen_fusion_digests(model) == frozen
    assert len(optimizer.state) == len(native_before)
    assert optimizer_ownership(model, optimizer) == [list(native_before)]
    for name, p in model.named_parameters():
        if name in native_before: assert not torch.equal(p, native_before[name])


@pytest.mark.parametrize("mode", (FBTMode(), FBTMode(enabled=False, num_passes=1, rt_mode=RTMode()),
                                  FBTMode(enabled=False, num_passes=2, beta=0, rt_mode=RTMode(()))))
def test_observed_step_rejects_noncontrol_modes_without_mutation(mode):
    model = _model(); common.configure_trainable(model)
    optimizer = common.build_optimizer(model); before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="only the single-pass"):
        common.observed_step(model, optimizer, [_batch()], backbone_kwargs={"mode": mode})
    _equal(before, model.state_dict()); assert not optimizer.state


def test_control_warmup_uses_base_lr_at_fiftieth_update():
    model = _model(); common.configure_trainable(model)
    optimizer = common.build_optimizer(model)
    scheduler = common.build_scheduler(optimizer)
    lrs = []
    for _ in range(51):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step(); scheduler.step()
    assert lrs[0] == pytest.approx(1e-5/50)
    assert lrs[48] == pytest.approx(49e-5/50)
    assert lrs[49:] == pytest.approx([1e-5, 1e-5])


@pytest.mark.parametrize("error", ("fusion_unfreeze", "native_freeze", "predictor", "dtype"))
def test_wrong_ownership_is_rejected_before_optimizer_creation(error):
    model = _model(); common.configure_trainable(model)
    if error == "fusion_unfreeze": model.backbone.fusion.state_proj.weight.requires_grad_(True)
    elif error == "native_freeze": model.backbone.readout_weight.requires_grad_(False)
    elif error == "predictor": model.predictor = torch.nn.Linear(32, 32)
    elif error == "dtype": model.to(dtype=torch.bfloat16)
    with pytest.raises(ValueError): common.build_optimizer(model)


def test_model_only_source_import_keeps_bytes_and_starts_with_empty_native_optimizer(tmp_path):
    original = _model()
    optimizer = prior.build_optimizer(original, dict(lr=1e-5, fusion_lr=1e-4, weight_decay=.1,
                                                      betas=(.9, .95), eps=1e-8))
    counters = TrainingCounters()
    prior.observed_step(original, optimizer, [_batch()], counters=counters,
                         backbone_kwargs={"mode": prior.make_mode(.7)})
    config = {"arm": "fbt", "storage_prefix": "gs://fixture", "nextlat_enabled": False, "rt_layers": [],
        "num_passes": 2, "gamma": 1, "model_config": original.backbone.config.to_dict(),
        "nextlat_config": original.config.to_dict(), "fusion_config": original.backbone.fusion_config.to_dict(),
        "fusion_output_scale": float(original.backbone.fusion.output_scale), "attention_backend": "math",
        "attention_precision": "mixed", "schedule": {"total_updates": 1, "total_tokens": 10, "used_windows": 2}}
    source = {"checkpoint_sha256": "a"*64, "runtime": {"torch": torch.__version__}}
    path = tmp_path/"source.pt"
    record = save_training_checkpoint(path, original, optimizer, counters=counters, data_cursor={"next_window": 2},
                                      configuration=config, source_fingerprint=source)
    before_rng = torch.get_rng_state().clone()
    model, provenance = common.load_control(device="cpu", endpoint={"checkpoint": record,
        "checkpoint_path": path, "configuration": config, "source_fingerprint": source})
    assert torch.equal(before_rng, torch.get_rng_state())
    _equal(original.state_dict(), model.state_dict())
    assert provenance["checkpoint_sha256"] == record["sha256"]
    assert provenance["frozen_fusion_digests"] == common.frozen_fusion_digests(model)
    assert all(r["name"].startswith("backbone.backbone.") for r in provenance["trainable_parameters"])
    assert not common.build_optimizer(model).state


def test_checkpoint_replay_restores_exact_future_update_optimizer_scheduler_counters_and_rng(tmp_path):
    model = _model(); common.configure_trainable(model)
    frozen = common.frozen_fusion_digests(model)
    optimizer = common.build_optimizer(model)
    scheduler = common.build_scheduler(optimizer, warmup_updates=3)
    counters = TrainingCounters()
    config = {"mode": asdict(common.make_mode()), "lr": 1e-5, "runtime": {"torch": torch.__version__}}
    source = {"checkpoint_sha256": "a"*64, "runtime": {"torch": torch.__version__}}
    def step(m, o, s, c): return common.observed_step(m, o, [_batch()], scheduler=s, counters=c)
    step(model, optimizer, scheduler, counters); step(model, optimizer, scheduler, counters)
    path = tmp_path/"control.pt"
    receipt = common.save_checkpoint(path, model, optimizer, counters=counters, scheduler=scheduler,
        configuration=config, source_fingerprint=source, data_cursor={"next_update": 2, "bad_evals": 0})
    assert type(torch.load(path, weights_only=True)["configuration"]["runtime"]["torch"]) is str
    expected_result = step(model, optimizer, scheduler, counters)
    expected = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), asdict(counters)))
    expected_random = torch.rand(5)
    recovered = _model(); common.configure_trainable(recovered)
    recovered_optimizer = common.build_optimizer(recovered)
    recovered_scheduler = common.build_scheduler(recovered_optimizer, warmup_updates=3)
    restored = common.load_checkpoint(path, recovered, recovered_optimizer, scheduler=recovered_scheduler,
        configuration=config, source_fingerprint=source, expected_sha256=receipt["sha256"])
    assert restored["data_cursor"] == {"next_update": 2, "bad_evals": 0}
    actual = step(recovered, recovered_optimizer, recovered_scheduler, restored["counters"])
    assert actual == expected_result
    _equal(expected, (recovered.state_dict(), recovered_optimizer.state_dict(),
                      recovered_scheduler.state_dict(), asdict(restored["counters"])))
    assert torch.equal(torch.rand(5), expected_random)
    assert common.frozen_fusion_digests(recovered) == frozen


def test_inventory_contains_frozen_parent_math_and_new_control_runtime(tmp_path, monkeypatch):
    assert set(common.O5C_SOURCES) <= set(common.SOURCE_FILES)
    assert {"scripts/olmo_o5e_common.py", "scripts/olmo_o5e_train.py", "scripts/olmo_o5e_preflight.py",
            "docs/reports/olmo1b-o5e/protocol.md"} <= set(common.SOURCE_FILES)
    for name in common.SOURCE_FILES:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    assert set(common.source_hashes()) == set(common.SOURCE_FILES)


def test_retention_wrapper_rejects_other_lineages_and_passes_hash_authority(tmp_path, monkeypatch):
    from scripts import olmo_o4_common
    calls = []
    def fake(path, uri, **kwargs):
        calls.append((path, uri, kwargs)); return {"verified": True}
    monkeypatch.setattr(olmo_o4_common, "retain_file", fake)
    path = tmp_path/"control.pt"; path.write_text("fixture")
    with pytest.raises(ValueError, match="designated"):
        common.retain_file(path, "gs://fast-chunks/other/file")
    uri = common.PREFIX_ROOT+"fixture/control.pt"
    assert common.retain_file(path, uri, expected_sha256="a"*64) == {"verified": True}
    assert calls == [(path, uri, {"expected_sha256": "a"*64})]
