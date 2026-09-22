"""Fusion-only gradient routing, pinned import, ownership and exact recovery."""

import base64
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import (LMTrainingConfig, TrainingCounters,
    save_training_checkpoint, optimizer_ownership)
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_o5b_common as prior
from scripts import olmo_o5c_common as common


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(6521)


def _model():
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math")
    return prior.ObservedFBTLM(OLMoFBT(base), NextLatConfig(32, vocab_chunk_size=3), enabled=False, gamma=1)


def _batch():
    ids = torch.tensor([[2, 3, 5, 8, 13, 21], [7, 10, 17, 22, 1, 1]])
    valid = torch.tensor([[True]*6, [True]*4+[False]*2])
    docs = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    return NextLatBatch(ids, valid, docs)


def _equal(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: _equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): _equal(x, y)
    else: assert a == b


def _checkpoint_fixture(tmp_path):
    model = _model()
    optimizer = prior.build_optimizer(model, dict(lr=1e-5, fusion_lr=1e-4, weight_decay=.1, betas=(.9, .95), eps=1e-8))
    counters = TrainingCounters()
    prior.observed_step(model, optimizer, [_batch()], counters=counters, backbone_kwargs={"mode": prior.make_mode(.7)})
    configuration = dict(arm="fbt", storage_prefix="gs://fixture", nextlat_enabled=False, rt_layers=[], num_passes=2,
        gamma=1, model_config=model.backbone.config.to_dict(), nextlat_config=model.config.to_dict(),
        fusion_config=model.backbone.fusion_config.to_dict(), fusion_output_scale=float(model.backbone.fusion.output_scale),
        attention_backend="math", attention_precision="mixed", schedule={"total_updates": 1, "total_tokens": 10, "used_windows": 2})
    source = {"checkpoint_sha256": "9"*64, "runtime": {"torch": torch.__version__}}
    path = tmp_path/"prior.pt"
    receipt = save_training_checkpoint(path, model, optimizer, counters=counters, data_cursor={"next_window": 2},
                                       configuration=configuration, source_fingerprint=source)
    return model, path, receipt, configuration, source


def test_freeze_keeps_native_function_and_pass0_has_no_graph_but_fusion_has_credit():
    model = _model()
    before = copy.deepcopy(model.state_dict())
    for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
    layout = common.freeze_native(model)
    assert {record["name"] for record in layout} == set(common.FUSION_NAMES)
    assert sum(record["numel"] for record in layout) == 2*32**2
    assert all(parameter.grad is None for parameter in model.parameters())
    _equal(model.state_dict(), before)
    assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
    assert not model.backbone.readout_weight.requires_grad
    result = model.loss_sums(_batch(), backbone_kwargs={"mode": common.make_mode(1)})
    assert not result.pass_losses[0].sums["ce"].requires_grad
    assert result.pass_losses[1].sums["ce"].requires_grad
    parameters = tuple(model.backbone.fusion.parameters())
    combined = torch.autograd.grad(result.total, parameters, retain_graph=True)
    final = torch.autograd.grad(result.pass_losses[1].total, parameters)
    for left, right in zip(combined, final):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert torch.isfinite(left).all() and left.norm() > 0
    assert all(parameter.grad is None for parameter in model.backbone.backbone.parameters())


def test_updates_change_both_fusion_matrices_only_and_keep_pass0_fixed():
    model = _model()
    common.freeze_native(model)
    fixed = common.frozen_state_digests(model)
    fusion_before = {name: dict(model.named_parameters())[name].clone() for name in common.FUSION_NAMES}
    optimizer = common.build_optimizer(model)
    scheduler = common.build_scheduler(optimizer)
    assert optimizer_ownership(model, optimizer) == [list(common.FUSION_NAMES)]
    assert optimizer.param_groups[0]["group_name"] == "fusion"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-6)
    results = [common.observed_step(model, optimizer, [_batch()], scheduler=scheduler,
               backbone_kwargs={"mode": common.make_mode(1)}) for _ in range(3)]
    assert len(optimizer.state) == 2
    assert all(result["group_gradient_norm_after_clip"]["fusion"] <= 1.000001 for result in results)
    assert all(result["pass_ce_means"][0] == results[0]["pass_ce_means"][0] for result in results)
    assert common.frozen_state_digests(model) == fixed
    assert all(not torch.equal(dict(model.named_parameters())[name], value) for name, value in fusion_before.items())


def test_fusion_scheduler_reaches_configured_lr_on_update50():
    model = _model(); common.freeze_native(model)
    optimizer = common.build_optimizer(model)
    scheduler = common.build_scheduler(optimizer)
    used = []
    for _ in range(51):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step(); scheduler.step()
    assert used[0] == pytest.approx(1e-4/50)
    assert used[48] == pytest.approx(49e-4/50)
    assert used[49:] == pytest.approx([1e-4, 1e-4])


def test_endpoint_load_is_model_only_exact_and_fresh_optimizer_has_no_prior_moments(tmp_path):
    original, path, receipt, config, source = _checkpoint_fixture(tmp_path)
    before_rng = torch.get_rng_state().clone()
    model, provenance = common.load_frozen_endpoint(path, receipt["sha256"], expected_configuration=config,
                                                  expected_source_fingerprint=source, device="cpu")
    assert torch.equal(before_rng, torch.get_rng_state())
    _equal(model.state_dict(), original.state_dict())
    assert provenance["checkpoint_sha256"] == receipt["sha256"]
    assert len(provenance["trainable_parameters"]) == 2
    assert provenance["frozen_state_digests"] == common.frozen_state_digests(model)
    assert type(provenance["endpoint_source_fingerprint"]["runtime"]["torch"]) is str
    fresh = common.build_optimizer(model)
    assert not fresh.state
    assert len(fresh.param_groups) == 1 and len(fresh.param_groups[0]["params"]) == 2


@pytest.mark.parametrize("failure", ["hash", "config", "source", "class", "shape", "nonfinite", "scale"])
def test_endpoint_load_rejects_wrong_authority_or_tensor_geometry(tmp_path, failure):
    _, path, receipt, config, source = _checkpoint_fixture(tmp_path)
    digest = receipt["sha256"]
    if failure == "hash": digest = "0"*64
    elif failure == "config": config = {**config, "gamma": 2}
    elif failure == "source": source = {"checkpoint_sha256": "0"*64}
    else:
        payload = common._load_weights_only(path)
        if failure == "class": payload["model_type"] = "other.Model"
        elif failure == "shape": payload["model"][common.FUSION_NAMES[0]] = torch.zeros(3, 3)
        elif failure == "nonfinite": payload["model"][common.FUSION_NAMES[0]][0, 0] = float("nan")
        elif failure == "scale": payload["model"]["backbone.fusion.output_scale"] *= 2
        torch.save(payload, path); digest = sha256_file(path)
    with pytest.raises(ValueError):
        common.load_frozen_endpoint(path, digest, expected_configuration=config, expected_source_fingerprint=source, device="cpu")


def test_checkpoint_recovery_preserves_frozen_state_and_exact_future_fusion_update(tmp_path):
    model = _model(); common.freeze_native(model)
    frozen = common.frozen_state_digests(model)
    optimizer = common.build_optimizer(model)
    scheduler = common.build_scheduler(optimizer, warmup_updates=3)
    counters = TrainingCounters()
    config = {"lr": 1e-4, "trainable_layout": common.trainable_layout(model), "runtime": {"torch": torch.__version__}}
    source = {"checkpoint_sha256": "1"*64, "runtime": {"torch": torch.__version__}}
    def step(m, o, s, c):
        return common.observed_step(m, o, [_batch()], scheduler=s, counters=c, backbone_kwargs={"mode": common.make_mode(1)})
    step(model, optimizer, scheduler, counters)
    step(model, optimizer, scheduler, counters)
    path = tmp_path/"fusion.pt"
    receipt = common.save_checkpoint(path, model, optimizer, scheduler=scheduler, counters=counters,
                configuration=config, source_fingerprint=source, data_cursor={"next_update": 2})
    # Future O5c files load safely without the legacy TorchVersion allowlist.
    payload = torch.load(path, weights_only=True)
    assert type(payload["source_fingerprint"]["runtime"]["torch"]) is str
    expected_result = step(model, optimizer, scheduler, counters)
    expected = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), asdict(counters)))
    expected_rng = torch.rand(5)
    resumed = _model(); common.freeze_native(resumed)
    resumed_optimizer = common.build_optimizer(resumed)
    resumed_scheduler = common.build_scheduler(resumed_optimizer, warmup_updates=3)
    restored = common.load_checkpoint(path, resumed, resumed_optimizer, scheduler=resumed_scheduler,
                configuration=config, source_fingerprint=source, expected_sha256=receipt["sha256"])
    assert restored["data_cursor"] == {"next_update": 2}
    actual_result = step(resumed, resumed_optimizer, resumed_scheduler, restored["counters"])
    assert actual_result == expected_result
    _equal((resumed.state_dict(), resumed_optimizer.state_dict(), resumed_scheduler.state_dict(), asdict(restored["counters"])), expected)
    assert torch.equal(torch.rand(5), expected_rng)
    assert common.frozen_state_digests(resumed) == frozen


def test_assertion_rejects_accidental_native_unfreezing_before_optimizer_creation():
    model = _model(); common.freeze_native(model)
    model.backbone.readout_weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Exactly"):
        common.build_optimizer(model)


def test_source_inventory_covers_new_data_recipe_and_inherited_frozen_math(tmp_path, monkeypatch):
    assert set(prior.SOURCE_FILES) <= set(common.SOURCE_FILES)
    required = {"scripts/olmo_o5c_common.py", "scripts/olmo_o5c_train.py", "scripts/olmo_o5c_data.py",
                "scripts/olmo_o5c_preflight.py", "docs/reports/olmo1b-o5c/protocol.md"}
    assert required <= set(common.SOURCE_FILES)
    for name in common.SOURCE_FILES:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    assert set(common.source_hashes()) == set(common.SOURCE_FILES)


def test_own_prefix_retention_is_create_only_and_rejects_conflicting_remote_bytes(tmp_path, monkeypatch):
    from google.cloud import storage
    class Blob:
        present = False
        generation = "22"
        metadata = None
        uploads = 0
        def exists(self): return self.present
        def upload_from_filename(self, path, **kwargs):
            assert kwargs["if_generation_match"] == 0 and kwargs["checksum"] == "md5"
            data = Path(path).read_bytes(); self.size = len(data)
            self.md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode()
            self.present = True; self.uploads += 1
        def reload(self): pass
    blob = Blob()
    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=lambda name: SimpleNamespace(blob=lambda key: blob)))
    path = tmp_path/"checkpoint.pt"; path.write_bytes(b"fusion checkpoint fixture")
    uri = common.PREFIX_ROOT+"fixture/code/checkpoint.pt"
    first = common.retain_file(path, uri, expected_sha256=sha256_file(path))
    assert common.retain_file(path, uri) == first and blob.uploads == 1
    blob.md5_hash = "wrong"
    with pytest.raises(ValueError, match="remote object differs"): common.retain_file(path, uri)
    assert blob.uploads == 1
    with pytest.raises(ValueError, match="designated"): common.retain_file(path, "gs://fast-chunks/other/checkpoint.pt")
    with pytest.raises(ValueError, match="changed"): common.retain_file(path, uri, expected_sha256="0"*64)
