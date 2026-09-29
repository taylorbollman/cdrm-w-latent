"""CPU oracles for fusion-only CE, compact state authority and exact recovery."""
import copy
from contextlib import contextmanager
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import TrainingCounters, _rng_state
from scripts import olmo_fusion_startup_train as train
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_precision_components import component_backward
from scripts.olmo_campaign_recurrence_precision import arm_contract
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def bounded_cpu(monkeypatch):
    torch.set_num_threads(1)
    # The same production counter/normalization code with a small exact target
    # budget; no production CLI or checkpoint uses this CPU-only test override.
    monkeypatch.setitem(train.TRAINING, "ce_targets_per_update", 25)


def tiny():
    model, recipe, source, ids, eos = construct(SimpleNamespace(scale="tiny", length=16), "NF", torch.device("cpu"))
    source = {**source, "sha256": "a"*64}
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
        length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
    batches = tuple(batch for bs, _ in fixtures for batch in bs)
    noise = tuple(n for _, ns in fixtures for n in ns)
    return model, recipe, source, fixtures, batches, noise


def prepared():
    model, recipe, source, fixtures, batches, noise = tiny()
    contract = arm_contract(model, recipe)
    execution = train.configure_full_fp32(model)
    train.freeze_for_startup(model)
    configuration = train.checkpoint_configuration(model, recipe, source,
        data_manifest={"test": "fixed tokens and masks"}, data_manifest_sha256="b"*64,
        sources={"source.py": "c"*64}, determinism={"test": True}, runtime={"cpu_test": True},
        execution=execution, initial_contract=contract)
    fingerprint = {"checkpoint_sha256": source["sha256"], "code": configuration["sources"],
                   "data_manifest_sha256": configuration["data_manifest_sha256"]}
    optimizer, scheduler = train.build_optimizer(model)
    return model, recipe, source, fixtures, batches, noise, configuration, fingerprint, optimizer, scheduler


def cursor(update):
    return {"schema": "olmo-fusion-startup-data-v1", "manifest_sha256": "b"*64, "next_update": update}


def test_ce_only_matches_full_zero_aux_gradient_and_preserves_feedback_autograd(monkeypatch):
    model, recipe, _, fixtures, batches, noise = tiny()
    train.freeze_for_startup(model)
    frozen, rng = train.frozen_state_pins(model), tree_digests(_rng_state(None))
    reference = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
    observed = []
    handle = model.backbone.register_forward_hook(lambda module, args, kwargs, output:
        observed.append(tuple(h.requires_grad for h in output.pass_hidden_states)), with_kwargs=True)
    # Omitted auxiliary arithmetic must not accidentally call the predictor.
    monkeypatch.setattr(model.predictor, "forward", lambda *a: (_ for _ in ()).throw(AssertionError("predictor executed")))
    try:
        result = train.ce_backward(model, recipe, batches, noise, ce_targets=25)
    finally:
        handle.remove()
    assert result["objective"] == reference["objective"]
    assert result["ce_sum"] == reference["loss_sums"]["ce"]
    assert observed == [(False, True, True, True)]*2
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=0, atol=0)
        else:
            assert not parameter.requires_grad and parameter.grad is None
    assert train.frozen_state_pins(model) == frozen and tree_digests(_rng_state(None)) == rng
    with pytest.raises(ValueError, match="denominator"):
        train.ce_backward(model, recipe, batches, noise, ce_targets=24)


def test_backward_and_checkpoint_replay_stay_inside_forced_math_and_disabled_autocast(monkeypatch):
    model, recipe, _, _, batches, noise = tiny()
    train.freeze_for_startup(model)
    # Test the runtime context at real leaf-gradient callbacks, not only at
    # forward dispatch; CPU defaults previously hid the CUDA replay mismatch.
    original = train.sdpa_kernel
    active, seen = [], []

    @contextmanager
    def observed_backend(backend):
        assert backend == train.SDPBackend.MATH
        with original(backend):
            active.append(True)
            try:
                yield
            finally:
                active.pop()

    def observe_gradient(gradient):
        assert active == [True]
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.is_autocast_enabled("cpu")
        seen.append(True)

    monkeypatch.setattr(train, "sdpa_kernel", observed_backend)
    handles = [p.register_hook(observe_gradient) for p in train.assert_fusion_only(model)]
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            train.ce_backward(model, recipe, batches, noise, ce_targets=25)
            assert torch.is_autocast_enabled("cpu")
    finally:
        for handle in handles:
            handle.remove()
    assert len(seen) == 4 and not active


def test_adam_warmup_and_compact_checkpoint_exact_next_update(tmp_path):
    model, recipe, source, _, batches, noise, config, fingerprint, optimizer, scheduler = prepared()
    counters = TrainingCounters()
    old_fusion = tree_digests(model.backbone.fusion.state_dict())
    first = train.train_update(model, recipe, batches, noise, optimizer, scheduler, counters, ce_targets=25)
    assert first["lr_used"] == [1e-4/16]
    assert first["lr_next"] == [2e-4/16]
    assert first["gradient_norm_before_clip"] > 0
    assert all(value > 0 for value in first["fusion_update_norms"].values())
    assert tree_digests(model.backbone.fusion.state_dict()) != old_fusion
    assert train.frozen_state_pins(model) == config["frozen_state_pins"]
    path = tmp_path/"update-000001.pt"
    record = train.save_fusion_checkpoint(path, model, optimizer, scheduler, counters,
        configuration=config, source_fingerprint=fingerprint, data_cursor=cursor(1))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload["model"]) == {"state_proj.weight", "token_gate.weight", "output_scale"}
    assert payload["optimizer_ownership"] == [["state_proj.weight", "token_gate.weight"]]
    assert sum(t.numel() for t in payload["model"].values()) < sum(p.numel() for p in model.parameters())/5
    assert all(p.grad is None for p in model.parameters())
    second = train.train_update(model, recipe, batches, noise, optimizer, scheduler, counters, ce_targets=25)
    expected = train.boundary_digests(model, optimizer, scheduler, counters, cursor(2))
    # Reconstructing the full tiny model and optimizer exercises fresh ownership,
    # including restoration after caller RNG consumption, not an in-place reset.
    other, recipe2, source2, _, batches2, noise2, config2, fingerprint2, optimizer2, scheduler2 = prepared()
    torch.rand(7)
    restored = train.load_fusion_checkpoint(other, path, source2, expected_sha256=record["sha256"],
        configuration=config2, source_fingerprint=fingerprint2, optimizer=optimizer2, scheduler=scheduler2)
    counters2 = TrainingCounters(**restored["counters"])
    assert restored["data_cursor"] == cursor(1) and all(restored["checks"].values())
    replay = train.train_update(other, recipe2, batches2, noise2, optimizer2, scheduler2, counters2, ce_targets=25)
    assert replay == second
    assert train.boundary_digests(other, optimizer2, scheduler2, counters2, cursor(2)) == expected
    assert train.frozen_state_pins(other) == config["frozen_state_pins"]
    assert counters2.optimizer_updates == 2 and counters2.ce_positions == 50
    assert counters2.latent_pairs == counters2.kl_triples == 0
    with pytest.raises(FileExistsError):
        train.save_fusion_checkpoint(path, other, optimizer2, scheduler2, counters2,
            configuration=config2, source_fingerprint=fingerprint2, data_cursor=cursor(2))


def test_weights_only_probe_import_preserves_full_trainability_modes_rng_and_identity(tmp_path):
    model, recipe, source, _, batches, noise, config, fingerprint, optimizer, scheduler = prepared()
    counters = TrainingCounters()
    train.train_update(model, recipe, batches, noise, optimizer, scheduler, counters, ce_targets=25)
    path = tmp_path/"fusion.pt"
    record = train.save_fusion_checkpoint(path, model, optimizer, scheduler, counters,
        configuration=config, source_fingerprint=fingerprint, data_cursor=cursor(1))
    diagnostic, _, source2, _, _, _ = tiny()
    diagnostic.backbone.fusion.token_gate.eval()  # Deliberately different caller mode.
    flags = {name: p.requires_grad for name, p in diagnostic.named_parameters()}
    modes = {name: m.training for name, m in diagnostic.named_modules()}
    ids = {name: id(p) for name, p in diagnostic.named_parameters()}
    rng = tree_digests(_rng_state(None))
    restored = train.load_fusion_checkpoint(diagnostic, path, source2, expected_sha256=record["sha256"])
    assert all(restored["checks"].values())
    assert tree_digests(diagnostic.backbone.fusion.state_dict()) == tree_digests(model.backbone.fusion.state_dict())
    assert flags == {name: p.requires_grad for name, p in diagnostic.named_parameters()}
    assert all(flags.values()) and modes == {name: m.training for name, m in diagnostic.named_modules()}
    assert ids == {name: id(p) for name, p in diagnostic.named_parameters()}
    assert rng == tree_digests(_rng_state(None))
    assert all(p.grad is None for p in diagnostic.parameters())
    json.dumps(restored)


@pytest.mark.parametrize("corruption", ["hash", "source", "missing_scale", "extra_tensor", "dtype", "shape",
    "nonfinite", "scale", "frozen", "cursor", "ce_count", "negative_counter", "recipe", "recipe_hash",
    "source_fingerprint", "ownership", "source_config", "schedule"])
def test_compact_authority_rejects_corruption_before_model_mutation(tmp_path, corruption):
    model, _, source, _, _, _, config, fingerprint, optimizer, scheduler = prepared()
    path = tmp_path/"initial.pt"
    train.save_fusion_checkpoint(path, model, optimizer, scheduler, TrainingCounters(),
        configuration=config, source_fingerprint=fingerprint, data_cursor=cursor(0))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    source = copy.deepcopy(source)
    if corruption == "source":
        source["sha256"] = "d"*64
    elif corruption == "missing_scale":
        del payload["model"]["output_scale"]
    elif corruption == "extra_tensor":
        payload["model"]["native.wte"] = torch.tensor(1.)
    elif corruption == "dtype":
        payload["model"]["state_proj.weight"] = payload["model"]["state_proj.weight"].bfloat16()
    elif corruption == "shape":
        payload["model"]["state_proj.weight"] = payload["model"]["state_proj.weight"][:1]
    elif corruption == "nonfinite":
        payload["model"]["token_gate.weight"][0, 0] = float("nan")
    elif corruption == "scale":
        payload["model"]["output_scale"] += .1
    elif corruption == "frozen":
        payload["configuration"]["frozen_state_pins"] = {}
    elif corruption == "cursor":
        payload["data_cursor"]["next_update"] = 1
    elif corruption == "ce_count":
        payload["counters"]["ce_positions"] = 1
    elif corruption == "negative_counter":
        payload["counters"]["input_tokens"] = -1
    elif corruption == "recipe":
        payload["configuration"]["recipe"]["feedback_jitter"] = .03
    elif corruption == "recipe_hash":
        payload["configuration"]["recipe_sha256"] = "incorrect"
    elif corruption == "source_fingerprint":
        payload["source_fingerprint"]["code"] = {}
    elif corruption == "ownership":
        payload["optimizer_ownership"][0].reverse()
    elif corruption == "source_config":
        payload["configuration"]["nextlat_config"]["seed"] += 1
    elif corruption == "schedule":
        payload["configuration"]["training"]["lr"] *= 2
    changed = tmp_path/"changed.pt"
    torch.save(payload, changed)
    digest = "0"*64 if corruption == "hash" else sha256_file(changed)
    before, rng = tree_digests(model.state_dict()), tree_digests(_rng_state(None))
    with pytest.raises(ValueError):
        train.load_fusion_checkpoint(model, changed, source, expected_sha256=digest)
    assert tree_digests(model.state_dict()) == before and tree_digests(_rng_state(None)) == rng


def test_freeze_and_checkpoint_refuse_wrong_ownership_or_uncommitted_cursor(tmp_path):
    model, _, _, _, _, _, config, fingerprint, optimizer, scheduler = prepared()
    with pytest.raises(ValueError, match="cursor"):
        train.save_fusion_checkpoint(tmp_path/"bad.pt", model, optimizer, scheduler, TrainingCounters(),
            configuration=config, source_fingerprint=fingerprint, data_cursor=cursor(1))
    next(model.backbone.backbone.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="two trainable"):
        train.assert_fusion_only(model)
    train.freeze_for_startup(model)
    next(model.predictor.parameters()).grad = torch.zeros_like(next(model.predictor.parameters()))
    with pytest.raises(ValueError, match="Frozen parameters"):
        train.assert_fusion_only(model)
