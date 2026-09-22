"""CPU-only checks of paired FBT pilot optimization, accounting and recovery."""

import copy
from dataclasses import asdict

import pytest
import torch

from cdrm.pretrained.lm_schedule import alpha_for_update, build_schedule
from cdrm.pretrained.lm_training import (LMTrainingConfig, TrainingCounters, optimizer_step,
    optimizer_ownership, save_training_checkpoint, load_training_checkpoint)
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_o5b_common as common


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(6052)


def _model():
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math")
    return common.ObservedFBTLM(OLMoFBT(base), NextLatConfig(32, vocab_chunk_size=3), enabled=False, gamma=1)


def _config(warmup=2, fusion_warmup=3):
    return dict(lr=1e-5, fusion_lr=1e-4, weight_decay=.1, betas=[.9, .95], eps=1e-8,
                fusion_warmup_updates=fusion_warmup,
                schedule=build_schedule([6] * 4000, batch_size=4, warmup_updates=warmup,
                                        ramp_min_tokens=72, ramp_min_updates=200 if warmup == 100 else 3))


def _batch():
    ids = torch.tensor([[2, 3, 5, 8, 13, 21], [1, 1, 7, 10, 17, 22],
                        [6, 9, 1, 1, 1, 1], [14, 17, 20, 23, 26, 1]])
    valid = torch.tensor([[True]*6, [False, False]+[True]*4,
                          [True]*2+[False]*4, [True]*5+[False]])
    docs = torch.arange(4)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    ce = valid.clone()
    ce[0, 2] = False
    return NextLatBatch(ids, valid, docs, ce_mask=ce, latent_mask=valid.clone())


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def test_native_and_fusion_groups_have_exact_parameter_ownership_and_distinct_lrs():
    model, config = _model(), _config()
    optimizer = common.build_optimizer(model, config)
    names = optimizer_ownership(model, optimizer)
    assert [group["group_name"] for group in optimizer.param_groups] == ["backbone", "fusion"]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    assert all(group["weight_decay"] == .1 for group in optimizer.param_groups)
    assert len(names[0]) == 9 and len(names[1]) == 2
    assert all(name.startswith("backbone.backbone.") for name in names[0])
    assert set(names[1]) == {"backbone.fusion.state_proj.weight", "backbone.fusion.token_gate.weight"}
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(owned) == len({id(p) for p in owned}) == len(list(model.parameters()))
    assert sum(p is model.backbone.readout_weight for p in owned) == 1
    assert model.predictor is None


def test_beta_zero_control_never_changes_or_initializes_fusion_optimizer_state():
    model = _model()
    optimizer = common.build_optimizer(model, _config())
    before = copy.deepcopy(model.backbone.fusion.state_dict())
    result = common.observed_step(model, optimizer, [_batch()], backbone_kwargs={"mode": common.make_mode(0)})
    assert result["group_gradient_norm_after_clip"]["fusion"] == 0
    _equal(model.backbone.fusion.state_dict(), before)
    assert all(parameter not in optimizer.state for parameter in model.backbone.fusion.parameters())


@pytest.mark.parametrize("invalid", ["frozen_fusion", "scalar_parameter"])
def test_optimizer_rejects_unsupported_parameter_group_shapes(invalid):
    model = _model()
    if invalid == "frozen_fusion":
        model.backbone.fusion.requires_grad_(False)
    else:
        model.register_parameter("unexpected_scalar", torch.nn.Parameter(torch.ones(())))
    with pytest.raises(ValueError):
        common.build_optimizer(model, _config())


def test_native_and_fusion_warmups_align_with_first_nonzero_beta_update102():
    model, config = _model(), _config(warmup=100, fusion_warmup=100)
    optimizer = common.build_optimizer(model, config)
    scheduler = common.build_scheduler(optimizer, config)
    schedule = config["schedule"]
    values = []
    for completed in range(203):
        beta = alpha_for_update(schedule, completed, schedule["batch_token_prefix"][completed])
        values.append((beta, [group["lr"] for group in optimizer.param_groups]))
        optimizer.step()
        scheduler.step()
    assert values[0][1] == pytest.approx([1e-7, 0])
    assert values[99][1] == pytest.approx([1e-5, 0])
    assert values[100] == (0, [1e-5, 0])  # Next update is101.
    assert values[101][0] > 0
    assert values[101][1] == pytest.approx([1e-5, 1e-6])
    assert values[199][1] == pytest.approx([1e-5, 9.9e-5])
    assert values[200][1] == pytest.approx([1e-5, 1e-4])
    assert values[202][1] == pytest.approx([1e-5, 1e-4])


@pytest.mark.parametrize("beta", [0.0, .37, 1.0])
def test_observation_matches_plain_update_and_logs_separate_pass_ce(beta):
    observed = _model()
    plain = copy.deepcopy(observed)
    configuration = _config()
    opts = [common.build_optimizer(m, configuration) for m in (observed, plain)]
    batch, mode = _batch(), common.make_mode(beta)
    with torch.no_grad():
        losses = observed.loss_sums(batch, backbone_kwargs={"mode": mode})
        expected_passes = [float(p.means["ce"]) for p in losses.pass_losses]
    options = dict(config=LMTrainingConfig(), backbone_kwargs={"mode": mode})
    actual = common.observed_step(observed, opts[0], [batch], **options)
    expected = optimizer_step(plain, opts[1], [batch], **options)
    for key in expected:
        assert actual[key] == expected[key], key
    assert actual["pass_ce_means"] == pytest.approx(expected_passes, abs=1e-6)
    assert actual["objective"] == pytest.approx(sum(actual["pass_ce_means"]), abs=1e-6)
    assert actual["stack_input_tokens"] == 2 * int(batch.valid_mask.sum()) == 34
    norms = actual["group_gradient_norm_after_clip"]
    assert sum(v*v for v in norms.values()) <= 1.000002
    assert (norms["fusion"] == 0) is (beta == 0)
    assert observed._observations is None
    assert not opts[0]._optimizer_step_pre_hooks
    _equal(observed.state_dict(), plain.state_dict())
    _equal(opts[0].state_dict(), opts[1].state_dict())


def test_two_microbatches_match_one_effective_update_with_unequal_valid_target_counts():
    whole = _model()
    split = copy.deepcopy(whole)
    config = _config()
    optimizers = [common.build_optimizer(m, config) for m in (whole, split)]
    batch = _batch()
    options = dict(config=LMTrainingConfig(), backbone_kwargs={"mode": common.make_mode(.6)})
    full = common.observed_step(whole, optimizers[0], common.microbatches(batch, 4), **options)
    parts = common.observed_step(split, optimizers[1], common.microbatches(batch, 2), **options)
    assert full["counts"] == parts["counts"] == {"ce": 12, "latent": 0, "kl": 0}
    assert full["counters"]["input_tokens"] == parts["counters"]["input_tokens"] == 17
    assert full["stack_input_tokens"] == parts["stack_input_tokens"] == 34
    assert full["pass_ce_means"] == pytest.approx(parts["pass_ce_means"], abs=2e-6)
    assert full["objective"] == pytest.approx(parts["objective"], abs=2e-6)
    for p, q in zip(whole.parameters(), split.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=3e-5)


def test_observation_hooks_are_removed_when_the_forward_fails(monkeypatch):
    model, config = _model(), _config()
    optimizer = common.build_optimizer(model, config)
    before = copy.deepcopy(model.state_dict())
    def fail(*args, **kwargs):
        raise RuntimeError("fixture forward failure")
    monkeypatch.setattr(model.backbone, "forward", fail)
    with pytest.raises(RuntimeError, match="fixture"):
        common.observed_step(model, optimizer, [_batch()], backbone_kwargs={"mode": common.make_mode(.5)})
    assert model._observations is None
    assert not optimizer._optimizer_step_pre_hooks
    assert all(p.grad is None for p in model.parameters())
    _equal(model.state_dict(), before)
    assert not optimizer.state


def test_zero_lr_profile_initializes_both_group_moments_without_changing_model_bytes():
    model, config = _model(), _config()
    optimizer = common.build_optimizer(model, config, zero_lr=True)
    before = copy.deepcopy(model.state_dict())
    for _ in range(2):
        result = common.observed_step(model, optimizer, [_batch()], backbone_kwargs={"mode": common.make_mode(1)})
        assert result["lr_used"] == [0, 0]
    _equal(model.state_dict(), before)
    assert common.finite_optimizer(model, optimizer)
    assert len(optimizer.state) == len(list(model.parameters()))


def test_two_group_checkpoint_resume_recovers_exact_update_and_future_fusion_warmup(tmp_path):
    model, config = _model(), _config()
    optimizer = common.build_optimizer(model, config)
    scheduler = common.build_scheduler(optimizer, config)
    counters = TrainingCounters()
    batch = _batch()
    fingerprint = {"checkpoint_sha256": "6"*64, "code": "O5b tiny CPU boundary"}
    def step(m, o, s, c):
        # The first two updates have no feedback; then mirror the schedule's
        # extra beta-zero transition update before fusion becomes active.
        beta = 0.0 if c.optimizer_updates <= 2 else .4
        return common.observed_step(m, o, [batch], scheduler=s, counters=c,
                                    backbone_kwargs={"mode": common.make_mode(beta)})
    for _ in range(4):
        step(model, optimizer, scheduler, counters)
    path = tmp_path / "boundary.pt"
    saved = save_training_checkpoint(path, model, optimizer, scheduler=scheduler, counters=counters,
             data_cursor={"next_window": 16, "bad_evals": 1}, configuration=config, source_fingerprint=fingerprint)
    expected_steps = [step(model, optimizer, scheduler, counters) for _ in range(4)]
    expected = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), asdict(counters)))
    expected_rng = torch.rand(7)
    resumed = _model()
    resumed_optimizer = common.build_optimizer(resumed, config)
    resumed_scheduler = common.build_scheduler(resumed_optimizer, config)
    restored = load_training_checkpoint(path, resumed, resumed_optimizer, scheduler=resumed_scheduler,
                configuration=config, source_fingerprint=fingerprint, expected_sha256=saved["sha256"])
    assert restored["data_cursor"] == {"next_window": 16, "bad_evals": 1}
    actual_steps = [step(resumed, resumed_optimizer, resumed_scheduler, restored["counters"]) for _ in range(4)]
    assert actual_steps == expected_steps
    _equal((resumed.state_dict(), resumed_optimizer.state_dict(), resumed_scheduler.state_dict(), asdict(restored["counters"])), expected)
    assert torch.equal(torch.rand(7), expected_rng)
    assert actual_steps[0]["lr_used"] == pytest.approx([1e-5, 2e-4/3])
    assert actual_steps[-1]["lr_used"] == pytest.approx([1e-5, 1e-4])


@pytest.mark.parametrize("change", ["fusion_lr", "fusion_warmup_updates", "weight_decay", "source"])
def test_resume_rejects_changed_optimizer_recipe_before_model_mutation(tmp_path, change):
    model, config = _model(), _config()
    optimizer = common.build_optimizer(model, config)
    scheduler = common.build_scheduler(optimizer, config)
    fingerprint = {"checkpoint_sha256": "7"*64}
    path = tmp_path / "boundary.pt"
    save_training_checkpoint(path, model, optimizer, scheduler=scheduler, counters=TrainingCounters(),
              data_cursor={}, configuration=config, source_fingerprint=fingerprint)
    altered = copy.deepcopy(config)
    if change == "source":
        fingerprint = {"checkpoint_sha256": "8"*64}
    else:
        altered[change] *= 2
    resumed = _model()
    before = copy.deepcopy(resumed.state_dict())
    opt = common.build_optimizer(resumed, altered)
    sched = common.build_scheduler(opt, altered)
    with pytest.raises(ValueError, match="differs"):
        load_training_checkpoint(path, resumed, opt, scheduler=sched, configuration=altered, source_fingerprint=fingerprint)
    _equal(resumed.state_dict(), before)
    assert not opt.state


@pytest.mark.parametrize("code,retention,expected", [
    ([4, 2], [5, 3], True), ([2, 4], [3, 5], True),
    ([4, 2], [3, 5], False), ([2, 4], [5, 3], False),
    ([3.5, 2], [4.5, 3], False), ([3.5001, 2], [4.5001, 3], True),
])
def test_catastrophic_gate_requires_both_domains_to_fail_for_the_same_pass(code, retention, expected):
    def record(values):
        return {"passes": [{"mean_nll": value} for value in values]}
    initial = {"dev": record([2, 2]), "retention_dev": record([3, 3])}
    current = {"dev": record(code), "retention_dev": record(retention)}
    assert common.catastrophic(current, initial, 1.5) is expected


def test_batch_slicing_preserves_all_masks_ids_and_exact_prefix_for_paired_online():
    batch = _batch()
    clipped = common.slice_batch(batch, slice(1, 3), length=3)
    for name, tensor in batch.__dict__.items():
        actual = getattr(clipped, name)
        if tensor is None:
            assert actual is None
        else:
            assert torch.equal(actual, tensor[1:3, :3])
    parts = common.microbatches(batch, 2)
    for name, tensor in batch.__dict__.items():
        if tensor is not None:
            assert torch.equal(torch.cat([getattr(p, name) for p in parts]), tensor)
    for value in (0, -1, 3, True):
        with pytest.raises(ValueError):
            common.microbatches(batch, value)


def test_frozen_source_inventory_covers_both_schedules_data_objectives_and_evaluation(tmp_path, monkeypatch):
    required = {"cdrm/pretrained/olmo_fbt.py", "cdrm/pretrained/fbt_training.py",
                "cdrm/pretrained/fbt_evaluation.py", "cdrm/pretrained/lm_evaluation.py",
                "cdrm/pretrained/lm_schedule.py", "cdrm/pretrained/lm_data.py",
                "scripts/olmo_o5b_common.py", "scripts/olmo_o5b_preflight.py",
                "scripts/olmo_o5b_train.py", "docs/reports/olmo1b-o5b/protocol.md"}
    assert required <= set(common.SOURCE_FILES)
    assert len(common.SOURCE_FILES) == len(set(common.SOURCE_FILES))
    for name in common.SOURCE_FILES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source fixture " + name)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    before = common.source_hashes()
    path = tmp_path/"cdrm/pretrained/fbt_evaluation.py"
    path.write_text("changed evaluation")
    after = common.source_hashes()
    assert [name for name in before if before[name] != after[name]] == ["cdrm/pretrained/fbt_evaluation.py"]


def test_resume_record_requires_latest_boundary_and_allows_moved_identical_filename(tmp_path):
    earlier = {"path": str(tmp_path/"update-000100.pt"), "sha256": "1"*64}
    latest = {"path": str(tmp_path/"update-000200.pt"), "sha256": "2"*64}
    report = {"checkpoints": [earlier, latest]}
    assert common.resume_record(report, tmp_path/"update-000200.pt") is latest
    assert common.resume_record(report, tmp_path/"copied"/"update-000200.pt") is latest
    for path in (tmp_path/"update-000100.pt", tmp_path/"unrecorded.pt"):
        with pytest.raises(ValueError, match="latest"):
            common.resume_record(report, path)
    with pytest.raises(ValueError, match="latest"):
        common.resume_record({"checkpoints": []}, tmp_path/"update-000200.pt")


def _completed():
    prefix = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5b-code-pilot/fixture"
    runtime = {"device": "intentional mocked fixture"}
    config = {"source_hashes": {"file.py": "3"*64}, "checkpoint_sha256": "4"*64,
              "data_manifest_sha256": "5"*64,
              "schedule": {"total_updates": 7, "total_tokens": 190, "used_windows": 28}}
    checkpoint = {"path": "/fixture/update-000007.pt", "optimizer_updates": 7,
                  "sha256": "6"*64, "size_bytes": 1024,
                  "storage": {"uri": prefix+"/ordinary/update-000007.pt", "sha256": "6"*64,
                              "size_bytes": 1024, "generation": "13", "md5_base64": "md5-fixture",
                              "verification": "server verified fixture"}}
    report = {"status": "completed", "arm": "ordinary", "finished_utc": "2026-09-22T00:00:00Z",
              "configuration": copy.deepcopy(config), "storage_prefix": prefix,
              "source_fingerprint": {"code": copy.deepcopy(config["source_hashes"]), "runtime": runtime,
                    "checkpoint_sha256": config["checkpoint_sha256"], "data_manifest_sha256": config["data_manifest_sha256"]},
              "counters": {"optimizer_updates": 7, "input_tokens": 190}, "data_cursor": 28,
              "checkpoints": [checkpoint]}
    return report, config, prefix, runtime


def test_completed_queue_arm_requires_matching_lineage_and_verified_final_checkpoint():
    report, config, prefix, runtime = _completed()
    common.validate_completed_arm(report, config, "ordinary", prefix, runtime)


@pytest.mark.parametrize("change", ["arm", "configuration", "source", "runtime", "counters", "cursor",
                                   "missing_final", "duplicate_final", "missing_receipt", "wrong_digest",
                                   "wrong_uri", "missing_digest", "missing_size"])
def test_completed_queue_arm_rejects_incomplete_or_foreign_results(change):
    report, config, prefix, runtime = _completed()
    if change == "arm":
        report["arm"] = "fbt"
    elif change == "configuration":
        report["configuration"]["new_unrecorded_option"] = True
    elif change == "source":
        report["source_fingerprint"]["code"]["file.py"] = "8"*64
    elif change == "runtime":
        report["source_fingerprint"]["runtime"] = {"device": "different"}
    elif change == "counters":
        report["counters"]["optimizer_updates"] -= 1
    elif change == "cursor":
        report["data_cursor"] -= 4
    elif change == "missing_final":
        report["checkpoints"] = []
    elif change == "duplicate_final":
        report["checkpoints"].append(copy.deepcopy(report["checkpoints"][0]))
    elif change == "missing_receipt":
        del report["checkpoints"][0]["storage"]
    elif change == "wrong_digest":
        report["checkpoints"][0]["storage"]["sha256"] = "0"*64
    elif change == "wrong_uri":
        report["checkpoints"][0]["storage"]["uri"] = prefix+"/fbt/update-000007.pt"
    elif change == "missing_digest":
        del report["checkpoints"][0]["sha256"]
        del report["checkpoints"][0]["storage"]["sha256"]
    elif change == "missing_size":
        del report["checkpoints"][0]["size_bytes"]
        del report["checkpoints"][0]["storage"]["size_bytes"]
    with pytest.raises(ValueError, match="Completed arm"):
        common.validate_completed_arm(report, config, "ordinary", prefix, runtime)
