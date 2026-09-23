"""CPU checks for integration experiment bounds, ownership and matrix accounting."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.resource_estimates import LossWork, estimate_training_resources
from cdrm.pretrained.rt_block_resources import estimate_rt_block_resources
from scripts import olmo_rt_author_integration as harness
from scripts import olmo_rt_efficiency as stage_a
from scripts.olmo_ordinary_throughput import make_batch


def command(stage="verify", *, case="rt", backend="author", batch=8):
    return ["--stage", stage, "--case", case, "--backend", backend,
            "--batch-size", str(batch), "--output-dir", "/tmp/unused-author-integration"]


def test_cli_fixes_comparison_and_common_capacity_shapes():
    verify = harness.parse_args(command())
    assert verify.stage == "verify" and verify.backend == "author" and verify.batch_size == 8
    for case in ("rt", "combined"):
        for backend in ("native", "author"):
            for batch in (32, 64):
                args = harness.parse_args(command("capacity", case=case, backend=backend, batch=batch))
                assert (args.case, args.backend, args.batch_size) == (case, backend, batch)


@pytest.mark.parametrize("arguments", [
    command(backend="native"), command(batch=32), command(case="ordinary"),
    command("capacity"), command("capacity", batch=128), command("profile"),
    command() + ["--author-precision", "fp32_state"],
    command() + ["--length", "32"], command() + ["--native-arm", "control"],
])
def test_cli_rejects_unauthorized_shapes_and_arithmetic_variants(arguments):
    with pytest.raises(SystemExit):
        harness.parse_args(arguments)


def small_model():
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math")
    return FBTNextLatLM(OLMoFBT(base), NextLatConfig(model_dim=32, vocab_chunk_size=128))


def test_backend_switch_reuses_actual_native_tied_parameters_and_only_execution_settings():
    model = small_model()
    initial = harness.parameter_signature(model)
    keys = tuple(model.state_dict())
    original_config = model.config.to_dict()
    for backend in ("author", "native"):
        harness.set_backend(model, backend, compiled_helpers=False)
        base = model.backbone.backbone
        assert base.rt_implementation == backend
        assert base.reuse_rope and base.kv_only_writes
        assert base.author_precision == "author_legacy"
        assert not base.author_compiled_helpers
        assert base.author_bwd_mlp_chunks == 4 and base.author_autocast_cache
        assert harness.parameter_signature(model) == initial
        assert tuple(model.state_dict()) == keys
        assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
        assert model.config.to_dict() == {**original_config, "ce_chunk_size": 2048}


def test_parameter_guard_allows_real_updates_but_detects_parameter_replacement():
    model = small_model()
    initial = harness.parameter_signature(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01)
    assert harness.parameter_signature(model) == initial
    layer = model.backbone.backbone.layers[0]
    layer.ff_out.weight = torch.nn.Parameter(layer.ff_out.weight.detach().clone())
    assert harness.parameter_signature(model) != initial


def test_full_batch_factory_is_used_for_all_six_update_parity_steps(monkeypatch):
    original = make_batch(2, 8, vocab_size=67, supervision="half")
    requested = []

    def fixture(tokenizer, case, update):
        requested.append(update)
        return original

    monkeypatch.setattr(stage_a, "changed_batch", fixture)
    model = torch.nn.Module()
    model.ff_out = torch.nn.Linear(2, 2, bias=False)
    observed = []

    class Plan:
        def __init__(self):
            self.model = model

        def optimizer_step(self, optimizer, batch, *, replay, scheduler, counters):
            assert batch.ce_mask.all()
            for name in ("input_ids", "valid_mask", "document_ids", "latent_mask", "kl_mask"):
                assert torch.equal(getattr(batch, name), getattr(original, name))
            observed.append(replay)
            optimizer.zero_grad(set_to_none=True)
            objective = self.model.ff_out.weight.square().sum() * batch.ce_mask.sum()
            objective.backward()
            optimizer.step()
            scheduler.step()
            counters.optimizer_updates += 1
            return {"objective": float(objective.detach()), "updates": counters.optimizer_updates}

    report = {"physical_optimizer_updates": 0}
    result = harness.complete_update_parity(Plan(), None, SimpleNamespace(), report=report,
        persist=lambda: None, batch_factory=harness.full_batch)
    assert result["passed"] and report["physical_optimizer_updates"] == 6
    assert requested == [5, 6, 7, 5, 6, 7]
    assert observed == [False, False, False, True, True, True]


@pytest.mark.parametrize("passes", [1, 2, 3])
def test_author_ledger_replaces_rt_work_only_and_counts_reused_calls_not_parameters(passes):
    config = OLMoConfig.tiny()
    mode = FBTMode(enabled=passes > 1, num_passes=passes, rt_mode=RTMode((0, 1)))
    nextlat = NextLatConfig(model_dim=config.model_dim) if passes > 1 else None
    work = LossWork.full_document(2, 8, nextlat)
    kwargs = dict(batch_size=2, length=8, mode=mode, nextlat=nextlat, work=work)
    native = harness.integration_estimate(config, backend="native", **kwargs)
    author = harness.integration_estimate(config, backend="author", **kwargs)
    independent_native = estimate_training_resources(config, batch_size=2, sequence_length=8,
        mode=mode, nextlat=nextlat, loss_work=work, ordinary_checkpointing=True,
        backward_memory="recompute", kv_only_writes=True).to_dict()
    assert native == {**independent_native, "rt_implementation": "native", "rt_backward_memory": "recompute"}
    assert author["parameter_counts"] == native["parameter_counts"]
    for key in ("input_tokens_per_update", "pass_token_work_per_update", "objective_positions_per_update",
                "objective_positions_across_passes", "ordinary_block_calls_per_microbatch"):
        assert author[key] == native[key]
    native_nonrt = [row for row in native["components"] if not row["name"].startswith("rt_")]
    author_nonrt = [row for row in author["components"] if not row["name"].startswith("author_rt_")]
    assert author_nonrt == native_nonrt
    native_rt = [row for row in native["components"] if row["name"].startswith("rt_")]
    replacement = author["rt_component_replacement"]
    assert replacement["removed_native_components"] == native_rt
    expected_calls = 2 * max(passes - 1, 1)
    assert replacement["calls"] == author["rt_block_calls_per_microbatch"] == expected_calls
    per_call = estimate_rt_block_resources(replace(config, num_layers=1), batch_size=2,
                                           sequence_length=8, backend="author")
    added = [row for row in author["components"] if row["name"].startswith("author_rt_")]
    assert len(added) == 4
    assert sum(row["minimum"] for row in added) == expected_calls * per_call["total_matrix_flops"]
    assert all(row["minimum"] == row["maximum"] for row in added)
    for limit in ("minimum", "maximum"):
        assert author[f"matrix_flops_{limit}"] == (native[f"matrix_flops_{limit}"]
            - sum(row[limit] for row in native_rt) + expected_calls * per_call["total_matrix_flops"])
    assert author["rt_backward_memory"] == "materialized"


def test_invalid_backend_fails_before_changing_model_or_ledger():
    model = small_model()
    initial = harness.parameter_signature(model)
    with pytest.raises(ValueError, match="implementation"):
        harness.set_backend(model, "unknown")
    assert harness.parameter_signature(model) == initial
    with pytest.raises(ValueError, match="ledger"):
        harness.integration_estimate(OLMoConfig.tiny(), batch_size=2, length=8,
            mode=FBTMode(enabled=False), nextlat=None, work=LossWork(14), backend="unknown")


def test_source_snapshot_contains_integration_dependencies_and_new_selector_sources():
    required = {"scripts/olmo_rt_author_integration.py", "scripts/olmo_rt_author_compare.py",
                "cdrm/pretrained/olmo_author.py", "cdrm/pretrained/olmo_tiled.py",
                "cdrm/pretrained/olmo_static.py", "cdrm/pretrained/static_training.py",
                "cdrm/pretrained/rt_block_resources.py", "cdrm/pretrained/resource_estimates.py"}
    assert required <= set(harness.SOURCES)
    assert all((harness.ROOT / path).is_file() for path in harness.SOURCES)
