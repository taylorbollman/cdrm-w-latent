"""The depth entry point preserves the existing scheduled training math."""
from types import SimpleNamespace

import pytest

from scripts import rt_nextlat_a5_fuzzy_lr_train as base
from scripts import rt_nextlat_a5_fuzzy_depth_train as variant


def test_scoped_factory_restores_after_exception_and_nested_scope():
    original = base.build_model, base.read_configuration, base.source_manifest
    unchanged = (base.train_step, base.task_loss, base.save_checkpoint, base.load_checkpoint,
                 base.evaluate_a5, base.learning_rate, base.schedule_configuration)
    with pytest.raises(RuntimeError, match="fixture"):
        with variant.depth_model_factory():
            with variant.depth_model_factory():
                assert base.build_model is variant.depth.build_model
                assert base.read_configuration is variant.depth.read_configuration
                assert base.source_manifest is variant.source_manifest
            assert base.build_model is variant.depth.build_model
            assert unchanged == (base.train_step, base.task_loss, base.save_checkpoint,
                                 base.load_checkpoint, base.evaluate_a5, base.learning_rate,
                                 base.schedule_configuration)
            raise RuntimeError("fixture")
    assert original == (base.build_model, base.read_configuration, base.source_manifest)


def test_manifest_preserves_every_old_hash_and_adds_exact_new_sources():
    original = base.source_manifest()
    actual = variant.source_manifest()
    assert all(actual[name] == digest for name, digest in original.items())
    assert set(actual) - set(original) == {
        *variant.depth.SOURCE_PATHS, "scripts/rt_nextlat_a5_fuzzy_depth_train.py",
    }
    with variant.depth_model_factory():
        assert base.source_manifest() == actual


def test_entry_point_rejects_two_layer_configuration():
    with pytest.raises(ValueError, match="explicit three-layer"):
        variant.run(SimpleNamespace(config=base.CONFIG_PATH))


def test_entry_point_forwards_to_existing_loop_and_restores(monkeypatch):
    args = SimpleNamespace(config=variant.depth.CONFIG_PATH)
    original = base.build_model, base.read_configuration, base.source_manifest
    sentinel = object()
    def fake_run(actual):
        assert actual is args
        assert base.build_model is variant.depth.build_model
        assert base.read_configuration is variant.depth.read_configuration
        return sentinel
    monkeypatch.setattr(base, "run", fake_run)
    assert variant.run(args) is sentinel
    assert original == (base.build_model, base.read_configuration, base.source_manifest)


def test_parser_has_explicit_three_layer_default_and_retains_schedule():
    args = variant.parser().parse_args([
        "--mode", "mixed", "--a5-data", "/tmp/unused-data",
        "--output", "/tmp/unused-depth-check",
    ])
    assert args.config == variant.depth.CONFIG_PATH
    assert args.learning_rate == 3e-4
    assert args.warmup_start_lr == 1e-4
    assert args.warmup_updates == 100
