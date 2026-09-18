"""The variant entry point must leave the frozen trainer and task loss intact."""
from types import SimpleNamespace

import pytest

from scripts import rt_nextlat_a5_fuzzy_train as base
from scripts import rt_nextlat_a5_fuzzy_embedding_train as variant


def test_scoped_factory_restores_baseline_even_after_error():
    original = base.build_model, base.read_configuration, base.source_manifest
    unchanged = base.train_step, base.task_loss, base.save_checkpoint, base.evaluate_a5
    with pytest.raises(RuntimeError, match="fixture"):
        with variant.variant_model_factory():
            assert base.build_model is variant.embeddings.build_model
            assert base.read_configuration is variant.embeddings.read_configuration
            assert base.source_manifest is variant.source_manifest
            assert unchanged == (base.train_step, base.task_loss, base.save_checkpoint, base.evaluate_a5)
            raise RuntimeError("fixture")
    assert original == (base.build_model, base.read_configuration, base.source_manifest)


def test_manifest_preserves_all_frozen_baseline_hashes_and_captures_adapter():
    original = base.source_manifest()
    actual = variant.source_manifest()
    assert all(actual[name] == digest for name, digest in original.items())
    assert set(actual) - set(original) == {
        *variant.embeddings.SOURCE_PATHS, "scripts/rt_nextlat_a5_fuzzy_embedding_train.py",
    } - set(original)


def test_variant_entry_point_requires_explicit_model_route():
    with pytest.raises(ValueError, match="explicit embedding variant"):
        variant.run(SimpleNamespace(config=base.CONFIG_PATH))
