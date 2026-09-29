"""Host RNG isolation and captured-owner lifetime regression without CUDA."""
import gc
import random
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from scripts import olmo_campaign_loop_guarded as guarded
from scripts.olmo_campaign_loop import LifecycleError
from scripts.olmo_lm_common import tree_digests


def _rng_state(cuda):
    return tree_digests({"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(), "cuda": cuda[0]})


@pytest.mark.parametrize("fail", [False, True])
def test_retention_preserves_actual_host_rng_and_mocked_device_rng(monkeypatch, fail):
    cuda = [torch.arange(8, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda[0].clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: cuda.__setitem__(0, value.clone()))
    random.seed(57); np.random.seed(58); torch.manual_seed(59)
    before = _rng_state(cuda)
    def retain(receipt, prefix):
        assert receipt == {"update": 1} and prefix == "test"
        random.random(); np.random.random(); torch.rand(3)
        cuda[0].add_(1)
        if fail:
            raise OSError("upload failed")
        return {"generation": 123}
    if fail:
        with pytest.raises(OSError, match="upload failed"):
            guarded.retain_without_rng({"update": 1}, "test", retain=retain)
    else:
        assert guarded.retain_without_rng({"update": 1}, "test", retain=retain) == {"generation": 123}
    after = _rng_state(cuda)
    # tree_digests intentionally leaves NumPy arrays unchanged; compare their
    # explicit contents without relying on ambiguous ndarray dictionary ==.
    assert before["python"] == after["python"]
    assert before["torch"] == after["torch"] and before["cuda"] == after["cuda"]
    assert before["numpy"][0] == after["numpy"][0]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert before["numpy"][2:] == after["numpy"][2:]


def test_coordinated_failure_releases_graph_owners_and_keeps_text_before_destroy():
    references, report, events = [], {}, []
    class GraphOwner:
        pass
    def failing_stage(*args):
        graph = GraphOwner()
        # A realistic runner/closure cycle additionally requires collection.
        graph.cycle = lambda: graph
        references.append(weakref.ref(graph))
        try:
            raise OSError("original rank-zero callback")
        except OSError as cause:
            raise LifecycleError("coordinated logging failure") from cause
    error = None
    try:
        guarded.run_stage_releasing_failure(None, None, None, None, None, report, None,
                                             stage=failing_stage)
    except LifecycleError as caught:
        error = caught  # Mirror legacy main retaining the error until teardown.
    assert error is not None and error.__cause__ is None and error.__context__ is None
    assert references[0]() is None
    diagnostic = report["coordinated_failure_diagnostic"]
    assert diagnostic["message"] == "coordinated logging failure"
    assert "original rank-zero callback" in diagnostic["traceback"]
    assert "failing_stage" in diagnostic["traceback"]
    def destroy(*args, **kwargs):
        assert references[0]() is None
        events.append("destroy")
    guarded._DistributedProxy(SimpleNamespace(destroy_process_group=destroy)).destroy_process_group()
    assert events == ["destroy"]
    assert report["lifecycle_adapter"]["version"] == guarded.VERSION


def test_unknown_update_failure_keeps_original_error_and_external_policy():
    error, report = RuntimeError("unclassified CUDA failure"), {}
    def fail(*args):
        raise error
    with pytest.raises(RuntimeError) as caught:
        guarded.run_stage_releasing_failure(None, None, None, None, None, report, None, stage=fail)
    assert caught.value is error
    assert "coordinated_failure_diagnostic" not in report


def test_success_return_unchanged_and_scope_restores_only_legacy_bindings():
    names = ("source_hashes", "retain_checkpoint", "run_stage", "dist")
    original = {name: getattr(guarded.legacy, name) for name in names}
    global_dist = torch.distributed.destroy_process_group
    with pytest.raises(ValueError):
        with guarded.guarded_driver_scope():
            assert guarded.legacy.source_hashes is guarded.source_hashes
            assert guarded.legacy.retain_checkpoint is guarded.retain_without_rng
            assert guarded.legacy.dist is not torch.distributed
            assert torch.distributed.destroy_process_group is global_dist
            report = {}
            assert guarded.run_stage_releasing_failure(None, None, None, None, None, report,
                                                       None, stage=lambda *args: 42) == 42
            raise ValueError("scope exit")
    assert all(getattr(guarded.legacy, name) is value for name, value in original.items())
    assert torch.distributed.destroy_process_group is global_dist


def test_driver_sources_pin_new_adapter_test_protocol_and_original_sources():
    sources = guarded.source_hashes()
    assert all(sources[name] == digest for name, digest in guarded._BASE_SOURCES().items())
    for name in ("scripts/olmo_campaign_loop_guarded.py", "tests/test_campaign_loop_guarded.py",
                 "docs/reports/olmo-campaign-lifecycle/guarded-loop-protocol.md"):
        assert sources[name] == guarded.sha256_file(guarded.ROOT / name)
