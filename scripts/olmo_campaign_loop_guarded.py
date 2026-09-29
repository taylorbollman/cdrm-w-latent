#!/usr/bin/env python3
"""Additive RNG and failed-graph lifetime guards for the tiny lifecycle runner.

The original acceptance implementation remains frozen evidence. This driver
uses the same CLI, data, updates, graph and checkpoint implementation, but pins
its own sources and isolates host retention randomness from training state.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import gc
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import sha256_file
from scripts import olmo_campaign_loop_run as legacy
from scripts.olmo_campaign_loop import LifecycleError

VERSION = "olmo-campaign-loop-guarded-v1"
PROTOCOL = ROOT / "docs/reports/olmo-campaign-lifecycle/guarded-loop-protocol.md"
_BASE_SOURCES = legacy.source_hashes
_BASE_RETAIN = legacy.retain_checkpoint
_BASE_STAGE = legacy.run_stage


def source_hashes():
    sources = _BASE_SOURCES()
    for path in (Path(__file__), ROOT / "tests/test_campaign_loop_guarded.py", PROTOCOL):
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
    return dict(sorted(sources.items()))


def retain_without_rng(receipt, prefix, *, retain=None, preserve=None):
    """SDK retries/IDs may draw RNG after a checkpoint; isolate that host work."""
    retain = _BASE_RETAIN if retain is None else retain
    preserve = legacy.preserve_local_rng if preserve is None else preserve
    with preserve():
        return retain(receipt, prefix)


def clear_completed_exception_frames(error):
    """Drop graph-owning completed frames, including chained host exceptions.

    Capture a text diagnostic before calling this. An active caller's frame is
    not cleared by traceback.clear_frames; the stage's completed frames are.
    This is deliberately only used for coordinated LifecycleError, never an
    unknown update/CUDA/NCCL failure.
    """
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(value for value in (current.__cause__, current.__context__)
                       if value is not None)
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


def run_stage_releasing_failure(args, coordinator, device, runtime, determinism,
                                report, tracker, *, stage=None):
    report["lifecycle_adapter"] = {
        "version": VERSION,
        "retention_preserves_local_rng": True,
        "coordinated_failure_releases_completed_frames_before_teardown": True,
        "unknown_failure_policy": "unchanged_external_launcher_teardown",
    }
    stage = _BASE_STAGE if stage is None else stage
    failure = None
    try:
        return stage(args, coordinator, device, runtime, determinism, report, tracker)
    except LifecycleError as error:
        # Strings retain diagnosis without retaining graph/runner/model objects.
        failure = {"type": type(error).__name__, "message": str(error),
                   "traceback": "".join(traceback.format_exception(error))}
        report["coordinated_failure_diagnostic"] = failure
        clear_completed_exception_frames(error)
    # Raising outside except avoids attaching the original exception context.
    # Completed closure cycles can also keep a captured graph alive; collect
    # them before the legacy main reaches destroy_process_group.
    gc.collect()
    raise LifecycleError(failure["message"]) from None


class _DistributedProxy:
    """Module-local teardown guard; never changes torch.distributed globally."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def destroy_process_group(self, *args, **kwargs):
        gc.collect()
        return self.original.destroy_process_group(*args, **kwargs)


@contextmanager
def guarded_driver_scope():
    """Install only legacy-module bindings, restoring them even after errors."""
    with ExitStack() as stack:
        for name, value in (
            ("source_hashes", source_hashes),
            ("retain_checkpoint", retain_without_rng),
            ("run_stage", run_stage_releasing_failure),
            ("dist", _DistributedProxy(legacy.dist)),
        ):
            stack.enter_context(patch.object(legacy, name, value))
        yield


def main(argv=None):
    with guarded_driver_scope():
        return legacy.main(argv)


if __name__ == "__main__":
    main()
