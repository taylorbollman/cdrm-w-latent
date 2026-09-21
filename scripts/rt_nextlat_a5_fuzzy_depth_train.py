#!/usr/bin/env python3
"""Fresh three-layer mixed-task RT using the validated linear-LR training loop.

The model factory and source closure change. Task losses, data order, optimizer
steps, LR schedule, evaluation and checkpoint guards remain unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager

from cdrm import rt_nextlat_task_depth as depth
from scripts import rt_nextlat_a5_fuzzy_lr_train as base


_BASE_SOURCE_MANIFEST = base.source_manifest


def source_manifest():
    sources = _BASE_SOURCE_MANIFEST()
    sources.update(depth.source_manifest())
    relative = "scripts/rt_nextlat_a5_fuzzy_depth_train.py"
    sources[relative] = base.file_sha256(base.ROOT / relative)
    return dict(sorted(sources.items()))


@contextmanager
def depth_model_factory():
    """Scope the construction override and restore it on every exit path."""
    original = base.build_model, base.read_configuration, base.source_manifest
    base.build_model = depth.build_model
    base.read_configuration = depth.read_configuration
    base.source_manifest = source_manifest
    try:
        yield
    finally:
        base.build_model, base.read_configuration, base.source_manifest = original


def run(args):
    depth.read_configuration(args.config)
    with depth_model_factory():
        return base.run(args)


def parser():
    result = base.parser()
    result.description = __doc__
    result.set_defaults(config=depth.CONFIG_PATH)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
