#!/usr/bin/env python3
"""Run embedding variants through the unchanged mixed A5/Fuzzy training loop.

Only configuration/model construction and the executed source manifest differ.
The adapter is scoped to this process; the frozen baseline trainer stays intact.
"""
from __future__ import annotations

from contextlib import contextmanager

from cdrm import rt_nextlat_task_embeddings as embeddings
from scripts import rt_nextlat_a5_fuzzy_train as base


_BASE_SOURCE_MANIFEST = base.source_manifest


def source_manifest():
    sources = _BASE_SOURCE_MANIFEST()
    sources.update(embeddings.source_manifest())
    relative = "scripts/rt_nextlat_a5_fuzzy_embedding_train.py"
    sources[relative] = base.file_sha256(base.ROOT / relative)
    return dict(sorted(sources.items()))


@contextmanager
def variant_model_factory():
    """Preserve all objective, batching, checkpoint and evaluation code."""
    original = base.build_model, base.read_configuration, base.source_manifest
    base.build_model = embeddings.build_model
    base.read_configuration = embeddings.read_configuration
    base.source_manifest = source_manifest
    try:
        yield
    finally:
        base.build_model, base.read_configuration, base.source_manifest = original


def run(args):
    config = embeddings.read_configuration(args.config)
    if "embedding_injection" not in config:
        raise ValueError("This entry point requires an explicit embedding variant")
    with variant_model_factory():
        return base.run(args)


if __name__ == "__main__":
    parser = base.parser()
    parser.description = __doc__
    run(parser.parse_args())
