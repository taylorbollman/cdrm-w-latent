"""Bounded crossed-backend contract and prior-anchor checks, CPU only."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.olmo_campaign_backend_cross import (
    CROSS, MATH, TRITON, anchor_comparison, configure_cross_path, load_reference, parse_args,
)


def production():
    return dict(attention_backend="sdpa", ordinary_attention_backend="sdpa", attention_precision="mixed",
                tile_backend="triton", backward_tile_backend="triton", backward_memory="recompute",
                rt_implementation="native", ordinary_pointwise_backend="eager", ordinary_rope_backend="native",
                ordinary_activation_checkpointing=True, cast_weights_once=True, reuse_rope=True, kv_only_writes=True)


def test_cross_is_only_flash_dispatch_change_from_bf16_math_eager():
    flags = production()
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2)))
    model.backbone = SimpleNamespace(backbone=SimpleNamespace(**flags))
    baseline = configure_cross_path(model, flags, MATH)
    triton = configure_cross_path(model, flags, TRITON)
    cross = configure_cross_path(model, flags, CROSS)
    assert baseline["ordinary_sdpa"] == "MATH"
    assert cross["ordinary_sdpa"] == triton["ordinary_sdpa"] == "FLASH_ATTENTION"
    assert cross["runtime_flags"] == baseline["runtime_flags"]
    assert cross["precision"] == "bf16_mixed"
    assert cross["runtime_flags"]["attention_precision"] == "mixed"
    assert {key for key in flags if cross["runtime_flags"][key] != triton["runtime_flags"][key]} == {
        "tile_backend", "backward_tile_backend"}
    with pytest.raises(ValueError):
        configure_cross_path(model, flags, "fp32_math_eager")


def previous_row(path):
    return {"path": path, "objective": "ce", "passed": True,
            "metrics": {"objective": 4.}, "forward_fingerprints": {"hidden": "hash"},
            "gradients_vs_fp32": {"finite": True, "groups": {"backbone": {"norm": 3.}},
                "originally_missing_zero_materialized": [], "participation_intact": True}}


def test_prior_anchor_exactness_checks_do_not_claim_saved_full_vectors():
    previous = previous_row(MATH)
    gradients = previous["gradients_vs_fp32"]
    good = anchor_comparison(previous["metrics"], previous["forward_fingerprints"], gradients, previous)
    assert good["passed"] and all(good["checks"].values())
    assert not anchor_comparison({"objective": 4.00001}, previous["forward_fingerprints"], gradients, previous)["passed"]
    changed = copy.deepcopy(gradients)
    changed["groups"]["backbone"]["norm"] += .1
    assert not anchor_comparison(previous["metrics"], previous["forward_fingerprints"], changed, previous)["passed"]
    assert not anchor_comparison(previous["metrics"], {"hidden": "different"}, gradients, previous)["passed"]


def reference_payload():
    return {"schema": "olmo-campaign-precision-bridge-v1", "status": "passed_operational_diagnostic", "passed": True,
            "sources": {"source.py": "a"*64}, "determinism": {"deterministic_algorithms": True},
            "integrity": {"weights_unchanged": True, "sources_unchanged": True},
            "rows": [previous_row(MATH), previous_row(TRITON)]}


def write_reference(tmp_path, payload):
    path = tmp_path/"report.json"
    path.write_text(json.dumps(payload))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_sha_source_and_success_pins(tmp_path):
    payload = reference_payload()
    path, sha = write_reference(tmp_path, payload)
    assert load_reference(path, sha, payload["sources"]) == payload
    with pytest.raises(ValueError, match="SHA256"):
        load_reference(path, "0"*64, payload["sources"])
    with pytest.raises(ValueError, match="source pins"):
        load_reference(path, sha, {"source.py": "b"*64})


@pytest.mark.parametrize("mutation", ["failed", "nondeterministic", "state_changed", "missing", "duplicate"])
def test_invalid_prior_report_rejected(tmp_path, mutation):
    payload = reference_payload()
    if mutation == "failed": payload["passed"] = False
    elif mutation == "nondeterministic": payload["determinism"]["deterministic_algorithms"] = False
    elif mutation == "state_changed": payload["integrity"]["weights_unchanged"] = False
    elif mutation == "missing": payload["rows"].pop()
    else: payload["rows"].append(previous_row(MATH))
    path, sha = write_reference(tmp_path, payload)
    with pytest.raises(ValueError):
        load_reference(path, sha, payload["sources"])


def test_cli_requires_pinned_reference_and_rejects_scope_expansion():
    args = ["--reference-report", "prior.json", "--reference-sha256", "a"*64, "--output-dir", "/tmp/cross"]
    assert parse_args(args).reference_sha256 == "a"*64
    for extra in (["--steps", "2"], ["--length", "1024"], ["--objective", "combined"], ["--batch-size", "8"]):
        with pytest.raises(SystemExit):
            parse_args(args+extra)
    with pytest.raises(SystemExit):
        parse_args(["--reference-report", "prior.json", "--reference-sha256", "A"*64, "--output-dir", "/tmp/cross"])
