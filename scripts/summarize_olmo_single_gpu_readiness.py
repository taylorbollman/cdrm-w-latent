#!/usr/bin/env python3
"""Verify and retain explicitly selected one-GPU objective/recovery evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_rt_efficiency_report import require, digest, regular, safe_source, check_summary
from scripts.olmo_rt_author_integration_report import resolve_commit, frozen_digest
from scripts.olmo_rt_efficiency_retain import safe_relative
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference

GROUPS = {
    "distributed": {"name": "olmo-distributed-prepare", "schema": "olmo-distributed-prepare-v1",
                    "script": "scripts/olmo_distributed_prepare.py"},
    "recovery": {"name": "olmo-graph-recovery", "schema": "olmo-graph-recovery-v1",
                 "script": "scripts/olmo_graph_recovery.py"},
}
DOCS = "docs/reports/olmo-single-gpu-readiness"
RUNTIME = ".runtime/olmo-single-gpu-readiness"
SUMMARY_SCHEMA = "olmo-single-gpu-readiness-summary-v1"
RETENTION_SCHEMA = "olmo-single-gpu-readiness-retention-v1"
ARTIFACT = ".runtime/olmo1b-step60000/artifacts/artifact-manifest.json"
RECEIPT = "docs/reports/olmo1b-o1/storage-receipt.json"
MAX_BYTES = 64 * 1024**2
TERMS = ("ce", "latent", "kl")
ACCUMULATION_BUDGETS = {"global_relative_l2": 2e-6,
                        "tensor_relative_l2": 2e-6, "tensor_max_relative": 1e-5}
RECOVERY_GATES = {
    "preparation_eager_graph", "preparation_capture_rng", "boundary_released_graph_eager",
    "save_execution_contract", "save_preserves_boundary", "reference_boundary_loss_gradients",
    "reference_eager_graph", "reference_capture_rng", "reference_final_released_graph_eager",
    "reference_state_health", "restored_boundary", "restored_boundary_loss_gradients",
    "restored_eager_graph", "restored_capture_rng", "restored_final_released_graph_eager",
    "restored_final_state_rng_cursor", "restored_update_metrics", "restored_state_health",
    "physical_update_accounting", "frozen_sources_dependencies",
}
ESSENTIAL_SOURCES = {
    "scripts/olmo_rt_large_batch.py", "scripts/olmo_large_batch_validation.py",
    "scripts/olmo_f1_common.py", "scripts/olmo_lm_common.py", "scripts/experiment_tracking.py",
    "scripts/docker_shell.sh", "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_tiled.py",
    "cdrm/pretrained/olmo_fbt.py", "cdrm/pretrained/fbt_training.py",
    "cdrm/pretrained/nextlat.py", "cdrm/pretrained/lm_training.py",
    "cdrm/pretrained/static_training.py", "cdrm/pretrained/static_nextlat.py",
}
PROJECT_FILES = (
    "scripts/summarize_olmo_single_gpu_readiness.py", "tests/test_olmo_single_gpu_readiness_report.py",
    "scripts/olmo_rt_efficiency_report.py", "scripts/olmo_rt_author_integration_report.py",
    "scripts/olmo_rt_efficiency_retain.py", "scripts/openelm_retain.py", "scripts/olmo_tiled_retain.py",
    "scripts/docker_shell.sh", "docker/requirements-docker.txt", RECEIPT,
    "docs/native-rt-single-to-two-gpu-plan.md", "docs/fbt-rt-nextlat-handoff.md",
    "docs/fbt-rt-nextlat-research-plan-v4.md", "AGENTS.md",
)
GROUP_TESTS = {
    "distributed": ("tests/test_distributed_training_preparation.py", "tests/test_olmo_distributed_prepare.py"),
    "recovery": ("tests/test_olmo_graph_recovery.py",),
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def parse_selection(value):
    require(isinstance(value, str) and re.fullmatch(r"(?:distributed|recovery)/[A-Za-z0-9][A-Za-z0-9_.-]*", value),
            "Selection must be distributed/NAME or recovery/NAME")
    group, name = value.split("/")
    return group, name


def objective_counts(values):
    require(isinstance(values, dict) and set(values) == set(TERMS)
            and all(type(values[t]) is int and values[t] >= 0 for t in TERMS), "Invalid objective counts")
    return values


def validate_gradient_gate(check, names, *, exact):
    require(all(check.get(key) is True for key in ("passed", "ownership_matches", "finite",
        "expected_participation_matches", "losses_finite", "losses_and_counts_exact")),
        "Incomplete adapter gradient/loss/participation gate")
    require(check.get("missing_expected_gradients") == check.get("unexpected_gradients") == [],
            "Adapter gradient participation differs")
    require(check.get("reference_losses") == check.get("candidate_losses")
            and check.get("reference_losses"), "Exact detached loss records differ")
    rows = check.get("gradients", {})
    require(set(rows) == names and bool(rows), "Gradient inventory differs from expected active names")
    require(check.get("exact_required") is exact, "Gradient gate's precision contract differs")
    for row in rows.values():
        require(row.get("passed") is True and row.get("finite") is True
                and all(finite(row.get(key)) for key in ("relative_l2", "max_relative", "max_absolute",
                                                       "delta_sq", "reference_sq")), "Invalid gradient tensor gate")
        if exact:
            require(row.get("bitwise_equal") is True and row["delta_sq"] == row["max_absolute"] == 0,
                    "Exact tensor-gradient gate contains a difference")
        else:
            require(row["relative_l2"] <= ACCUMULATION_BUDGETS["tensor_relative_l2"]
                    and row["max_relative"] <= ACCUMULATION_BUDGETS["tensor_max_relative"],
                    "CPU VJP sum exceeded the frozen accumulation budget")
    delta, reference = sum(row["delta_sq"] for row in rows.values()), sum(row["reference_sq"] for row in rows.values())
    expected = math.sqrt(delta/reference) if reference else (0. if delta == 0 else float("inf"))
    require(finite(check.get("global_gradient_relative_l2"))
            and math.isclose(check["global_gradient_relative_l2"], expected, rel_tol=1e-12, abs_tol=1e-30),
            "Global gradient error differs from tensor evidence")
    require(check.get("budgets") == (None if exact else ACCUMULATION_BUDGETS), "Accumulation budgets changed")
    require(not exact or check.get("all_bitwise_equal") is True, "Exact gradient summary failed")
    require(exact or expected <= ACCUMULATION_BUDGETS["global_relative_l2"], "Global accumulation budget failed")


def validate_distributed(raw):
    config, checks = raw["configuration"], {c["name"]: c for c in raw["checks"]}
    updates = config.get("updates")
    require(type(updates) is int and updates in (1, 2), "Invalid bounded adapter update count")
    require(config.get("world_size") == 1 and config.get("real_distributed_execution") is False
            and config.get("cuda_graphs") is False and config.get("arm") == "compiled-native",
            "Adapter preparation must remain explicitly single-GPU eager")
    expected = {"world1_adapter_exact", "same_order_accumulation_exact", "independent_cpu_vjp_sum"}
    expected |= {f"adapter_update_{i}_{suffix}" for i in range(1, updates+1)
                 for suffix in ("gradient_ownership", "health")}
    require(set(checks) == expected and raw["physical_optimizer_updates"] == updates,
            "Adapter gates or physical update accounting differ")
    names = raw.get("expected_active_parameter_names")
    require(isinstance(names, list) and names and len(names) == len(set(names)), "Missing active-name declaration")
    for name in ("world1_adapter_exact", "same_order_accumulation_exact", "independent_cpu_vjp_sum"):
        validate_gradient_gate(checks[name], set(names), exact=name != "independent_cpu_vjp_sum")
    require(config.get("accumulation_budgets") == ACCUMULATION_BUDGETS, "Configuration budgets differ")
    batches = raw.get("batches", {})
    locals_ = batches.get("microbatch_counts")
    require(isinstance(locals_, list) and len(locals_) == 2, "Missing two microbatch count records")
    for value in locals_: objective_counts(value)
    counts = objective_counts(batches.get("global_counts"))
    require(counts == {t: sum(value[t] for value in locals_) for t in TERMS}, "Global denominators do not sum")
    require(locals_[1]["latent"] == locals_[1]["kl"] == 0 and all(value["ce"] > 0 for value in locals_),
            "Expected locally empty auxiliary objectives were not tested")
    if config["case"] == "combined":
        require(locals_[0]["latent"] > 0 and locals_[0]["kl"] > 0, "Combined auxiliary coverage missing")
    require(raw.get("independent_reference_microbatches_completed") == 2, "Independent references incomplete")
    rows = raw.get("update_records", [])
    require(len(rows) == updates, "Adapter update records missing")
    for i, row in enumerate(rows, 1):
        ownership, health = checks[f"adapter_update_{i}_gradient_ownership"], checks[f"adapter_update_{i}_health"]
        require(ownership.get("expected_participation_matches") is True
                and ownership.get("missing_expected_gradients") == ownership.get("unexpected_gradients") == [],
                "Adapter update gradient ownership failed")
        require(row.get("update") == i and row.get("microbatches") == 2
                and row.get("global_counts") == counts
                and row.get("input_tokens") == 2*config["batch_size"]*config["length"], "Adapter update accounting differs")
        require(row.get("weights_changed") is True and health.get("weights_changed") is True
                and row.get("state_health", {}).get("passed") is True
                and row["state_health"].get("nonfinite_parameters") == row["state_health"].get("nonfinite_optimizer_tensors") == []
                and finite(row.get("gradient_norm_before_clip")), "Unhealthy adapter update")
    return updates


def validate_recovery(raw, directory):
    config, checks = raw["configuration"], {c["name"]: c for c in raw["checks"]}
    require(set(checks) == RECOVERY_GATES and raw["physical_optimizer_updates"] == 6
            and raw.get("branch_physical_optimizer_updates") == {"preparation": 2, "reference": 2, "restored": 2},
            "Recovery gates or physical update accounting differ")
    require(config.get("world_size") == 1 and config.get("accumulation") == 1
            and config.get("reference_graph_rebuilt") is True, "Recovery scope differs")
    for name, check in checks.items():
        if name.endswith("_eager_graph") or name.endswith("_released_graph_eager"):
            require(all(check.get(key) is True for key in ("all_bitwise_equal", "loss_names_match",
                    "ownership_matches", "storage_matches")), "Recovery graph gate incomplete")
            require(check.get("gradients") and check.get("losses")
                    and all(row.get("bitwise_equal") is True for rows in (check["gradients"], check["losses"])
                            for row in rows.values()), "Recovery loss/gradient evidence differs")
            if name.endswith("_eager_graph"):
                require(check.get("replays_checked") == 2, "Recovery replay coverage differs")
            else:
                require(check.get("graph_released_before_eager") is True, "Terminal eager check retained graph")
        elif "bitwise_manifest_equal" in check:
            require(check["bitwise_manifest_equal"] and all(value is True for value in check["bitwise_manifest_equal"].values()),
                    "Recovery manifest equality failed")
    for key, update in (("boundary", 2), ("reference_final", 4), ("restored_final", 4)):
        boundary = raw.get(key, {})
        require(boundary.get("state", {}).get("counters", {}).get("optimizer_updates") == update
                and boundary.get("cursor", {}).get("next_update") == update
                and boundary.get("rng") and boundary.get("state", {}).get("model")
                and boundary["state"].get("optimizer") and boundary["state"].get("scheduler"),
                "Recovery logical state/cursor/ownership evidence missing")
    require(raw["reference_final"] == raw["restored_final"], "Restored final state/RNG/cursor differs")
    require(len(raw.get("reference_records", [])) == len(raw.get("restored_records", [])) == 2
            and raw["reference_records"] == raw["restored_records"]
            and len(raw.get("preparation_records", [])) == 2, "Recovery branch update records differ")
    for key, offset in (("preparation_records", 0), ("reference_records", 2), ("restored_records", 2)):
        for index, row in enumerate(raw[key], offset+1):
            require(row.get("metrics", {}).get("update_completed") is True
                    and row["metrics"].get("counters", {}).get("optimizer_updates") == index
                    and row.get("cursor", {}).get("next_update") == index, "Recovery recorded update is incomplete")
    receipt, disposal = raw.get("recovery_checkpoint", {}), raw.get("checkpoint_disposal", {})
    require(receipt.get("schema") == "olmo-lm-training-checkpoint-v1"
            and receipt.get("optimizer_updates") == 2 and type(receipt.get("size_bytes")) is int
            and receipt["size_bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", receipt.get("sha256", "")),
            "Recovery checkpoint receipt missing or invalid")
    expected_suffix = f"/.runtime/olmo-graph-recovery/{directory.name}/diagnostic-boundary.pt"
    path = PurePosixPath(receipt.get("path", ""))
    require(path.is_absolute() and ".." not in path.parts and str(path).endswith(expected_suffix), "Checkpoint receipt path differs")
    require(disposal.get("deleted_after_success") is True
            and all(disposal.get(key) == receipt[key] for key in ("sha256", "size_bytes"))
            and not (directory / "diagnostic-boundary.pt").exists(), "Verified checkpoint disposal is missing or inconsistent")
    return 4


def validate_finished(raw, group, directory):
    require(raw.get("schema") == GROUPS[group]["schema"] and raw.get("finished_utc")
            and raw.get("status") in {"passed", "failed", "oom"}, "Require a finished report in the selected group")
    require(type(raw.get("physical_optimizer_updates")) is int and raw["physical_optimizer_updates"] >= 0,
            "Invalid physical optimizer-update count")
    checks = raw.get("checks")
    require(isinstance(checks, list) and all(isinstance(c, dict) and isinstance(c.get("name"), str)
            and type(c.get("passed")) is bool for c in checks), "Invalid check inventory")
    require(len({c["name"] for c in checks}) == len(checks), "Duplicate checks")
    if raw["status"] != "passed":
        require(raw.get("error") or raw.get("error_type") or raw.get("tracking_error_type")
                or raw.get("tracking_finish_error"), "Failed report lacks an explicit error")
        return None
    require(checks and all(c["passed"] for c in checks), "Passing report has a failed or missing gate")
    config = raw.get("configuration", {})
    case = config.get("case") if group == "distributed" else config.get("case", {}).get("name")
    require(case in {"rt", "combined"} and config.get("precision") == "bf16_mixed", "Unsupported model/precision scope")
    if group == "distributed":
        specification = config.get("case_specification", {})
        require(specification.get("name") == case
                and all(specification.get(key) == config.get(key) for key in ("batch_size", "length")),
                "Adapter case/shape specification differs")
    wandb = raw.get("wandb", {})
    require(wandb.get("status") == "synced" and wandb.get("mode") == "online"
            and wandb.get("entity") == "taylorbollman" and wandb.get("project") == "pretrained-fbt-rt-nextlat"
            and wandb.get("group") == GROUPS[group]["name"] and wandb.get("run_url"), "Passing report lacks synchronized W&B evidence")
    return validate_distributed(raw) if group == "distributed" else validate_recovery(raw, directory)


def verify_dependencies(raw, directory, root):
    record = raw.get("dependencies")
    if record is None and raw["status"] != "passed":
        return {"present": False, "files_checked": 0, "qualification": "Run failed before dependency inventory"}
    require(isinstance(record, dict) and isinstance(record.get("packages"), dict), "Dependency inventory missing")
    require(record["packages"].get("torch") and record["packages"].get("triton"), "Core dependency versions missing")
    checked = 0
    for key, prefix in (("dao_sources", "flash_attn"), ("fa4_sources", "flash_attn/cute")):
        sources = record.get(key)
        require(isinstance(sources, dict), "Dependency source inventory missing")
        for name, row in sources.items():
            relative = safe_relative(name)
            require(relative.suffix == ".py", "Unexpected dependency source")
            snapshot = regular(directory / "dependency-snapshot" / prefix / relative, root)
            require(digest(snapshot) == row.get("sha256"), "Dependency snapshot differs")
            checked += 1
    return {"present": True, "files_checked": checked, "packages": record["packages"],
            "scope": "Installed package versions and recorded source snapshots; native/SDPA arms request no Dao/FA4 source overlay"}


def load_run(root, selection, revision):
    root = Path(root)
    group, name = parse_selection(selection)
    revision = resolve_commit(root, revision)
    directory = root / ".runtime" / GROUPS[group]["name"] / name
    path = regular(directory / "report.json", root)
    raw = json.loads(path.read_text())
    logical = validate_finished(raw, group, directory)
    require(resolve_commit(root, raw.get("runtime_commit")) == revision, "Runtime commit differs from explicit selection")
    hashes = raw.get("source_hashes")
    essential = ESSENTIAL_SOURCES | {GROUPS[group]["script"]}
    if group == "distributed": essential |= {"cdrm/pretrained/distributed_training.py"}
    # Both harnesses freeze every top-level pretrained module. Check that a
    # claimed passing snapshot cannot silently omit a module from that commit.
    tree = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, "cdrm/pretrained"], cwd=root, text=True)
    essential |= {name for name in tree.splitlines() if name.endswith(".py") and name.count("/") == 2}
    require(isinstance(hashes, dict) and essential <= hashes.keys(), "Essential runtime sources omitted")
    differences = {}
    for source, expected in hashes.items():
        safe_relative(source)
        if source != "scripts/docker_shell.sh": safe_source(source)
        require(digest(regular(directory / "source-snapshot" / source, root)) == expected
                == frozen_digest(str(root), revision, source), "Frozen source/report/snapshot mismatch: " + source)
        current = digest(root/source) if (root/source).is_file() else None
        if current != expected: differences[source] = {"reported_sha256": expected, "current_sha256": current}
    protocol = f"docs/reports/{GROUPS[group]['name']}/protocol.md"
    require(digest(regular(directory / "protocol.md", root)) == raw.get("protocol_sha256")
            == frozen_digest(str(root), revision, protocol), "Frozen protocol/report/snapshot mismatch")
    dependencies = verify_dependencies(raw, directory, root)
    receipt = json.loads(regular(root / RECEIPT, root).read_text())
    source_checkpoint = None if "checkpoint" not in raw else checkpoint_reference(receipt, raw["checkpoint"])
    require(source_checkpoint is not None or raw["status"] != "passed", "Successful run lacks native checkpoint provenance")
    config = raw.get("configuration", {})
    specification = config.get("case_specification", {}) if group == "distributed" else config.get("case", {})
    row = {"selection": selection, "group": group, "name": name, "runtime_commit": revision,
           "report_path": path.relative_to(root).as_posix(), "report_sha256": digest(path), "status": raw["status"],
           "case": config.get("case") if group == "distributed" else specification.get("name"),
           "batch_size": specification.get("batch_size"), "length": specification.get("length"),
           "physical_optimizer_updates": raw["physical_optimizer_updates"], "logical_endpoint_updates": logical,
           "checks": [check_summary(c) for c in raw["checks"]], "source_pairs_checked": len(hashes),
           "current_source_differences": differences, "dependency_verification": dependencies,
           "checkpoint_reference": source_checkpoint, "wandb_url": raw.get("wandb", {}).get("run_url"),
           "error": raw.get("error"), "failure_stage": raw.get("stage") if raw["status"] != "passed" else None,
           "checkpoint_disposal": raw.get("checkpoint_disposal"),
           "diagnostic_checkpoint_present": (directory / "diagnostic-boundary.pt").exists()}
    return row, raw


def summarize(selections, revision, *, overrides=None, root=ROOT):
    require(selections and len(set(selections)) == len(selections), "Require a nonempty explicit unique run selection")
    overrides = {} if overrides is None else overrides
    require(set(overrides) <= set(selections), "Runtime overrides include an unselected run")
    rows = [load_run(root, value, overrides.get(value, revision))[0] for value in selections]
    completed = {(row["group"], row["case"]) for row in rows if row["status"] == "passed"
                 and row["batch_size"] == 2 and row["length"] == 512
                 and row["physical_optimizer_updates"] == (2 if row["group"] == "distributed" else 6)}
    return {"schema": SUMMARY_SCHEMA, "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_commit": resolve_commit(root, revision), "runs": rows,
            "statuses": dict(Counter(row["status"] for row in rows)),
            "physical_optimizer_updates": sum(row["physical_optimizer_updates"] for row in rows),
            "source_pairs_checked": sum(row["source_pairs_checked"] for row in rows),
            "primary_scope_complete": completed == {(g, c) for g in GROUPS for c in ("rt", "combined")},
            "scope": "Single-GPU objective/accumulation and graph-rebuild recovery only; no two-GPU/DDP/NCCL/sharding validation",
            "qualification": "Failed reports remain failed. Recovery has six actual steps across branches and a four-update logical endpoint; branches both rebuild graphs. No throughput or quality claim."}


def collect_evidence(root=ROOT):
    root = Path(root)
    summary = json.loads(regular(root / DOCS / "summary.json", root).read_text())
    require(summary.get("schema") == SUMMARY_SCHEMA and summary.get("status") == "completed", "Completed summary required")
    refreshed = summarize([row["selection"] for row in summary["runs"]], summary["runtime_commit"],
                          overrides={row["selection"]: row["runtime_commit"] for row in summary["runs"]}, root=root)
    def stable(value):
        value = json.loads(json.dumps(value))
        value.pop("created_utc", None)
        for row in value["runs"]: row.pop("current_source_differences", None)
        return value
    require(stable(summary) == stable(refreshed), "Selected reports/derived summary changed since selection")
    artifact = json.loads(regular(root / ARTIFACT, root).read_text())
    receipt = json.loads(regular(root / RECEIPT, root).read_text())
    checkpoint = checkpoint_reference(receipt, artifact["checkpoint"])
    members = {}
    def add(path, name):
        relative = safe_relative(name)
        require(relative.suffix not in {".pt", ".pth", ".safetensors", ".bin"}, "Weights are excluded from evidence")
        require(name not in members, "Duplicate retained member")
        members[name] = regular(path, root)
    groups = {row["group"] for row in refreshed["runs"]}
    for row in refreshed["runs"]:
        group, name = row["group"], row["name"]
        path = root / row["report_path"]
        raw = json.loads(path.read_text())
        require(row["checkpoint_reference"] in (None, checkpoint), "Selected runs use different native checkpoint")
        prefix = f"runtime/{group}/{name}/"
        add(path, prefix + "report.json")
        add(path.parent / "protocol.md", prefix + "protocol.md")
        runtime = root / ".runtime" / GROUPS[group]["name"]
        logs = [candidate for candidate in (runtime/"logs"/(name+".log"), runtime/(name+".log")) if candidate.exists()]
        require(len(logs) == 1, "Require one unambiguous selected run log")
        add(logs[0], f"logs/{group}/{name}.log")
        for source in raw["source_hashes"]:
            add(path.parent / "source-snapshot" / source, prefix+"source-snapshot/"+source)
        for key, base in (("dao_sources", "flash_attn"), ("fa4_sources", "flash_attn/cute")):
            for source in raw.get("dependencies", {}).get(key, {}):
                relative = "dependency-snapshot/"+base+"/"+source
                add(path.parent/relative, prefix+relative)
    for group in sorted(groups):
        docs = root / "docs/reports" / GROUPS[group]["name"]
        for name in ("protocol.md", "test-results.txt"):
            add(docs/name, f"report/{GROUPS[group]['name']}/{name}")
        for name in ("usage.md", "results.md"):
            if (docs/name).exists(): add(docs/name, f"report/{GROUPS[group]['name']}/{name}")
        for test in GROUP_TESTS[group]: add(root/test, "project/"+test)
    for name in ("summary.json", "results.md", "test-results.txt", "usage.md"):
        add(root / DOCS / name, "report/readiness/"+name)
    for name in PROJECT_FILES: add(root/name, "project/"+name)
    add(root / ARTIFACT, "native-reference/artifact-manifest.json")
    add(root / RUNTIME / "final-gpu.log", "logs/final-gpu.log")
    require(sum(path.stat().st_size for path in members.values()) < MAX_BYTES, "Evidence exceeds 64 MiB")
    return refreshed, checkpoint, members


def upload_verified(bucket, path, prefix):
    expected = file_digest(path)
    blob = bucket.blob(prefix+"/"+path.name)
    blob.metadata = {"sha256": expected["sha256"], "artifact_schema": RETENTION_SCHEMA}
    blob.upload_from_filename(str(path), if_generation_match=0, checksum="md5")
    blob.reload()
    _check_remote(blob, expected)
    require(hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation)).hexdigest() == expected["sha256"],
            "Downloaded retained object differs")
    return {"uri": "gs://fast-chunks/"+blob.name, "generation": str(blob.generation), **expected}


def retain(*, root=ROOT, dry_run=False):
    root = Path(root)
    summary, checkpoint, members = collect_evidence(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / RUNTIME / ("retention-"+stamp)
    output.mkdir(parents=True, exist_ok=False)
    archive = output/"evidence.tar.gz"
    inventory = build_evidence_archive(archive, [(path, name) for name, path in sorted(members.items())],
        "# Single-GPU readiness evidence\n\nSelected successes/failures, frozen runtime/protocol/dependency snapshots, "
        "tests and logs. No model/optimizer weights, credentials or datasets are included. The source model is reused "
        "by its immutable O1 checkpoint receipt. Recovery diagnostic weights are disposable after verified success; "
        "any failed-run checkpoint stays local and is not automatically uploaded. No genuine distributed claim.\n")
    require(archive.stat().st_size < MAX_BYTES, "Compressed evidence exceeds 64 MiB")
    manifest = {"schema": RETENTION_SCHEMA, "members": inventory, "weights_uploaded": False,
                "checkpoint_reference": checkpoint, "runs": summary["runs"], "statuses": summary["statuses"],
                "physical_optimizer_updates": summary["physical_optimizer_updates"], "evidence": file_digest(archive)}
    manifest_path = output/"retention-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    if dry_run:
        result = {"status": "dry_run", "members": len(inventory), "archive": str(archive)}
        print(json.dumps(result))
        return result
    from google.cloud import storage
    bucket = storage.Client().bucket("fast-chunks")
    prefix = "cdrm-w-latent/fbt-rt-nextlat/olmo-single-gpu-readiness/"+stamp
    receipt = {"schema": RETENTION_SCHEMA, "status": "verified", "weights_uploaded": False,
               "members": len(inventory), "runs": len(summary["runs"]), "statuses": summary["statuses"],
               "physical_optimizer_updates": summary["physical_optimizer_updates"],
               "checkpoint_reference": verify_checkpoint_reference(bucket, checkpoint),
               "objects": [upload_verified(bucket, path, prefix) for path in (archive, manifest_path)]}
    receipt_path = output/"storage-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    receipt["receipt_object"] = upload_verified(bucket, receipt_path, prefix)
    (root/DOCS/"storage-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": "verified", "receipt": receipt["receipt_object"]["uri"]}))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--runtime-commit")
    parser.add_argument("--run-commit", action="append", default=[])
    parser.add_argument("--retain", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.retain:
        if args.runs or args.runtime_commit or args.run_commit: parser.error("Retention reads the existing selection")
        return retain(dry_run=args.dry_run)
    if not args.runs or not args.runtime_commit or args.dry_run:
        parser.error("Summarization requires --runs/--runtime-commit; --dry-run requires --retain")
    overrides = {}
    for value in args.run_commit:
        name, separator, revision = value.partition("=")
        require(separator and revision and name not in overrides, "Invalid/duplicate runtime override")
        parse_selection(name)
        overrides[name] = revision
    summary = summarize(args.runs, args.runtime_commit, overrides=overrides)
    directory = ROOT/DOCS
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: summary[key] for key in ("statuses", "physical_optimizer_updates", "primary_scope_complete")}))


if __name__ == "__main__":
    main()
