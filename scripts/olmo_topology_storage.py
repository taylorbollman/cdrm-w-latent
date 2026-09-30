"""Campaign-identity CPU retention hook for the unchanged async manager.

The historical pilot worker validates a different execution identity. This
explicit hook retains the campaign validator used by SSDCheckpointStorage;
neither validator nor the historical worker is modified or monkeypatched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_campaign_execution_restore import (
    MANIFEST_FIELDS, REMOTE_ROOT, committed_metadata, pin, publication_metadata,
)
from scripts.olmo_campaign_ssd_storage import _write_durable
from scripts.olmo_pilot_async_storage import (
    AsyncRetentionError, RESULT_SCHEMA, SCHEMA, _frozen_sources, _sha,
    cpu_child_environment,
)


def source_hashes():
    names = ("olmo_topology_storage.py", "olmo_pilot_async_storage.py",
             "olmo_campaign_ssd_storage.py", "olmo_campaign_execution_restore.py",
             "olmo_campaign_recovery_bundle.py", "olmo_two_gpu_retain.py", "openelm_retain.py")
    return {"scripts/" + name: _sha(ROOT / "scripts" / name) for name in names}


def regular(path):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
            or any(parent.is_symlink() for parent in path.parents)):
        raise ValueError("Require an owned regular retention file: " + str(path))
    return path


def authenticate_job(job_path, job_sha256):
    """Authenticate the campaign authority before any SDK operation."""
    pin(job_sha256)
    if _sha(regular(job_path)) != job_sha256:
        raise ValueError("Retention job SHA256 differs")
    job = json.loads(Path(job_path).read_text())
    if job.get("schema") != SCHEMA:
        raise ValueError("Unknown async retention job schema")
    supplied = {str((Path(name) if Path(name).is_absolute() else ROOT / name).resolve()): value
                for name, value in job["source_pins"].items()}
    for name, expected in source_hashes().items():
        if supplied.get(str(ROOT / name)) != expected:
            raise ValueError("Campaign retention dependency not pinned: " + name)
    _frozen_sources(supplied)
    receipt = job["receipt"]
    if set(receipt) != MANIFEST_FIELDS | {"directory", "manifest_sha256"}:
        raise ValueError("Require a complete local committed checkpoint receipt")
    manifest = {name: receipt[name] for name in MANIFEST_FIELDS}
    committed_metadata(manifest)
    directory = Path(receipt["directory"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Require a real committed checkpoint directory")
    manifest_path = regular(directory / "manifest.json")
    regular(directory / "state.pt")
    if (_sha(manifest_path) != receipt["manifest_sha256"]
            or json.loads(manifest_path.read_text()) != manifest):
        raise ValueError("Committed local manifest differs from submitted receipt")
    prefix = job["storage_prefix"]
    if (not isinstance(prefix, str) or not prefix.startswith(REMOTE_ROOT)
            or any(part in ("", ".", "..") for part in prefix[5:].split("/"))
            or any(char in prefix for char in ("?", "#", "%"))):
        raise ValueError("Require an explicit project checkpoint prefix")
    return job


def retain_in_child(job_path, job_sha256, result_path, log_path, *, timeout_seconds,
                    cancel_event=None):
    """Same hook API as the pilot: isolated CPU child, bounded/reaped session."""
    if (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 480):
        raise ValueError("CPU retention timeout must be positive and at most 480 seconds")
    if Path(result_path).exists() or Path(result_path).is_symlink():
        raise FileExistsError("Require a fresh retention result path")
    with Path(log_path).open("xb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker",
            str(job_path), "--job-sha256", job_sha256, "--result", str(result_path)],
            cwd=ROOT, env=cpu_child_environment(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout_seconds
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise AsyncRetentionError("CPU retention child cancelled by trainer teardown")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout_seconds)
                try:
                    returncode = process.wait(timeout=min(0.25, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    if returncode != 0:
        raise AsyncRetentionError(f"CPU retention child exited {returncode}; inspect {log_path}")
    result = json.loads(regular(result_path).read_text())
    if result.get("schema") != RESULT_SCHEMA or result.get("job_sha256") != job_sha256:
        raise AsyncRetentionError("CPU retention result does not bind the submitted job")
    return result


def worker(job_path, job_sha256, result_path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Retention child requires explicitly disabled CUDA visibility")
    if any(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")):
        raise RuntimeError("Retention child must not inherit a process-group environment")
    job = authenticate_job(job_path, job_sha256)
    # Cloud and torch imports stay in the fresh CPU child only.
    from google.cloud import storage
    from scripts.openelm_retain import file_digest
    from scripts.olmo_two_gpu_retain import upload_verified
    import torch
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("Retention child unexpectedly initialized CUDA or distributed state")
    receipt = job["receipt"]
    directory = Path(receipt["directory"])
    digests = {name: file_digest(regular(directory / name)) for name in ("state.pt", "manifest.json")}
    for name, digest in digests.items():
        expected = receipt["state"] if name == "state.pt" else {"sha256": receipt["manifest_sha256"]}
        if any(digest[key] != value for key, value in expected.items() if key in digest):
            raise ValueError("Committed local checkpoint bytes changed before upload")
    bucket_name, prefix = job["storage_prefix"][5:].split("/", 1)
    bucket = storage.Client().bucket(bucket_name)
    started = time.perf_counter()
    objects = [upload_verified(bucket, f"{prefix}/{directory.name}/{name}", directory / name,
               digests[name], download_sha256=True) for name in ("state.pt", "manifest.json")]
    retention = {"objects": objects, "create_only": True, "download_sha256_verified": True}
    publication_metadata({**receipt, "retention": retention})
    _frozen_sources(job["source_pins"])
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("Retention child unexpectedly initialized CUDA or distributed state")
    result = {"schema": RESULT_SCHEMA, "job_sha256": job_sha256, "retention": retention,
        "worker": {"pid": os.getpid(), "cuda_initialized": False, "distributed_initialized": False,
            "sdk_seconds": time.perf_counter() - started, "cuda_visible_devices": "",
            "identity_validator": "olmo-campaign-execution-identity-v1"}}
    _write_durable(result_path, result, once=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    worker(args.worker, args.job_sha256, args.result)


if __name__ == "__main__":
    main()
