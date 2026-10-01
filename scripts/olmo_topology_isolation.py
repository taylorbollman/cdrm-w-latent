#!/usr/bin/env python3
"""Bounded isolation of two independent one-rank CUDA graph jobs.

The host supervisor uses the project Docker launcher. A local Docker shim adds
only per-job names/ownership labels, allowing cleanup before worker startup.
Workers use a tiny ordinary campaign model: this tests process isolation and
checkpoint usability, not native-model precision or training quality.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "olmo-topology-isolation-v1"
OWNER_LABEL = "cdrm.topology-isolation-job"
ABRUPT_EXIT = 73
ROLES = ("victim", "peer")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_pins():
    scripts = ("olmo_topology_isolation.py", "olmo_allocation_benchmark.py", "olmo_pilot_ordered_data.py",
        "olmo_campaign_loop.py", "olmo_campaign_probe.py", "olmo_distributed_prepare.py",
        "olmo_packed_campaign_run.py", "olmo_f2_graph_backend_probe.py", "olmo_validation.py",
        "olmo_two_gpu_validate.py", "experiment_tracking.py")
    paths = set((ROOT / "cdrm/pretrained").rglob("*.py"))
    paths.update(ROOT / "scripts" / name for name in scripts)
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    host = sub.add_parser("supervise")
    host.add_argument("--scenario", choices=("controlled", "abrupt"), required=True)
    host.add_argument("--output-dir", type=Path, required=True)
    host.add_argument("--timeout-seconds", type=int, default=600)
    worker = sub.add_parser("worker")
    worker.add_argument("--role", choices=ROLES, required=True)
    worker.add_argument("--control", type=Path, required=True)
    worker.add_argument("--control-sha256", required=True)
    args = parser.parse_args(argv)
    if args.mode == "supervise":
        if not 120 <= args.timeout_seconds <= 900:
            parser.error("Require a bounded timeout between 120 and 900 seconds")
        args.output_dir = args.output_dir.resolve()
        if not args.output_dir.is_relative_to(ROOT) or args.output_dir.exists():
            parser.error("Use a new output directory inside the persistent project")
    elif not re.fullmatch(r"[0-9a-f]{64}", args.control_sha256):
        parser.error("Require a lowercase SHA256 control pin")
    return args


def make_control(output, scenario, timeout, *, identity=None):
    identity = uuid.uuid4().hex if identity is None else identity
    if not re.fullmatch(r"[0-9a-f]{32}", identity):
        raise ValueError("Cell identity must be a UUID hexadecimal token")
    return {"schema": SCHEMA, "cell_id": identity, "scenario": scenario, "timeout_seconds": timeout,
        "output_relative": str(Path(output).relative_to(ROOT)), "before_updates": 2, "peer_after_updates": 4,
        "sources": source_pins(), "jobs": {role: {
            "job_id": identity + "-" + role, "visible_devices": str(index),
            "container_name": "cdrm-isolation-" + identity + "-" + role,
            "rendezvous_id": "isolation-" + identity + "-" + role,
            "retention_identity": "olmo-topology-isolation/" + identity + "/" + role}
            for index, role in enumerate(ROLES)}}


def docker_shim(docker, job):
    """Keep launcher behavior; add ownership before Docker creates a container."""
    command = shlex.join(docker)
    owned = shlex.join(["run", "--name", job["container_name"], "--label", OWNER_LABEL + "=" + job["job_id"]])
    return ("#!/bin/sh\nset -eu\nif [ \"${1:-}\" = run ]; then\n"
            f"  shift\n  exec {command} {owned} \"$@\"\nfi\nexec {command} \"$@\"\n")


def worker_command(control, control_sha, role):
    job = control["jobs"][role]
    container_root = Path("/workspace/cdrm-w-latent")
    output = container_root / control["output_relative"]
    invocation = ["env", "-u", "GOOGLE_APPLICATION_CREDENTIALS",
        "CUDA_VISIBLE_DEVICES=" + job["visible_devices"], "OMP_NUM_THREADS=1",
        "NCCL_ASYNC_ERROR_HANDLING=0", "TORCH_NCCL_ASYNC_ERROR_HANDLING=0",
        "torchrun", "--nproc_per_node=1", "--max-restarts=0", "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0", "--rdzv-id=" + job["rendezvous_id"], "-m",
        "scripts.olmo_topology_isolation", "worker", "--role", role,
        "--control", str(output / "control.json"), "--control-sha256", control_sha]
    preflight = ("set -euo pipefail\n"
        "test -f /.dockerenv\n"
        "test \"$PWD\" = /workspace/cdrm-w-latent\n"
        f"nvidia-smi -i {job['visible_devices']} --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader\n"
        f"test -z \"$(nvidia-smi -i {job['visible_devices']} --query-compute-apps=pid --format=csv,noheader)\"\n"
        "exec " + shlex.join(invocation))
    inside = ["timeout", "--signal=TERM", "--kill-after=15s", str(control["timeout_seconds"]) + "s",
              "bash", "-lc", preflight]
    return ["bash", "scripts/docker_shell.sh", *inside]


def check_marker(marker, control, *, role=None, kind=None):
    if marker.get("schema") != SCHEMA or marker.get("cell_id") != control["cell_id"]:
        raise ValueError("Marker cell identity differs")
    if role is not None and marker.get("job_id") != control["jobs"][role]["job_id"]:
        raise ValueError("Marker job identity differs")
    if kind is not None and marker.get("kind") != kind:
        raise ValueError("Marker operation differs")
    return marker


def wait_marker(path, control, *, role=None, kind=None):
    start = time.monotonic()
    while not path.exists():
        if time.monotonic() - start > control["timeout_seconds"] - 30:
            raise TimeoutError("Bounded worker marker wait expired")
        time.sleep(.1)
    return check_marker(read_json(path), control, role=role, kind=kind)


def validate_outcome(control, victim, peer, *, victim_exit, peer_exit, victim_log, fault_observed_unix):
    if peer_exit != 0 or peer.get("status") != "completed":
        raise ValueError("Peer did not complete independently")
    for role, report in (("victim", victim), ("peer", peer)):
        if report.get("schema") != SCHEMA or report.get("job_id") != control["jobs"][role]["job_id"]:
            raise ValueError("Worker identity differs")
        if report.get("sources") != control["sources"] or report.get("world_size") != 1:
            raise ValueError("Worker runtime scope differs")
        if report.get("rendezvous_id") != control["jobs"][role]["rendezvous_id"]:
            raise ValueError("Worker rendezvous differs")
        if report.get("retention_identity") != control["jobs"][role]["retention_identity"] or report.get("visible_devices") != control["jobs"][role]["visible_devices"]:
            raise ValueError("Worker device or retention identity differs")
        if not report.get("checkpoints") or any(not row.get("cpu_readback_exact") or not row.get("adam_steps_match") for row in report["checkpoints"]):
            raise ValueError("Worker lacks a verified retained checkpoint")
        if not report.get("parameters_changed") or any(not row["finite"] or not row["same_graph_objects"] for row in report["updates"]):
            raise ValueError("Worker did not complete finite parameter-changing graph updates")
    if not peer.get("clean_teardown") or victim.get("wandb", {}).get("run_id") == peer.get("wandb", {}).get("run_id"):
        raise ValueError("Peer teardown or independent tracking identity failed")
    if victim.get("container_id") == peer.get("container_id"):
        raise ValueError("Jobs did not use separate containers")
    before = control["before_updates"]
    after = control["peer_after_updates"]
    if len(victim["updates"]) != before or len(peer["updates"]) != before + after:
        raise ValueError("Before/after optimizer update inventory differs")
    for index, row in enumerate(peer["updates"]):
        if row["update"] != index + 1 or not row["finite"] or not row["same_graph_objects"]:
            raise ValueError("Peer update failed or replaced its live graphs")
        if index < before and row["finished_unix"] >= fault_observed_unix:
            raise ValueError("Missing peer progress before victim exit")
        if index >= before and row["started_unix"] <= fault_observed_unix:
            raise ValueError("Missing peer progress after observed victim exit")
    if not peer.get("post_fault_parameters_changed"):
        raise ValueError("Peer parameters did not change between pre-fault and final checkpoints")
    if not peer.get("fresh_restore_exact") or peer["checkpoints"][-1]["optimizer_updates"] != before + after:
        raise ValueError("Peer checkpoint did not restore into fresh model and Adam")
    if control["scenario"] == "controlled":
        if victim_exit != 0 or victim.get("status") != "controlled_stopped" or not victim.get("clean_teardown"):
            raise ValueError("Cooperative victim stop failed")
    elif (victim_exit in (0, 124, 137) or victim.get("status") != "deliberate_abrupt_exit"
          or victim.get("intended_worker_exit") != ABRUPT_EXIT
          or not re.search(r"exitcode\s*:\s*73\b", victim_log)):
        raise ValueError("Did not observe the intended abrupt worker exit 73")
    return {"passed": True, "peer_updates_before_failure": before, "peer_updates_after_failure": after,
            "victim_launcher_exit_code": victim_exit, "peer_launcher_exit_code": peer_exit,
            "victim_worker_exit_code": 0 if control["scenario"] == "controlled" else ABRUPT_EXIT,
            "peer_fresh_checkpoint_restore": True, "peer_graphs_survived": True,
            "peer_post_fault_parameters_changed": True,
            "fault_scope": "Peer waits at a completed optimizer boundary during victim exit, then reuses its live graphs",
            "restore_scope": "Fresh CPU model and Adam in the surviving worker; not fresh-process CUDA recovery"}


def _docker_cli():
    executable = shutil.which("docker")
    if executable is None:
        raise RuntimeError("Docker is required")
    base = [executable]
    if subprocess.run(base + ["info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20).returncode:
        base = ["sudo", "-n", executable]
        subprocess.run(base + ["info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=True)
    return base


def cleanup_owned(docker, job):
    """Kill only a container bearing this supervisor's exact ownership label."""
    result = subprocess.run(docker + ["inspect", "--format", "{{json .Config.Labels}}", job["container_name"]],
                            text=True, capture_output=True, timeout=15)
    if result.returncode == 0:
        labels = json.loads(result.stdout)
        if labels.get(OWNER_LABEL) != job["job_id"]:
            raise RuntimeError("Refuse cleanup of a container with another ownership label")
        subprocess.run(docker + ["rm", "--force", job["container_name"]],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=True)
    remaining = subprocess.run(docker + ["ps", "-aq", "--filter", "label=" + OWNER_LABEL + "=" + job["job_id"]],
                               text=True, capture_output=True, timeout=15, check=True)
    if remaining.stdout.strip():
        raise RuntimeError("Owned container remains after cleanup")
    return {"job_id": job["job_id"], "owned_containers_remaining": 0}


def supervise(args):
    output = args.output_dir
    docker = _docker_cli()
    control = make_control(output, args.scenario, args.timeout_seconds)
    output.mkdir(parents=True)
    write_json(output / "control.json", control)
    control_sha = sha256(output / "control.json")
    for name, pin in control["sources"].items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha256(target) != pin:
            raise ValueError("Sources changed during snapshot")
    report = {"schema": SCHEMA, "status": "running", "scenario": args.scenario,
              "control_sha256": control_sha, "cell_id": control["cell_id"], "started_unix": time.time(),
              "scope": "Independent one-rank tiny campaign jobs; not native precision, capacity, throughput or same-job rank failure", "jobs": {}}
    processes, streams = {}, {}
    started = time.monotonic()
    previous_handlers = {}
    def persist():
        write_json(output / "supervisor.json", report)
    def interrupted(signum, frame):
        raise RuntimeError("Supervisor interrupted by signal " + str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        for role in ROLES:
            job = control["jobs"][role]
            shim_dir = output / (role + "-launcher")
            shim_dir.mkdir()
            shim = shim_dir / "docker"
            shim.write_text(docker_shim(docker, job))
            shim.chmod(0o700)
            command = worker_command(control, control_sha, role)
            # Bound setup as well as the command inside the container.
            command = ["timeout", "--signal=TERM", "--kill-after=15s", str(args.timeout_seconds + 30) + "s", *command]
            streams[role] = (output / (role + ".log")).open("x")
            environment = {**os.environ, "PATH": str(shim_dir) + os.pathsep + os.environ["PATH"],
                "CDRM_ROOT": str(ROOT), "CDRM_DOCKER_GPUS": "all", "CDRM_FLASH_ATTENTION_SOURCE": "installed"}
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=streams[role],
                                     stderr=subprocess.STDOUT, start_new_session=True)
            processes[role] = child
            report["jobs"][role] = {**job, "host_pid": child.pid, "command": command, "started_unix": time.time()}
            persist()
        requested = released = False
        while True:
            if time.monotonic() - started > args.timeout_seconds:
                raise TimeoutError("Isolation supervisor exceeded its bounded lifetime")
            peer_exit, victim_exit = processes["peer"].poll(), processes["victim"].poll()
            if peer_exit is not None and (not released or peer_exit != 0):
                raise RuntimeError("Peer exited before completing the expected post-failure work")
            if not requested:
                if victim_exit is not None:
                    raise RuntimeError("Victim exited before fault injection")
                if all((output / role / "ready.json").is_file() for role in ROLES):
                    for role in ROLES:
                        marker = check_marker(read_json(output / role / "ready.json"), control, role=role, kind="ready")
                        if marker["optimizer_updates"] != control["before_updates"]:
                            raise ValueError("Worker readiness clock differs")
                    write_json(output / "fault-request.json", {"schema": SCHEMA, "cell_id": control["cell_id"],
                        "job_id": control["jobs"]["victim"]["job_id"], "kind": "fault-request", "scenario": args.scenario,
                        "requested_unix": time.time()})
                    requested = True
            elif not released and victim_exit is not None:
                victim = read_json(output / "victim" / "report.json")
                if args.scenario == "controlled":
                    if victim_exit != 0 or victim.get("status") != "controlled_stopped":
                        raise RuntimeError("Controlled victim exit failed")
                elif (victim_exit in (0, 124, 137) or victim.get("status") != "deliberate_abrupt_exit"
                      or not re.search(r"exitcode\s*:\s*73\b", (output / "victim.log").read_text())):
                    raise RuntimeError("Unexpected victim exit instead of deliberate exit73")
                observed = time.time()
                report["fault_observed_unix"] = observed
                write_json(output / "peer-release.json", {"schema": SCHEMA, "cell_id": control["cell_id"],
                    "job_id": control["jobs"]["peer"]["job_id"], "kind": "peer-release",
                    "victim_launcher_exit": victim_exit, "fault_observed_unix": observed})
                released = True
                persist()
            if released and peer_exit is not None:
                break
            time.sleep(.2)
        victim, peer = (read_json(output / role / "report.json") for role in ROLES)
        report["checks"] = validate_outcome(control, victim, peer,
            victim_exit=processes["victim"].returncode, peer_exit=processes["peer"].returncode,
            victim_log=(output / "victim.log").read_text(), fault_observed_unix=report["fault_observed_unix"])
        for role, child in processes.items():
            worker_report = read_json(output / role / "report.json")
            for receipt in worker_report["checkpoints"]:
                if sha256(output / role / receipt["path"]) != receipt["sha256"]:
                    raise ValueError("Retained checkpoint bytes differ")
            report["jobs"][role].update(exit_code=child.returncode, report_sha256=sha256(output / role / "report.json"),
                                        wandb=worker_report["wandb"], checkpoints=worker_report["checkpoints"])
        if source_pins() != control["sources"]:
            raise ValueError("Source files changed during isolation test")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        # Stop launchers first so cleanup cannot race a not-yet-created container.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        for role, child in processes.items():
            try:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=5)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=10)
            except BaseException as error:
                report["status"] = "failed"
                report.setdefault("launcher_cleanup_errors", []).append({"role": role, "error_type": type(error).__name__})
        cleanup = []
        for role in processes:
            try:
                cleanup.append(cleanup_owned(docker, control["jobs"][role]))
            except BaseException as error:
                report["status"] = "failed"
                cleanup.append({"role": role, "error_type": type(error).__name__, "error": str(error)})
        report["cleanup"] = cleanup
        report["finished_unix"] = time.time()
        persist()
        for stream in streams.values():
            stream.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if report["status"] != "completed":
        raise RuntimeError("Isolation cleanup did not complete")
    return report


def tree_equal(left, right):
    """Exact tensor/container comparison, used after CPU checkpoint readback."""
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def tree_finite(value):
    import torch
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(tree_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(tree_finite(v) for v in value)
    if isinstance(value, float):
        import math
        return math.isfinite(value)
    return True


def model_changed_between_checkpoints(before, after):
    """Counter or optimizer changes alone do not establish model progress."""
    first, last = before["model"], after["model"]
    if not first or first.keys() != last.keys():
        raise ValueError("Checkpoint model ownership differs")
    for name, value in first.items():
        other = last[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError("Checkpoint model tensor contract differs: " + name)
    return not tree_equal(first, last)


def worker(args):
    # GPU imports and all CUDA actions occur only after container verification.
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("GPU workers require the project container")
    if os.environ.get("WORLD_SIZE") != "1" or os.environ.get("LOCAL_RANK") != "0":
        raise RuntimeError("Each isolation job must own an independent one-rank process group")
    if any(os.environ.get(k) != "0" for k in ("NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING")):
        raise RuntimeError("Captured NCCL requires explicit async-error flags and external timeout")
    if sha256(args.control) != args.control_sha256:
        raise ValueError("Control bytes differ")
    control = read_json(args.control)
    job = control["jobs"][args.role]
    if source_pins() != control["sources"] or os.environ.get("TORCHELASTIC_RUN_ID") != job["rendezvous_id"]:
        raise ValueError("Source or rendezvous identity differs")
    import gc
    import math
    import socket
    from datetime import timedelta
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
    from cdrm.pretrained.campaign_training import CampaignObjective
    from cdrm.pretrained.distributed_training import sum_objective_counts
    from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
    from scripts.olmo_allocation_benchmark import construct_tiny, tiny_batches, release_completed_graph_runner
    from scripts.olmo_packed_campaign_run import configure_cuda_runtime
    from scripts.experiment_tracking import OnlineTracker
    from scripts.olmo_two_gpu_validate import preserve_local_rng
    device, runtime, determinism = configure_cuda_runtime(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly its assigned GPU")
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(seconds=control["timeout_seconds"]), device_id=device)
    cell = ROOT / control["output_relative"]
    output = cell / args.role
    output.mkdir(exist_ok=False)
    report = {"schema": SCHEMA, "status": "running", "role": args.role, "job_id": job["job_id"],
        "retention_identity": job["retention_identity"], "scenario": control["scenario"], "world_size": 1,
        "rendezvous_id": os.environ["TORCHELASTIC_RUN_ID"], "master_port": os.environ.get("MASTER_PORT"),
        "container_id": socket.gethostname(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sources": control["sources"], "runtime": runtime,
        "determinism": determinism, "updates": [], "checkpoints": [], "started_unix": time.time(),
        "scope": "Tiny FP32 ordinary campaign CUDA graph/Adam process isolation only"}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=output,
        group="olmo-topology-isolation", name=job["job_id"] + "-" + control["scenario"], preserve_state=preserve_local_rng)
    def persist():
        report["wandb"] = tracker.record
        write_json(output / "report.json", report)
    runner = model = optimizer = adapter = None
    try:
        tracker.start({"scope": report["scope"], "role": args.role, "scenario": control["scenario"], "job_id": job["job_id"]})
        persist()
        tiny = SimpleNamespace(arm="B", length=8, effective_rows=4)
        model, optimizer, recipe, _ = construct_tiny(tiny, device)
        initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
        def batches_for(index):
            return tiny_batches(recipe=recipe, logical_update=index, rank=0, world_size=1,
                batch_size=2, effective_rows=4, width=model.config.model_dim)[:2]
        batches, noises = batches_for(0)
        counts = sum_objective_counts([model.counts(batch) for batch in batches])
        adapter = CampaignObjective(model, batches[0], mode=recipe.mode(), global_counts=counts,
            world_size=1, feedback_noise=noises[0], config=LMTrainingConfig(precision="fp32", max_grad_norm=1.))
        runner = CampaignDDPGraphTraining(adapter)
        runner.capture(warmup=11)
        graph_ids = (id(runner.local_graph), id(runner.sync_graph))
        pointers = tuple(p.data_ptr() for p in model.parameters())
        counters = TrainingCounters()
        def save_checkpoint():
            owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            adam_steps_match = all(parameter in optimizer.state and
                optimizer.state[parameter]["step"].item() == counters.optimizer_updates for parameter in owned)
            if not owned or not adam_steps_match:
                raise ValueError("Adam ownership/step clocks differ from completed optimizer updates")
            payload = {"model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                       "optimizer": optimizer.state_dict(), "counters": asdict(counters),
                       "job_id": job["job_id"], "retention_identity": job["retention_identity"]}
            path = output / f"update-{counters.optimizer_updates:06d}.pt"
            if path.exists():
                raise ValueError("Refuse checkpoint overwrite")
            temporary = path.with_suffix(".tmp")
            with temporary.open("xb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            exact = tree_finite(restored) and tree_equal(payload, restored)
            if not exact:
                raise ValueError("CPU checkpoint readback differs")
            receipt = {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                "optimizer_updates": counters.optimizer_updates, "cpu_readback_exact": exact,
                "adam_steps_match": adam_steps_match,
                "retention_identity": job["retention_identity"] + "/" + path.stem}
            report["checkpoints"].append(receipt)
            write_json(output / (path.stem + "-receipt.json"), receipt)
            persist()
            return restored
        total = control["before_updates"] + (control["peer_after_updates"] if args.role == "peer" else 0)
        restored = before_fault_checkpoint = None
        for index in range(total):
            if index == control["before_updates"]:
                release = wait_marker(cell / "peer-release.json", control, role="peer", kind="peer-release")
                report["peer_release"] = release
            started = time.time()
            batches, noises = batches_for(index)
            result = runner.backward(batches, feedback_noises=noises, replay=True)
            metrics = runner.step(result, optimizer, counters=counters)
            torch.cuda.synchronize(device)
            finite = math.isfinite(metrics["objective"]) and math.isfinite(metrics["gradient_norm_before_clip"])
            same = graph_ids == (id(runner.local_graph), id(runner.sync_graph)) and pointers == tuple(p.data_ptr() for p in model.parameters())
            if not finite or not same:
                raise RuntimeError("Peer graph/update health failed")
            row = {"update": index + 1, "phase": "before_fault" if index < control["before_updates"] else "after_fault",
                "started_unix": started, "finished_unix": time.time(), "finite": finite,
                "same_graph_objects": same, "objective": metrics["objective"], "gradient_norm": metrics["gradient_norm_before_clip"],
                "counters": asdict(counters)}
            report["updates"].append(row)
            report["parameters_changed"] = any(not torch.equal(initial[name], value.detach().cpu())
                for name, value in model.named_parameters())
            persist()
            tracker.log({"update": index + 1, "train/objective": row["objective"],
                         "train/gradient_norm": row["gradient_norm"], "train/after_fault": row["phase"] == "after_fault"}, step=index + 1)
            if index + 1 == control["before_updates"]:
                restored = save_checkpoint()
                before_fault_checkpoint = restored
                write_json(output / "ready.json", {"schema": SCHEMA, "cell_id": control["cell_id"],
                    "job_id": job["job_id"], "kind": "ready", "optimizer_updates": counters.optimizer_updates,
                    "ready_unix": time.time()})
        if args.role == "victim":
            request = wait_marker(cell / "fault-request.json", control, role="victim", kind="fault-request")
            if request["scenario"] != control["scenario"]:
                raise ValueError("Fault scenario differs")
            report["fault_request"] = request
            if source_pins() != control["sources"]:
                raise ValueError("Source changed before intentional fault")
            if control["scenario"] == "abrupt":
                report.update(status="deliberate_abrupt_exit", intended_worker_exit=ABRUPT_EXIT)
                tracker.summary({"expected_failure": True, "worker_exit_code": ABRUPT_EXIT, "retained_update": counters.optimizer_updates})
                # Deliberately synchronize logging, then abort without graph,
                # process-group or Python finally cleanup. Local evidence is authoritative.
                tracker.finish(succeeded=False)
                persist()
                os._exit(ABRUPT_EXIT)
            report["status"] = "controlled_stopped"
        else:
            restored = save_checkpoint()
            report["post_fault_parameters_changed"] = model_changed_between_checkpoints(before_fault_checkpoint, restored)
            if not report["post_fault_parameters_changed"]:
                raise ValueError("Peer model did not change after the observed victim exit")
            fresh, fresh_optimizer, _, _ = construct_tiny(tiny, torch.device("cpu"))
            fresh.load_state_dict(restored["model"], strict=True)
            fresh_optimizer.load_state_dict(restored["optimizer"])
            restored_counters = TrainingCounters(**restored["counters"])
            report["fresh_restore_exact"] = (tree_equal(fresh.state_dict(), restored["model"])
                and tree_equal(fresh_optimizer.state_dict(), restored["optimizer"])
                and asdict(restored_counters) == asdict(counters))
            if not report["fresh_restore_exact"]:
                raise ValueError("Peer checkpoint fresh model/Adam restore differs")
            del fresh, fresh_optimizer, restored_counters
            report["status"] = "completed"
        if source_pins() != control["sources"]:
            raise ValueError("Source changed during test")
        tracker.summary({"optimizer_updates": counters.optimizer_updates, "completed": report["status"] == "completed",
                         "cooperative_stop": report["status"] == "controlled_stopped"})
        tracker.finish(succeeded=True)
        release_completed_graph_runner(runner)
        runner = adapter = optimizer = model = None
        gc.collect()
        dist.destroy_process_group()
        report["clean_teardown"] = True
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        persist()
        try:
            tracker.finish(succeeded=False)
        finally:
            persist()
        raise
    finally:
        report["finished_unix"] = time.time()
        persist()
    return report


def main(argv=None):
    args = parse_args(argv)
    return supervise(args) if args.mode == "supervise" else worker(args)


if __name__ == "__main__":
    main()
