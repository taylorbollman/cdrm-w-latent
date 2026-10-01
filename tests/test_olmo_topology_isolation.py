"""Isolation contracts and owned cleanup, without launching CUDA or Docker."""
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts import olmo_topology_isolation as isolation


@pytest.fixture
def control(monkeypatch):
    monkeypatch.setattr(isolation, "source_pins", lambda: {"scripts/isolation.py": "a" * 64})
    return isolation.make_control(isolation.ROOT / ".runtime/test-isolation-not-created", "controlled", 600,
                                  identity="0123456789abcdef" * 2)


def worker_report(control, role):
    job = control["jobs"][role]
    rows = [{"update": i + 1, "finite": True, "same_graph_objects": True,
             "started_unix": i + 1 if i < 2 else i + 11,
             "finished_unix": i + 1.5 if i < 2 else i + 11.5}
            for i in range(2 if role == "victim" else 6)]
    return {"schema": isolation.SCHEMA, "job_id": job["job_id"], "world_size": 1,
            "sources": control["sources"], "rendezvous_id": job["rendezvous_id"],
            "visible_devices": job["visible_devices"], "retention_identity": job["retention_identity"],
            "container_id": role + "-container", "updates": rows, "parameters_changed": True,
            "checkpoints": [{"cpu_readback_exact": True, "adam_steps_match": True,
                             "optimizer_updates": 2 if role == "victim" else 6}],
            "status": "controlled_stopped" if role == "victim" else "completed", "fresh_restore_exact": True,
            "post_fault_parameters_changed": True,
            "clean_teardown": True, "wandb": {"run_id": role + "-run"}}


def validate(control, victim=None, peer=None, **kwargs):
    return isolation.validate_outcome(control, victim or worker_report(control, "victim"),
        peer or worker_report(control, "peer"), victim_exit=kwargs.get("victim_exit", 0), peer_exit=kwargs.get("peer_exit", 0),
        victim_log=kwargs.get("victim_log", ""), fault_observed_unix=10.)


def test_independent_command_device_rendezvous_output_and_bound(control):
    jobs = control["jobs"]
    assert jobs["peer"]["visible_devices"] != jobs["victim"]["visible_devices"]
    assert jobs["peer"]["retention_identity"] != jobs["victim"]["retention_identity"]
    assert jobs["peer"]["rendezvous_id"] != jobs["victim"]["rendezvous_id"]
    commands = {role: isolation.worker_command(control, "b" * 64, role) for role in isolation.ROLES}
    for role, command in commands.items():
        assert command[:2] == ["bash", "scripts/docker_shell.sh"]
        assert "timeout" in command and "600s" in command
        script = command[-1]
        assert "test -f /.dockerenv" in script and "nvidia-smi" in script
        assert "CUDA_VISIBLE_DEVICES=" + jobs[role]["visible_devices"] in script
        assert "--rdzv-id=" + jobs[role]["rendezvous_id"] in script
        assert "--max-restarts=0" in script
        assert "--role " + role in script


@pytest.mark.parametrize("timeout", [0, -1, 119, 901, 999999])
def test_supervisor_rejects_unbounded_timeout(timeout):
    with pytest.raises(SystemExit):
        isolation.parse_args(["supervise", "--scenario", "abrupt", "--output-dir",
            str(isolation.ROOT / ".runtime/test-isolation-not-created"), "--timeout-seconds", str(timeout)])


def test_docker_shim_labels_only_run_and_preserves_argument_boundaries(tmp_path, control):
    fake = tmp_path / "fake docker"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    fake.chmod(0o700)
    shim = tmp_path / "docker"
    job = control["jobs"]["peer"]
    shim.write_text(isolation.docker_shim([str(fake)], job))
    shim.chmod(0o700)
    run = subprocess.run([str(shim), "run", "--rm", "image", "echo", "two words"], text=True, capture_output=True, check=True)
    assert run.stdout.splitlines() == ["run", "--name", job["container_name"], "--label",
        isolation.OWNER_LABEL + "=" + job["job_id"], "--rm", "image", "echo", "two words"]
    info = subprocess.run([str(shim), "image", "inspect", "image"], text=True, capture_output=True, check=True)
    assert info.stdout.splitlines() == ["image", "inspect", "image"]


def test_owned_cleanup_refuses_another_job(monkeypatch, control):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='{"cdrm.topology-isolation-job":"another-job"}')
    monkeypatch.setattr(isolation.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="another ownership"):
        isolation.cleanup_owned(["docker"], control["jobs"]["peer"])
    assert len(calls) == 1 and "rm" not in calls[0]


def test_owned_cleanup_removes_only_matching_container_and_checks_absence(monkeypatch, control):
    calls = []
    job = control["jobs"]["peer"]
    def run(command, **kwargs):
        calls.append(command)
        text = '{"cdrm.topology-isolation-job":"' + job["job_id"] + '"}' if "inspect" in command else ""
        return SimpleNamespace(returncode=0, stdout=text)
    monkeypatch.setattr(isolation.subprocess, "run", run)
    assert isolation.cleanup_owned(["docker"], job)["owned_containers_remaining"] == 0
    assert calls[1] == ["docker", "rm", "--force", job["container_name"]]
    assert calls[2][-1] == "label=" + isolation.OWNER_LABEL + "=" + job["job_id"]


def test_cooperative_and_abrupt_outcomes_distinguish_worker_and_launcher_exit(control):
    assert validate(control)["peer_updates_after_failure"] == 4
    abrupt = copy.deepcopy(control)
    abrupt["scenario"] = "abrupt"
    victim = worker_report(abrupt, "victim")
    victim.update(status="deliberate_abrupt_exit", intended_worker_exit=73, clean_teardown=False)
    result = validate(abrupt, victim=victim, victim_exit=1, victim_log="exitcode  : 73 (pid: 123)")
    assert result["victim_launcher_exit_code"] == 1 and result["victim_worker_exit_code"] == 73
    with pytest.raises(ValueError, match="intended abrupt"):
        validate(abrupt, victim=victim, victim_exit=124, victim_log="exitcode  : 73")
    with pytest.raises(ValueError, match="intended abrupt"):
        validate(abrupt, victim=victim, victim_exit=1, victim_log="exitcode : 1")


@pytest.mark.parametrize("change", [
    lambda r: r["updates"][2].update(started_unix=9.),
    lambda r: r["updates"][1].update(finished_unix=11.),
    lambda r: r["updates"][3].update(same_graph_objects=False),
    lambda r: r["updates"][3].update(finite=False),
    lambda r: r.update(fresh_restore_exact=False),
    lambda r: r.update(post_fault_parameters_changed=False),
    lambda r: r.update(parameters_changed=False),
    lambda r: r.update(clean_teardown=False),
    lambda r: r["checkpoints"][0].update(cpu_readback_exact=False),
    lambda r: r["checkpoints"][0].update(adam_steps_match=False),
])
def test_peer_must_make_healthy_progress_on_same_graphs_on_both_sides_of_exit(control, change):
    peer = worker_report(control, "peer")
    change(peer)
    with pytest.raises(ValueError):
        validate(control, peer=peer)


def test_control_files_cannot_cross_release_jobs(control):
    marker = {"schema": isolation.SCHEMA, "cell_id": control["cell_id"],
              "job_id": control["jobs"]["victim"]["job_id"], "kind": "peer-release"}
    with pytest.raises(ValueError, match="job identity"):
        isolation.check_marker(marker, control, role="peer", kind="peer-release")
    marker["job_id"] = control["jobs"]["peer"]["job_id"]
    assert isolation.check_marker(marker, control, role="peer", kind="peer-release") == marker


def test_checkpoint_tree_check_catches_moment_corruption_and_nonfinite_values():
    original = {"model": {"weight": torch.tensor([1., 2.])},
                "adam": {0: {"step": torch.tensor(6.), "exp_avg": torch.tensor([.1, .2])}},
                "counters": {"optimizer_updates": 6}}
    restored = copy.deepcopy(original)
    assert isolation.tree_equal(original, restored) and isolation.tree_finite(restored)
    restored["adam"][0]["exp_avg"][1] += 1
    assert not isolation.tree_equal(original, restored)
    restored["adam"][0]["exp_avg"][1] = float("inf")
    assert not isolation.tree_finite(restored)


def test_post_fault_progress_requires_changed_model_not_only_advanced_counter_or_adam():
    before = {"model": {"weight": torch.tensor([1., 2.])},
              "optimizer": {"step": 2}, "counters": {"optimizer_updates": 2}}
    after = copy.deepcopy(before)
    after["optimizer"]["step"] = 6
    after["counters"]["optimizer_updates"] = 6
    assert not isolation.model_changed_between_checkpoints(before, after)
    after["model"]["weight"][1] += .01
    assert isolation.model_changed_between_checkpoints(before, after)
    after["model"]["weight"] = torch.zeros(3)
    with pytest.raises(ValueError, match="tensor contract"):
        isolation.model_changed_between_checkpoints(before, after)
