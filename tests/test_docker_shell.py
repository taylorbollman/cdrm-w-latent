"""Launcher selection tests use a fake Docker CLI and never open a container."""
import json
import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/docker_shell.sh"
PROJECT = "/workspace/cdrm-w-latent"


def launch(tmp_path, selection=None, *, exit_code=0):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['TEST_DOCKER_CALLS'], 'a') as file:
    file.write(json.dumps(sys.argv[1:]) + '\\n')
sys.exit(int(os.environ['TEST_RUN_EXIT']) if sys.argv[1] == 'run' else 0)
""")
    docker.chmod(0o755)
    findmnt = binaries / "findmnt"
    findmnt.write_text("#!/bin/sh\nexit 1\n")
    findmnt.chmod(0o755)
    env = dict(os.environ)
    env.pop("CDRM_FLASH_ATTENTION_SOURCE", None)
    env.update(PATH=f"{binaries}:{env['PATH']}", CDRM_ROOT=str(tmp_path),
        CDRM_DOCKER_IMAGE="fixture-image", CDRM_DOCKER_GPUS="none",
        CDRM_ENV_FILE=str(tmp_path / "missing.env"),
        CDRM_CONTAINER_HOME=str(tmp_path / "container-home"),
        XDG_CACHE_HOME=str(tmp_path / "cache"), CLOUDSDK_CONFIG=str(tmp_path / "missing-gcloud"),
        TEST_DOCKER_CALLS=str(tmp_path / "calls.jsonl"), TEST_RUN_EXIT=str(exit_code))
    if selection is not None:
        env["CDRM_FLASH_ATTENTION_SOURCE"] = selection
    result = subprocess.run(["bash", str(LAUNCHER), "bash", "-lc", "echo fixture"],
        env=env, capture_output=True, text=True)
    calls = tmp_path / "calls.jsonl"
    return result, [] if not calls.exists() else [json.loads(row) for row in calls.read_text().splitlines()]


@pytest.mark.parametrize("selection", [None, "", "vendor", "installed"])
def test_only_selected_flash_vendor_path_changes(tmp_path, selection):
    result, calls = launch(tmp_path, selection)
    assert result.returncode == 0, result.stderr
    argv = next(args for args in calls if args[0] == "run")
    mode = selection or "vendor"
    environment = [argv[i+1] for i, arg in enumerate(argv) if arg == "-e"]
    paths = [PROJECT, PROJECT + "/vendors/apex"]
    if mode == "vendor":
        paths.append(PROJECT + "/vendors/flash-attention")
    paths.append(PROJECT + "/vendors/vllm")
    assert "PYTHONPATH=" + ":".join(paths) in environment
    assert "CDRM_FLASH_ATTENTION_SOURCE=" + mode in environment
    assert "PYTHONNOUSERSITE=1" in environment
    assert "--gpus" not in argv
    assert argv[-4:] == ["fixture-image", "bash", "-lc", "echo fixture"]


@pytest.mark.parametrize("selection", ["wheel", "Installed", "vendor:installed"])
def test_invalid_selection_fails_before_docker(tmp_path, selection):
    result, calls = launch(tmp_path, selection)
    assert result.returncode != 0
    assert "expected vendor or installed" in result.stderr
    assert calls == []


def test_launcher_preserves_child_exit_status(tmp_path):
    result, calls = launch(tmp_path, "installed", exit_code=37)
    assert result.returncode == 37
    assert any(args[0] == "run" for args in calls)
