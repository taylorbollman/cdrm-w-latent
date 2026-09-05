# Docker environment

The Dockerfile and pinned dependencies were preserved from the existing environment.
The default image is `cdrm-w-latent:dev`, based on `nvcr.io/nvidia/pytorch:26.06-py3`.

Build with `bash scripts/docker_build.sh`. Open a GPU shell with
`bash scripts/docker_shell.sh`, or a CPU shell with
`CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh`.
Append a command such as `bash -lc 'pwd'` for noninteractive use.

The project is mounted at `/workspace/cdrm-w-latent` and used as the working
directory. PYTHONPATH includes the project and all three vendor directories.
The launcher loads `.env`, mounts host Google Cloud configuration when present,
and persists the container home in `.docker-home`. Host caches remain at
`~/.cache`; mounted Local SSD caches use `/mnt/localssd/cdrm_runtime`.

Overrides: `CDRM_ROOT`, `CDRM_DOCKER_IMAGE`, `CDRM_BASE_IMAGE`,
`CDRM_DOCKER_GPUS`, `CDRM_CONTAINER_USER`, `CDRM_CONTAINER_HOME`,
`CDRM_ENV_FILE`, `CDRM_LOCALSSD_MOUNT`, `CDRM_LOCAL_RUNTIME_ROOT`.
CPU startup explicitly disables GPU passthrough. GPU requests do not fall back to CPU.

## Editable recurrent-transformer installation

Initialize `vendors/recurrent-transformer` with `git submodule update --init --recursive`
before building. The Dockerfile installs its dependencies from
`requirements-docker.txt`, then runs an editable installation with `--no-deps`
at `/workspace/cdrm-w-latent/vendors/recurrent-transformer`. Dependency resolution
is explicit in the requirements file; the source path matches the runtime mount.

Python 3.12.3 is retained. The package requires NumPy below 2, pinned here to
1.26.4. PyTorch and CUDA remain supplied by the NGC environment. Rebuild after
changing dependencies; ordinary Python source edits need only a new interpreter.

For the initial installation on this VM, the already populated image was tagged
`cdrm-w-latent:pre-recurrent` and reused with
`CDRM_BASE_IMAGE=cdrm-w-latent:pre-recurrent bash scripts/docker_build.sh`.
Fresh machines use the default NGC base and install the same pinned requirements.

## Verification on this VM

The editable package is `ai2-olmo==0.6.0`, sourced from the mounted submodule.
Python 3.12.3 and the original NGC PyTorch build are retained. The installed
requirements pass `pip check`, and the five upstream configuration tests pass:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh python -m pip check
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'cd vendors/recurrent-transformer && python -m pytest -q tests/config_test.py'
```

These are installation/configuration checks; GPU execution has not been tested.
