# cdrm-w-latent

Research workspace with a persistent Docker environment and an editable local
checkout of Recurrent OLMo.

## Clone and open the environment

```bash
git clone --recurse-submodules https://github.com/taylorbollman/cdrm-w-latent.git
cd cdrm-w-latent
bash scripts/docker_build.sh
bash scripts/docker_shell.sh
```

The project is mounted at `/workspace/cdrm-w-latent`. The existing environment
uses Python 3.12; no additional Conda environment is needed. Configure local
credentials in `.env` using `.env.example`; `.env` and `.docker-home` are ignored
by Git and excluded from the Docker build context.

For an explicit CPU-only shell:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh
```

## Recurrent transformer development

`vendors/recurrent-transformer` is a Git submodule pointing to
[taylorbollman/recurrent-transformer](https://github.com/taylorbollman/recurrent-transformer),
a fork of [geniucos/recurrent-transformer](https://github.com/geniucos/recurrent-transformer).
The Docker image installs its `ai2-olmo` package in editable mode. Changes to
its Python sources are visible in new Python processes without reinstalling.
The original Apache-2.0 license remains in the submodule.

The submodule's `origin` points to the personal fork. Work is independent of
the upstream authors; publish changes to the personal fork, then commit the
updated submodule pointer in this repository. No upstream push remote is
configured in this workspace.

After cloning without `--recurse-submodules`, initialize it before building:

```bash
git submodule update --init --recursive
```

## Existing VM entry points

The home commands delegate to scripts in this repository:

```bash
bash /home/taylorbollman/start.sh       # GPU instance bootstrap and shell
bash /home/taylorbollman/start_cpu.sh   # CPU + Local SSD bootstrap and shell
bash /home/taylorbollman/remote_tunnel.sh
```

Local SSD runtime: `/mnt/localssd/cdrm_runtime`.
The remote tunnel retains its existing `roadmap-vm2` default name.

See [Docker details](docker/README.md), [SSD setup](scripts/gcp/README.md),
and [migration notes](MIGRATION.md).
