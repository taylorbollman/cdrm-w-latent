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

`recurrent-transformer` is a Git submodule pointing to
[taylorbollman/recurrent-transformer](https://github.com/taylorbollman/recurrent-transformer),
a fork of [geniucos/recurrent-transformer](https://github.com/geniucos/recurrent-transformer).
The Docker image installs its `ai2-olmo` package in editable mode. Changes to
its Python sources are visible in new Python processes without reinstalling.
The original Apache-2.0 license remains in the submodule. The pinned starting
revision is `a21b42d2bc292edb86ed1b62cee4bcab809a9d21`; the fork is developed
as a top-level component while supporting dependencies remain in `vendors/`.

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

GPU startup reuses an existing image and mounted Local SSD; rebuild explicitly
with `bash scripts/docker_build.sh` when changing dependencies or install paths.
For an already bootstrapped machine, use `bash scripts/docker_shell.sh` directly.

Local SSD runtime: `/mnt/localssd/cdrm_runtime`.
The remote tunnel retains its existing `roadmap-vm2` default name.

See [Docker details](docker/README.md), [SSD setup](scripts/gcp/README.md),
and [migration notes](MIGRATION.md).

Project inputs are preserved under [docs/inputs](docs/inputs/README.md). The initial
run protocol is provisional; recorded NUM/OPS results live under
[docs/reports/stage-a](docs/reports/stage-a/).

The first R3 implementation, conversion API, tests, and bounded GPU commands are
documented in [Stage A usage](docs/stage-a-usage.md). Scientific and numerical
choices are recorded in [the decision log](docs/semantic-decisions.md).
See [Stage A results](docs/reports/stage-a/results.md) for validated behavior,
measured H100 performance, retained failures, and the next milestone.

The completed Stage B synthetic pilot pairs SEQ and R3 on associative recall,
noisy recall, and ordered state updates. See [the results](docs/reports/stage-b/results.md),
its [frozen plan](docs/stage-b-plan.md),
[task definition](docs/stage-b-state-task.md), and
[execution and recovery guide](docs/stage-b-usage.md). Checkpoints and fixed data
live in the persistent project `.runtime` directory and are archived under
`gs://fast-chunks/cdrm-w-latent/stage-b/20260906T190223Z/`; local SSD caches are
disposable. Actual cloud retention is recorded in
[the storage manifest](docs/reports/stage-b/storage.json).

The subsequent [R3 backward investigation](docs/reports/r3-backward/results.md)
clears the tested FP32/no-accumulation MQAR path at D256/T128/B2 and B64,
with classified floating-point and first-step Adam discrepancies retained.
No backward or trainer patch was needed. Its diagnostics and FP64 references
are archived separately under `gs://fast-chunks/cdrm-w-latent/r3-backward/20260906T210249Z/`.

The [BF16 mixed-precision milestone](docs/reports/r3-bf16/results.md) adds an
opt-in FP32-state recurrent policy and [reproducible commands](docs/r3-bf16-usage.md).
Paired training and midpoint recovery work, but numerical maximum-error flags
remain; FP32 stays the default. At B64/T128, BF16 lowers memory and slows updates.
