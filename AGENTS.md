# Workspace

Use `/home/taylorbollman/cdrm-w-latent` as the active project. The old projects
are unrelated archives and must not be searched unless explicitly requested.
Do not print credentials or environment-file values.

# Container execution

Never run CUDA, training, GPU evaluation or profiling in the host shell and
never silently fall back to CPU. Bootstrap GPU instances with
`bash /home/taylorbollman/start.sh`; CPU instances with Local SSD use
`bash /home/taylorbollman/start_cpu.sh`.

After bootstrap, use `bash /home/taylorbollman/cdrm-w-latent/scripts/docker_shell.sh`.
For a single command, append `bash -lc '<command>'`. The container working
directory must be `/workspace/cdrm-w-latent`. Before GPU work, verify that the
command is inside the container and that `nvidia-smi` succeeds there.
Use `CDRM_DOCKER_GPUS=none` explicitly for CPU container work.

Add dependencies to `docker/requirements-docker.txt` and rebuild with
`scripts/docker_build.sh`. Keep `.env` and `.docker-home` out of Git.

# Experiment tracking

For future training and evaluation runs with graphable metrics, log online to
Weights & Biases under the `taylorbollman` entity. The user authorizes creating
appropriately named projects and runs there. Use the credentials in `.env`
without printing them, and include the project or run URLs in progress updates
and results. Keep the existing local records and GCS artifact retention alongside
W&B tracking.

# RT numerical handoff

For future changes to the base Recurrent Transformer or its numerical tests,
read `docs/rt-numerical-handoff.md` first. It records the completed precision
baseline, reusable validation methods, harness adaptation constraints and
retained artifacts. The historical precision lineage is closed; future
experiments should have new output directories and an explicit scope.

# A5 experiment handoff

For the two-layer A5 experiment, read `docs/rt-a5-handoff.md` and
`docs/rt-a5-usage.md`. The first paired 10,000-update development pilot is
complete. The user chose full FP32 and minimal bounded correctness checks;
do not automatically restart mixed-precision or compiler studies. Final
confirmation remains unevaluated.
