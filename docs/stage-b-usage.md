# Running and recovering the frozen Stage B pilot

The frozen plan is [configs/stage_b/pilot.json](../configs/stage_b/pilot.json).
It defines one paired SEQ/R3 seed for each of MQAR, noisy recall and ordered state
updates: six runs, 2,000 updates each, 12 blocks, width 256, FP32, and
`global_batch=microbatch=64`. R3 replaces block index 3 and uses rho=1 from
initialization. Labels already align with logits; only designated answers are
scored. See [the pilot plan](stage-b-plan.md),
[retrieval contracts](reports/stage-b/task-sources.md), and
[state-task definitions](stage-b-state-task.md).

This is an operational guide, not a statement that every phase has completed.
Actual status, commands and errors are retained in the execution ledgers and run
manifests. Existing artifacts must be preserved when repeating or recovering work.

## Container and CPU checks

Start commands from the host project directory:

```bash
cd /home/taylorbollman/cdrm-w-latent
```

On a fresh GPU instance, `bash /home/taylorbollman/start.sh` bootstraps the shared
infrastructure and opens the project container. After bootstrap, enter directly:

```bash
bash scripts/docker_shell.sh
```

For an individual GPU command, use the launcher and check the environment inside
it. A missing GPU is an error; the runner never falls back to CPU:

```bash
bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L'
```

The host path is bind-mounted at `/workspace/cdrm-w-latent`. Model training,
model evaluation and profiling belong inside this container. The controller
shown below may run on the host because it only launches container commands and
copies artifacts; it does not execute models on the host.

Run the task, runner-helper and report tests in an explicitly CPU-only container:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 python -m pytest -q tests/test_synthetic_retrieval.py tests/test_synthetic_state.py tests/test_stage_b_training.py tests/test_stage_b_report.py --tb=short'
```

These CPU tests do not replace the separate GPU backend and recovery gates.
Their scope and retained limitations are documented in
[backend-summary.md](reports/stage-b/backend-summary.md) and
[runner-validation.md](reports/stage-b/runner-validation.md). The frozen pilot
uses no gradient accumulation; optional R3 accumulation has not passed the full
parameter-equivalence check. BF16 is not the selected precision for this pilot.

## Freeze and archive the data

The original local run root is:

```text
.runtime/stage-b/20260906T190223Z/
```

For a clean execution before fixtures exist, the following commands generate
held-out fixtures, audit the complete training horizon, then materialize the
same audited training arrays:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 python scripts/stage_b_prepare.py --plan configs/stage_b/pilot.json --output-dir .runtime/stage-b/20260906T190223Z/fixtures'

CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 python scripts/stage_b_archive_training.py --plan configs/stage_b/pilot.json --fixtures-dir .runtime/stage-b/20260906T190223Z/fixtures'
```

These are creation commands, not rerun commands for an already populated run:
preparation refuses an existing output directory, and archival refuses existing
training archives. Reuse the retained fixtures for the existing pilot. For a new
experiment, choose a new run ID and GCS prefix, freeze a separate plan and create
new fixtures; changing a plan or fixture manifest changes checkpoint identity.

Each task retains 128,000 training examples, 1,024 dev examples per condition,
4,096 test examples per condition, calibration arrays, and JSON metadata and
manifests. Training arrays are consumed in the exact archived order by both
architectures. The archive pass checks the original training-stream digest,
input-only split separation, and complete retrieval-mapping separation. It
preserves intentional semantic reuse between delay conditions within one split.
The state task shares fixed operation permutations across splits; its holdout is
an ordered operation pattern, not a holdout of all resulting transition functions.

## Execute the three phases

For a fresh execution with prepared fixtures, run these **host controller**
commands in order. Do not overlap them or launch another model job on the same GPU:

```bash
python3 scripts/stage_b_execute.py \
  --plan configs/stage_b/pilot.json \
  --run-root .runtime/stage-b/20260906T190223Z \
  --phase calibrate

python3 scripts/stage_b_execute.py \
  --plan configs/stage_b/pilot.json \
  --run-root .runtime/stage-b/20260906T190223Z \
  --phase train

python3 scripts/stage_b_execute.py \
  --plan configs/stage_b/pilot.json \
  --run-root .runtime/stage-b/20260906T190223Z \
  --phase evaluate
```

Calibration uses a separate 100-update schedule and repeated fixed batch. Its
checkpoints are labeled OPS. The controller stops if the fixed-batch CE has not
halved for any arm. Inspect this calibration before proceeding; its learning
curve is not held-out SYN evidence and its weights do not initialize SYN runs.

Training starts each arm from corresponding random SEQ weights, converted
exhaustively for R3, with a fresh AdamW optimizer. Every 200 updates it evaluates
all dev conditions. Best-development selection uses mean answer CE over the
plan's primary conditions: MQAR `iid`, noisy recall `low` and `moderate`, and
state tracking `iid`. The 2,000-update final checkpoint remains the primary
endpoint. Evaluation uses separate test fixtures and retains predictions for
every scored answer; test evaluation rejects a checkpoint before the full horizon.

The controller writes `execution/{calibrate,train,evaluate}.json` and per-job logs.
It refuses to rerun a phase that already has a ledger, including a stopped phase;
recover individual jobs explicitly. Its `execution/budget.json` starts a shared
7,200-second wall window at the first phase, covering calibration, training,
evaluation, uploads and elapsed time between invocations. It is not reset by
starting the next phase. Each runner also enforces the plan's 3,600-second limit
using its accumulated training-run elapsed counter. A budget stop is recorded as
incomplete, not as a completed endpoint.

## Checkpoints and exact resume

A training directory such as
`runs/SYN-mqar-R3-seed0/` retains the following checkpoint roles:

| Role | Behavior |
| --- | --- |
| `SYN-mqar-R3-seed0-init.pt` | Initial paired weights and optimizer state; preserved |
| `SYN-mqar-R3-seed0-bestdev.pt` | Replaced when the declared development CE improves |
| `SYN-mqar-R3-seed0-latest.pt` | Rolling recovery checkpoint, updated every 200 updates and at clean exit |
| `SYN-mqar-R3-seed0-uNNNN.pt` | Preserved checkpoint at an explicitly stopped or interrupted completed-update boundary |
| `SYN-mqar-R3-seed0-final.pt` | Preserved checkpoint after all 2,000 updates |

The same naming applies to other tasks and SEQ. Resolved configuration, a run
manifest with checkpoint hashes, `learning-curve.jsonl`, and dev metrics are kept
beside these files. Initialization, final and numbered boundary checkpoints
refuse overwrite; `bestdev`, `latest` and the evolving manifest are intentionally
updated within their owning trajectory.

`--resume` restores model and Adam state, Python/NumPy/Torch/CUDA RNG state,
completed-update and token counters, the next-batch digest, recurrence rho, and
development-selection state. It verifies the exact plan, settings, task,
topology, fixture-manifest hash, source hashes and model configuration. Preserve
the training source snapshot and the fixture manifest bytes; even a source edit
that appears numerically harmless can invalidate exact resume.

For a gracefully paused R3 MQAR run whose `latest` checkpoint has completed at
least one update and fewer than 2,000, the direct GPU-container command is:

```bash
bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/stage_b_train.py --plan configs/stage_b/pilot.json --task mqar --topology r3 --fixtures-dir .runtime/stage-b/20260906T190223Z/fixtures --output-dir .runtime/stage-b/20260906T190223Z/runs/SYN-mqar-R3-seed0 --resume .runtime/stage-b/20260906T190223Z/runs/SYN-mqar-R3-seed0/SYN-mqar-R3-seed0-latest.pt --stop 2000'
```

`--stop` is the **total completed update count**, not additional updates, and does
not shorten the 2,000-update learning-rate schedule. Same-directory resume
requires the same run owner and no learning-curve entries later than the selected
checkpoint. If an older checkpoint must be recovered, use a new empty output
directory and retain its lineage rather than deleting later history. A no-op
resume at the requested stop is rejected. Reaching the accumulated per-run wall
limit does not gain a fresh budget through resume.

Compiled R3 **resume from update zero is explicitly disabled**: that path did not
pass bitwise validation. Start a fresh paired R3 run instead, or resume after
completed updates. The retained midpoint recovery validation is for a small
fixture, not a claim that every full-size replay will be bitwise identical.

SIGINT or SIGTERM asks the runner to stop at the next complete optimizer-update
boundary and save recovery state. An abrupt process or machine loss can only
recover from a checkpoint already written. Direct runner commands do not upload
automatically; synchronize recovered artifacts as described below.

## Final evaluation and reporting

The `evaluate` controller phase evaluates all six final checkpoints. For one
completed arm, the equivalent direct command is below; the output directory
must not already contain these predictions or its evaluation manifest:

```bash
bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/stage_b_train.py --plan configs/stage_b/pilot.json --task mqar --topology r3 --fixtures-dir .runtime/stage-b/20260906T190223Z/fixtures --output-dir .runtime/stage-b/20260906T190223Z/runs/SYN-mqar-R3-seed0/evaluation-final --evaluate-only .runtime/stage-b/20260906T190223Z/runs/SYN-mqar-R3-seed0/SYN-mqar-R3-seed0-final.pt --split test'
```

After all six endpoints and final evaluations exist, build the report in the
CPU container. This checks paired batches/LRs, checkpoint hashes, fixture labels,
prediction ordering and stored metrics before writing its tables and plots:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 python scripts/stage_b_report.py --plan configs/stage_b/pilot.json --run-root .runtime/stage-b/20260906T190223Z --output-dir .runtime/stage-b/20260906T190223Z/report'
```

The report refuses a nonempty destination. Its paired bootstrap intervals describe
variation over test examples for this trained seed; they do not include
training-seed uncertainty. Report generation does not itself verify or perform
GCS upload.

## Local persistence, GCS retention and restore

The requested archive prefix is:

```text
gs://fast-chunks/cdrm-w-latent/stage-b/20260906T190223Z/
```

| Suffix | Contents |
| --- | --- |
| `plan/pilot.json` | Exact frozen plan |
| `source/` | Source archives, revisions and per-file/archive hashes |
| `fixtures/` | Training/dev/test/calibration arrays, metadata and audits |
| `calibration/OPS-{task}-{SEQ\|R3}-seed0/` | Separate fixed-batch operational runs |
| `runs/SYN-{task}-{SEQ\|R3}-seed0/` | Research checkpoints, curves and final evaluation artifacts |
| `execution/` | Phase ledgers, budget, concrete commands and job/upload logs |

`.runtime` is gitignored but resides under the project's persistent home/root
filesystem, bind-mounted into the container. Removing a container does not remove
these files. The local SSD paths used for package/model/Triton caches are
**disposable** and are not the checkpoint archive. Local persistence does not
replace cloud retention when a machine or disk is discarded.

The controller uploads each completed run and synchronizes execution logs at
phase exit. It does not continuously upload a running job, and a stopped job can
leave its recovery checkpoint only on local storage. To archive that specific
run from the host, use its existing dedicated destination:

```bash
gcloud --quiet storage rsync --recursive \
  .runtime/stage-b/20260906T190223Z/runs/SYN-mqar-R3-seed0 \
  gs://fast-chunks/cdrm-w-latent/stage-b/20260906T190223Z/runs/SYN-mqar-R3-seed0
```

For a fresh restored workspace, recover fixtures at the same project-relative
path, since their manifest entries retain those paths:

```bash
gcloud --quiet storage rsync --recursive \
  gs://fast-chunks/cdrm-w-latent/stage-b/20260906T190223Z/fixtures \
  .runtime/stage-b/20260906T190223Z/fixtures
```

Likewise restore the required run directory and the exact source snapshot
recorded by its checkpoint. Use a fresh checkout/destination rather than copying
a cloud checkpoint over a newer local recovery file. The original training
source archive is `source/training-source-v2.tar.gz`, with hashes in the matching
JSON sidecar. Restore the original container software environment as well as
source bytes; the source archive alone is not a Docker image or dependency cache.
The runner checks local plan/source/fixture identities when loading a checkpoint.

[storage.json](reports/stage-b/storage.json) records actual upload status and
verification; a URI in a plan or report by itself does not establish successful
retention. Credentials, environment files and unrelated workspace files are not
part of the source or experiment archives.
