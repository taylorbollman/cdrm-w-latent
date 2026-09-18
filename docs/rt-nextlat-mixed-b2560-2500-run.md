# Fresh mixed baseline: batch 2,560 per task, 2,500 updates

**Completed:** the initial run stopped at 2,500 and verified GCS retention
at 23:39 UTC on 2026-09-17. Full A5 L36 whole-word accuracy was 0% and
Fuzzy answer accuracy 46.9152%. The user then authorized an exact continuation
to **5,000 total updates**; see [the active continuation](rt-nextlat-mixed-b2560-5000-run.md).
The original protocol below describes the completed first stage.

Authorized and launched on 2026-09-17. The user selected 2,560 examples per
data source to leave room for later model changes, and reduced the initial
learning budget to **2,500 optimizer updates, then stop for review**. This
supersedes the earlier proposed 10k larger-batch baseline. No additional
training, harder-task calibration or embedding variants are queued.

Supervisor PID at launch: **146761**, 22:14 UTC. Read the runtime status and
training report below for current progress; a launch is not a completed run.
Startup was observed past update 25 with finite losses/gradient norms and
approximately 1.97 seconds per update. An independent audit confirmed the
training contract differs from the original only in logical and physical
batch sizes, with identical initialization metadata and source closure.
[Live W&B run](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/fuq95voo).

## Unchanged model and objective

Fresh initialization, paired with the original mixed baseline: seeds 1234
(backbone), 1235 (NextLat predictor), 1236 (Fuzzy vocabulary rows). There is
no trained parent checkpoint or warm start. Two tiled RT blocks, D128/H16/
FFN512, ALiBi, Mitchell initialization, window two at index 0 and full RT at
index 1, NextLat weight one, no embedding propagation and no latent rollout.
There are 479,616 trainable parameters.

Keep full FP32 eager execution, no TF32/autocast/compile/CUDA graphs, and
AdamW LR 1e-4, betas (0.9, 0.95), epsilon 1e-8, matrix decay .01 and global
gradient clip one. The model configuration and complete trainer source
closure match the original mixed baseline. The tested 3,072-per-task shape
provides capacity evidence; no new precision study was required.

Each optimizer update processes 2,560 A5 length-12 examples and 2,560 Fuzzy
length-400 examples in separate physical batches. Microbatch equals batch
per task: 2,560. Accumulate half of each task's mean CE-plus-NextLat loss,
then clip once and update Adam once. No cross-task padding/concatenation or
state carried between examples. Native Fuzzy dense training supervision,
masked teacher-forced evaluation, disjoint symbol ranges and task-local
output distributions remain unchanged.

## Data, budget and evaluations

Reuse the original 800,000-word A5 training set and 12,800-example Fuzzy
training set, with order seeds 5432 and 2026091604. The sampler carries
epoch remainders deterministically. Fuzzy traverses an epoch every five
updates; A5 every 312.5 updates. Presentations are repeated visits to these
fixed corpora, not newly generated unique examples.

| New update | Presentations per task | Nominal exposure at batch 128/task |
| ---: | ---: | ---: |
| 500 | 1,280,000 | 10,000 updates |
| 750 | 1,920,000 | 15,000 updates |
| 1,000 | 2,560,000 | 20,000 updates |
| 2,500 | 6,400,000 | 50,000 updates |

The previous mixed run actually stopped at 19,810 updates with 2,535,680
presentations per task; its final checkpoint is slightly short of the nominal
20k exposure above. Larger-batch results are a different optimization regime.
Same exposure does not imply the same optimizer trajectory or learning speed.

Log training metrics every 25 updates and evaluate every 100. Also retain
explicit checkpoints at 0, 250, 500, 750, 1,000, 1,500, 2,000 and 2,500;
every evaluated checkpoint is saved too. Checkpoint zero records exact
initialization, without a separate zero-step evaluation. Routine A5 monitoring
uses fixed 4,096-word development subsets at lengths 12 and 36. Full A5
evaluations use 102,400 words per role at 500, 750, 1,000, 1,500, 2,000 and
2,500, and confirm an earlier positive subset observation if one occurs.
Fuzzy evaluations use all 1,280 existing development examples.

The final saved checkpoint is evaluated on both tasks and reported together.
Confirmation stays unused. The historical positive-A5 annotation is descriptive
here and cannot extend the authorized budget. **Stop at 2,500 regardless of
accuracy**, then review. A zero L36 score at this short larger-batch budget is
not proof that the architecture cannot learn the mixed task.

Expected runtime is roughly 85–90 minutes including evaluation, based on
nearby measured batch sizes; replace this estimate with observed timing.
Old batch-128 runs remain historical references. They are deliberately omitted
from the reporter's strict same-batch control interface. No cross-batch control
validation is bypassed or mislabeled as a matched comparison.

## Execution and retention

Runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260917T221257Z-d128-b2560-mixed-2500/`.

- `execution-config.json`: exact CLI, fixed endpoint and task counts.
- `ready.json`: bound source/config hashes and independent sampler/report audit.
- `launch.json`, `status.json`, `supervisor.log`: launch and current phase.
- `train-mixed/`: snapshots, history, metrics and resumable checkpoints.
- `STOP`: request a clean early stop, checkpoint and full endpoint evaluation.
- `STOP_QUEUE`: cancellation before launch only; use `STOP` during training.

All training runs through the required GPU Docker container. The supervisor
then runs the existing reporter without single-task control arguments, logs
report figures to W&B, and verifies final GCS retention. Graphable evidence
uses `taylorbollman/rt-nextlat-fuzzy-a5`. Use the logged `update` metric for
optimizer-count comparisons rather than W&B's default logging-event index.

Report destination:
`docs/reports/rt-nextlat-fuzzy-a5/d128-b2560-mixed-2500/`.
Retention prefix:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260917T221257Z-d128-b2560-mixed-2500/`.
`retention/final-receipt.json` will bind the saved endpoint to the verified
archive. Checkpoints retain optimizer, RNG and task-stream cursors for a later
explicitly requested continuation into a fresh output directory.
