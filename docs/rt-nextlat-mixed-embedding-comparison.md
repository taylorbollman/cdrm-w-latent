# Mixed A5/Fuzzy embedding comparison: three fresh 15k runs

**Revised execution order:** the user subsequently requested **value → head
→ input**. The early input run was cleanly stopped and retained at 452; it
will resume last to 15k. Read the
[reordered queue handoff](rt-nextlat-mixed-embedding-reordered.md) for current
supervisors and paths. The model and comparison protocol below is unchanged;
the original launch/order notes are historical.

Authorized on 2026-09-18: after the batch-2,560 baseline finishes at 15,000
total optimizer updates and its results are summarized, run all three
previously discussed embedding routes sequentially. Each variant starts
from the paired original initialization, trains **15,000 updates**, and is
summarized and retained before the next starts. There is no performance gate
or automatic budget extension. The user does not require review between arms.

**Launch:** the detached queue supervisor started at 07:33:12 UTC on
2026-09-18, PID **342639**, initially waiting for the baseline's 15k summary
and retention. All **42 focused CPU tests** and the independent CLI/queue
audit passed. GPU preflight remains an automatic requirement before each arm.
Read the runtime `status.json` for current progress and each training
`report.json` for its W&B URL; this launch note is not a result claim.

**Baseline and first transition verified:** the baseline completed all 15k
updates, authoritative reporting and GCS retention. Its full A5 L36 whole-word
accuracy was 92.8535%; Fuzzy answer / sequence accuracies were 99.9112% /
97.8906%. The input arm's GPU preflight passed both small backward checks and
two actual B2560 mixed updates (59.17 GiB peak allocated, 61.13 GiB reserved).
Fresh input training started normally with paired baseline initialization:
[W&B input arm](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/u1otrxf4).

The user subsequently authorized shared held-out length512/1024 evaluations
of all four models' 15k checkpoints. A separate
[length-evaluation supervisor](rt-nextlat-fuzzy-length-evaluation.md) runs
after this training queue completes. This queue's training and routine
evaluation datasets remain unchanged. Vocabulary32 training remains a later
decision; it is not part of the authorized addition.

## Four arms

All arms use two tiled RT blocks with width 128, 16 heads, GELU FFN512,
Mitchell initialization and ALiBi. Block 0 retains window two; block 1 is
full recurrent attention. NextLat weight remains one and evaluation uses
the backbone; no autonomous latent rollout is introduced.

Only block **index 1** changes:

| Arm | Upper-block change | Gate | Parameters |
| --- | --- | --- | ---: |
| Baseline | No embedding route | None | 479,616 |
| Input | Add `lambda * P_e(e_t)` to the block input, before its existing normalization | Fixed 0.01 | 496,000 |
| Value | Add `lambda * P_e(e_t)` only to the permanent stored value | Fixed 0.01 | 496,000 |
| Head | Replace one existing permanent value head with its value projection applied to `e_t` | None | 479,616 |

`e_t` is the attached raw token embedding. The input and value arms share the
same independently initialized, learned, bias-free D128-to-D128 projection
(isolated seed 1237, normal standard deviation `128**-0.5`). Lambda is fixed,
with no warmup and no learned scalar. Existing normalization conventions
remain unchanged. The projection adds 16,384 parameters (about 3.4%).

The head arm reassigns the **last existing head, index 15**, following the
earlier helper's convention. ALiBi slopes differ across heads, so retaining
that convention matters. The head uses 8 dimensions out of 128, versus the
earlier D512/H8 experiment's 64 out of 512. Its existing value-projection
rows supply embedding values; no projection parameters, cache width or head
are added. Keys remain contextual, queries keep the current lower-output
route, and temporary self K/V stay unchanged. The value-addition arm also
leaves keys and temporary self K/V unchanged.

## Matched training and measurement

Each optimizer update contains **2,560 A5 T12 examples plus 2,560 Fuzzy T400
examples**, with physical microbatch 2,560 and equal task-mean loss weights.
Each arm processes 38.4 million presentations per task, including repeat
visits to the same finite datasets. Preserve task-local output distributions,
native dense Fuzzy training targets, masked Fuzzy answer evaluation and all
within-example NextLat transitions. No data, vocabulary, padding, difficulty
or task-weight change is introduced for this comparison.

Preserve backbone seed 1234, predictor seed 1235, appended Fuzzy-row seed 1236,
A5 order 5432 and Fuzzy order 2026091604. All shared initial parameter tensors
must match the baseline exactly. AdamW, LR 1e-4, clipping, FP32 eager execution
and all runtime settings remain unchanged. Optimizer state is fresh for each
arm; no variant starts from the trained baseline or another variant.

Logging every 25 and routine evaluations every 100 match the baseline. Routine
A5 checks use fixed 4,096-word subsets; Fuzzy uses all 1,280 development examples.
Full A5 evaluations use 102,400 words per role at the union of the baseline's
three stage schedules: 500, 750, 1,000, 1,500, 2,000, 2,500, 3,000, 3,500, 4,000, 4,500,
5,000, 6,000, 7,500, 10,000, 12,500 and 15,000, plus the actual early stopping endpoint
if needed. The same first-positive confirmation rule remains active.
Checkpoint milestones also include 0, 250, 2,750 and 5,500; evaluated states are
saved. Final confirmation data remain unused.

Compare joint same-checkpoint A5 length 36 whole-word accuracy, token/prefix
curves, training-length A5 and Fuzzy answer/sequence/retrieval metrics. Plot
optimizer updates and cumulative training time separately. The baseline's
training time is summed across all three resumed stages; continuation-only
time would be misleading. End-to-end elapsed time includes evaluation and
startup overhead, and is not identical to the training-time axis.

Fuzzy may approach a ceiling. Time to reach accuracy thresholds and remaining
retrieval errors can be informative even when endpoint token accuracies are
close. Do not combine the two task accuracies into one score or select each
task's best checkpoint separately. This is one paired seed on reused
development data: close rankings and causal explanations remain provisional.
Harder Fuzzy training conditions and seed replications are later decisions.
The authorized longer-length evaluation is a separate follow-on.

## Implementation and bounded checks

The frozen baseline model, trainer and reporter sources are unchanged. The
new adapter `cdrm/rt_nextlat_task_embeddings.py` reuses the previously tested
permanent-write and embedding-head implementations. The thin entry point
`scripts/rt_nextlat_a5_fuzzy_embedding_train.py` substitutes only model/config
construction and source capture in the existing mixed training loop.

Bounded CPU checks cover paired initialization, disabled-route baseline
limits, intended gradient paths, causality and forward/example isolation.
Before each arm, after the GPU is free, perform a small same-function
naive/tiled FP32 backward check and two discarded updates at the actual
B2560/T12/T400 shapes. This checks the route adaptation and available memory;
it does not reopen mixed-precision, compiler or numerical-policy studies.
Any failed check stops the queue rather than silently changing settings.

## Queue, reporting and retention

Queue runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260918T072040Z-d128-b2560-embedding-comparison/`.
Read `status.json` for actual launch/progress; this document specifies the
authorized protocol. `execution-config.json` contains resolved commands and
`ready.json` binds the source hashes and readiness evidence.

Order: **input, value, head**. The supervisor waits for the baseline's complete
15k report and verified GCS retention before taking the GPU. Each arm has
`<variant>/validation/` and `<variant>/train/` beneath the queue runtime.
Baseline lineage:
`.runtime/rt-nextlat-fuzzy-a5/20260918T015724Z-d128-b2560-mixed-15000/`.

New comparative reports appear at
`docs/reports/rt-nextlat-fuzzy-a5/d128-b2560-embedding-comparison/after-input/`,
`after-value/` and `after-head/`, with online W&B plots in
[rt-nextlat-fuzzy-a5](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5).
Each includes the baseline and all completed variants so far. The dedicated
reporter describes the actual mechanism; the old single-arm report's
hardcoded no-bypass narrative is not used for these variants.

After each summary, retain its runtime/checkpoints and comparative report to
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260918T072040Z-d128-b2560-embedding-comparison/<variant>/`
and verify the endpoint checkpoint hash before starting the next arm.
Persistent project copies remain available too.

Create queue-root `STOP` to request a clean stop of the active training arm,
full endpoint evaluation, report/retention, and cancellation of later arms.
`STOP_QUEUE` alone lets an active arm finish and prevents the next launch.
No baseline stop marker is changed by cancelling this queue.

At the baseline's roughly 2 seconds/update, each 15k arm takes about 8.5 hours
including evaluations; three arms are roughly a day, subject to route overhead.
