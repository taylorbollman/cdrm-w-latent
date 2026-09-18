# A5 warm start with gradual Fuzzy Recall introduction

Prospective protocol, 2026-09-17. The user accepted the smaller-batch initial
test from the trained A5 checkpoint. This experiment tests retaining A5 while
learning Fuzzy Recall. It does not replace the separate question of whether
the previous fresh mixed run would eventually learn A5 with a longer budget.

Stopped and retained on 2026-09-17 at the user's request to prioritize the
original fresh mixed continuation. The A5-only control completed 10,000
additional updates (20,000 global); the mixed curriculum stopped cleanly at
**6,288 additional updates (16,288 global)**. Supervisor PID 53073 finished
reporting and verified GCS retention at 18:03 UTC. Neither arm remains running.

At the saved mixed checkpoint, full development evaluation gave:

| Metric | Result | Evaluation size |
| --- | ---: | ---: |
| A5 length-12 whole-word accuracy | 99.4473% | 102,400 words |
| A5 length-36 whole-word accuracy | 65.3574% | 102,400 words |
| A5 length-36 mean token accuracy | 93.9634% | 102,400 words |
| Fuzzy answer accuracy | 17.6892% | 34,897 answer tokens |
| Fuzzy all-answers sequence accuracy | 0% | 1,280 examples |

This supports A5 retention and continued improvement, but does not establish
successful Fuzzy acquisition at this budget. The A5-only control has no
full evaluation at phase 6,288; do not claim a matched endpoint comparison.

Saved checkpoint, relative to the execution lineage below:
`train-mixed-curriculum/checkpoints/phase-006288.pt`, SHA256
`ceacc5450c7cd92e5f80437106911fb5e7e81ea85cd0f208bc17d8e197a717b5`.
The complete resume packet includes optimizer, RNG and both data cursors.
Its hash matches the verified archive member in `retention/final-receipt.json`:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260917T150900Z-d128-a5-warmstart-curriculum/final-evidence.tar.gz`.

[Mixed curriculum W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/uyg8ngm1),
[A5-only control W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/jpytgyqd),
and [completed report](reports/rt-nextlat-fuzzy-a5/d128-a5-warmstart-curriculum/mixed/report.md).

The original fresh mixed model subsequently resumed from 10k toward 20k.
See [the continuation handoff](rt-nextlat-mixed-20k-continuation.md).
The protocol below records the originally planned 10k-per-arm budget; the
mixed arm's actual shorter endpoint is the user-authorized change above.

## Why the earlier result remains inconclusive about eventual learning

The A5-only model began improving near 8,000–8,500 optimizer updates and
reached 23.6865% length-36 whole-word accuracy at 10,000. The fresh mixed
run also completed 10,000 optimizer updates, only 1,500–2,000 updates beyond
that transition. Its endpoint remained at a two-operation shortcut: A5
length-12 mean accuracy 18.2314%, length-36 mean accuracy 7.1777%, and
zero completely correct words in each full 102,400-word development sample.
Fuzzy answer accuracy was 99.4383%. Later learning remains possible.

The user's W&B screenshot has `Step` on its horizontal axis. That is the
logging-event index, not the number of Adam updates, and its origin resets
for the resumed run. Use the logged `update` metric for optimizer-count
comparisons. The local completed-run figures already use optimizer updates.

## Model and starting checkpoint

Retain the two-layer D128/H16/FFN512 RT, first layer window two and second
full recurrence, ALiBi, Mitchell initialization lineage, NextLat weight one,
and no embedding bypass. There are 479,616 trainable parameters. Execution
remains full FP32 eager; no precision, compiler or architecture changes.

Parent:
`.runtime/rt-nextlat-fuzzy-a5/20260916T182200Z-d128-a5-mixed/train-a5/checkpoints/step-010000.pt`.
SHA256:
`0f246e2f39f8a26c8a39fcb336ee1bac4da12990f14bdfaf5676b45ce1364701`.

Restore all model/predictor parameters, Adam state, RNG and the A5 data
cursor/order chain. Preserve LR 1e-4, AdamW settings and global clip one.
The new curriculum stage is not an exact continuation of the old objective;
it has a new explicit contract and lineage. Existing frozen trainers and
their strict resume rules remain unchanged.

## Paired continuations

- **A5-only control:** 10,000 additional updates from the parent, with 128
  length-12 A5 examples per update and the existing A5 CE + NextLat objective.
- **Mixed curriculum:** 10,000 additional updates, each containing 128 A5
  length-12 examples and 128 native length-400 Fuzzy examples. Physical
  microbatch is 128 per task. Average losses separately, then perform one
  global clip and Adam update.

For additional-update index `s`, use

`w_FR(s) = 0.5 * min(s / 3000, 1)`

and `L(s) = (1 - w_FR(s))*L_A5 + w_FR(s)*L_FR`, where each task loss retains
its original CE plus NextLat terms. At phase zero, the weight is zero;
the first optimizer update uses `s=1`. Equal weighting begins at phase 3,000
and remains through phase 10,000. Both arms keep all layers trainable.

Keep three counters distinct and persist them across any stage resume:

| Quantity | Start | End |
| --- | ---: | ---: |
| Phase updates | 0 | 10,000 |
| Global Adam updates | 10,000 | 20,000 |
| A5 example presentations | 1,280,000 | 2,560,000 |
| Fuzzy presentations, mixed only | 0 | 1,280,000 |

Continue A5 order seed 5432 from its retained offset. Start Fuzzy order seed
2026091604 at offset zero. Keep the original datasets and task-local output
distributions; no sequence concatenation or A5 target shifting.

## Evaluation and interpretation

Evaluate the restored baseline before training. Monitor A5 token,
final-state and cumulative-prefix/whole-word accuracy at lengths 12 and 36,
alongside Fuzzy answer-token and all-answers sequence accuracy. Use early
checks at phases 50 and 100, then every 250 updates; save the corresponding
checkpoint. Full A5 evaluations use 102,400 development words at the baseline,
phase 3,000, phase 5,000 and endpoint; routine checks use the fixed 4,096-word
subset. Fuzzy checks use all 1,280 development examples. Confirmation is unused.

Log the phase and global optimizer count, Fuzzy weight, task losses, gradient
norm and example counts to W&B. Plot the weight next to both tasks' outcomes.
Review intermediate declines without assuming they are irreversible. No
automatic accuracy-based early-stop threshold was requested; obey a user
stop request and save/evaluate the stopping checkpoint.

Success means retaining useful A5 performance while learning Fuzzy at the
same checkpoint, particularly after reaching equal weights. The control
shows whether continued A5 training itself changes length generalization.
This first test combines pretraining and a ramp; a later abrupt-mix branch
would be required to distinguish their effects. Failure does not establish
an architectural impossibility or prove the fresh mixed run cannot learn later.

## Bounded checks and execution

Validate weight endpoints, independent stream origins and stage-resume
counters with focused CPU tests. Before training, use a small discarded
H100 check from the real parent, including fresh-process split versus
uninterrupted continuation. Verify parent restoration and actual mixed
batch shapes; reuse the existing FP32 kernel qualification.

Run through the project Docker launcher, verifying container location and
GPU availability. The supervisor runs the cheap A5 continuation first, then
the mixed curriculum, reporting and GCS retention. Stop files are recorded
in the execution lineage. No extension of the older fresh mixed run and no
larger-batch learning experiment was queued by the original protocol; the
subsequent fresh-mixed extension is documented separately above.

Checkpoints remain under the persistent project directory. Retain completed
stage evidence under a new
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/` prefix. W&B uses entity
`taylorbollman`, project `rt-nextlat-fuzzy-a5`.

Execution lineage:
`.runtime/rt-nextlat-fuzzy-a5/20260917T150900Z-d128-a5-warmstart-curriculum/`.
Read `status.json` for the supervisor phase and `launch.json` for its PID.
Stop files: `STOP_A5_CONTROL`, `STOP_MIXED`, and `STOP_QUEUE` (the latter
prevents the next arm from starting). `execution-config.json` records the
CLI settings; `ready.json` binds validation to source hashes.

The new trainer is `scripts/rt_nextlat_a5_fuzzy_curriculum.py`; reporting is
`scripts/rt_nextlat_a5_fuzzy_curriculum_report.py`. Focused trainer/independent
CPU checks passed. Actual H100 B128 checks restored the parent model, Adam
and RNG exactly and reproduced a three-update trajectory after a one-update
restart, including task offsets and order chains. See
`preflight/state-proof.json`. The full restored baseline reproduced the
parent A5 accuracies exactly, including 23.6865% length-36 whole-word accuracy.
These discarded checks do not advance either production arm.

Training supports exact stage resumes into fresh directories. The initial
reporter deliberately requires phase-zero-to-end histories; if an interruption
requires a staged resume, add an explicitly audited phase-history join rather
than silently presenting its suffix as the entire experiment.
