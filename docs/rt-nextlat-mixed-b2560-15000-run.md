# Batch-2,560 mixed continuation to 15,000 updates

**Training completed on 2026-09-18:** exactly 15,000 total optimizer updates,
with 38.4 million presentations per task. The full endpoint A5 L12 token /
whole-word accuracies are 99.9993% / 99.9961%; A5 L36 token / whole-word
accuracies are **99.2369% / 92.8535%** (102,400 words per role). Fuzzy answer
accuracy is **99.9112%**, and whole-sequence exactness is **97.8906%**
(1,280 examples). The added 10k updates used 5.409 training hours and
5.576 elapsed hours. The exact 5k boundary-state audit passed.
[Authoritative endpoint report](reports/rt-nextlat-fuzzy-a5/d128-b2560-mixed-15000/report.md).
Closeout and verified GCS retention are complete; the archive is
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260918T015724Z-d128-b2560-mixed-15000/final-evidence.tar.gz`.
Read runtime `status.json` and `retention/final-receipt.json` for endpoint
identity and storage verification. The launch notes below are historical.

**Follow-on authorization:** the user subsequently requested three fresh
15k embedding-variant runs after this baseline's completed summary. A separate
[comparison queue](rt-nextlat-mixed-embedding-comparison.md) implements that
request; the frozen baseline supervisor itself remains unchanged. The
no-follow-on language below records this stage's original authorization.

Authorized and launched on 2026-09-18. Continue the completed 5,000-update
checkpoint for **10,000 additional optimizer updates, stopping at 15,000
total**. Keep batch **2,560 per task** (5,120 examples per update), physical
microbatch 2,560, and all learning settings unchanged. The user may interrupt
if performance rises sufficiently; there is no automatic accuracy-based stop.
No batch reduction, architecture change, or later experiment is queued.

Supervisor PID at launch: **199688**, 01:59:50 UTC. Startup was verified past
update **5,090**, with finite losses and gradient norms at about 1.9 seconds
per training update. The loaded training contract exactly equals the parent
contract. This is a launch handoff, not a completed-results report.
[Continuation W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/qygud2ze).
Use the `update` metric for optimizer counts; W&B's logging-event index starts
over for this continuation.

## Starting evidence

The [5,000-update stage](rt-nextlat-mixed-b2560-5000-run.md) completed and
verified GCS retention at 01:17 UTC. Its full same-checkpoint development
results were:

| Metric | Accuracy | Evaluation size |
| --- | ---: | ---: |
| A5 L12 whole-word | 0% | 102,400 words |
| A5 L12 mean token | 18.1993% | Same words |
| A5 L36 whole-word | 0% | 102,400 words |
| A5 L36 mean token | 7.1791% | Same words |
| Fuzzy answer tokens | 90.5379% | 1,280 examples |
| Fuzzy all-answers sequence exact | 6.1719% | Same examples |

Parent checkpoint:
`.runtime/rt-nextlat-fuzzy-a5/20260917T234932Z-d128-b2560-mixed-5000/train-mixed-continue5000/checkpoints/step-005000.pt`.
SHA256: `3e7cfea69fdacd02a77df46e331499a4676fd6df23ac74168ff10e99522a3b74`.
The local hash matches the verified retained archive member. The strict
loader restores model, AdamW, RNG, example counters and deterministic stream
cursors. At the boundary there are 12.8 million presentations per task:
A5 epoch 16 position 0 and Fuzzy epoch 1,000 position 0. At 15,000 updates
there will be 38.4 million presentations per task (including repeat visits
to the finite corpora), A5 epoch 48 and Fuzzy epoch 3,000.

## Fixed experiment

Two tiled RT blocks, D128/H16/FFN512, window two at index 0 and full RT at
index 1; Mitchell initialization, ALiBi, NextLat weight one, no embedding
bypass or embedding-access head. There are 479,616 trainable parameters.
Full FP32 eager execution, no TF32/autocast/compile/CUDA graphs. AdamW LR
1e-4 and all optimizer settings, clipping and task weights stay unchanged.

Each update processes 2,560 A5 length-12 examples and 2,560 Fuzzy length-400
examples separately, averages the two task losses equally, and performs
one optimizer update. A5 is not padded to length 400. Preserve the same
finite corpora, data order, task-local output distributions, native dense
Fuzzy training labels, masked answer evaluation and NextLat transitions.
Final confirmation data remain unused. Old batch-128 runs are descriptive
references, not strict matched controls.

## Evaluation, stopping and reporting

Training logging remains every 25 updates; routine evaluation remains every
100, with fixed A5 4,096-word subsets and all 1,280 Fuzzy development examples.
Full A5 length-12 and length-36 development evaluations use 102,400 words
per role at **6,000, 7,500, 10,000, 12,500 and 15,000**, plus the actual
endpoint if stopped early. A first positive routine A5 result is confirmed
using the existing full-evaluation rule.

Explicit checkpoint milestones are 5,000 (restored boundary), 5,500, 6,000,
7,500, 10,000, 12,500 and 15,000; evaluated checkpoints are also saved.
Create `STOP` in the runtime root to request a clean early stop, endpoint
checkpoint and full endpoint evaluation. Do not kill the process for a
normal user-requested stop. The user selected a fixed maximum endpoint;
positive-A5 gate annotations do not authorize additional training.

The previous extra 2,500 updates took 81.7 training minutes and 84.7 elapsed
minutes. Budget roughly **5.5–6 hours** for this extra 10,000 updates including
evaluations, followed by report/retention closeout.

After training, the CPU-only boundary audit compares the entire parent 5k
packet with the re-saved child 5k packet, allowing serialization bytes to
differ only when the actual packet contents are exactly equal. Reporting
joins all three stages, 0–2,500–5,000–actual endpoint, once. For an endpoint
above 10k, the existing extension reporter corrects legacy 10k-gate wording
and publishes the authoritative figures to W&B. For an earlier stop through
10k, the existing direct reporter is used. No old batch-128 controls are
passed to either strict report. No-new-update stops retain the checkpoint
without attempting a positive-length history report.

## Execution and retention

Runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260918T015724Z-d128-b2560-mixed-15000/`.
Resolved CLI: `execution-config.json`; launch validation: `ready.json`;
live phase/PID: `status.json` and `launch.json`.
Training output: `train-mixed-continue15000/`.
Detached driver: `execute.py`; final boundary proof:
`recovery-boundary-state-audit.json`.

The complete frozen training source closure equals the parent's. Independent
CPU-container CLI review confirmed that changes are limited to continuation
paths, endpoint, checkpoint/full-evaluation milestones and W&B labels.
The required GPU Docker environment and H100 80GB were verified before launch.

On completion or clean early stop, the supervisor audits the boundary,
produces the joined report, and verifies final GCS retention and the retained
endpoint checkpoint SHA. Authoritative report destination:
`docs/reports/rt-nextlat-fuzzy-a5/d128-b2560-mixed-15000/`.
Retention prefix:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260918T015724Z-d128-b2560-mixed-15000/`.
The runtime/checkpoints also remain in the persistent project directory.
