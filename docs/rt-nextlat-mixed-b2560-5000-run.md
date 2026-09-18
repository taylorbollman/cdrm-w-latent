# Batch-2,560 mixed continuation to 5,000 updates

**Completed:** all 5,000 total updates, joined reporting, exact resume-boundary
audit and verified GCS retention finished at 01:17 UTC on 2026-09-18. At the
full endpoint, A5 L12/L36 whole-word accuracy remained 0% (102,400 words each),
with mean token accuracies 18.1993% / 7.1791%. Fuzzy answer accuracy reached
90.5379%, with all-answers sequence exactness 6.1719% (1,280 examples).
The extra 2,500 updates took 81.7 training minutes / 84.7 elapsed minutes.
[Final report](reports/rt-nextlat-fuzzy-a5/d128-b2560-mixed-5000/report.md).

**New authorization:** the user requested the exact same run continue to
15,000 total updates. That continuation is live; see the
[15k run handoff](rt-nextlat-mixed-b2560-15000-run.md). The original 5k cap
and launch-status language below are the historical record for this stage.

Authorized and launched on 2026-09-17 after the user reviewed the completed
2,500-update baseline. Resume that exact checkpoint for **2,500 additional
updates, stopping at 5,000 total**. Keep batch 2,560 per task, physical
microbatch 2,560, original model, FP32 runtime, LR 1e-4, AdamW state, RNG,
task weights and deterministic task streams unchanged. No next experiment
or extension beyond 5,000 is queued.

Supervisor PID at launch: **168028**, 23:51 UTC. Read the runtime status for
live progress; this launch note is not a completed-results report.
Training was observed past update **2,525** with finite losses/gradient norms,
at roughly two seconds per update. The loaded training contract exactly
matches the completed parent contract.
[Continuation W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/otyh4yb7).

## Starting evidence

The [initial larger-batch baseline](rt-nextlat-mixed-b2560-2500-run.md)
completed 2,500 updates and verified GCS retention at 23:39 UTC. Its full
same-checkpoint development results were:

| Metric | Accuracy | Evaluation size |
| --- | ---: | ---: |
| A5 L12 whole-word | 0% | 102,400 words |
| A5 L12 mean token | 18.1488% | Same words |
| A5 L36 whole-word | 0% | 102,400 words |
| A5 L36 mean token | 7.1664% | Same words |
| Fuzzy answer tokens | 46.9152% | 1,280 examples |
| Fuzzy all-answers sequence exact | 0% | Same examples |

The extension tests whether more optimizer updates yield useful mixed-task
learning. Its endpoint is user-selected, independent of positive-A5 gate
annotations. Confirmation remains unused, and no numerical/precision study
or learning-rate change is introduced.

Parent checkpoint:
`.runtime/rt-nextlat-fuzzy-a5/20260917T221257Z-d128-b2560-mixed-2500/train-mixed/checkpoints/step-002500.pt`.
SHA256: `743cbd63f87ecaab116a9b543589a776b5b6c3fcc3c49d1eed0e8ca6c42fff6a`.
The local hash matches the parent's verified GCS archive member. It contains
6.4 million consumed examples per task and all 25 finite FP32 model/Adam
parameter states. A5 resumes at epoch 8 position 0; Fuzzy at epoch 500
position 0. The new endpoint will have 12.8 million presentations per task.

## Evaluation and reporting

Retain training logging every 25 updates and routine evaluations every 100.
Full A5 length-12 and length-36 development evaluations use 102,400 words
per role at 3,000, 3,500, 4,000, 4,500 and 5,000. Routine checks use the
same fixed 4,096-word subsets and confirm a first positive subset if needed.
Every Fuzzy evaluation uses the existing 1,280-example development set.
Save explicit checkpoints at the restored 2,500 boundary, 2,750, 3,000,
3,500, 4,000, 4,500 and 5,000; evaluated checkpoints are also saved.

The existing strict checkpoint loader restores all training state. Before
final reporting, `audit.py boundary` verifies exact equality of the entire
parent packet and the re-saved child 2,500-step packet in a CPU-only container.
This handles differing serialization hashes without weakening the reporter's
resume checks. The recursive report joins original updates 1–2,500 with
new updates 2,501–5,000 once, preserving the actual endpoint if stopped early.
Old batch-128 runs remain descriptive references, not strict matched controls.

The original stage used 81.3 training minutes and 84.4 elapsed minutes.
Expect about 85 minutes for this continuation plus reporting/retention.
Use the logged `update` metric for optimizer counts; this new W&B run's
logging-event index starts over, while optimizer updates start at 2,500.

## Execution and retention

Runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260917T234932Z-d128-b2560-mixed-5000/`.
Resolved CLI and parent identity: `execution-config.json`.
Source and parent validation: `ready.json`.
Live phase/PID: `status.json` and `launch.json`.
Training outputs: `train-mixed-continue5000/`.
Create `STOP` there in the runtime root to request an earlier clean stop
with saved checkpoint and full endpoint evaluation.

The complete training source closure is unchanged from the parent.
Independent CPU-container CLI review confirmed changes are limited to
continuation paths, endpoint, checkpoint/evaluation milestones and W&B labels.
All GPU training runs through the required Docker environment.

On completion the supervisor verifies the boundary state, produces the
joined report with online W&B figures, and verifies final GCS retention.
Report destination:
`docs/reports/rt-nextlat-fuzzy-a5/d128-b2560-mixed-5000/`.
Retention prefix:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260917T234932Z-d128-b2560-mixed-5000/`.
The checkpoint and evidence remain in the persistent project directory too.
