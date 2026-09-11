# A5 pilot handoff

Latest work: the bounded RT continuation to **100,000 total updates is
complete**, including full checkpoint evaluation, plots and retention.
No training remains running. Read [the 100k run note](rt-a5-100k-run.md)
and [results](reports/rt-a5/budget-100k/report.md) first. Training-length
accuracy improved, but the fixed 100k endpoint extrapolates worse than 10k:
E(13) is 48.7432%, E(14) is 7.0781%, and observed E(36) is zero. Intermediate
checkpoints fluctuate substantially; this is not a monotonic trend claim.
**Stop here for the user's decision.** Do not continue to 400k automatically;
that earlier request was revised. Confirmation remains unevaluated.

The subsequent RT-first length evaluation is also complete. Read the
[length follow-up handoff](rt-a5-length-followup-usage.md) and
[results](reports/rt-a5/length-followup/report.md) before further A5 work.
It records the 5k/10k extrapolation tradeoff and the scoped acceptance of
accuracy measurements despite retained strict logit-screen failures.

The rest of this document records the original paired 10,000-update pilot,
completed on 2026-09-11. Its implementation and runs are finished; no jobs
from that closed pilot remain active. The later 100,000-update continuation
noted above is a separate active lineage. The full 400,000-update experiment
has not been run.

Read [the PR summary](reports/rt-a5/pr-summary.md),
[the paired results and plots](reports/rt-a5/pilot/report.md),
[usage](rt-a5-usage.md), and [the approved plan](rt-a5-experiment-plan.md).

**Preserve the user's simplification.** Full FP32 is used for attention,
projections, MLPs, head, gradients and Adam state. Autocast, TF32, whole-model
and helper `torch.compile`, and CUDA graphs are off. The user explicitly
wants to avoid unnecessary numerical analysis. The bounded task/backward
checks passed; do not automatically restart BF16 qualification, local FP64
diagnostics, compiler studies or a capture implementation. Eager FP32 was
fast enough for this pilot.

The primary models have two blocks, D512/H8/GELU-FFN2048 and 6,357,504
parameters each. Both RT blocks are tiled recurrent at rho 1. They retain
our LayerNorm/full-width learned QK normalization and ALiBi. They have no
CDRM components or NextLat objective. Their corresponding initial weights are
exactly paired by checked conversion from a canonical SEQ initialization.
All existing vendor/model implementations were retained.

The A5 input/target alphabet is 60 even permutations. Targets are cumulative
products at the **same** positions, with no BOS, target feedback or LM shift.
Training length is 12; evaluation length is 36. Cumulative exactness E(t)
requires all states through t to be correct. Isolated accuracy A(t) does not.
Keep these distinct, especially when interpreting long words with correct
early prefixes and incorrect later ones.

The closed pilot lineage is `.runtime/rt-a5/20260911T154748Z`:

- `data/`: frozen corpus and manifests; 800,000 short training words,
  200,000 short development words, separate long pool and independent
  confirmation. Normal OOD development uses 102,400 words. All recorded
  complete-word and 12-prefix intersections are zero. Native NumPy PCG64
  generation deliberately differs from upstream Python sampling.
- `checks/`, `profile/`, `overfit/`: successful bounded GPU evidence with
  snapshots. `cpu-test-result.json` records 44 initial tests;
  `reporter-test-result.json` records 12 additional reporter tests.
- `resume-proof.json`: RT D128/B16/T12, fresh-process update 1 continuation
  through 3 versus uninterrupted 3, exact model/Adam/RNG/order agreement.
- `train-seq/`, `train-rt/`: complete seed 1234 runs, physical B1024, LR 1e-4,
  constant schedule, AdamW (.9,.95), epsilon 1e-8, matrix decay .01, clip 1.
  Both consumed the same 10,240,000 ordered training words. Checkpoints at
  `checkpoints/step-{000000,001000,005000,010000}.pt` include full resume state.
- `saved-state-check.json`: final parameters and every Adam state are finite
  FP32; paired final source/order hashes match.
- `storage-data.json`: verified GCS generations, hashes and sizes for the
  18 data objects (120,205,938 bytes). Evidence retention is recorded in the
  companion storage receipt created after the final archive.

The final development results use 102,400 words per role:

| Model | L12 token / exact-word | L36 token / exact-word |
| --- | --- | --- |
| SEQ | 39.72% / 0% | 14.35% / 0% |
| RT | 99.88% / 99.18% | 37.36% / 0% |

On the OOD words, RT cumulative exactness drops from about 99.17% at prefix
12 to 79.87% at 13, 21.52% at 14, 2.81% at 15 and zero by 18 in this sample.
Its final-state accuracy at 36 is about 1.70%, near the 1/60 chance level.
Thus RT's higher mean long-word token accuracy does not establish successful
length generalization. Long-word CE is actually higher for RT (4.6701) than
SEQ (3.5511). These finite FP32 outcomes do not themselves establish a
precision bug, an architectural impossibility or a final-convergence result.

Total run times were 176.76 seconds (SEQ) and 571.57 seconds (RT); the actual
training-loop portions were 167.09 and 550.81 seconds. The separate profile
measured 15.55/53.81 ms synchronized update times after 25 warmup updates. Use
the actual pilot timing for future budget estimates; compiler/graph tuning
was unnecessary for this milestone.

W&B project: [rt-a5-state-tracking](https://wandb.ai/taylorbollman/rt-a5-state-tracking).
Main runs: [SEQ](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/wrszf3u7),
[RT](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/mgt2i6cv),
[paired plots](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/50ofd0n8).
The final confirmation set remains **unevaluated**. This is one seed at
2.5% of the 400,000-update reference budget. Subsequent choices may use
development results, while confirmation must remain separate.

Use fresh continuation/output directories if extending a compatible run.
The trainer validates source, data, model, runtime and optimizer mapping;
changing those is not an exact resume. Specify appropriate future
`--checkpoint-steps` when extending beyond 10,000: its defaults contain only
the initial pilot checkpoints plus the requested final endpoint. The reporter
expects one complete matched history window per arm, so resumed multi-part
curves need deliberate assembly without double-counting an interrupted tail.

All GPU commands go through the project Docker launcher, with container/GPU
verification; CPU-only artifact inspection uses the GPU-disabled container.
Keep reusable artifacts under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260911T154748Z/` and retain historical
precision evidence separately. The working tree already contained substantial
uncommitted earlier work. New A5 code, configs, tests and documentation are
reviewable locally; no external PR, commit or push was created for this pilot.
