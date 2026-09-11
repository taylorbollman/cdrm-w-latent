# RT + NextLat A5 pilot: results and continuation handoff

Completed 2026-09-11. The user approved training the RT backbone with NextLat
and explicitly excluded autonomous MLP latent recurrence experiments. Both
new arms stopped at the fixed **10,000-update** endpoint. No continuation
or independent confirmation evaluation is authorized by this milestone.

The main result is a promising improvement just beyond the training length:
RT + NextLat outperforms pure RT and ordinary Transformer + NextLat there
at the matched endpoint. It does **not** solve length 36. The ordinary
Transformer + NextLat control performs somewhat worse than the ordinary
Transformer at this budget.

## What was implemented

The existing two-block D512/H8/GELU-FFN2048/ALiBi backbone is unchanged.
Both RT blocks use tiled recurrence at rho=1. NextLat adds one shared
training auxiliary after final LayerNorm: RMSNorm of the concatenated next
operation embedding and current latent, followed by a three-linear-layer
GELU residual predictor. Its hidden width 512 follows the pinned released
A5 code; the paper's Table 5 lists 1,024. Each backbone has 6,357,504 parameters;
the predictor adds 1,049,600, for 7,407,104 during training.

The joint loss is same-position state CE plus weight-one SmoothL1(beta=1),
averaged over all 11 adjacent transitions and 512 coordinates per length-12
word. Target-role latents are detached; source latents and next-operation
embeddings remain attached. A single backward and global clip/Adam step
updates shared parameters once. KL and predicted-state CE have zero training
weight. There is no target encoder, EMA, auxiliary rollout, BOS, LM label
shift or EOS masking. All 60 operation IDs, including identity 0, are valid.

Accuracy evaluation calls only the original backbone. Separate bounded
one-step loss/scale/CE/KL diagnostics never carry a predicted latent to the
next prediction and are not an additional inference arm.

See [implementation and usage](rt-nextlat-a5-usage.md) and
[the approved design](rt-nextlat-a5-plan.md).

## Fixed-endpoint evidence

The following are percentages from the same 102,400 held-out length-36
development words, evaluated by each backbone at 10k updates. E(t) requires
every state through position t to be correct; M(36) averages token accuracy
across the whole word. These are prefix metrics from length-36 outputs,
not separate evaluation runs at each shorter length.

| Backbone | E(12) | E(13) | E(14) | E(16) | M(36) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RT | 99.1709 | 79.8730 | 21.5156 | 0.2441 | 37.3588 |
| RT + NextLat | 98.3770 | 96.8652 | 62.2373 | 1.3350 | 39.1194 |
| Transformer | 0 | 0 | 0 | 0 | 14.3542 |
| Transformer + NextLat | 0 | 0 | 0 | 0 | 12.9946 |

RT + NextLat improves E(13) by 16.99 percentage points and E(14) by 40.72
points over pure RT, while its E(12) is 0.79 points lower. The separate
length-12 development set gives whole-word exactness 98.3047% versus 99.1846%
for pure RT. At length-36, all four arms have zero observed whole-word
exactness and final-state accuracy near 1/60 chance. RT + NextLat has only
5 fully correct prefixes at position 18 and none from 19 onward.

This is one paired seed at 2.5% of the paper's 400k-update budget. The endpoint
was fixed before training. At 5k RT + NextLat was behind pure RT: E(13) was
34.67% versus 90.08%, and E(14) 6.68% versus 41.44%. Do not describe the benefit
as consistent throughout training or evidence of faster learning. The
earlier pure-RT 100k run showed large checkpoint variation in extrapolation;
its different budget is not an equal-budget comparison here.

Backbone/predictor initialization and each training minibatch are paired.
All four use the original 800k training words, batch 1024, seed 1234, constant
AdamW LR 1e-4, and full FP32 eager execution with TF32/compile/CUDA graphs
disabled. Each new arm saw 10,240,000 training words. Predictor seed is 1235.
Both new runs execute sequentially on the H100 80GB: RT + NextLat first.
Measured complete-run durations were 9.77 and 3.47 minutes, compared with 9.53
and 2.95 minutes for the original RT and Transformer pilots. These are
elapsed runs, not controlled hardware-normalized benchmarks.

The auxiliary diagnostics do not indicate a collapsed representation:
RT's final bounded length-12 probe has latent RMS 1.022, variation RMS 1.015
and latent relative L2 error 0.138. This is a limited observation, not a
general representation-quality or numerical guarantee. The actual backbone
task results above determine the performance conclusion.

## Validation and preservation

- 33 model/trainer CPU tests passed; 14 reporting tests passed, followed by
  the relevant targeted test after adding the third comparison row.
- The GPU combined-loss tiled/naive FP32 check passed; maximum absolute
  gradient difference was 3.80e-7 in the small B2/T12/D128 fixture.
- Disabling the auxiliary exactly reproduced both baselines' logits, CE,
  gradients and one Adam update. A raising predictor hook confirmed that
  backbone inference bypasses it.
- The actual B1024/D512 update fixtures had finite FP32 model, gradient and
  Adam state. Both small models memorized the discarded overfit fixture.
- A fresh-process 1→3 resume exactly matched uninterrupted 3 for backbone,
  predictor, Adam, RNG, contract, counters and word order.
- Final saved tensors and Adam states passed the independent CPU inspection;
  all active Adam counters equal 10,000. Initial backbone tensors match their
  corresponding historical checkpoints, and both predictors start identical.
- The historical 43-source identity is unchanged. The new 46-source frozen
  training identity is
  `1e6d0c63289f01525bc0c19bba6b2646d61df10ddb815bc74f5c111b46f9f961`.

The run lineage is `.runtime/rt-a5/20260911T191702Z-nextlat/`. Primary
directories are `train-rt-nextlat` and `train-seq-nextlat`. Each retains
full resumable checkpoints at 0, 1k, 5k, 10k. Independent confirmation stays
untouched. Historical baselines remain in `.runtime/rt-a5/20260911T154748Z/`.
Do not alter old A5 sources or add JSON to `configs/rt_a5/`, because that
would invalidate the old pure-RT strict resume identity.

Read [the complete report and plots](reports/rt-a5/nextlat-pilot/report.md),
[metrics CSV](reports/rt-a5/nextlat-pilot/metrics.csv), and the runtime
`saved-state-validation.json`, `final-evidence-audit.json`,
`checkpoint-storage.json` and `evidence-storage.json` before future work.
The 1k baseline evaluations used 4,096 words, whereas the new arms used 102,400;
the report explicitly carries those denominators. At 5k and 10k all arms use
the same 102,400 words in each development role. E/A intervals are pointwise
over words; no paired-significance or training-seed uncertainty is claimed.

W&B:

- [RT + NextLat training](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/j1pce7q9)
- [Transformer + NextLat training](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/ocnfvfb9)
- [Four-backbone comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/14gci5ue)

All retained primary checkpoints and the final evidence archive belong under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260911T191702Z-nextlat/`, with verified
receipts. The original dataset is already retained in its earlier lineage.
No autonomous MLP latent recurrence experiment should be added on a future
continuation unless the user explicitly changes that scope.

The appropriate next decision is whether to test the encouraging boundary
result across more seeds and/or longer matched training. No such extension
has been started. Preserve this 10k endpoint as the preselected pilot result.
