# A5-only L1R RT + NextLat: matched BF16 repeat

**Complete and reviewed:** both precision arms reached 80,000 updates. The
BF16 run, paired report and verified GCS retention finished on 2026-09-18.
At 80k both arms have **100% token and whole-word accuracy at lengths 12 and
36**, on 102,400 development words per length. The two initializations and
all 80,000 data-order records match. Final BF16 model/Adam tensors are finite
FP32, all Adam counters are 80,000, and the checkpoint matches its retained
archive member. No following experiment is queued; the user will provide
next directions.

[Completed comparison](reports/rt-a5/l1r-nextlat-bf16-repeat80k/README.md) ·
[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/w9jalp09) ·
[Comparison W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/p4un9rd1).

| Full checkpoint | FP32 length-36 whole word | BF16 length-36 whole word |
|---:|---:|---:|
| 5,000 | 74.0674% | 72.8818% |
| 10,000 | 85.7910% | 86.2666% |
| 20,000 | 96.3369% | 97.4062% |
| 25,000 | 96.2090% | 97.6084% |
| 30,000 through 80,000 | 100% | 100% |

The full checkpoints do not imply uninterrupted perfection. BF16's routine
4,096-word evaluations had an additional dip at 51.5k–56k, reaching **97.6318%**
length-36 whole-word accuracy at 51.5k. This coincided with a finite training
spike (update 51,270: loss 0.01943, pre-clipping gradient norm 3.79; clip bound
one). Every BF16 evaluation from 56.5k through 80k was perfect. FP32 also
fluctuated earlier, but its last nonperfect evaluation was at 37k. All logged
training values in both runs are finite. This supports preserved learning and
final state tracking under the protected BF16 recipe, with some additional
transient variation; it does not establish multi-seed precision equivalence.

BF16 took **79.63 training-loop minutes** versus FP32 **74.48** (+6.92%);
total times were **82.33 versus 76.74 minutes** (+7.29%). This small T12 model
did not demonstrate a BF16 speedup; the historical runs are not a replicated
throughput benchmark. The result applies to protected `bf16_fp32_state`, not
released-style `legacy` BF16 or a mixed A5/Fuzzy training experiment.

Final checkpoint SHA256:
`ccd393ae2e30398ddfa7e519de298fcdc8b268a147a1746e5cc52918ff0a42f1`.
Verified archive:
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260918T175000Z-l1r-nextlat-bf16-repeat/final-evidence.tar.gz`
(SHA256 `0d88f7f95a93de282ed63e9b12c3dd556c4f2fd1490753aebad38c1a8d4566f2`).
The subsequent read-only endpoint/trajectory audit is recorded locally as
`completion-review.json` in the runtime. The original archived report remains
unchanged.

**First full checkpoint, 1,000 updates:** on the same 102,400-word pools,
BF16 length-12 whole-word accuracy is **95.1260%** versus FP32 **95.6348%**;
length-36 whole-word accuracy is **45.6699%** versus **46.7471%**. Length-36
token accuracy is **87.8207%** versus **88.3541%**. This is an early matched
observation, not the final 80k comparison. BF16 checkpoint SHA256:
`3a169d790fc9fc9643420081791b3dc6b0cdaf25307dac0f1393957deee704ea`.

The selected reference is the longest completed A5-only run of our two-layer
RT with its first layer restricted to window two, full recurrence in its second
layer, NextLat training, and no embedding injection: **80,000 optimizer updates
at width 512**. It is larger than the width-128 model in the recent mixed-task
experiments. The longest width-128 A5-only lineage reached 20,000 global updates
(10,000 initial updates plus a 10,000-update A5-only control continuation).

The BF16 run repeated the selected 80k experiment from its original
initial tensors and fresh Adam state. It was not a continuation from the trained
80k checkpoint. Implementation and launch records live in
`.runtime/rt-a5/20260918T175000Z-l1r-nextlat-bf16-repeat/`.

## Exact reference

- Training: `.runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/train-window-first/`.
- [W&B run](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/m2klbr0d).
- [Original report](reports/rt-a5/nextlat-depth-order-80k/report.md) and
  [completed experiment handoff](rt-nextlat-depth-order-80k.md).
- [Machine-readable selection and verification](../.runtime/rt-a5/20260918T175000Z-l1r-nextlat-bf16-repeat/reference-selection.json)
  includes the entire original contract, initialization, resolved arguments,
  checkpoint identities, and source comparison.

| Setting | Matched value |
|---|---|
| Backbone | Two tiled RT blocks; index 0 window two, index 1 full prefix; rho 1 |
| Width / heads / MLP | 512 / 8 / 2,048; GELU |
| Positions / normalization | ALiBi; learned LayerNorm and full-width Q/K normalization |
| Parameters | 6,357,504 backbone + 1,049,600 predictor = **7,407,104** |
| Vocabulary | 60; untied embedding and output head; no biases or dropout |
| Initialization | Mitchell; backbone seed 1234, predictor seed 1235 |
| Data | Frozen 800,000 length-12 A5 training words; data-order seed 1234 |
| Batch / budget | 1,024 / 80,000 updates; 81,920,000 word presentations |
| Optimizer | AdamW, constant LR 1e-4, betas (0.9, 0.95), epsilon 1e-8 |
| Decay / clipping | Matrix decay 0.01, vector decay zero; global norm clip 1 |
| Objective | State CE + weight-one next-latent SmoothL1, beta 1 |
| Gradient paths | Only target-role latents detached; source latents and next-token embeddings attached |
| Runtime controls | Eager execution; TF32, compilation and CUDA graphs off |

NextLat uses the original three-linear-layer GELU/RMSNorm residual predictor,
hidden width 512, conditioned on the next operation embedding and current
post-final-normalization latent. Evaluation uses the RT backbone; there is no
autonomous latent rollout.

The original invocation was `python -m scripts.rt_a5_depth_order_train --variant
rt_window2_first --data-dir .runtime/rt-a5/20260911T154748Z/data --output-dir
.runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/train-window-first
--updates 80000`, with W&B group/run-name arguments. The saved `config.json`,
copied into the selection record, resolves every default.

## Reference metrics and saved identities

These are full development evaluations: **102,400 words per length**.

| Update | Length-12 whole word | Length-36 whole word | Length-36 token accuracy |
|---:|---:|---:|---:|
| 1,000 | 95.6348% | 46.7471% | 88.3541% |
| 5,000 | 99.3350% | 74.0674% | 95.4556% |
| 10,000 | 99.4775% | 85.7910% | 97.3167% |
| 20,000 | 99.8066% | 96.3369% | 99.2835% |
| 25,000 | 99.6807% | 96.2090% | 99.1667% |
| 30,000 | 100% | 100% | 100% |
| 80,000 | 100% | 100% | 100% |

Every retained full evaluation from 30k through 80k had zero observed errors in
either development pool. This is a finite-sample, single-seed result.

All **55 original live source files match their frozen copies and recorded
hashes**, and all 12 retained reference checkpoints match their recorded size
and SHA256. The dataset manifest also matches. These checks are recorded in
`reference-selection.json`; historical files were not modified.

- Original source-manifest digest:
  `cc9fbfa50ebd57387f4ad95803bb2db9d42b69a13875268c0400759858fe4176`.
- Dataset manifest SHA256:
  `944c7a2e86a9329611c0fee74aaad59dfec1e77604c58a7ae7a465c8d529f9eb`.
- Initial `step-000000.pt` SHA256:
  `9ef67f1d02d614516feea997d905efc483ddfd1dca3a735e79a09dd3dcc8c735`.
- Final `step-080000.pt` SHA256:
  `b1767887ecd3ae12f1d55a3b8cb0c71e17799bc3ff25a9074a301da2351540bf`.

The original run took 4,468.6 training-loop seconds and 4,604.2 total seconds
on an H100 80GB, PyTorch `2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3. Those are
historical measurements, not a promised BF16 speedup.

## Precision change and evaluation

Use the project's protected **BF16 autocast plus `bf16_fp32_state`** policy,
keeping learned parameters, gradients, Adam state and protected recurrent
attention calculations in FP32. This follows the completed
[precision decision](rt-numerical-handoff.md). The reference recorded the
`legacy` recurrent policy but disabled autocast and used FP32 throughout;
that policy label does not mean the reference ran BF16 arithmetic.

All architecture, data/order, initialization, objective, optimizer and budget
settings remain matched. The new code must explicitly enable the mixed
precision operations: wrapping the old trainer in autocast is insufficient
because its update function explicitly enters an FP32 context. Preserve the
old files and use a separately recorded driver/model precision wrapper.

Retain the original evaluation cadence: every 500 updates on 4,096 words,
with full 102,400-word evaluations at checkpoints 1k, 5k, 10k, 20k, 25k, 30k,
40k, 50k, 60k, 70k and 80k. Save step zero as well. One-step diagnostic pools
contain 1,024 words. Compare matched full checkpoints and display routine
subsets with their own denominators. Final confirmation remains untouched.

The result tests learning under one changed precision policy; it does not
establish seed-independent equivalence or reproduce the current D128 mixed
experiment. New runtime/precision provenance may differ where necessary to
describe that change, but must not silently alter model or training semantics.

## Completed bounded preflight

Seven focused CPU checks passed. The GPU preflight also passed on the H100:

- All original step-zero tensors and initialization metadata match exactly.
- With mixed precision disabled, the new adapter reproduces original FP32
  logits, losses and gradients exactly on the small real-data fixture.
- At D512/B2/T12, BF16 global raw-gradient relative L2 versus FP32 was
  **1.279846%**; no tensors crossed the historical per-tensor screens. Those
  screens are engineering diagnostics, not paper-published tolerances.
- Both blocks were observed using BF16 projections, FP32 residuals and
  protected FP32 attention. Observers were absent from the operational checks.
- B1024/T12 updates had finite FP32 parameters, gradients and Adam state;
  strict checkpoint reload reproduced the next changed-input update exactly.
- B1024/T36 inference ran without calling the NextLat predictor.

[Preflight W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/hrz1u1l4);
full evidence: `validation/report.json` under the new runtime. These discarded
updates do not warm-start the training repeat. The learning comparison remains
the substantive test; this bounded preflight is not a convergence guarantee.

`contract.precision` and `contract.precision_contract` describe actual BF16
execution. Historical initialization/experiment metadata is preserved verbatim
and can contain the original FP32 label; `model_config.precision=fp32` also
continues to describe parameter storage.

The new host supervisor is `scripts/rt_a5_bf16_queue.py`. It trains one fresh
run, generates the paired report, and retains checkpoints and evidence under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260918T175000Z-l1r-nextlat-bf16-repeat/`.
Creating the runtime's `STOP` requests a saved stopping checkpoint and terminal
development evaluation. There are no following experiments in this queue.
