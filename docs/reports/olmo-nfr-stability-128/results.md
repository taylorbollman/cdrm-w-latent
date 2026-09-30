# NFR settling and continuation through update 128

The reduced-KL FBT + native RT + NextLat continuation completed update128.
All64 resumed updates were finite. Fourth-pass development CE improved from
5.5201 to3.0585 while first-pass CE stayed near2.77. The gap fell from2.7452
to0.3024nats; feedback still worsens prediction relative to the first pass.
Training is stopped, cloud128 verified, W&B synced and host exit0. The final
saved-state K1–32 probe is pending; the completed64 probes are below.

[Training](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/2zu5jloq),
[training-summary plots](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/9xienwmk)
and [paired64 curves](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/3p91owsx).
Local figures: [development](figures/development.pdf),
[training dynamics](figures/training-dynamics.pdf),
[paired64 endpoint curves](figures/endpoint-curves.pdf).
See [protocol](protocol.md), [validation](validation.md),
[storage receipts](storage-receipt.md) and [progress/recovery](progress.md).

## Continued adaptation, without a useful-refinement claim

These are common FP32/no-jitter measurements on the regular64-row development
panel, separate from the eight-row deep-pass panel. CE is nats per target.

| Optimizer update | Pass1 | Pass2 | Pass3 | Pass4 | Pass4 minus pass1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.774970 | 5.375003 | 5.481358 | 5.520121 | 2.745152 |
| 80 | 2.787306 | 4.407264 | 4.511726 | 4.563218 | 1.775912 |
| 96 | 2.774441 | 3.630273 | 3.676090 | 3.700264 | 0.925822 |
| 100 | 2.769629 | 3.510893 | 3.552609 | 3.574547 | 0.804918 |
| 112 | 2.763501 | 3.231428 | 3.253204 | 3.263256 | 0.499755 |
| 128 | 2.756071 | 3.045481 | 3.053890 | 3.058500 | 0.302429 |

Improvement continues after warmup100, but slows: the gap drops0.3052 over
100–112 and0.1973 over112–128. This supports investigating further adaptation;
it does not establish that additional passes will eventually beat pass1.

The final raw latent losses are0.193384/0.154142/0.154367/0.154215;
raw KL is2.514281/2.069738/2.077383/2.075170. Later-pass auxiliary losses
increase while CE improves. NextLat predicts the next token position's hidden
state/distribution **within each pass**, not the next FBT iteration. Its
stop-gradient teachers evolve with the model. These opposite trends alone
establish neither numerical failure nor causes such as predictor lag or
gradient conflict. CE passweights remain1/2,1/6,1/6,1/6; auxiliaries are averaged
across passes before latent1/KL0.1 coefficients. No fourfold KL summation.

## Execution, integrity and cost

The run restored exact reduced64 model, populated Adam, rank RNG, cursor and
schedule, and reproduced development64 exactly. It retained native RT at
layers0/15, FBT K4/beta1/jitter0.02, latent weight1, KL0.1, T1024, BF16 mixed
training, FP32 master weights/Adam, fused Adam, CUDA graphs and activation
checkpointing. Ordinary attention remains Flash SDPA; native RT uses the
accepted Triton path, not FA4. Evaluation uses common FP32/no jitter.

Two H10080GB ranks used physical B12 each and22 accumulation slots:512 real
rows and524,288 real input tokens per update (528 physical slots including
16dummy slots). The64 new updates add33,554,432 inputs; the original128-update
lineage totals67,108,864 inputs, excluding fusion preparation/pretraining.
The model has1,267,879,936 trainable optimizer-owned parameters, including the
NextLat predictor; without that training-only predictor,1,185,153,024.

All64 new updates were finite;57 clipped. Pre-clipping norm min/median/max/final
is0.8794/2.0880/5.1992/0.9442. Updates119 and123–128 were below threshold1.
Warmup ended100; peak2e-4 began101 and remained unchanged. All scheduled
evaluations preserved the complete training boundary on both ranks.
The independent final audit passed15,557 checks with zero failures, including
all222 frozen source pins,71 Adam tensors, clocks/counters/data order,
repeated64 state,64 updates, development checks and16 verified publications.

Measured two-GPU throughput was3,560 real input tokens/s over materialization,
forward/backward and optimizer/cursor regions. Executor throughput was2,542
inputs/s over13,198s (3h40m), including startup, evaluations and checkpointing,
but excluding final W&B/host closeout. Tokens are counted once despite K4.
Peak reserved memory was59.08GiB and peak allocated42.83GiB per GPU. This is a
checkpoint-heavy diagnostic, not a new optimized training throughput sweep.

Final128 is retained on SSD and in GCS. The original report and history are
immutable; stale W&B checkpoint-summary fields were reconciled separately.
See the storage receipt for exact authorities and recovery paths.

## Saved NFR64 endpoint measurements

Both use the same eight packed T1024 development rows: 8,192 inputs and 8,184
CE targets, common FP32, no jitter, native RT at layers 0/15, no predictor
execution and no optimizer updates. This is the deep-pass panel, separate
from the regular 64-row development panel used for training progress.

| Metric | KL 1 | KL 0.1 |
| --- | ---: | ---: |
| CE at K1 | 2.935495 | 2.788031 |
| CE at K4 | 6.606082 | 5.412258 |
| CE at K8 | 6.622874 | 5.430928 |
| CE at K32 | 6.627102 | 5.431623 |
| CE K32 minus K4 | +0.021020 | +0.019366 |
| Tail consecutive change at K4 | 1.5173% | 3.7064% |
| Tail consecutive change at K8 | 0.1611% | 0.1540% |
| Tail consecutive change at K32 | 0.0006970% | 0.00004587% |
| Direct tail K4–K32 hidden residual | 1.5480% | 2.1792% |
| Direct tail K8–K32 hidden residual | 0.5902% | 0.1766% |
| Direct all-position K4–K32 residual | 1.7495% | 2.7539% |

Tail means the final 128 positions of each row. Consecutive change is pooled
RMS(hK − hK−1) / RMS(hK−1). Direct residual is pooled
RMS(hK − h32) / RMS(h32), measured from the same canonical forward, rather
than inferred from the consecutive differences. K32 is a finite reference,
not an exact-online target.

Both pooled tail curves decrease across passes. Reduced KL leaves more
remaining change at K4 but is closer to its K32 state at K8. Therefore, lower
KL does not simply slow settling at every depth. The control's slower deep
tail is primarily associated with one panel row; it continues decaying at
K32 rather than flattening at the earlier checkpoint's numerical floor.

The normalized hidden RMS is approximately one by construction, so it is not
sufficient evidence of bounded internal scales. Pre-normalization RMS remains
bounded too: at K4/K32 it is about 8.795/8.775 for control and 4.676/4.687 for
reduced. Input RMS remains approximately 0.037–0.040. K32 entropy is 6.545
versus 5.666. No runaway growth or non-decaying oscillation appears on this
panel. Every weights/RNG/gradient/runtime preservation check passed.

The practical conclusion is that K4 retains a measurable hidden-state
residual, but simply iterating these saved weights more does not recover the
large first-to-later-pass prediction gap. This does not rule out a role for
finite-pass mismatch during learning, and is not a generation or BF16
optimization-equivalence result.

## Recommended next decision

The user asked whether continued improvement warrants extending the same
experiment. Subject to the final saved128 pass curve remaining healthy, the
recommendation is a bounded continuation to192 with an observation at160,
unchanged model/loss/precision and the existing2e-4 plateau. This has **not**
been authorized or launched; current training stops128.

The current scheduler and ordered data plan explicitly end128. A clean
extension must strict-load128 under its original authority, establish an
explicit192 plan with an identical first128 membership/allocation/noise/LR
prefix, and record the deliberate schedule-horizon metadata transition while
preserving weights, Adam, RNG, cursor and currentLR. It must not reset the
optimizer or repeat warmup. Avoid modifying the completed pinned runtime.

No matched KL1 control was trained to128, and the separate FBT-only lineage is
not a causal RT comparison. Settling, finite updates and improved CE do not
establish general BF16 equivalence, RT benefit or useful feedback refinement.
