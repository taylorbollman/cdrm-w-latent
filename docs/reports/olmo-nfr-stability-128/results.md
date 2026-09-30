# NFR settling and continuation through update 128

The reduced-KL FBT + native RT + NextLat continuation completed update 128.
All 64 resumed updates were finite. Fourth-pass development CE improved from
5.5201 to 3.0585 while first-pass CE stayed near 2.77. The gap fell from 2.7452
to 0.3024 nats; feedback still worsens prediction relative to the first pass.
Training is stopped, cloud 128 verified, W&B synced and host exit 0. The final
saved-state K1–32 probe also completed: practical finite-pass mismatch shrank,
but additional inference passes still slightly worsen CE.

[Training](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/2zu5jloq),
[training-summary plots](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/9xienwmk)
and [paired 64 curves](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/3p91owsx).
Local figures: [development](figures/development.pdf),
[training dynamics](figures/training-dynamics.pdf),
[paired 64 endpoint curves](figures/endpoint-curves.pdf).
Final comparison: [64 versus 128 pass curves](figures/reduced64-to128-curves.pdf)
and [W&B overlay](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/tc74nrn4).
See [protocol](protocol.md), [validation](validation.md),
[storage receipts](storage-receipt.md) and [progress/recovery](progress.md).

## Continued adaptation, without a useful-refinement claim

These are common FP32/no-jitter measurements on the regular 64-row development
panel, separate from the eight-row deep-pass panel. CE is nats per target.

| Optimizer update | Pass 1 | Pass 2 | Pass 3 | Pass 4 | Pass 4 minus pass 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.774970 | 5.375003 | 5.481358 | 5.520121 | 2.745152 |
| 80 | 2.787306 | 4.407264 | 4.511726 | 4.563218 | 1.775912 |
| 96 | 2.774441 | 3.630273 | 3.676090 | 3.700264 | 0.925822 |
| 100 | 2.769629 | 3.510893 | 3.552609 | 3.574547 | 0.804918 |
| 112 | 2.763501 | 3.231428 | 3.253204 | 3.263256 | 0.499755 |
| 128 | 2.756071 | 3.045481 | 3.053890 | 3.058500 | 0.302429 |

Improvement continues after warmup at 100, but slows: the gap drops 0.3052 over
100–112 and 0.1973 over 112–128. This supports investigating further adaptation;
it does not establish that additional passes will eventually beat pass 1.

The final raw latent losses are 0.193384/0.154142/0.154367/0.154215;
raw KL is 2.514281/2.069738/2.077383/2.075170. Later-pass auxiliary losses
increase while CE improves. NextLat predicts the next token position's hidden
state/distribution **within each pass**, not the next FBT iteration. Its
stop-gradient teachers evolve with the model. These opposite trends alone
establish neither numerical failure nor causes such as predictor lag or
gradient conflict. CE pass weights remain 1/2,1/6,1/6,1/6; auxiliaries are averaged
across passes before latent 1 / KL 0.1 coefficients. No fourfold KL summation.

## Execution, integrity and cost

The run restored exact reduced 64 model, populated Adam, rank RNG, cursor and
schedule, and reproduced development at 64 exactly. It retained native RT at
layers 0/15, FBT K4 / beta 1 / jitter 0.02, latent weight 1, KL 0.1, T1024, BF16 mixed
training, FP32 master weights/Adam, fused Adam, CUDA graphs and activation
checkpointing. Ordinary attention remains Flash SDPA; native RT uses the
accepted Triton path, not FA4. Evaluation uses common FP32/no jitter.

Two H100 80GB ranks used physical B12 each and 22 accumulation slots: 512 real
rows and 524,288 real input tokens per update (528 physical slots including
16 dummy slots). The 64 new updates add 33,554,432 inputs; the original 128-update
lineage totals 67,108,864 inputs, excluding fusion preparation/pretraining.
The model has 1,267,879,936 trainable optimizer-owned parameters, including the
NextLat predictor; without that training-only predictor, 1,185,153,024.

All 64 new updates were finite; 57 clipped. Pre-clipping norm min/median/max/final
is 0.8794/2.0880/5.1992/0.9442. Updates 119 and 123–128 were below threshold 1.
Warmup ended at 100; peak 2e-4 began at 101 and remained unchanged. Scheduled
evaluations passed cursor/graph and model/RNG/runtime preservation checks on
both ranks; these lean checks are distinct from complete tensor-byte hashing
at restore/preparation/save boundaries.
The independent final audit passed 15,557 checks with zero failures, including
all 222 frozen source pins, 71 parameter tensors with populated Adam state,
clocks/counters/data order,
repeated 64 state, 64 updates, development checks and 16 verified publications.

Measured two-GPU throughput was 3,560 real input tokens/s over materialization,
forward/backward and optimizer/cursor regions. Executor throughput was 2,542
inputs/s over 13,198 s (3 h 40 m), including startup, evaluations and checkpointing,
but excluding final W&B/host closeout. Tokens are counted once despite K4.
Peak reserved memory was 59.08 GiB and peak allocated 42.83 GiB per GPU. This is a
checkpoint-heavy diagnostic, not a new optimized training throughput sweep.

Final 128 is retained on SSD and in GCS. The original report and history are
immutable; stale W&B checkpoint-summary fields were reconciled separately.
See the storage receipt for exact authorities and recovery paths.

## Final saved-state pass curves

The same eight packed T1024 rows, tensor hashes, FP32/no-jitter computation
and definitions were used at both checkpoints. This panel is deliberately
separate from the larger development panel above.

| Metric | Reduced KL at 64 | Reduced KL at 128 |
| --- | ---: | ---: |
| CE at K1 | 2.788031 | 2.772209 |
| CE at K4 | 5.412258 | 3.024883 |
| CE at K8 | 5.430928 | 3.030894 |
| CE at K32 | 5.431623 | 3.031271 |
| CE K32 minus K4 | +0.019366 | +0.006388 |
| Direct all-position K4–K32 hidden residual | 2.7539% | 2.0843% |
| Direct tail K4–K32 hidden residual | 2.1792% | 0.7445% |
| Direct all-position K8–K32 hidden residual | 0.3228% | 0.1486% |
| Direct tail K8–K32 hidden residual | 0.1766% | 0.0170% |

The practical residual decreases, particularly in the tail. Consecutive tail
change at K4 is 1.8820%, at K8 0.02859%, and reaches about 1e-6 in relative
units by K16. Every row's K32 tail change lies around 0.90–1.17e-6. The deep
numerical floor itself need not become smaller as the model trains: the
earlier reduced64 floor was about 4.6e-7. Pre-normalization RMS at K32 falls
from 4.6865 to 2.4909; entropy falls from 5.6657 to 3.2744. Bounded internal
scales and decaying changes support the stability observation, beyond the
approximately unit normalized hidden RMS imposed by normalization.

K32 remains a finite reference, not exact online inference. The K4 residual
is still measurable, especially across all positions. More inference passes
slightly worsen CE at both checkpoints, so they do not explain away the
remaining first-to-later-pass gap. Improved settling and improved prediction
co-occur during training; this does not establish that one caused the other.

The observer performed no optimizer updates or predictor execution. Full
weights/RNG hashes and gradient absence matched before/after, and the runtime
preservation checks passed. Its host exited 0 and W&B synced. Two initial
CPU preflights failed historical full-stat checks while exact data-authority
content hashes remained unchanged; an unchanged third attempt passed after
stable reads. Access-time changes are the supported explanation, with the
original failed attempts retained; no source/data edit or relaxed check was
needed. See validation/storage notes for the evidence.

## Earlier saved NFR64 endpoint measurements

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
experiment. The healthy final saved128 pass curve and continued CE improvement
support a bounded continuation to 192 with an observation at 160,
unchanged model/loss/precision and the existing 2e-4 plateau. This has **not**
been authorized or launched; current training stops at 128.

The current scheduler and ordered data plan explicitly end at 128. A clean
extension must strict-load 128 under its original authority, establish an
explicit 192 plan with an identical first 128 membership/allocation/noise/LR
prefix, and record the deliberate schedule-horizon metadata transition while
preserving weights, Adam, RNG, cursor and current LR. It must not reset the
optimizer or repeat warmup. Avoid modifying the completed pinned runtime.

No matched KL 1 control was trained to 128, and the separate FBT-only lineage is
not a causal RT comparison. Settling, finite updates and improved CE do not
establish general BF16 equivalence, RT benefit or useful feedback refinement.
