# RT precision comparison: first 100-update development pilot

**Continue the frozen diagnostic comparison to 500 updates and the second
preselected seed, while keeping the protected policy as the default.** Both
trained-state numerical checks pass at physical B512, and both training runs
remain finite. However, the legacy-trained model is worse on this first
seed's development set: **+0.009403 nats per supervised token** under common
FP32 evaluation. That difference is visible in the paired document interval
and exceeds the proposed 0.005 quality margin. Legacy has not earned promotion.

This is the 100-update screen specified in the
[frozen plan](../../rt-precision-alignment-plan.md), rather than the final
confirmation comparison. The extension will determine whether this early
difference persists and agrees across the two fixed initialization seeds.
It does not change the margin or reinterpret the present deficit as a pass.
No confirmation data was opened for this report or figure build.

![Development gaps and trained-state numerical comparisons](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pilot-100-report/pilot-differences.png)

[Interactive W&B figure](https://wandb.ai/taylorbollman/rt-precision-alignment/runs/up0n5zl1)
and [standalone SVG](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pilot-100-report/pilot-differences.svg).

## Matched execution

The two runs use the standard RT with all 12 layers recurrent, D1024, 16
heads, FFN4096, rho 1, causal ALiBi and learned Q/K normalization. There are
151,045,120 backbone parameters and 216,843,264 parameters including the
untied embedding and output tables. Each update uses physical B512/T512,
four internal backward MLP chunks, head-only microbatch 2, and a captured
forward/backward with clipping and AdamW outside the graph.

A uses BF16 autocast plus `bf16_fp32_state`; B uses BF16 autocast plus
`legacy`. Parameters, residual storage, gradients and Adam moments remain
FP32 in both. The prior
[dtype/change audit](change-audit.md) and
[attention localization](attention-findings.md) explain the exact precision
differences; B is not an entirely BF16 computation.

Both start from seed 20260910 and the same initial model digest
`8b87ebbad81f24103e8b3c96c69fc2fd8f97195cf42a72b65ed7442177f7668c`.
They use identical C4 token rows, optimizer settings, schedule and frozen
protocol. Training CE is shifted token CE summed and divided by B×512;
the reported token CE below multiplies that value by 512/511. Evaluation
divides by the number of supervised tokens directly.

All 100 updates in each run report finite FP32 parameters, gradients and
moments. At update 100 the learning rate is 0.000118, following the released
5000-update warmup. The proposed 500-update endpoint reaches only 0.00019;
neither endpoint tests the 0.001 peak learning rate or final convergence.

## Development comparison

Both evaluation policies use physical B512 on the same 1024 development
sequences: 523,264 supervised targets, aggregated into 1,043 documents.
The primary diagnostic evaluates the two trained weight sets with the same
FP32 forward computation. The secondary comparison evaluates each under its
native mixed-precision policy.

| Evaluation | A CE | B CE | B−A | Paired two-sided 95% interval | One-sided 95% upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common FP32 | 6.21191467 | 6.22131791 | **0.00940324** | [0.00786219, 0.01095135] | **0.01073532** |
| Native mixed precision | 6.21198226 | 6.22140499 | 0.00942273 | [0.00787450, 0.01096924] | 0.01075699 |

The frozen bootstrap resamples paired documents uniformly with replacement,
then divides the sum of their B−A losses by the sum of their target counts.
It uses 10,000 draws, seed 20260912 and linear quantiles. The paired analysis
checks matching checkpoint identity, seed, update, evaluation batch, source,
protocol, tokenizer and data; it verifies token/document artifact hashes and
independently reconstructs the document aggregates from token losses.

The common-FP32 interval is entirely above 0.005 for this development sample.
There is a clear one-seed deficit at this checkpoint, which a plot showing
overlapping full-scale loss curves would obscure. Its near equality under
native evaluation means evaluation-time rounding alone does not explain it.
Document resampling does not measure uncertainty over training seeds, and
this predeclared 100-update development checkpoint is not the untouched
confirmation test.

The earlier A-only FP32 evaluation at batch 32 is retained as a grouping
control. Its CE differs from A's batch-512 CE by only `1.68e-9` nats per
supervised token. The paired primary and native comparisons both use B512.

Training differences are not monotonic. The mean B−A native token-CE gap
over updates 61–80 was −0.003803; over 81–100 it was +0.003862. The final
batch gap was +0.015204. Clipping occurred on 49/100 A updates and 56/100 B
updates; the last-20 median raw gradient norms were 0.9137 and 1.1242.
These differences deserve continued monitoring, but are not nonfinite or
explosive training failures.

## Numerical checks at both trained anchors

Each comparison starts A, B and FP32 reference C from the **same weights and
nonempty step-100 Adam moments**. One anchor is the A-trained checkpoint;
the other is the B-trained checkpoint. They are separate conditional tests,
not a comparison of the gradients of two different learned weight sets.

All three arms use the same physical B512 diagnostic batch. Physical FP32
fits, so no accumulated-microbatch reference is involved. Every comparison
covers all 111 parameter tensors and 216,843,264 gradient and applied-update
coordinates. C has autocast and TF32 disabled. The driver applies the common
clipping/Adam contract, verifies exactly one optimizer-step advance, and
compares the resulting actual parameter delta.

These conditional numerical probes use the **saved step-100 learning rate
0.000118**, rather than the scheduled step-101 rate 0.00011818. Within each
anchor, all three arms use the same saved moments and learning rate. Thus
they compare a common update rule without claiming to reproduce the next
scheduled training update exactly. Clipping is independently verified as
inactive in all three arms at the A-trained anchor (coefficient 1), and
active in all three at the B-trained anchor (approximately 0.8106–0.8127).

| Trained anchor | Arithmetic vs C | Global gradient relative L2 | Worst tensor gradient relative L2 | Worst tensor max/reference-max | Actual Adam-delta relative L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| A-trained, update 100 | A | 0.5060% | 1.0697% | 1.4634% | 0.3165% |
| A-trained, update 100 | B | 0.5068% | 1.2002% | 1.8206% | 0.3365% |
| B-trained, update 100 | A | 0.5898% | 1.0966% | 1.7219% | 0.3087% |
| B-trained, update 100 | B | 0.6016% | 1.2023% | 1.9060% | 0.3267% |

There are **no global or tensor gradient review flags at either anchor**,
and both applied-Adam distance screens pass. The frozen triggers are 1.5625%
global gradient L2, 3.125% tensor gradient L2, 6.25% tensor maximum ratio and
1.5625% trained Adam-delta L2. They are engineering review rules, not paper
tolerances. B/C Adam-delta cosines exceed 0.999994 at both anchors. B/A
applied-delta distances are 0.1848% at the A anchor and 0.1612% at the B anchor.

These results support testing the learning trajectory further. They do not
prove that small per-step precision differences cannot accumulate into a
quality difference. Likewise, they do not erase the initialized B2 query/key
flags or the localized 4.3098% fixed-operand query-adjoint error documented
in the attention report. Those remain reproducible evidence with a narrower
scope; at the tested trained operational shape they do not produce a failed
gradient or trained-Adam screen. There is currently no new trained-state
failure requiring another B2 localization run or an arithmetic patch.

## Cost and disposition

Updates 3–100 average 6.1741 seconds for A and 5.4792 seconds for B in these
matched training runs, approximately 11.3% less elapsed update time for B.
The recorded update timing includes verification checks and is not an
isolated kernel benchmark. Peak reserved GPU memory over those updates is
49.00 versus 39.17 GiB; minimum sampled free memory is 27.38 versus 37.37 GiB.
These are useful operational differences on this H100, subject to run-order
and timing scope. Reservation and sampled physical free memory are more
informative here than allocation peaks alone, which can undercount workspace
reused by CUDA graph replay.

The evidence supports a **diagnostic extension, not promotion of B**:

1. Verify fresh-process resume at the existing B checkpoint, then continue
   both preserved first-seed runs to the fixed 500-update endpoint with the
   same data order, schedule and precision sources.
2. Run the already selected second seed, 20260911, under the same protocol.
   Preserve both seed outcomes; do not select whichever happens to agree.
3. Reassess development learning curves and trained-state numerical checks
   before opening confirmation. Keep the 0.005 margin unchanged and maintain
   A as the default while B remains experimental.

If the deficit persists across the planned evidence, the useful speed/memory
benefit alone does not meet the stated quality goal. At that point keeping A
or testing a narrowly justified protection is more defensible than declaring
the difference harmless. Conversely, a short warmup checkpoint does not yet
establish that B will have worse final convergence. Extending the frozen
comparison is the bounded way to resolve that uncertainty.

## Evidence and independent review scope

The CPU figure build snapshots and hashes its six closed input reports and
its own source; its
[report](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pilot-100-report/report.json)
is complete and W&B is synced. It includes both trained anchors and both
development policies, with no skipped incomplete cases.

The trained reports are
[A anchor](../../../.runtime/rt-precision-alignment/20260910T191100Z/numerics/trained-A-seed0-100-b512/report.json)
and
[B anchor](../../../.runtime/rt-precision-alignment/20260910T191100Z/numerics/trained-B-seed0-100-b512/report.json).
Their SHA256s are respectively
`b2c16ef5f85565e70d122d2d2cb20c672cb14e58b116ad06ef50495875f4ca40`
and
`a2332601592d4885ab951e24e51d3a384234ac47e19d893f06babdfa49c55223`.
I independently checked their completed status, physical reference scope,
state-step/precision records and all 38 current source hashes, and reproduced
the global gradient and Adam relative-L2 values from their per-tensor FP64
reductions. This review did not independently reload every multi-gigabyte
trained packet. Separate CPU
[A-anchor raw-packet audit](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/trained-A-seed0-100-b512-audit.json)
and
[B-anchor raw-packet audit](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/trained-B-seed0-100-b512-audit.json)
both passed: they independently reloaded the packets, checked their integrity
and reproduced the all-coordinate metrics. Those packet-level checks are
additional evidence beyond this report-level review.

The paired
[FP32 analysis](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pair-seed0-100-dev-fp32/report.json)
and
[native analysis](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pair-seed0-100-dev-native/report.json)
retain input hashes, document-pairing checks and bootstrap identities.
Training histories are available in W&B for
[A](https://wandb.ai/taylorbollman/rt-precision-alignment/runs/mshen0oe)
and [B](https://wandb.ai/taylorbollman/rt-precision-alignment/runs/in42ga2x).
