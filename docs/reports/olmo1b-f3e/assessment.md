# F3e assessment: multiple selected RT layers

The primary two- and four-layer integration checks pass without changing native
OLMo, Q/K math, precision, kernel arithmetic or acceptance thresholds. The
user's main experiment will use multiple RT layers but has not selected their
positions or committed to all-layer recurrence. These checks establish bounded
execution behavior, not a preferred learning architecture.

**Complete, 2026-09-23:** 16 GPU reports pass 48/48 gates, comprising eight
correctness and eight capacity runs. They contain 96 physical optimizer updates
(48 eager + 48 graph), excluding warmup-only backwards. All 474 scoped CPU tests
pass. No failed attempts or numerical fixes were needed. Runtime/protocol commit
`543d243` stayed fixed throughout; report/retention helpers are `de3e6b9`.
See [results](results.md), [capability ledger](capability-ledger.json),
[resource ledger](resource-ledger.json), [plots](multi-rt-resources.pdf),
[protocol](protocol.md) and [final selection](final-inputs.json).

## What is checked

Original OLMo-1B step 60000 (~252B tokens), 16 layers, width 2048, native RoPE,
tied embeddings/readout and no Q/K normalization are unchanged. Selected RT
layers use the F3d bounded-workspace backward, cast reuse and Triton historical
tiles. Ordinary layers use deterministic PyTorch Flash and activation
checkpointing. BF16 autocast uses FP32 parameters/gradients/Adam, with TF32 off.
CUDA graphs capture forward/loss/backward; copy, validation, clipping, AdamW and
scheduler remain outside the graph and inside measured full-step time.

Primary layouts are adjacent `(0,1)`, separated `(0,15)` and spread-four
`(0,5,10,15)`, using zero-based indices. Combined means FBT K2 plus NextLat:
ordinary bootstrap followed by one feedback stack containing the selected RT
layers. K3 uses two shared feedback stacks. All16 is a separate stress case.

Independent four-layer CPU references reconstruct native sequential RT
histories, feedback and dense CE/latent/KL objectives, checking all parameter
gradients. They also check attached caches through a frozen upper RT layer and
changes that invalidate a prepared execution plan. Actual-checkpoint GPU runs
compare recompute against materialized BF16 backward, then compare same-candidate
eager and graph execution, including changed tokens/weights, gradient overwrite
and three-versus-three complete AdamW updates. This is not a new full-native-model
BF16-versus-FP32 study.

## Numerical interpretation

All seven primary actual-checkpoint correctness cases pass all five gates.
Initial forward losses are bitwise identical. Every same-candidate graph check
is bitwise identical, and all complete-update comparisons match weights, Adam
moments, scheduler, counters and metrics exactly.

| Case | B/T | Initial gradient global relative L2 |
| --- | ---: | ---: |
| Combined spread2 smoke | 1/32 | 0 |
| Combined adjacent2 | 8/512 | 0.004630 |
| Combined spread2 | 8/512 | 0.007292 |
| RT-only spread2 | 8/512 | 0.005920 |
| Combined spread4 | 4/512 | 0.006483 |
| Combined K3 spread2 | 1/32 | 0 |
| Combined spread2, native context limit | 1/2048 | 0.007477 |

Unchanged budgets are global relative L2 <=0.015625, per tensor <=0.03125 and
maximum absolute error divided by reference tensor maximum <=0.0625. The largest
observed primary values are 0.007477, 0.014221 and 0.045455 respectively. Zero
references require exact zero. Stricter diagnostic flags remain in the raw
reports; recompute-versus-materialized gradients are generally not bitwise.

Multi-layer discrepancies are larger than F3d's single-layer measurements.
They do not grow monotonically in this screen; spread4 also changes batch size,
so this is not a controlled numerical depth-scaling study. The largest coordinate
differences occur in MLP projections, not evidence uniquely implicating Q/K.
There are no new dedicated Q/K activation probes here. Retain native Q/K math;
these results do not establish sustained training stability at arbitrary depth.
Large but finite startup gradients still require the configured clipping.

The all16 B1/T32 stress case passes all five gates exactly, including its
materialized-versus-recompute gradients. Its B8/T512 capacity check also passes,
but is separate from this short all-gradient comparison.

## Resource interpretation

Fresh common-shape results for combined K2+NextLat, B64/T512:

| Selected RT layers | Input tokens/s | Seconds/update | Peak allocated GiB | Matrix TFLOPs/update |
| --- | ---: | ---: | ---: | ---: |
| Single anchor `(0)` | 10,933 | 2.997 | 39.093 | 644.65–689.37 |
| Adjacent2 `(0,1)` | 9,895 | 3.312 | 39.091 | 657.84–701.12 |
| Spread2 `(0,15)` | 9,889 | 3.313 | 39.094 | 657.84–701.12 |
| Spread4 `(0,5,10,15)` | 8,310 | 3.943 | 39.094 | 684.23–724.62 |

Two layers cost about 9.5% throughput and four about 24% versus the fresh single
anchor. Adjacent and spread two-layer timings differ by only 0.058%; three timed
updates do not establish a placement advantage. Nearly identical whole-update
allocated peaks do not imply zero extra RT activation cost. Peak reserved memory
during setup is 60.3–60.9 GiB, distinct from current postcapture 41.4–41.8 GiB.

RT-only spread2 B128/T512 passes at 22,720 input tokens/s, allocated 46.106 GiB,
reserved peak 76.797 GiB and current reserved 49.039 GiB. Because setup reservation
approaches the device limit, the frozen protocol's half-batch follow-up checks
B64 before recommending a comfortable RT-only operating point. This additional
B64 run passes at 19,445 tokens/s, allocated 32.247 GiB, reserved peak 47.963 GiB
and current reserved 34.500 GiB. B64 retains about 85.6% of B128 throughput with
substantially more setup headroom. Use it as the conservative RT-only development
reference. The B128 result remains successful evidence, not an OOM; it is useful
when its tighter setup reservation is acceptable.

Combined spread2 B8/T2048 passes at 4,708 tokens/s, allocated 31.289 GiB, reserved
peak 44.432 GiB and current reserved 33.254 GiB. Both B and T differ from the common
table; this is not isolated sequence-length scaling or a maximum-batch search.
Capacity health at these larger batches is not a full gradient-equivalence
comparison at each batch size. Every timing is a three-update median.

All16 combined B8/T512 passes at 939 tokens/s, allocated 25.440 GiB, reserved
peak 32.543 GiB and current reserved 26.838 GiB. This small-batch stress case
does not estimate an optimized all-layer operating point and is not directly
comparable to the common B64 table. Larger-batch all16 throughput remains open.

Combined parameter counts are 1,267,879,936 active training weights and
1,185,153,024 deployable inference weights; the 82,726,912-parameter NextLat
predictor is training-only. RT-only has 1,176,764,416 active weights; its shared
test wrapper retains 8,388,608 frozen fusion weights, explicitly counted as
resident. Selecting additional RT layers and reusing FBT passes add no weights.
Capacity cards inspect actual optimizer ownership; correctness cards leave
that inventory unavailable after the parity helper releases its optimizer.

The updated analytic estimator adds recompute's extra QK work while preserving
its materialized default. Actual CE/latent/KL/predictor-union counts feed the
cards. Matrix FLOPs exclude pointwise operations, clipping/Adam, communication,
launch overhead and hardware padding; they are not hardware-utilization claims.

## Backends and remaining work

At T512 all historical forward rectangles use the fused RT kernel. At T2048,
two selected RT layers execute 4,088 fused plus 6 eager forward tiles. The six
large rectangles cover 75.0366% of historical attention pair area: per layer,
two 512x512 plus one 1024x1024 rectangles out of T(T-1)/2 pairs. Few fallback calls
therefore do not imply that long-context forward is almost entirely fused.
This is not 75% of total model arithmetic or elapsed time. Historical recompute
backward tiles are fused throughout the checked contexts.

Profile long-context device time and memory before prioritizing kernel extension;
large fallback matmuls may already be efficient. Native RT uses Triton, not FA4.
No new FA4/CuTE integration claim follows from coexistence with ordinary Flash.

Next: complete the remaining feature-combination runtime cards using a documented
multi-layer selection; then broaden graph recovery/accumulation, padding and
online execution readiness. Genuine two-GPU checks require a second GPU. The
existing AccumulateGrad stream warning remains visible, with exact measured
single-GPU parity; DDP ownership is still a separate check. No quality training
or architecture selection follows automatically from this milestone.

## Durable evidence and review

The [W&B project](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat)
contains every run under group `olmo1b-f3e-multi-rt`; individual links are in
the results table. The local selection is
`.runtime/olmo1b-step60000/f3e-final-inputs.json`. Queue66490 and headroom follow-up
81397 completed successfully. The original native checkpoint is reused, and
no disposable few-update weights replace it. Small evidence retention uses
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3e-multi-rt/20260923T145449Z/`, with a
verified [receipt](storage-receipt.json).

Independent reviews checked sequential-reference tests, pass/parameter/FLOP
accounting, reporting/retention integrity, actual gradient records and resource
interpretation. Source inventories, per-tensor error ratios, graph counters and
full optimizer boundaries are validated independently of aggregate pass flags.
The half-batch headroom follow-up was the protocol's anticipated adaptation;
it did not change numerical budgets or replace the successful B128 evidence.
Implementation and closeout: [PR19](https://github.com/taylorbollman/cdrm-w-latent/pull/19).
