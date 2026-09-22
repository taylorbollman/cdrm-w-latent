# F3 assessment: canonical combined training in CUDA graphs

2026-09-22. The CUDA-graph integration milestone is complete: all seven
correctness cases and six paired capacity measurements passed. No job is queued.
Read the [protocol](protocol.md), [usage](../../olmo1b-f3-usage.md), and
[PR15](https://github.com/taylorbollman/cdrm-w-latent/pull/15).

## What is implemented

The opt-in `StaticFBTTraining` path captures native stack execution, finite-pass
FBT fusion, selected-position CE/NextLat losses and backward. Input validation
and staging, gradient clipping, AdamW and scheduler execution remain outside
capture. Timings include those external operations when labeled complete steps.
The original public eager model and objective APIs keep their behavior.

A prepared layout validates one independent document per row, fixed masks,
positions and target counts before capture. Integer gathers select exactly the
canonical CE positions, latent pairs and KL triples, including the same union
and order of predictor inputs. This avoids dynamic Boolean selection and host
scalar reads during capture. Loss weights, pass coefficients, target/readout
detachments, attached conditioning and cross-pass gradients are unchanged.

All-valid rows use equivalent implicit causal attention so deterministic Flash
can execute the ordinary layers. CPU tests compare this lowering with the
canonical explicit-mask path. Padding keeps the explicit mask and has separate
tiny FP32 coverage; padded GPU Flash graphs are not established here.

Ordinary activation checkpointing reuses the existing non-reentrant block
wrapper. RT still saves inputs/completed outputs and reconstructs its custom
backward without replaying the sequential forward. The selected RT attention
itself remains native eager dyadic tiling inside the graph, not a Flash/CuTE RT
kernel. No Q/K normalization or pretrained model math changed.

## Correctness evidence

All seven actual-checkpoint cases passed:

- RT, B1/T32, ordinary checkpointing off and on.
- RT+FBT+NextLat K2, B1/T32, checkpointing off and on.
- Combined K3, B1/T32, checkpointing on.
- RT and combined K2, B8/T512, checkpointing on.

For each, original tokens, changed tokens, repeated replay and changed weights
produce exactly matching per-pass loss sums and all participating gradients.
The strict F2 relative-L2 and maximum-error budgets remain unchanged. Each case
also compares three complete eager AdamW updates with three graph-backed updates
from the same initial parameter/optimizer/scheduler state: every loss metric,
model tensor, optimizer moment, scheduler state and counter matches exactly.
These are 42 physical optimizer updates, 21 in each execution arm, excluding
warmup/capture/backward-only diagnostic work. They are functionality fixtures,
not model-quality or learning-efficiency evidence.

The longer RT case covers 65 gradients; combined covers 71. Independent review
confirms both B8/T512 reports and source snapshots. Actual dispatch traces show
PyTorch Flash in ordinary blocks. Each uses BF16 autocast, FP32 parameters,
gradients and moments, TF32 off, deterministic algorithms/cuBLAS settings, and
no autocast weight cache. These recorded settings matter; F2's nondeterministic
backend qualifications are not reclassified as passing results.

267 distinct scoped CPU tests pass: prepared loss and forwarding parity,
full-update equivalence, existing native RT/FBT/checkpoint regressions,
layout/ownership guards, reporting and retention. Replay preserves persistent
gradient buffers only for graph participants. Unused parameters retain
`grad=None`, so AdamW does not introduce unintended decay.

## Capacity and interpretation

The separate complete optimizer-step measurements use three preparation updates,
ten backward warmups and three timed updates per arm, plus independently timed
forward/loss/backward. Both arms begin from corresponding pretrained and freshly
initialized branch states. All six paired cells passed finite endpoint checks;
they add 72 physical updates to the 42 correctness updates, for 114 total.
Independent final review verified all 13 reports and matching source inventories.
The designated comparison arms each contain 57 updates; actual execution is
75 eager and 39 graph updates because graph capacity arms prepare Adam eagerly.

All rows below use T512 and ordinary-block checkpointing on one H100 80GB.
Input tokens count each example once, including in the two-pass combined case.

| Configuration | Batch | Eager input tokens/s | Graph input tokens/s | Full-step speedup | Graph peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| RT | 32 | 7,960 | 16,672 | 2.09x | 24.32 |
| RT | 64 | 12,914 | 21,553 | 1.67x | 30.25 |
| RT | 128 | 18,127 | 24,684 | 1.36x | 42.10 |
| RT + FBT K2 + NextLat | 32 | 5,779 | 9,152 | 1.58x | 32.17 |
| RT + FBT K2 + NextLat | 64 | 7,833 | 10,409 | 1.33x | 40.84 |
| RT + FBT K2 + NextLat | 128 | 9,184 | 10,801 | 1.18x | 58.20 |

**Use B64 as the common development point for the next combined checks.** It
retains 96.4% of combined B128 throughput at substantially lower memory cost.
B128 is a validated capacity point and remains useful for RT-only work, where
it improves throughput by 14.5% over B64. This is a resource recommendation,
not evidence about the batch size best suited to learning. No attempt was made
to find the largest fitting batch.

Memory needs two distinct readings. Combined B128 reached a 77.95 GiB reserved
high-water mark during setup/warmup; current reserved memory after capture was
64.85 GiB. The corresponding B64 values were 64.27 and 43.33 GiB. All runs
completed, but B64 provides more room for changes. Reserved peaks are allocator
measurements, not the continuing graph-private footprint. Installed PyTorch's
graph context already clears unused cached allocations before capture; adding
another call there would be redundant. No allocator/runtime change was needed.

These speedups compare matched prepared-eager and graph arms using deterministic
Flash and disabled autocast weight caching. F2 used different execution-policy
settings, so its timing table is historical context, not an exact paired control.
Graphs have the largest relative benefit at smaller batches; after capture,
combined throughput approaches a plateau between B64 and B128. A short device
profile should identify the remaining work before choosing a kernel rewrite.

The raw `capture_seconds` field means setup/warmup plus capture, not the isolated
CUDA capture block. Correctness-audit memory peaks must not substitute for the
separate capacity measurements. All-gradient parity is established at the
recorded B1/T32 and B8/T512 shapes, not at every large capacity shape.
See [results](results.md), [plot](throughput.pdf) and [raw summary](summary.json).

## Execution boundaries and next step

Only RT layer index 0 is selected; the other 15 native layers are ordinary.
FBT K2 includes its ordinary bootstrap and one feedback/RT pass; K3 adds one
more feedback pass. K denotes shared-stack passes, not layer count. The
inference backbone remains the original OLMo-1B step60000 (~252B tokens):
16 layers, width 2048, 16 full-MHA heads, SwiGLU8192, native RoPE, tied 50,304
embedding/readout rows, non-affine LayerNorm and no Q/K normalization.

RT has 1,176,764,416 active trainable/inference parameters. Combined training has
1,267,879,936: fusion adds 8,388,608 and the training-only NextLat predictor adds
82,726,912. Combined inference has 1,185,153,024. Additional passes and RT do not
add parameters. The RT-only harness retains an inactive frozen fusion module;
the report distinguishes resident from active/inference counts.

The graph freezes layout, mode, objective configuration, parameter ownership,
precision and backend settings. New tokens and in-place optimizer updates are
supported. Different masks/shapes/settings require a new validated graph.
GPU graph padding, accumulation, online caches, all-layer RT, nonzero predictor
dropout, distributed execution and graph checkpoint/resume are not cleared by
these measurements. Standard saved checkpoints require cleared gradients;
dispose the plan, clear gradients and rebuild capture at such a boundary.

Next, take brief device profiles at useful batches and build the common
parameter/throughput/memory/FLOP ledger across the feature combinations. Use
those profiles to select a bounded native RT fused-tile prototype; this milestone
does not complete the broader F3 kernel work. Broader RT-layer selections and
genuine multi-GPU checks remain in the approved plan. A second GPU is not
currently exposed. Substantial learning comparisons stay deferred.

Small source, report, trace, test and plot evidence is retained under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3-graph-training/`;
the [storage receipt](storage-receipt.json) pins the exact objects and hashes.
The original retained native checkpoint is reused; disposable fixture updates
did not create additional multi-gigabyte weight archives.
