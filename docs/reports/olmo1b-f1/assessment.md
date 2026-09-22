# F1 assessment: integration passes; eager RT execution is the main efficiency lead

Completed 2026-09-22. **F1 passes within its declared scope.**
All 18 actual-checkpoint cases passed in one execution, with 72 retained-counter
updates plus two recovery replays: 74 physical optimizer executions. Total wall
time was about 10.3 minutes, including checkpoint I/O, hashing and profiling.
See the [full results](results.md), [capability ledger](capability-ledger.json),
[protocol](protocol.md) and [usage](../../olmo1b-f1-usage.md).
[W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8patkvpi).

## What this establishes

The actual original OLMo-1B checkpoint supports short BF16 mixed optimizer
execution for all eight RT/FBT/NextLat combinations, with expected gradients
and active parameter changes, exact inactive-state invariance, and finite
parameters, optimizer moments and endpoint losses. Native input/readout tying
is preserved. Existing independent tiny mathematical tests were reused;
this is not a new full-scale FP32 gradient-equivalence campaign.

Additional actual-model cases passed for K3, fractional alpha/beta,
zero-to-fractional-to-full controls, RT layers (0,15), and representative
T128/T512 execution. In the transition case, 69 tensors participate at beta0
and 71 once fusion becomes active, as intended.

Both ordinary and RT+FBT+NextLat restore their BF16 update2 checkpoint exactly,
then reproduce update3 model/optimizer/scheduler/counters/metrics and the next
CPU/CUDA RNG draws exactly. The combined model's B1/T8 online output matches
split cached execution exactly and rejects an incompatible cache mode.
The two disposable recovery files were deleted after verification; their hashes
and comparison results are retained.

Scoped validation comprises **244 distinct CPU tests**: 212 integration,
observer, retention and retained model/platform tests, plus 32 reporter tests.
An independent raw-report audit checked source hashes, loss aggregation,
parameter ownership, counters, replay and cache evidence.

## Early throughput finding

These are B1/T512 complete optimizer updates on one H10080GB, using the same
operational token/mask fixture. RT selects layer0; FBT uses K2. CE has 256 targets
per 512 valid input tokens. The new branches start from their default random
initialization; this is not a task-performance comparison.

| Model | Seconds/update | Valid input tokens/s | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Ordinary | 0.0676 | 7,571 | 18.40 |
| RT | 1.6266 | 315 | 18.40 |
| FBT | 0.1043 | 4,910 | 18.68 |
| RT + FBT + NextLat | 1.6803 | 305 | 20.40 |

At this tiny batch, RT is approximately 24.1 times slower than ordinary
execution. Adding FBT and NextLat to RT increases measured step time by only
about 3.3%. This suggests the current eager recurrence path dominates this
setting; it does not establish that the other mechanisms are cheap at larger
batches or that RT intrinsically has this throughput disadvantage.

All four traces show cuDNN fused SDPA in ordinary blocks. Selected RT remains
eager PyTorch tiling and custom backward. The RT trace has large self-CPU
durations in the custom backward, launch dispatch and forward control, alongside
many small matrix/copy operations. This is a concrete reason to investigate
batch scaling and eager scheduling/replay overhead before undertaking a large
Flash/CuTE attention rewrite.

The profiler itself changes timing, and its device lists contain both
CPU-attributed scopes and raw kernels. Do not sum those mixed rows into a GPU
utilization percentage. Use synchronized unprofiled step medians for throughput.
The resource screen does not determine maximum comfortable batch, final FLOPs,
optimized kernels, graph behavior or multi-GPU scaling.

## Numerical-health qualification

All **56 updates with retained norm measurements** were clipped at norm1
(44 observed updates plus 12 timed updates). Warmup/profiler-only updates also
use clipping, but their individual preclip norms were not retained.
Large norms are particularly apparent with newly initialized feedback:

| Initial B2/T32 case | Preclip global gradient norm |
| --- | ---: |
| Ordinary | 31.6 |
| Ordinary + NextLat | 137.6 |
| FBT | 672.9 |
| RT + FBT + NextLat, K2 | 887.7 |
| RT + FBT + NextLat, K3 | 2,805.8 |

These are very small fixtures, with random fusion switched fully on at beta1
and unit auxiliary losses. The finite-pass objective also has two units of CE
weight. They are deliberately demanding operational checks, not a recommended
long-training startup schedule. K3 and K2 have the same total pass-loss weight,
so K3's larger norm is a useful lead for studying gradients through feedback.
It does not identify an erroneous derivative or prove a Q/K scale problem.

The evidence supports retaining current model math while investigating scale.
It does not establish unqualified stability over long contexts, large batches
or substantial adaptation. A falling loss on variants of two repeated prompts
does not establish language-model quality.

## Recommended next review milestone

Proceed with a bounded F2 / early-F3 investigation under the functionality plan:

1. Observe activation and Q/K/logit scale and separate CE/NextLat gradient
   contributions at startup, including the K3 case. Distinguish scale introduced
   by fusion, auxiliary losses and recurrence. Use a small same-semantics FP32
   reference only where a concrete concern remains.
2. Measure T512 RT and all-three batches 1/2/4/8, stopping with comfortable
   memory headroom. This can show how much of the present cost is amortized by
   a larger batch before selecting an optimization.
3. Target the observed eager dispatch/replay overhead. Prototype compiled helpers
   or capture where feasible, preserving changed-input/weight behavior and
   returned parameter gradients. Keep the planned fused historical-tile and
   backward work explicit, but do not assume replacing ordinary attention
   will address the current bottleneck.

Keep native absence of Q/K normalization initially. Introduce a separate gradual
normalization adaptation only if the targeted health evidence warrants it.
The genuine two-GPU milestone can proceed on validated kernels when a second
GPU is available. Final resource/FLOP accounting and learning comparisons
remain later review points.

No further GPU job or learning comparison was queued when F1 completed.
All-16-layer RT, longer combined online execution, optimized RT/Flash integration,
compile/graphs and distributed execution remain unvalidated.

Small evidence is retained at
gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f1-integration/20260922T154616Z/;
the [verified storage receipt](storage-receipt.json) records generations/hashes.
The original native checkpoint is reused rather than uploaded again.
