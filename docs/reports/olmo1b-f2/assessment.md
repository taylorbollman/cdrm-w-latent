# F2 assessment: numerical health and usable physical batches

2026-09-22. Read with the [measured results](results.md),
[protocol](protocol.md), [usage](../../olmo1b-f2-usage.md) and
[current plan](../../fbt-rt-nextlat-research-plan-v4.md).

Implementation: [PR14](https://github.com/taylorbollman/cdrm-w-latent/pull/14).
Independent final review found no material numerical or scope issues.
The [storage receipt](storage-receipt.json) records verified evidence retention.

**Ordinary-block checkpointing gives us a practical B128/T512 starting point for
both RT and RT+FBT+NextLat on one H10080GB. Keep native Q/K behavior.** The scale
probes identify feedback/auxiliary startup sensitivity rather than a demonstrated
attention-scale defect. This milestone is operational evidence, not a quality
comparison or a claim that large batches have the same learning efficiency.

## What changed

The native model now has default-off ordinary_activation_checkpointing. It uses
non-reentrant checkpointing on ordinary block calls during grad-enabled,
cache-free training. Selected RT calls retain their existing saved-input/
completed-output reconstruction and custom first-order backward. Their slow
sequential forward is never replayed by the new wrapper. Tiny shared-pass tests
count those calls explicitly, and a complete actual-checkpoint BF16 update is
bitwise identical with the option off/on, including Adam moments and scheduler.

Two advanced-index scalar-zero assignments in RT backward were replaced by
equivalent diagonal-view zeroing. This removes a CUDA-capture CPU-scalar-copy
blocker without changing the diagonal/cache-prefix semantics. Existing native
tiled gradient oracles and new rectangular-prefix checks pass.

New bounded runners retain detached activation observations, per-pass/per-loss
gradient summaries, physical-batch timings and graph controls. Observers preserve
outputs, losses and all parameter gradients exactly on the actual combined
BF16 fixtures at both T32 and T128. No Q/K normalization, objective weight,
recurrence equation, FP32-state policy, or default attention backend changed.

## Batch size and memory

All measurements below are complete eager optimizer steps, including CE,
enabled auxiliary losses, backward, clipping, AdamW and scheduler. They use
BF16 autocast with FP32 parameters/gradients/moments, one physical microbatch,
and three warmup plus three timed changing-input/changing-weight updates.
Valid input tokens are counted once, irrespective of FBT pass count.

| Path | Ordinary checkpointing | Physical batch | Input tokens/s | Allocated peak GiB |
|---|---|---:|---:|---:|
| RT | Off | 1 | 367 | 18.4 |
| RT | Off | 32 | 9,194 | 43.3 |
| RT | On | 128 | 20,176 | 41.4 |
| RT | On | 256 | 24,196 | 65.1 |
| RT+FBT+NextLat | Off | 1 | 333 | 20.4 |
| RT+FBT+NextLat | Off | 16 | 4,150 | 46.9 |
| RT+FBT+NextLat | On | 128 | 10,490 | 51.9 |

B128 reserved peaks are42.3GiB for RT and58.2GiB for combined. RT B256 completed
but its65.116GiB allocated peak is just above the frozen65GiB cutoff; we retain
that label rather than moving the boundary after observing it. Combined B256
hit CUDA OOM. Without checkpointing, RT B64 and combined B32 completed but used
71.0 and76.9GiB allocated, respectively, above the intended comfortable range.

The sweep contains22 measured cells (132 complete updates) and one capacity-OOM
cell. All measured endpoints have finite model/optimizer state. No learned
checkpoint from these disposable timing fixtures is a candidate for later
experiments. The source checkpoint is already retained separately.

This supports the user's emphasis on physical batch. It also supports testing
graphs at the intended batch/memory budget rather than extrapolating from B1.
The [RT paper's Section6](https://arxiv.org/html/2604.21215v1#S6) explains why
accumulation cannot improve the per-position recurrent MLP's arithmetic intensity.
Its B512 setting used much smaller models; it is not a capacity promise for this
native1.18B model.

## Numerical interpretation and Q/K decision

At B2/T32, total initial gradient norms are31.6 ordinary,45.1 RT,137.6 NextLat,
672.9 FBT,887.7 combined K2 and2805.8 combined K3. These are unclipped gradients
at the original checkpoint plus newly initialized branches, not trained-model
loss curves. Every objective term uses its own valid-target denominator.

The T32 K3 increase is mainly in native/fusion gradients. Its final pass's CE
and KL contributions, already weighted by.5, have norms1802.6 and1216.3 and
cosines.948/.909 with the full gradient. K2's feedback-pass CE/KL norms are656.7
and909.4. K2 and K3 both have total pass-loss weight2; the comparison is not
explained by counting three full-weight losses instead of two.

This effect is fixture-dependent: at B2/T128 the ordinary/K2/K3 norms are9.70,
1205.1 and1171.5. K3 is not universally larger. The ordinary+NextLat T32 KL
gradient norm131.7 versus CE31.6 and latent2.46 makes auxiliary startup weighting
a concrete future investigation, alongside feedback blending/initialization.
These observations do not isolate one faulty Jacobian or justify automatic
changes to the established objective during this execution milestone.

Observed Q/K and hidden scales remain finite at sampled layers0,1,7,15 through
T128. Feedback layer0 Q/permanent-K RMS is roughly.44–.49, compared with native
ordinary~.96/1.29. Its reconstructed normalized attention entropy is~.96 atT32
and.985–.989 atT128: flatter, not more saturated, than the ordinary reference.
Feedback input RMS remains around the calibrated embedding scale. Keep native
no-QK-normalization and repeat these bounded probes when architecture or learned
state changes. This does not establish inspected attention health at all layers,
all lengths or later training stages.

The observed attention operands are real, but score/entropy summaries are
detached FP32 reconstructions rather than private fused-softmax values. BF16
component-gradient closure errors (up to1.45% here) arise from comparing separate
backward decompositions with one combined backward; they are reported
descriptively, not as independent BF16-versus-FP32 derivative errors. Tiny FP32
decomposition closure and prior independent native RT oracles remain the
correctness references.

## Graph scope and the remaining execution milestone

The short B1/T32 native-stack graph passes all changed-token/changed-weight and
gradient-overwrite checks exactly, with2.91x measured replay speedup. It captures
only stack forward and a fixed hidden cotangent backward; it excludes CE,
NextLat, FBT and optimizer work and disables autocast's weight cache in both
compared executions.

At B8/T512, automatic-dispatch graph forward remains bitwise exact but some
gradients differ by~.38% relativeL2. Controls reproduce~.48% variation between
identical eager RT executions and~.51% with RT entirely off; graph repeats vary
similarly. Traces identify ordinary cuDNN fused SDPA. These controls show a
nondeterministic execution component already present without RT or graphs;
they do not prove every observed difference comes from one specific kernel.
Forcing non-deterministic PyTorch Flash SDPA also fails the original near-bitwise
budget. The installed cuDNN path rejects strict deterministic SDPA execution.
Original failed reports and thresholds remain intact.

With deterministic algorithms, configured cuBLAS workspace and forced PyTorch
Flash SDPA, B8/T512 passes every original comparison bitwise, including changed
tokens, a real weight update and gradient overwrite across repeated replays.
Eager stack/VJP time is1.583s versus.481s graph replay (3.29x); timing allocated
peak17.72GiB and reserved memory after capture28.63GiB. This establishes a stable
reference under explicitly recorded settings without widening any error budget.
The default nondeterministic paths remain qualified; production nondeterminism
is not itself proof of an incorrect derivative. This backend-specific
microbenchmark still does not clear full combined
training, padding/caches, ordinary checkpointing inside capture, or B128 capture.

Next work should integrate a validated static loss layout and changed-state
replay into canonical training, preserving CE/latent/KL masks and reductions.
Then measure combined graph+checkpointing at practical physical batches, including
graph-private memory, before considering a major Flash/CuTE RT kernel rewrite.
Native RT attention still uses eager PyTorch dyadic tiles/custom VJP; ordinary
fused attention does not turn it into an FA4 RT kernel. Keep model-quality
comparisons deferred. Formal FLOP accounting and genuine two-GPU execution remain
in the approved plan; only one H100 is exposed on this machine.

## Model and parameter scope

Original OLMo-1B step60000 (~252B tokens):16layers, width2048,16MHA heads,
SwiGLU8192, native RoPE, tied50304-row vocabulary, non-affine LayerNorm,
no Q/K normalization. **RT selects only index0.** FBT K2 means an ordinary
bootstrap plus one feedback pass, with RT active in the extra pass. All16-layer
RT is not cleared by these measurements.

| Configuration | Trainable parameters | Inference parameters |
|---|---:|---:|
| Native / RT | 1,176,764,416 | 1,176,764,416 |
| RT+FBT+NextLat | 1,267,879,936 | 1,185,153,024 |

The RT-only harness additionally retains8,388,608 frozen fusion parameters;
registered storage therefore differs from active/inference counts. NextLat's
82,726,912 parameters are training-only. Shared/tied weights are counted once.
The complete per-cell accounting remains in the raw capacity report.
