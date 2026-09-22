# O5a: FBT reference implemented and checked

Completed 2026-09-22 on the original OLMo-1B step60000 (~252B tokens)
checkpoint. **The bounded correctness milestone passes.** No model learning,
checkpoint conversion, or changes to existing OLMo/RT math were performed.
The two new modules wrap the existing stack and training objectives.

The implementation supports FBT, RT and NextLat independently, including all
eight combinations. Finite passes share one backbone, keep cross-pass gradients
attached, and shift previous-pass post-finalnorm states by one token. Pass0
remains ordinary. Exact online execution consumes the freshly completed
previous-token state through a separate, provenance-checked cache. NextLat
remains auxiliary training only.

Fusion follows the pinned author's gate-product mechanism with explicit OLMo
adaptations: FP32 RMS reductions, epsilon1e-5, and fixed scale calibrated to the
native embedding matrix. Measured scale is **0.0370765589**. The two new
2048-by-2048 matrices add **8,388,608 parameters**, bringing FBT to
**1,185,153,024** and FBT+NextLat to **1,267,879,936** parameters. Native tying,
RoPE, LayerNorm, MLPs, vocab rows and all original weights are preserved.
The source [audit](../../../cdrm/pretrained/_fbt_reference/README.md) distinguishes
the mechanism from an exact Nanochat or paper reproduction.

## Actual-checkpoint checks

H100 80GB, B1/T8, FP32 with TF32 off, native ordinary math attention, layer0
when RT is selected. Independent reference composes direct fusion equations
with the sequential RT backend; the candidate uses tiled RT. Every active
parameter gradient is compared, including the new matrices and optional
predictor. Inactive fusion at ordinary endpoints is explicitly accounted for.

| Case | Outcome | Aggregate gradient relative L2 error |
| --- | --- | ---: |
| K1, configured RT extra passes | Exact ordinary endpoint | 0 |
| K3, beta0, no RT | Exact ordinary endpoint | 0 |
| K2, alpha=beta=0.37, RT | Pass | 1.35e-6 |
| K3, beta1, FBT-only | Pass | 9.50e-6 |
| K3, alpha=beta=1, FBT+RT | Pass | 7.72e-6 |

The separate K2 alpha=beta=0.37 objective checks also pass, with NextLat off
and on. They compare each loss term and all active gradients against an
independent dense objective, including detached targets/readout and the
`pass0 + mean(extra passes)` reduction. Loss sums count each data position once;
they do not silently multiply the denominator by pass count.

At alpha=beta=0.37, exact online states match independently recomputed prefixes
to **7.88e-7 relative L2**, and cached split continuation matches the whole
online call exactly. The finite-pass discrepancy on this same eight-token
fixture is:

| Complete passes | Hidden-state relative L2 versus exact online oracle |
| ---: | ---: |
| 1 | 0.27891 |
| 2 | 0.036674 |
| 3 | 0.005447 |
| 9 | 9.97e-7 |

This verifies causal convergence of the implementation. It does not show
that two or three passes will suffice for a trained model or long sequences.
The random feedback matrices have not learned useful behavior yet.

One K2 FBT+RT+NextLat **BF16 mixed** check has finite losses and every active
gradient. Aggregate gradient relative L2 versus the independent FP32 reference
is **0.01510 (1.51%)**. This is a descriptive short-fixture comparison, not
BF16 large-batch, long-context or learning clearance. All validation forwards
and backwards leave every model/buffer byte unchanged; native embedding/readout
tying remains intact.

## Tests, records and limits

CPU validation passes **549 distinct tests**: 526 model/platform tests and
23 evidence-retention tests. The model/platform group includes **60 new FBT
tests** (28 core, 9 independent adversarial, 23 objective/platform). Coverage
includes all switches, independent online replay and gradients, causal
Jacobians, padding/documents, multiple outstanding modes, finite convergence,
cache rejection after mode/weight/dtype changes, unequal microbatch reduction,
and bitwise optimizer/save-resume replay on a tiny FBT+RT+NextLat model.

A stale-cache risk from direct child-module dtype round trips was found and
fixed before actual-checkpoint execution. Final source hashes are in
[validation-summary.json](validation-summary.json); no sources changed during
the GPU run. [Test record](test-results.txt), [protocol](protocol.md),
[usage](../../olmo1b-fbt-usage.md), and
[W&B validation](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ii69nrrg)
provide reproduction details. The [verified GCS receipt](storage-receipt.json)
retains source/report evidence under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-fbt-reference/20260922T033600Z/`,
reusing the immutable original checkpoint without uploading a model copy.

Scope limits: one native checkpoint, short actual-checkpoint fixtures, tiny
optimizer recovery, first-order tiled gradients, and one GPU. No learned FBT
quality result, capacity benchmark, prefix mixing/switching, training jitter,
production generation interface, all-16-layer RT, distributed validation, or
new long learning run is claimed.

## Next review milestone

O4's NextLat deficit begins before recurrence activation. Start the next
learning investigation with **ordinary versus FBT-only**, initially without
NextLat, rather than combining two new adaptation problems. Keep the ability
to enable NextLat and RT independently; the negative short O4 result does not
rule out a later interaction.

The ordinary control must match the pass-loss weighting. With K2/gamma1, the
FBT objective has aggregate CE weight2. A new K2/beta0/no-RT control gives the
same objective mass and data/update accounting as the feedback arm. O4's
single-pass control is useful context but is not automatically a matched causal
baseline. Universal clipping in O4 makes it especially unwise to assume that
changing objective scale is inconsequential.

Before that learning run, fix the fusion warm start/beta transition, pass count,
any sampled plain-prefix mixing/jitter policy, and actual T512 batch capacity.
Evaluate individual pass NLL plus exact online prediction/retention; aggregate
training CE is not final-pass NLL. Keep token exposure separate from pass
compute. Review those choices before committing a large learning budget.
