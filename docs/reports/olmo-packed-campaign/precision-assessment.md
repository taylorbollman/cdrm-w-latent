# Bounded precision-component assessment

2026-09-29. Evidence: `.runtime/olmo-packed-campaign/precision-components-01/report.json`,
[W&B run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/6a48mv6r).
The run completed all 12 backward probes and the fixed-weight/source/RNG checks
in 77.5 seconds. “Passed operational diagnostic” means those checks passed; it
is not numerical acceptance of BF16.

## What this localizes

The earlier **3.40224%** BF16 sparse-versus-prepared raw-gradient difference is
reproduced. CE alone agrees exactly between those BF16 layouts. Differences
appear when either NextLat auxiliary loss contributes a gradient. They are much
larger in the backbone and fusion than in the predictor.

| Objective | BF16 prepared versus sparse gradient relative L2 | Gradient cosine | Difference norm | Sparse gradient norm |
|---|---:|---:|---:|---:|
| Combined | 3.40224% | 0.999422 | 32.1553 | 945.1221 |
| CE only | 0% | 1.000000 | 0 | 355.5881 |
| Latent only | 2.17298% | 0.999782 | 0.5197 | 23.9162 |
| KL only | 2.28388% | 0.999744 | 18.1540 | 794.8741 |

For the combined objective, the layout discrepancy is 3.39344% in the backbone,
3.83846% in fusion and 0.070034% in the predictor. Predictor discrepancies for
latent-only and KL-only are 0.024178% and 0.071670%, respectively. This narrows
the layout issue toward auxiliary-loss cotangents and their propagation through
the backbone. It does **not** establish the exact operation responsible. The
KL-only difference is the larger absolute auxiliary difference here, but one
cannot assign percentages of the combined discrepancy from these norms.

In particular, combined-backward BF16 arithmetic need not equal a sum of
separately executed component backwards: cotangent addition and rounding happen
at different points. The combined difference norm exceeds the sum of the two
individual auxiliary difference norms. A component-sum decomposition would
therefore be misleading. None of these probes use DDP, CUDA graphs, optimizer
updates or packing; those mechanisms are unnecessary for this discrepancy.

## A larger shared cross-precision difference

Both BF16 layouts also differ substantially from the common full-FP32 reference:

| Objective | BF16 sparse versus FP32 relative L2 | BF16 prepared versus FP32 relative L2 | Sparse/FP32 norm ratio | Sparse/FP32 cosine |
|---|---:|---:|---:|---:|
| Combined | 85.96% | 86.13% | 0.4951 | 0.5112 |
| CE only | 95.93% | 95.93% | 0.4019 | 0.3001 |
| Latent only | 104.17% | 104.35% | 0.6545 | 0.2621 |
| KL only | 71.95% | 72.55% | 0.5973 | 0.7024 |

Thus the BF16 sparse path is not a sufficiently independent precision reference
for the prepared path. Their close mutual agreement relative to FP32 does not
clear their shared sensitivity. CE alone shows the large cross-precision gap,
even though its two BF16 layouts agree exactly, so fixing only auxiliary-loss
layout arithmetic cannot explain or resolve the full observation.

Forward scalar losses move much less: combined objective is 14.434714 in FP32
versus 14.547510 in BF16 sparse; CE mean is 7.280284 versus 7.304820, latent mean
0.902395 versus 0.909805, and KL mean 6.252036 versus 6.332885. Small scalar-loss
differences do not by themselves bound gradient-direction differences.

This comparison changes precision **and execution kernels**: full FP32 uses
math SDPA and eager native RT, whereas BF16 uses Flash SDPA and Triton native
RT. It is not an isolated test of BF16 rounding alone, nor evidence that Flash,
Triton, RT, FBT or the lack of Q/K normalization specifically caused the gap.
The prior FP32 sparse/prepared check (~7.38e-7 relative L2) supports mathematical
agreement of those layouts in FP32; it does not resolve the backend/precision
question above.

## Scope and recommended follow-up

This is one initial pretrained NFR state, K4 with native RT at layers 0 and 15,
T16, physical B2, two virtual rank fixtures, 29 valid input tokens, 25 CE
and latent targets and 21 KL targets. All auxiliary weights and original
pass-loss weights are retained; components are isolated by zero cotangents on
other enabled losses. There is no updated-state, production-length or
training-quality conclusion. Neither a strict FP32 equality budget nor a new
arbitrary BF16 threshold should be used to declare these descriptive
cross-precision measurements accepted or rejected. The original 3.40224%
layout qualification remains open.

Before quality training, prioritize two bounded localization checks:

1. **Separate loss layout from backbone propagation.** Reuse identical detached
   hidden states, embeddings, readout and predictor inputs for sparse/prepared
   auxiliary losses, measuring their incoming hidden-state gradients directly
   in FP32 and BF16. Then replay a common cotangent through the same backbone.
   This distinguishes loss-layout rounding from sensitivity in propagation.
2. **Separate precision from kernel choice.** Add a BF16 eager-native-RT/math-SDPA
   bridge on the same state and data. Compare it with both current endpoints;
   if the gap remains, change one precision boundary at a time and record
   activations/cotangents by FBT pass and RT layer. Start with CE-only and the
   combined objective rather than a large sweep. A nearby real packed-data
   fixture can subsequently check whether the tiny fixture is representative.

These checks should precede any decision to change Q/K normalization or model
architecture. The packed data, distributed execution and restart readiness
work can continue under the explicit numerical qualification. Passing that
operational work does not clear this separate precision issue.
