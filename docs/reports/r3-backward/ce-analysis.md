# Actual mean-CE numerical evidence

These retained NUM runs compare naive autograd with compiled tiled R3 under the same answer-masked mean CE. All **510 intended parameter/input-gradient comparisons** are present, finite and shape matched, and pass the original elementwise rule. Every relative-L2 rule passes. **Twelve gradient max-error/RMS flags and one original logit-coordinate flag remain failed.** The evidence supports the tested backward path without a mathematical-backward patch; it does not establish that every declared numerical criterion passed.

Original criterion: `abs(error) <= 2e-6 + 2e-5*abs(reference)`. Separate fixed tensor criteria require relative L2 and max-error/reference-RMS each <= `2e-5`, with denominator floor `1e-12`. Near-zero means `abs(reference) <= 1e-3*reference_RMS`. No threshold was changed.

| Case | B / T / D | CE naive / tiled | Max gradient relative L2 | Max gradient error/RMS | Gradient original / scale failures | Logit original failures |
| --- | --- | --- | ---: | ---: | --- | ---: |
| tiny-ce | 2 / 32 / 32 | 7.478177547 / 7.478177071 | 1.166e-06 | 1.419e-05 | 0 / 0 | 0 |
| init-b2 | 2 / 128 / 256 | 7.571122169 / 7.571122646 | 1.392e-06 | 1.655e-05 | 0 / 0 | 0 |
| trained-b2 | 2 / 128 / 256 | 2.700320482 / 2.700320244 | 2.990e-06 | 3.394e-05 | 0 / 8 | 0 |
| init-b64 | 64 / 128 / 256 | 7.383334160 / 7.383334160 | 1.230e-06 | 1.688e-05 | 0 / 0 | 0 |
| trained-b64 | 64 / 128 / 256 | 2.847276688 / 2.847276688 | 2.066e-06 | 5.530e-05 | 0 / 4 | 1 |

All runs use 12 blocks, rho=1 at block3, FP32, disabled autocast/TF32, deterministic math SDPA and no accumulation. The tiny case is T32/D32; actual-width cases are T128/D256 with B2 and physical B64, at initialization and the retained R3 u2000 checkpoint. The trained optimizer diagnostic uses copied u2000 Adam state and the last scheduled LR for one NUM step, not a research continuation.

## Every pairwise gradient max/RMS failure

All entries below fail only the max/RMS gradient criterion; original elementwise and relative-L2 criteria pass. **None** of these worst-error reference coordinates is near zero under the fixed definition. A small RMS across a tensor can coexist with much larger individual coordinates; cancellation and sparse gradients must be assessed locally rather than labeled uniformly near-zero.

| Case / tensor | Max absolute gap | Relative L2 | Max gap/RMS | abs(ref at worst)/RMS | FP64 absolute error naive / tiled at that coordinate |
| --- | ---: | ---: | ---: | ---: | --- |
| trained-b2 / `transformer.wte.weight` | 2.015e-07 | 8.156e-07 | 3.135e-05 | 0.211 | 2.769e-07 / 7.539e-08 |
| trained-b2 / `transformer.blocks.3.ff_out.weight` | 1.537e-08 | 9.408e-07 | 2.628e-05 | 11.228 | 2.983e-09 / 1.238e-08 |
| trained-b2 / `transformer.blocks.3.ff_proj.weight` | 1.048e-08 | 8.903e-07 | 2.087e-05 | 4.518 | 8.264e-09 / 2.213e-09 |
| trained-b2 / `transformer.blocks.8.ff_out.weight` | 1.863e-08 | 8.251e-07 | 2.476e-05 | 10.236 | 5.522e-09 / 1.310e-08 |
| trained-b2 / `transformer.blocks.9.ff_out.weight` | 1.397e-08 | 8.488e-07 | 2.270e-05 | 8.311 | 1.220e-08 / 1.772e-09 |
| trained-b2 / `transformer.blocks.9.ff_proj.weight` | 9.313e-09 | 9.533e-07 | 2.292e-05 | 12.404 | 2.058e-08 / 1.127e-08 |
| trained-b2 / `transformer.blocks.10.ff_out.weight` | 1.583e-08 | 8.471e-07 | 2.255e-05 | 1.392 | 2.410e-08 / 8.268e-09 |
| trained-b2 / `transformer.ff_out.weight` | 2.235e-07 | 6.259e-07 | 3.394e-05 | 8.568 | 2.548e-07 / 3.129e-08 |
| trained-b64 / `transformer.blocks.2.att_proj.weight` | 1.059e-08 | 1.035e-06 | 2.087e-05 | 3.775 | 1.109e-08 / 2.169e-08 |
| trained-b64 / `transformer.blocks.3.ff_proj.weight` | 2.794e-09 | 1.078e-06 | 2.736e-05 | 12.040 | 6.619e-10 / 2.132e-09 |
| trained-b64 / `input/embedding_output` | 2.794e-08 | 8.652e-07 | 5.530e-05 | 40.504 | 3.127e-09 / 3.107e-08 |
| trained-b64 / `input/block3_input` | 2.910e-10 | 6.332e-07 | 2.555e-05 | 11.211 | 2.732e-10 / 5.642e-10 |

## Independent FP64 comparison

Each oracle reconstructs both arms' pre-step weights exactly from saved FP32 weights and FP64 deltas, verifies their equality and FP32 round trips, and evaluates the same saved labels/tokens with naive FP64 autograd. The reference changes forward and backward arithmetic precision together; it is not a same-rounded-forward isolated backward oracle. ALiBi/mask constants keep their original construction precision. Tiled FP64 is not claimed.

| Case / FP32 arm | Max gradient relative L2 vs FP64 | Max gradient error/FP64 RMS | Gradient original / scale failures | Original logit-coordinate failures |
| --- | ---: | ---: | --- | ---: |
| tiny-ce / naive | 1.342e-06 | 1.655e-05 | 0 / 0 | 0 |
| tiny-ce / tiled | 1.030e-06 | 2.259e-05 | 0 / 1 | 0 |
| trained-b2 / naive | 4.550e-06 | 8.323e-05 | 0 / 44 | 1 |
| trained-b2 / tiled | 4.559e-06 | 7.230e-05 | 0 / 38 | 5 |
| init-b64 / naive | 2.047e-06 | 2.389e-05 | 0 / 1 | 684 |
| init-b64 / tiled | 2.083e-06 | 2.439e-05 | 0 / 1 | 663 |
| trained-b64 / naive | 6.323e-06 | 1.284e-04 | 0 / 26 | 426 |
| trained-b64 / tiled | 6.785e-06 | 1.226e-04 | 0 / 27 | 410 |

Every oracle gradient passes the unchanged original and relative-L2 rules for both FP32 arms. The oracle-logit original-coordinate failures are separately retained in the last column; all whole-logit scale criteria pass. The max/RMS diagnostic also flags the ordinary naive FP32 arm, so a flag by itself does not localize a defect in tiled backward. In the trained B64 case the oracle discrepancies are larger than the pairwise FP32 backend gap. Complete names, exact coordinates, local values, original flags and classifications for **every oracle gradient max/RMS failure** are retained in `oracle_gradient_scale_failures` in [ce-analysis.json](ce-analysis.json).

## Original logit exception

`trained-b64` coordinate `[26, 112, 518]`: naive `0.0103462822735`, tiled `0.010343728587`, gap `2.554e-06` versus allowed `2.207e-06`. FP64 is `0.0103440348383`; absolute errors are naive `2.247e-06` and tiled `3.063e-07`. This original elementwise flag remains failed. Whole-logit scale diagnostics pass, and recorded FP32 CE is identical between arms.

## Clipping, optimizer moments and updates

| Case | Clip norm naive / tiled | Clipped-gradient scale flags | Moment original / scale flags | Update original / scale flags | Additional update-screen flags | Post-weight original / scale flags |
| --- | --- | ---: | --- | --- | ---: | --- |
| tiny-ce | 16.233800888 / 16.233800888 | 0 | 0 / 2 | 0 / 30 | 1 | 0 / 0 |
| init-b2 | 25.660949707 / 25.660949707 | 0 | 0 / 27 | 0 / 51 | 10 | 0 / 0 |
| trained-b2 | 7.504099369 / 7.504099369 | 8 | 0 / 5 | 0 / 52 | 0 | 0 / 0 |
| init-b64 | 5.207837105 / 5.207837582 | 0 | 0 / 26 | 0 / 51 | 14 | 0 / 3 |
| trained-b64 | 1.589402080 / 1.589402080 | 2 | 0 / 0 | 0 / 54 | 0 | 0 / 0 |

All original elementwise moment, update and post-weight criteria pass, but the stricter scale flags remain in the table and JSON. The additional update screen (`relative L2 <= 1e-3` and `max update gap <= .01*LR`) is separate and has initialization flags. Changing optimizer denominators can amplify small gradients; conversely, large weight magnitudes can hide small update differences. The completed [first-update analysis](first-update-analysis.md) traces the initialization flags to the retained gradients, Adam's sensitivity near epsilon, and final FP32 parameter storage. It preserves every failed screen and compares the available FP64 references. Gradient clipping and Adam are not linear cotangent-scaling tests.

The CE evidence is consistent with ordinary floating-point differences rather than an identified tiled-backward defect. The recommended supported scope is the tested FP32/no-accumulation MQAR regime, combined with the separate raw-scaling, persistent-write and Adam-sensitivity evidence. BF16, accumulation, T512, distributed execution and CDRM are outside this conclusion. No blanket all-tests-pass claim or numerical threshold waiver is justified.

[Machine-readable evidence, source report hashes and all failed coordinates](ce-analysis.json) · [CPU aggregation script](ce_analysis.py)
