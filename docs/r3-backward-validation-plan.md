# Bounded FP32 R3 backward investigation

This milestone implements the user's
[validation request](inputs/r3_backward_validation_request.md). It performs NUM
diagnostics, not a research-training sweep, and preserves all Stage B artifacts.
One H100 is available inside the required Docker container. Use at most 45
minutes of GPU command execution; stop and report unresolved discrepancies if
the bounded investigation cannot distinguish their causes.

The initial sequence is:

1. Audit the recorded raw failures. Reconstruct the D32/T128/B2/V256 fixture in
   its CUDA RNG order and verify the original token and cotangent hashes. Retain
   the historical automatic-SDPA profile separately from strict math SDPA.
2. At a fixed forward graph, compare naïve autograd and compiled tiled backward
   for the raw output cotangent at scales 1/32, 1 and 32, plus a scale matching
   the norm of a masked-CE cotangent. Preserve both the legacy elementwise rule
   and the additional normwise diagnostics. Capture embedding-output and actual
   block-3 input gradients and every canonical parameter gradient.
3. Run actual aligned, answer-only mean CE on MQAR: first a small D32/T32 fixture,
   then D256/12 blocks/T128/B2 using the retained Stage B R3 initialization and
   trained final checkpoint. If these fit comfortably, include physical B64
   checks at both checkpoints to cover the pilot's actual batch without
   accumulation. Use the checkpoint's next training batch, not final-test data.
4. Compare one NUM AdamW step from identical weights and optimizer state, with
   the Stage B betas, epsilon, weight decay and clipping. Initial weights use
   the first-update warmup LR; the trained final checkpoint uses its retained
   final LR for this diagnostic only. No research checkpoint is updated.
5. Demonstrate gradients through earlier persistent writes with a small
   last-position-only block-output cotangent. If any discrepancy remains
   ambiguous, use a small naïve FP64 reference or targeted directional check.

## Acceptance and interpretation, fixed before new GPU comparisons

The original elementwise rule remains `abs(error) <= 2e-6 + 2e-5*abs(reference)`;
its failures must not be relabeled as passes. Logits and CE retain this rule.
Additional gradient diagnostics use both relative L2 error and maximum absolute
error divided by reference RMS, each bounded by `2e-5`, reusing the original
relative-error budget at tensor scale. These are engineering checks, not a
floating-point error theorem. Norm reductions use FP64 and an explicitly
recorded `1e-12` denominator floor. Near-zero coordinates are reported separately.

For backward homogeneity, compare `G(alpha*v)/alpha` with `G(v)` before clipping
or Adam. Preserve raw error magnitudes too. Power-of-two scales are useful because
they do not introduce new rounding when scaling ordinary finite FP32 values.
Include the actual CE-derived gradient norm; a normalized random test alone
cannot clear the training path.

Optimizer diagnostics compare pre-clip gradients, clip norms/coefficients,
moments, actual parameter deltas and resulting weights. Moment comparisons use
the same normwise budget. A practical update screen flags a tensor if relative
delta L2 error exceeds `1e-3` or a coordinate's update discrepancy exceeds
`0.01 * learning_rate`; this separately labeled screen limits perturbations to
0.1% of tensor update magnitude and 1% of the LR scale. Passing is not by itself
a certificate: investigate flagged coordinates and distinguish FP32 subtraction
resolution and Adam sensitivity from gradient defects. No bitwise equivalence
between numerical evaluation orders is required.

BF16, accumulation, T512, distributed execution and CDRM are outside clearance.
Precision is FP32, autocast/TF32/dropout off, deterministic algorithms on, ordinary
attention math SDPA, and R3 rho=1 with four backward chunks and compiled helpers.
Record any departure in a historical reproduction. Source and fixture identities,
commands, all failure classifications and any focused fix belong in the report.

Artifacts live under `.runtime/r3-backward/20260906T210249Z/` on the persistent
home filesystem, with durable copies at
`gs://fast-chunks/cdrm-w-latent/r3-backward/20260906T210249Z/`.
