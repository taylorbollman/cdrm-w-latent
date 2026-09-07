> Discussion proposal only. The authoritative pre-B64 criteria are
> [confirmatory-criteria.md](confirmatory-criteria.md), written by the root agent
> before confirmation. That document requires **both** the L2 and maximum
> added-error budgets. This proposal's suggestion to make the maximum budget
> diagnostic was not adopted. Its timestamp is later than the authoritative
> freeze; it does not supersede or modify the confirmatory rules.

# Initial engineering-screen proposal (superseded)

Frozen at 2026-09-06T23:18:25.858972+00:00, after inspecting only legacy and candidate-v1 initialization
B2 results. No trained candidate or B64 result was inspected by this reviewer.
These are engineering screens for an approximate mixed-precision implementation,
not replacements for historical bounds or proof of training equivalence.

For each of all 100 canonical parameter gradients and the embedding-output and
block-3-input gradients, let `g32` be naive FP32, `gn` naive mixed, and `gt` tiled
mixed on the same checkpoint, batch, and aligned masked mean CE.

The principal screens are:

1. Every intended gradient is present, finite, and FP32. Parameters and Adam
   state remain FP32; the declared recurrent precision boundaries are observed.
2. Extra backend relative error remains within the existing BF16 bound:
   `||gt - gn||_2 / max(||gn||_2, 1e-12) <= 0.015625`.
3. Extra backend error must not exceed the ordinary naive precision error plus
   the original FP32 tolerance floor. Define the tolerance tensor exactly as
   `f_i = 2e-6 + 2e-5 * abs(g32_i)`. Require
   `||gt - gn||_2 <= ||gn - g32||_2 + ||f||_2` for every tensor separately.
4. Apply the analogous added-error budget to logits using RMS:
   `RMS(zt - zn) <= RMS(zn - z32) + RMS(f_logits)`, where
   `f_logits_i = 2e-6 + 2e-5 * abs(z32_i)`.
5. Preserve the original FP32 checks and verify normalized power-of-two
   gradient scaling at 1/32 and 32. BF16 homogeneity was exact on the exploratory
   initialization B2 fixture; continued checks must retain any discrepancy.

The additional tail diagnostic is fixed now as
`max(abs(gt - gn)) <= max(abs(gn - g32)) + max(f)` for each tensor.
It is reported separately rather than treated as a universally justified hard
maximum-error gate. A failure requires localization; passing this comparison
alone does not establish that a shared tail is harmless.

All historical gradient relative-L2/max-error-over-RMS and logit elementwise
flags remain visible for all three comparisons. In particular, satisfying the
extra-backend screens does not clear an excessive error shared by both mixed
backends against FP32. No global aggregate may hide a failed tensor.

For initial Adam interpretation, define the near-zero reference-gradient mask
per parameter as
`abs(g32_i) <= 0.015625 * RMS(g32_parameter) + 2e-6`.
The RMS is the FP32 **gradient** RMS for that parameter, not its weight RMS.
Report sign disagreements and parameter-update disagreement energy inside and
outside this mask, absolute delta/LR, global and per-block update alignment,
and exact-zero/active-support statistics. This classification does not remove
coordinates from the historical gradient checks. First-step Adam ratios or
nearly 2*LR coordinate differences are not by themselves a backward defect;
non-negligible-gradient sign flips or substantial disagreement energy still
require investigation. No fitted numeric Adam pass threshold is introduced.

Rationale: ordinary BF16 projections impose a precision floor shared by both
backends. The added-error budget asks whether online/custom recurrence adds
more error than that baseline, while retaining the prior relative bound. The
FP32 allowance is the exact pre-existing elementwise tolerance tensor reduced
in the same norm as the comparison, not a maximum selected from candidate
measurements. Sparse support can dilute full-tensor RMS, but active-support
statistics are diagnostic and cannot justify blanket dismissal of dense tails.

Fresh initialization and trained B64 batches, with fixed independent counters,
provide confirmation; retained initialization B2 is exploratory. Actual CE,
clipping/Adam effects, bounded paired training, midpoint recovery, and measured
performance remain separate requirements for an engineering clearance.
