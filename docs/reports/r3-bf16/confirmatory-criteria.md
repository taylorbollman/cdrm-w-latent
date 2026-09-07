# Frozen confirmatory screens

Frozen on 2026-09-06 before the fresh B64 fixtures are executed. The engineering
choice uses the initialization B2 exploration; neither B64 result informed it.
The trained B2 result is exploratory and is not held-out confirmation.

Use naive FP32 gradients `g32`, naive mixed `gn`, and tiled mixed `gt`, separately
for every parameter and the two retained input gradients. All must be present,
finite, and FP32. Keep every historical flag in the original reports.

1. Additional backend relative L2 remains bounded by the existing BF16 screen:
   `norm(gt-gn) / max(norm(gn), 1e-12) <= 0.015625` (twice BF16 epsilon).
2. Additional backend error must fit within the independently measured naive
   precision error plus the old FP32 tolerance. For
   `f = 2e-6 + 2e-5 * abs(g32)`, require both
   `norm(gt-gn) <= norm(gn-g32) + norm(f)` and
   `maxabs(gt-gn) <= maxabs(gn-g32) + max(f)`.
3. Apply the analogous RMS error budget to logits, using the same tolerance
   formula. Record actual answer-only mean CE independently.
4. Preserve historical max-error/reference-RMS `<=0.0625` and elementwise
   flags. Inspect failing coordinates, active support, magnitude and direction;
   the new error budget does not erase these failures. Norms use detached CPU
   FP64 reductions, with the existing `1e-12` denominator floor.
5. Require fixed-forward power-of-two homogeneity and isolated earlier-write
   credit. These support the comparison but cannot alone prove correctness.
6. Retain actual AdamW deltas, clipping and moments. For initial-step attribution,
   classify coordinates by `abs(g32) <= 0.015625 * RMS(g32_parameter) + 2e-6`;
   report sign changes and disagreement energy both inside and outside this
   set. This is an interpretation aid, not a new optimizer pass threshold.

The rationale is to bound the *additional* approximation introduced by tiled
execution against the observed precision conversion error, while retaining the
old relative-L2 ceiling. Neither BF16 backend is an exact oracle. This does not
claim that all BF16-versus-FP32 gradients satisfy the historical screens.
Historical tails need explicit interpretation and bounded training/recovery
evidence before an operational recommendation.

Confirmation uses the entire next unseen batch at counters 1 and 2001 for the
retained initialization and update-2000 checkpoints, respectively: B64/T128,
unchanged masks, normalization, ALiBi, rho1, optimizer state and runtime policy.
Run the three arms anew on these fixtures. Do not revise these screens in
response to their outcomes; record unresolved failures if they occur.
