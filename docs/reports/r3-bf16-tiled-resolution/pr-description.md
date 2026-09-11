Tiled R3 BF16 gradients are now evaluated against FP32 directly. The previous
maximum-error budget could reject a tiled result that was more accurate than
naïve BF16. Fixed-operand probes traced a substantial part of that disagreement
to repeated rounding in naïve's shared BF16 weight cast. These results do not
justify a tiled arithmetic correction.

The validation harness now includes the primary tiled comparison, controlled
cache/reduction settings, optional instrumentation, retained block boundaries,
and source snapshots. New diagnostics cover dense accumulation and saved
reference/optimizer packets; the FP64 fallback accepts those packets. Numerical
and operational runs log online to `taylorbollman/r3-bf16-tiled-validation`.

A separate recovery check exposed fresh compiler/kernel-selection drift across
containers. Sharing Inductor's cache restored exact recovery in a controlled
100-update/midpoint-resume pair. The launcher now persists that cache, and
operational identities record its directory/policy. This is a same-runtime
result, not a cross-machine determinism guarantee.

Validation: both fresh B64/T128 fixtures pass the frozen BF16-reference and
optimizer screens; global gradient errors are 0.836% and 1.282%. Four very small
FP32 max/RMS flags remain on the trained fixture, so the complete frozen machine
suite remains failed. All coordinate-wise FP32 checks pass, including an
independent full-batch FP64 reference. Original failures are preserved.
The paired 100-update runs are stable; cached midpoint recovery is bitwise
exact. Twenty-seven targeted CPU tests and launcher persistence checks pass.
BF16 uses 27.9% less allocated memory and takes 23.1% longer per update here.

Recommend scoped opt-in tiled BF16 use with the documented runtime and reviewed
FP32 exception. FP32 remains the default; CDRM and broader shapes await their
own validation. See [results](results.md), [frozen contract](validation-contract.md),
and [W&B summary](https://wandb.ai/taylorbollman/r3-bf16-tiled-validation/runs/stxue79z).
