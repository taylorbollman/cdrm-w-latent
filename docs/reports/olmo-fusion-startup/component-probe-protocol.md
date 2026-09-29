# Conditional post-startup component bridge

2026-09-29. Prepare this diagnostic while fusion-only warmup proceeds. Launch
only if root's review finds the startup endpoint useful enough to extend.
No existing helper, model, attention precision or checkpoint source changes.

Strictly import the SHA-pinned fusion-only update 128 into the original fully
trainable NF model, leaving the native backbone, tied readout, fresh predictor
and saved output scale unchanged. This is weights-only import, not optimizer
resume. Use the exact supplementary T128 held-out fixture: two B2 records,
512 input tokens, 508 CE/latent positions and 504 KL triples. Keep K4, beta 1,
jitter 0.02, isolated documents and the original keyed noise byte for byte.

Exactly three FP32/BF16 pairs are permitted, in this order:

1. NF combined CE + latent + KL.
2. NFR CE only, with latent/KL branches active but zero auxiliary cotangents.
3. NFR combined CE + latent + KL.

This is six aggregate cases and twelve physical backwards, with no optimizer
calls. RT is enabled only after strict NF checkpoint validation, by changing
the recipe arm to NFR. Existing native RT layers 0 and 15 then run on every
pass, with alpha 1. Prove that all tensor state, trainability, and every recipe
field other than arm remain unchanged. Do not introduce an RT precision change,
new kernel or custom backward.

Reuse the frozen component backward helpers and global target denominators.
CE pass weights are 1/2,1/6,1/6,1/6; latent and KL each average four passes,
with coefficients 1 for the combined objective. Stop-gradient rules remain
unchanged. The comparison uses full FP32/math/eager and current production
BF16/ordinary Flash/native Triton, with FP32 masters, deterministic controls,
TF32 disabled and autocast cache disabled. Forced SDPA remains active through
backward while forward autocast scopes follow the existing production helper.

Use the completed long-fixture NF-CE update128 report from the new context probe
(`olmo-fusion-startup-context-probe-v1`, `fixture_kind=long`) as a pinned saved-state,
source, runtime and fixture authority. Normalize tuple/list metadata through
the existing JSON-safe tree digests before exact contract comparison; this does
not alter tensor hashes or weaken source/state checks. NF-combined must reproduce its complete
forward fingerprints and each raw CE/latent/KL loss sum at the same precision.
NFR-combined must similarly match the new NFR-CE forward and raw losses. Changing
the objective changes backward only; gradients and incoming cotangents are not
expected to match between objectives. No additional NF-CE backward is scheduled.

Report descriptive full/backbone/fusion/predictor gradient geometry and norms,
each pass's forward and full incoming-cotangent geometry, and existing per-token
support observations. Use an objective-aware health check: predictor gradients
must be nonzero for combined and exactly zero for CE. Require finite gradients,
forward states, cotangents and loss sums, complete participation, exact counts,
four passes per record, unchanged weights/fixtures/modes/RNG, and restored
production flags. Retain existing numerical qualifications; these are functional
gates, not relaxed BF16 acceptance thresholds.

The fusion was trained without RT or active auxiliary losses, so this assesses a
new combination at that state. It is not evidence of successful RT training,
optimizer stability, quality gains or packed-context clearance. Cross-objective
norm changes do not uniquely attribute a source of error. Root owns the GPU
launch, bounded by an external 900-second timeout, online W&B, incremental
reports, source snapshots and immutable GCS retention. CPU tests use explicit
tiny CPU math paths; the production CLI requires the GPU container.
