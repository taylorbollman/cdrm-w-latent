# Ordinary-stack packed precision calibration

2026-09-29. One new N-only CE precision pair on the exact frozen packed T1024
fixture, after the startup NF packed and component checks. This stage calibrates
ordinary-stack arithmetic; it does not change or clear an acceptance threshold.

Pin the same two B1/T1024 records (2,048 valid inputs,2,046 CE targets,2,040
latent pairs,2,032 KL triples), fixture SHA256
`4932410f9fd370d9dae20a1075bf2a191c642e1563eb8557a6fe02fd7e83a975`.
Use cold NF authority report SHA256
`c1cfb4a0af1524872033828e8a802d26fb534543331785d11f30b42e9726ae02`.
Keep the original OLMo step60000 backbone/tied readout and seeded predictor.
All weights are unmodified. No fusion startup checkpoint is needed when fusion
is inactive, and no second saved-state comparison is authorized here.

Construct the original NF model, verify its cold state, runtime, recipe and
fixture against the retained report with the unchanged strict loader, and
load the fixture under its existing NF noise/continuous-stream contract. Then
explicitly select arm N: one ordinary pass, no RT, no feedback/noise, dormant
fusion frozen; native and predictor parameters remain trainable. Preserve all
packed token/document/mask values and tensor identities/state. NextLat branches
remain active with zero auxiliary cotangents; CE weight is1. K4 NF has a
different weighted objective, so raw cross-arm gradient norms are descriptive.

Execute exactly two aggregate cases/four physical backwards: FP32/math and
current production BF16/Flash, with unchanged FP32 master weights, ordinary
activation checkpointing, TF32 off, deterministic controls before CUDA, and
explicit autocast/forced-SDPA scopes through backward replay. No training,
optimizer, DDP, CUDA graph, new kernel, new normalization, clipping or LR sweep.
At each precision, require the baseline's entire output and token embeddings
to be byte-identical to the saved cold NF first pass on both records. Incoming
cotangents intentionally differ because later NF feedback is absent.

Report within-N gradient geometry, absolute norms/differences/cosines, forward
and cotangent position geometry including actual support and document-boundary
positions. Require positive finite backbone gradients, exactly zero predictor
gradients from the zero auxiliary cotangents, and no dormant fusion gradients.
Check model state, trainability, modes, RNG, source/fixture/reference bytes and
both first-pass anchors, then clear gradients and restore production flags.
Existing legacy numerical flags are descriptive only.

Root alone runs the GPU stage in the project container, with a900-second
external limit, online W&B, incremental reports, frozen source snapshots and
verified retention. CPU tests use actual tiny models, objective gradients,
precision-specific forward anchors, failure cleanup and explicit CPU math
oracles. No CPU fallback exists for the production CLI. A successful stage
establishes only this bounded calibration, not BF16 production readiness.
