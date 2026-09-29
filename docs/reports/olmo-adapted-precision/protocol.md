# Adapted-state NF precision comparison

2026-09-29. User approved the next step after PR42. Base `eb42ad2`, branch
`feat/olmo-adapted-precision`. Freeze this protocol and the new helper/tests
before GPU execution. Previous sources and protocols remain unchanged.

## Question and scope

Does the large initial NF BF16/full-FP32 gradient difference persist when using
the saved, already-adapted O5c mixed update512 backbone and fusion? This is a
new diagnostic in the current runtime, not a historical resume or causal
fusion-only ablation. O5c inherited an adapted O5b backbone; both backbone and
fusion differ from the cold source. O5c trained at K2 without RT, NextLat or
jitter. The present K4/jitter fixture intentionally tests a different operating
condition. No new training, optimizer, architecture or precision-policy change.

## Explicit model-only import

Pinned checkpoint `.runtime/olmo1b-step60000/o5c-pilot-01/mixed/update-000512.pt`:
4,807,843,871 bytes, SHA256
`7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a`.
Pinned report at the same directory, `report.json`, SHA256
`020a204ae02d3dfa1af2753f278cb465d73120ca8133258bc8a84272e15afbc8`.
Retained generation `1790059437165208` is prior storage authority; no new
download is needed if local file verification succeeds.

Construct the current deterministic NF model first and verify its cold state,
configuration and fixture against the completed recurrence matrix. Then use a
new explicit weights-only adapter, leaving the old O5d validator unchanged.
Record all historical/current source pins and explicitly account for the six
changed core files. Reject other unexplained source differences. Validate report
and checkpoint metadata, architecture and fusion configuration, exact tensor
keys/shapes/dtypes, finite values, tied ownership and complete imported state.
Preserve saved buffers, especially fusion.output_scale; never recompute that
scale from the adapted embedding. Validate before copying and preserve current
parameter objects/trainability. Tiny CPU fixtures test the import contract;
the actual pretrained diagnostic requires the GPU container.

Import no optimizer, scheduler, data cursor, training/eval flags, requires_grad
flags or RNG history. Keep the current fresh predictor, pinned to its original
seed/bytes; its auxiliary cotangents are zero. Verify RNG, runtime flags and
all non-imported state are unchanged. Record exactly which tensors were mapped.

## Two matched precision cases

Reuse the fixed isolated NF fixture: two physical B2/T16 records, valid lengths
(16,5)/(6,2), 29 inputs, 25 CE and latent targets and 21 KL targets. The campaign
recipe remains T1024; this is its short numerical fixture. Keep all masks,
positions, token IDs and keyed feedback noise identical to the cold matrix.
K4, beta1, jitter0.02; CE weights (1/2,1/6,1/6,1/6), global denominators; NextLat
branches execute with zero auxiliary cotangents. No temporal RT is selected.

Run full FP32/math and production BF16/Flash at identical adapted parameters
and buffers, four physical model backwards in total. Use existing component
backward and pass-observation helpers. FP32 master parameters, raw unclipped
gradients, TF32 off, highest FP32 precision, autocast cache off, deterministic
controls configured before CUDA. Restore production flags before each path;
ordinary checkpointing remains enabled, pointwise eager and RoPE native.

Report per-loss sums/counts, normalized CE objective, per-pass valid hidden-state
and total incoming-cotangent geometry, complete parameter-group gradient norms,
BF16/FP32 relative L2/cosine/norm ratio and descriptive per-parameter errors.
Pass cotangents include later feedback; they are not fixed across the precision
pair. Retain fingerprints/source snapshots and small numerical reports; no
complete parameter-gradient vectors or new model checkpoint are exported.

Cold results supply context only; do not require adapted losses or gradients to
equal them. Before import, verify the cold construction without another backward.
After import, compare precisions within the adapted state. Retain both raw and
relative errors so a changed gradient norm cannot conceal absolute differences.

## Operational checks, interpretation and stop

Require finite losses/gradients/states/cotangents, complete gradient participation,
zero predictor gradients, exact fixture counts, four passes per record, unchanged
parameters/buffers/RNG/inputs/sources and runtime flags restored. Import integrity
and finite execution are operational gates, not BF16 numerical clearance. Do not
relax or introduce a numerical acceptance budget based on the observed result.

Stop after these two cases and assess. Better forward/gradient agreement would
motivate controlled feedback-startup work, not prove that fusion adaptation fixes
BF16. Persistent disagreement would motivate selected ordinary-layer replays.
Neither outcome clears RT/NFR, auxiliary objectives, packed T1024, graphs, DDP,
optimizer updates or training quality. Further experiments need their own scope.

Root alone launches GPU0 inside the project container under a 900-second external
limit. Save atomic per-case reports and track online W&B. Retain evidence in
`gs://fast-chunks`, preserve failed stages, and push progress every20–30minutes.
GPU1 stays unused. No long uncheckpointed training is part of this milestone.
