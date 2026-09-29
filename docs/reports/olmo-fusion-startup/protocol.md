# Cold-start fusion learning and matched precision protocol

2026-09-29. User authorized about six hours of useful overnight numerical and
readiness work without another review. This is the first bounded learning stage;
see overnight-plan.md for the broader conditional continuation. Freeze this file,
new helpers/tests and existing sources beforeGPU execution. Original production
math, historical diagnostics and previous acceptance budgets remain unchanged.

## State and objective

Construct original OLMo-1B step60000 from its pinned prepared checkpoint, with
current deterministic fresh fusion and predictor. NF configuration: K4 feedback,
beta1, jitter0.02, first-pass policy configured-rt-v1, no temporal RT. CE pass
weights are(1/2,1/6,1/6,1/6), with a common global target denominator counted once.
Training freezes native weights, tied embedding/readout, predictor and fusion
output_scale. Only state_proj.weight and token_gate.weight train. Preserve
autograd through later backbone passes; freezing parameters is not a no_grad
context. Keep first-pass CE as its constant reporting/objective contribution.

Training uses the existing FBT forward and CE loss functions with auxiliary
arithmetic omitted. CPU tests compare its CE/fusion gradients with the existing
full objective's zero auxiliary cotangents. Numerical probes use a separate full
trainable NF model and the established sparse CE backward with latent/KL branches
still present but zero cotangents. No optimizer is instantiated by a probe.

FullFP32/math-SDPA training, TF32off, highest FP32 precision, deterministic controls
beforeCUDA, ordinary activation checkpointing, nativeRoPE, eagerpointwise; no
DDP orCUDA graphs. This stage is not a throughput benchmark. Matched BF16 probes
restore production runtime flags before selecting BF16 Flash; FP32 masters and
autocast-cache-off remain common.

## Data and update budget

Reuse verified prepared Dolma v1_5 coverage data, manifestSHA
`f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76`.
This is not the production mixture. Deterministic document shuffle seed20260929;
train split only, isolated rows, actual document boundaries and tokens preserved.
T128/B8 is the first physical layout; T256/B4 is a predeclared fallback only if
preflight establishes a concrete need, before any main training starts. No
physical-layout optimization sweep. Long documents use one context token of
overlap; do not fabricate EOS or targets across chunk/document boundaries.

Train128 logical updates ×8,192 supervised CE targets =1,048,576 targets. If an
update boundary cuts a window, repeat that context with complementary target
masks across updates: every target appears once, additional context exposure is
reported separately. A pure update lookup generates batches/noise and never
commits a cursor; commit only after a completed optimizer update. No cycling.
Record per-update physical slots, input/target counts, source/document exposure,
noise/input pins and committed cursor.

Four development documents, excluded from training, supply a fresh numerical
fixture with real slices of lengths(16,5)/(6,2), physicalB2/T16 twice. Preserve
their real tokens rather than appending a synthetic sliceEOS. Export immutable
JSON with data/doc/slice/mask/keyed-noise pins before probes. It matches the
original fixture's25CE/25latent/21KL counts but contains different content.
Neither fixture is optimized during training. Reserved confirmation split stays
unused. Data preparation adds no download/tokenization or new production corpus.

## Optimizer, recovery and preflight

Fresh AdamW on the two fusion matrices: LR1e-4, betas(.9,.95), epsilon1e-8,
weight decay0.1, raw global gradient clipping at1, foreach/fused false. The first
update usesLR/16 and the16th fullLR; then constantLR. Feedback beta stays1.
Log raw gradient and parameter-update norms, clipping and each pass's CE.
Stop on nonfinite loss/gradient/optimizer state, missing or zero fusion gradient,
frozen-state mutation, target/cursor mismatch or failed source integrity.

Preflight uses two unique updates and one replay: save afterupdate1, runupdate2,
reloadupdate1 and repeatupdate2. Compare complete fusion/Adam/scheduler/RNG/cursor,
counts, inputs/noise and metrics exactly. This is three physical optimizer steps,
two logical updates, disposable progress. It establishes restoration in one
process. Separately run a fresh-process resume from its update1 and compare the
next complete update with the retained uninterrupted endpoint. Restore the same
configuration and physical layout. Do not label same-process replay as a fresh
restart or silently omit a failed comparison.

Save compact atomic checkpoints at0,32,128, segment boundaries, requested stops,
and every600seconds if sooner. Include complete fusion state plus Adam,
scheduler,RNG,counters,cursor and immutable original/frozen model authority.
Never reconstruct output_scale from current embeddings or import historical
adapted weights. Existing generic checkpoint validation remains unmodified.
Store locally on persistent project disk and upload verified immutable GCS
checkpoints. Keep local copies. Save/push code and notes every20–30minutes.

Use the preflight's actual step time/memory to estimate duration. Root alone
schedulesGPU launches; allCUDA executes inside the verified project container.
GPU0 runs warmup. GPU1 may run the independent frozen-state precision probes,
but simultaneous work is not used for throughput comparison. OnlineW&B required.

## Saved-state observations and stopping boundary

Measure original and fresh fixtures at0,32,128: each state has two matched
FP32/BF16 pairs, four aggregate/eight physical backwards. The initial original
fixture must reproduce retained coldNF summaries and first-pass identities;
later original-fixture probes retain exact first-pass fingerprints because the
backbone has not trained. Preserve complete state, fresh predictor, modes,
trainability, RNG and fixture/noise bytes in all probes. Clear gradients afterward.

Report full/group relative and absolute gradient errors, direction/norm ratios,
losses, all-pass hidden/cotangent geometry and per-position support observations.
Support is the union of any nonzero cotangent across both precisions, not an
acceptance filter. Retain position details; a small gradient discrepancy alone
does not show a large hidden-state discrepancy is harmless. Do not tune the
warmup schedule based on interim precision results or fit either probe fixture.

At128, assess learnability and practical startup value. If agreement materially
improves on both fixtures, the next written stage may test a short matched
BF16/FP32 continuation. Otherwise stop this warmup and consider a bounded beta
startup check. No automatic extension of this128-update budget. Neither outcome
alone clears real NextLat losses, nativeRT/NFR, packedT1024 or distributed graph
training. Additional overnight stages require their own frozen scope, not a
retroactive edit to this protocol.
