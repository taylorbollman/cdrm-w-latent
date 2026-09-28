# Portable readiness probe protocol

2026-09-28. This is the first bounded implementation slice of campaign readiness,
not approval of the later training budget, corpus mixture or LR calibration.

CPU checks cover the opt-in campaign policy and legacy behavior: all eight arm
definitions, configured RT on the first pass, externally supplied keyed jitter,
separate CE/latent/KL pass coefficients, optimizer ownership/exclusions, token
warmup, document/window accounting, raw-byte ingestion and checkpoint contracts.
A tiny model is saved mid-warmup and loaded in a fresh process; the next update,
optimizer, scheduler, counters, data cursor and next RNG draw must match exactly.
This is intentionally a CPU fixture, not a distributed recovery claim.

The actual-checkpoint GPU probe uses the pinned OLMo-1B step60000 weights,
NFR (NextLat + FBT + RT), K4, native RT at layers 0 and 15 on **all four**
passes, and the new campaign objective. CE coefficients are
`[1/2, 1/6, 1/6, 1/6]`; latent and KL coefficients are each `[1/4]*4`.
Feedback jitter amplitude is 0.02 with externally generated deterministic
per-row/per-update/per-pass noise. No Q/K normalization is introduced.

Execution is B1/T16, native Triton RT forward/backward tiles and recomputation,
ordinary Flash SDPA, activation checkpointing, BF16 autocast with FP32 master
parameters/gradients/Adam, TF32 off. The fixture is fixed operational prose with
all valid targets. The recipe's planned T1024/effective batch are metadata;
this tiny smoke deliberately does not execute those production dimensions.
CUDA graphs and torch.compile are off for this semantic probe. It does not
estimate production memory, throughput, training quality or precision parity.

Gates, fixed before launch:

1. Canonical loss backward has finite objective and every active gradient.
   Both selected RT layers execute exactly four times; CUDA RNG is unchanged.
2. Repeat at identical weights/noise using an independently written literal
   coefficient expression. Objective absolute difference must be at most 1e-6;
   every gradient must satisfy `atol=3e-5, rtol=3e-4`. This compares loss assembly
   on the **same BF16 path**, not BF16 against FP32.
3. Two disposable fused-Adam updates have finite gradients/parameters, the
   configured clipping, and the correct token-based LR clock.
4. Source hashes at exit match the archived source snapshot at entry.

The launcher has a 15-minute limit. Each backward/update atomically publishes
its local report and W&B metrics. These short diagnostic updates are replayable
from the immutable source checkpoint; no new billion-parameter checkpoint is
needed. Save implementation and progress to Git throughout; retain source,
report and logs in GCS after writers stop. Any failed attempt remains evidence.

The next integration gates remain padded Flash execution, graph-safe changing
masks/jitter and accumulation, per-rank jitter inputs, globally normalized losses
with empty local target sets, actual distributed fresh-process recovery and
real-data throughput. The graph trainer rejects nonzero jitter until that work
is qualified. This milestone does not close prior native-RT BF16 qualifications.
