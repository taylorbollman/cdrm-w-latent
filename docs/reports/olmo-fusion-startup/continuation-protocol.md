# Bounded matched fusion continuation and BF16 replay

2026-09-29. Follow the useful startup128 precision and saved-Adam observations
with a short training-readiness check. No historical helper, model or loader
changes. All new source/protocol files must be reviewed and frozen before launch.

Two trajectories start from the same independently SHA-pinned original startup
update 128, restoring complete fusion weights, Adam moments, scheduler, RNG and
inherited counters. Both use original training data indices 128 through 143,
8,192 CE targets per update, physical B8/T128, NF K4, beta 1 and jitter 0.02.
Train only the two fusion matrices. Original backbone, tied readout, predictor
and saved output scale remain frozen while autograd traverses later backbone
passes. NextLat auxiliary arithmetic is omitted as in warmup.

The two paths are full FP32/math and production BF16/Flash with FP32 masters.
Keep the inherited AdamW LR at 1e-4, betas (.9,.95), epsilon 1e-8, decay .1,
gradient clipping 1 and the completed 16-update warmup scheduler. No reset,
precision-dependent tuning, extra data or extension beyond update 144. Forward
autocast scopes and checkpoint backward dispatch match the reviewed update probe.

Each path performs 16 updates, 128 to 144. Retain complete continuation
checkpoints at 136 and 144, optionally also 128 before the first new update,
and at any requested stop or ten-minute boundary. Checkpoints use a **new kind**,
source/configuration identity and immutable destinations, with inherited counters
rather than resetting to zero. The old startup128 loader still authenticates the
origin in every fresh process; the existing generic loader handles new-format
fusion/Adam/scheduler/RNG restore with additional frozen-state, finite-moment,
fixed-hyperparameter, cursor and inherited-step checks before mutation.

Run a third fresh process on BF16, loading its continuation checkpoint 136 before
execution, then re-executing updates 137 through 144. Match every input/noise pin,
raw and clipped gradient hash, numerical metric, model/Adam/scheduler/RNG/cursor
boundary bitwise against the uninterrupted BF16 golden report. This adds eight
physical optimizer calls: **40 total** across the three trajectories, **16 unique
new data updates**, and no scheduled fourth trajectory. Interrupted work may be
recovered from its last complete checkpoint; incomplete updates are never saved.

At 128,136,144, evaluate the same frozen four-document long development fixture
under a **common FP32/math** path, without gradients or updates. Restore the
caller runtime flags/autocast context and preserve module modes, parameters,
RNG, fixture/noise bytes and cursor. Fixed keyed jitter remains enabled in this
diagnostic; this is not a deterministic online-inference or quality benchmark.
The fresh136 replay repeats the 136/144 evaluations and must match the golden
FP32 evaluations exactly.

Record training CE and per-pass CE, raw gradient and actual master-update norms,
clipping scale, fixed LR, counts, data/noise hashes, complete boundary hashes and
common-FP32 held-out CE. Timings include diagnostic hashes and are not a throughput
benchmark. Compare the two learning trajectories descriptively, not as a quality
competition. Passing does not clear backbone unfreezing, active auxiliary losses,
RT, packed contexts, DDP or production BF16 training.

Root alone launches each single-GPU stage inside the required project container.
Deterministic controls precede CUDA; TF32 and autocast caches remain disabled.
Online W&B, incremental persistent reports, immutable checkpoints and verified
GCS retention are required. Signals and stop files take effect only at completed
update boundaries; no emergency save follows an unknown failed optimizer update.
Keep regular checkpoints no more than ten minutes apart at safe boundaries.
CPU tests must cover both precision closures, exact objective/gradient controls,
new-format fresh-model recovery, read-only FP32 evaluation and malformed import
rejection before any live-state mutation. GPU stages require an external bounded
timeout selected from the observed warmup update time.
