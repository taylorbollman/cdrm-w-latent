# Conditional sixteen-update full-NFR continuation

2026-09-29. Prepare this experiment now, but root launches only after reviewing
the four-update NFR outcome and the saved-state packed execution bridge. A failed
FP32 sparse/prepared or BF16 prepared/captured semantic gate takes priority over
more canonical sparse training. Do not launch if the remaining deadline or
checkpoint retention budget is insufficient.

Both independent trajectories start from the **same retained BF16 update4**,
including model, Adam moments, scheduler, RNG, counters and exact data cursor.
This is a comparison conditional on BF16-derived history, not independent cold
starts and not continuation of the earlier FP32 trajectory. First reconstruct
and strictly restore the unchanged original four-update isolated NFR checkpoint
contract. Match the entire saved boundary before doing anything else.

Create a new continuation lineage with a deliberate prefix-preserving scheduler
fork. Keep its original first four valid-token increments, completed tokens,
epoch4, step count5, current/base LRs, optimizer ownership and all Adam moments.
Append exactly sixteen increments for source training selections148..163.
Retain the original52,428,800-token warmup and floor fraction0.1; do not restart
warmup, accelerate the LR, reset Adam or pretend an8192-CE diagnostic update
consumes the production batch's tokens. No old loader or scheduler is changed.

Each path performs exactly sixteen further updates,4→20: isolated B8/T128,
8192 CE targets per update, all backbone/fusion/predictor parameters trainable,
native RT0/15 alpha1 on all K4 passes, beta1, jitter0.02, combined CE/latent/KL,
original pass weights, stop-gradients, non-fused campaign Adam and clip1.
FP32 uses ordinary math/eager RT; BF16 uses production ordinary Flash/native
Triton with FP32 masters. Same tokens, masks, keyed noise and denominators.
Forced SDPA scope surrounds backward and checkpoint recomputation. Autocast is
forward-only. Determinism is configured before CUDA, TF32 and autocast caches
are disabled. Root schedules one independent process per path; this is not DDP.

Per step, record term loss sums and means through fixed counts, combined
objective, LR, clip scale, and scalar raw-gradient/master-parameter norms for
backbone/fusion/predictor and the whole model. No full CPU parameter/gradient
snapshots, vector comparisons or hashes per step. Nonfinite losses/gradients or
masters, missing gradients, wrong counts/cursor/LR or altered input/RNG stop the
run. A large finite norm or a modest loss difference is evidence to assess, not
an invented new acceptance threshold. Later differences include trajectory
divergence rather than solely same-state rounding.

Evaluate the identical long held-out fixture using the same read-only FP32
path at4/12/20; report all three losses separately and their combined objective.
A narrow, restored instance-method observer also records CE, latent and KL
sums/means for each of the four passes from those exact existing forwards.
The frozen evaluator's aggregate fields remain unchanged, and observed call
counts must equal the physical fixture records. No extra model forward, change
to weighting, or replacement loss implementation is introduced; final-pass
differences remain visible even if earlier-pass changes offset them.
Preserve complete model/Adam/scheduler/RNG/cursor state. The frozen loader gets
an explicit NF data-construction recipe and must reproduce the original NFR
report's exact fixture/noise pins; actual execution remains NFR.

Reuse the already-retained update4 origin rather than duplicate its15GB bytes;
the new schedule fork is reconstructible from immutable source and metadata.
Publish full checkpoints12/20 and at completed boundaries after600seconds
since the previous save. Retain them with generation-pinned GCS verification.
A long update or save may exceed the cadence. Graceful stop requests save the
last fully completed boundary. Abrupt failure recovers from the last complete
retained checkpoint and replays unsaved work; no rollback of a failed optimizer
step is claimed. New-lineage fresh-process resume is supported and CPU-tested;
no additional GPU replay experiment is automatically authorized.

Hard scope: at most16 optimizer calls per invocation,32 across the two planned
paths,131072 new CE targets per path. Root external budget is at most one hour
per trajectory, with an internal3300-second check at safe boundaries and time
reserved for publication. Intermediate checkpoints are valid interruption
points; a stopped short segment must not be reported as sixteen-update success.
Source snapshots, reports and W&B metrics are persisted incrementally.

The decision is whether a short full-model BF16 continuation develops delayed
loss/gradient/optimizer instability beyond the first four fresh-Adam updates.
Fusion-only loss agreement did not test that question. Close losses would
justify moving to a modest later pilot; divergent parameter trajectories alone
do not establish a failure. This is not model-quality measurement, packed
training clearance, a throughput benchmark, long-run stability or general BF16
equivalence. Endpoint parameter/moment geometry may be compared once from the
saved checkpoints, without further training or a precision sweep.
