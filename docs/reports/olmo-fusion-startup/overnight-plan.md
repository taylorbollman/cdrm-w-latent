# Overnight numerical and training-readiness work

Authorized 2026-09-29 at about07:59UTC. User is away for5.5–6hours and explicitly
authorizes useful technical work and bounded follow-ups without another review.
Aim to close out around14:00UTC; do not consume time on empty repetition or extend
an uninformative experiment simply to fill the window. This expands the previous
four-case milestone's stopping boundary. Preserve completed diagnostics unchanged.

## First milestone

Implement the proposed cold-start full-FP32 fusion-only warmup. Original OLMo
step60000 backbone, fresh fusion, K4/beta1/jitter0.02, CE-only training; no temporal
RT or auxiliary updates. Train128updates of8,192CEtargets if the data contract
supports that exact count simply. Use existing pinned prepared training documents,
isolated rows and disjoint fresh precision probes. Start withT128/B8 and adjust
physical batching only after a bounded capacity preflight, before freezing the
training protocol. Report the actual row, input and target counts.

Use AdamW1e-4, 16-update linear LR warmup, betas(.9,.95), eps1e-8, wd.1, clip1.
Freeze all parameters except the two fusion matrices and retain output_scale.
Use fullFP32/math with checkpointing, no graphs/DDP in this first learning probe.
An efficient CE-only closure may omit zero-weight auxiliary arithmetic, provided
forward/CE gradients match the original contract. Precision probes continue to
use the original full trainability and zero-cotangent auxiliary branches.

Save compact fusion/optimizer/RNG/cursor checkpoints with immutable original
backbone authority at0,32,128 and every10minutes if sooner. Retain regular
checkpoints toGCS and keep local copies. Validate a fresh-process continuation,
including frozen-backbone and optimizer ownership checks. Numerical probes at
0,32,128 use both the original short fixture and a small predeclared fresh-data
fixture, recording full parameter and per-position state/cotangent geometry.

## Follow-up decisions within the authorization

If warmup materially improves agreement on both fixtures, perform a short
matched BF16/FP32 continuation from the same saved state and optimizer moments.
Compare update direction/scale, finite behavior and heldout losses, not quality
wins. If it does not, do not automatically lengthen training: use a small
predeclared feedback-strength startup check instead. Preserve unfavorable results.

After the first result, choose the smallest useful next extension toward actual
NextLat auxiliary gradients, multiple native RT layers and packedT1024. Do not
claim that NF/CE/T16 clears those combinations. Avoid broad precision sweeps or
changing Q/K normalization before evidence points there. User permits continuing
without review, but each new stage needs a written scope and source freeze.

Independent useful work can proceed alongsideGPU0: audit concrete remaining
checkpoint/layout/parameter/compute-accounting gaps and implement one bounded
readiness check if it fills a real gap. Do not rerun all the already-passed DDP,
graph and restart suites merely because another GPU is available. GPU1 may run
an independently reviewed check only after root schedules it; root owns launches.

## Persistence and reporting

Use the project container exclusively forCUDA, verify its cwd andGPU visibility,
and never fall back toCPU. OnlineW&B undertaylorbollman and immutableGCS evidence
remain required. Save and push every20–30minutes; checkpoint at most10minutes
apart during training. Profile before estimating long runs, and report any
multi-hour estimate in commentary. No production-quality campaign is authorized
by this diagnostic plan. Keep concise live updates and durable progress notes.

Working branch:`feat/olmo-fusion-startup`, base`baf33fb`/PR44. Root owns protocol,
precision probes,GPU scheduling, retention and closeout; runner reviewer owns
new warmup runner/tests; data reviewer owns new data helper/tests; precision
reviewer independently checks interpretation and remaining readiness gaps.
