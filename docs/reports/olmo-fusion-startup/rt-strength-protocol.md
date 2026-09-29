# Fixed-state native RT strength diagnostic

2026-09-29. Conditional follow-up to the completed fusion128 component probe.
New opt-in script and tests only; all prior runtime sources remain frozen.

Strictly import the saved NF fusion-only update128 into the original pretrained
NF model. Reuse the exact held-out T128 fixture, two physical B2 records, 512
input tokens and global counts CE508/latent508/KL504. Verify checkpoint, source,
state, runtime, original NF contract, fixture and keyed noise against the pinned
completed component report. All parameters remain trainable for measurement;
no optimizer is constructed and no weights change.

Run exactly four aggregate cases/eight physical backwards: combined CE + latent
+ KL at fixed native RT strengths alpha0 and alpha0.25, each full FP32/math/eager
then BF16/ordinary Flash/native Triton. Native RT layers0 and15 execute on all
four FBT passes, including at alpha0. Existing RT writes use the interpolation
`(1-alpha)*layer_input + alpha*completed_layer_output`. Alpha0's ordinary-model
limit does not bypass the sequential scan; reordered arithmetic remains a
possible source of numerical differences. Alpha0.25 changes the recurrence
function, so comparing gradient magnitudes across strengths is descriptive.

Use an immutable diagnostic wrapper containing the original NF recipe and an
explicit FBTMode/RTMode override. Log the original recipe/import contract and
actual effective mode separately. Do not pretend the historical alpha1-only
`arm_contract` describes this diagnostic. K4, beta1, jitter0.02, first-pass RT
policy, isolated documents, NextLat stop-gradients and all loss weights stay
unchanged. Reuse the frozen generic `component_backward` combined branch and
its `canonical_backward`; do not duplicate the loss normalization. CE weights
are 1/2,1/6,1/6,1/6; auxiliary losses average four passes with coefficients1.

CPU tests must show the alpha1 wrapper exactly reproduces the old NFR combined
forward fingerprints, objective and every parameter gradient. Test alpha0's
ordinary-model limit at strict FP32 roundoff budgets on a tiny model, retaining
the scan path, and exercise alpha0/.25 pairing, mode/state/fixture preservation,
gradient participation, cleanup and reference rejection. Alpha1 is permitted
in the helper only for this CPU oracle; no new alpha1 GPU case is authorized.

The completed component report supplies retained NF-combined and alpha1
NFR-combined context. Their gradient vectors were not retained; the new stage
does not claim a direct vector comparison against those old cases. Compare each
BF16 case to its own newly measured FP32 reference. Report full/group relative
and absolute gradient error, cosine, loss means, all-pass forward/cotangent and
supported-position geometry. Require finite values, positive predictor/backbone/
fusion participation, unchanged state/masks/noise/RNG, and restored runtime
flags. There is no new BF16 acceptance threshold or training-viability claim.

Deterministic setup precedes CUDA, TF32 is disabled, masters stay FP32 and
autocast weight cache is disabled. Forced attention selection remains active
through backward replay. Root alone launches the GPU stage after source/test
freeze, with external900s timeout, online W&B, incremental persistent reports,
source snapshots and immutable GCS retention. No DDP, CUDA graphs, packing,
optimizer updates or model-quality evaluation is added here.
