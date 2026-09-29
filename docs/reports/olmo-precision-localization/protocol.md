# Bounded numerical localization protocol

2026-09-29. Branch `feat/olmo-precision-localization`. This protocol is drafted
before execution and becomes immutable when root freezes the diagnostic source
inventory. Subsequent hypothesis-driven additions must be recorded separately
before their runs. This is a numerical diagnosis milestone, not quality
training or an architecture-selection experiment.

## Problem and existing qualifications

The completed packed-data milestone established actual-data update/cursor
functionality and exact fresh-process continuation under explicit deterministic
controls on two H100s. The original nondeterministic restart failure remains
retained. Those results establish repeatability within a fixed execution
contract; they do not establish agreement across precision or kernel choices.

The earlier initial-state NFR diagnostic found two distinct observations:

- BF16 prepared versus sparse losses produced a **3.40224%** combined raw-gradient
  relative L2 difference. CE-only gradients matched exactly; latent-only and
  KL-only differences were about 2.173% and 2.284%. Packed NFR separately retained
  a 1.6953% layout qualification failure.
- Both isolated-fixture BF16 layouts differed about **86% relative L2** from the
  full-FP32 combined gradient, with cosine about 0.51 and roughly half its norm.
  CE alone also showed a large gap. That comparison changed precision, ordinary
  attention dispatch and native RT tile kernels together.

The existing [component assessment](../olmo-packed-campaign/precision-assessment.md)
and [restart investigation](../olmo-packed-campaign/restart-repeatability.md)
remain authoritative for their recorded numbers and scope. Small scalar-loss
differences, successful distributed execution and exact recovery do not clear
either gradient qualification. Component gradient-error norms must not be added
to apportion a combined BF16 backward: cotangent addition and rounding occur at
different places in those graphs.

## Fixed state, data and execution controls

Use the same initial pretrained NFR fixture as the previous component diagnostic:

| Item | Frozen selection |
| --- | --- |
| Backbone | OLMo-1B step 60000, revision `81b71efbce6f4dada57c94860301af4298bcd351`, about 252B pretraining tokens |
| Native source weights | SHA256 `ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c` |
| Architecture | Existing 16-layer, width-2048 pretrained backbone; tied embeddings, RoPE, no added Q/K normalization |
| Arm | NFR: K4 feedback passes, native RT at layers 0 and 15 on every pass, existing NextLat predictor and both auxiliary losses |
| Data | Original isolated-document T16 fixture, physical B2 for each of two virtual-rank inputs, logical update zero |
| Global valid counts | 29 input tokens, 25 CE targets, 25 latent pairs, 21 KL triples |
| Loss weights | Campaign-v1: CE pass weights `(1/2, 1/6, 1/6, 1/6)`; latent/KL pass weights `(1/4, 1/4, 1/4, 1/4)`; latent and KL coefficients 1, auxiliary token CE coefficient 0 |
| Feedback | Existing shifted feedback, jitter amplitude 0.02, pinned initialization and keyed-noise seeds; same tensors reused for every relevant comparison |
| Training changes | None: no optimizer, parameter updates, DDP, CUDA graphs, clipping, scheduler/token-clock advance or real-data training loop |

Configure deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
CUDA initialization; enable cuDNN deterministic mode and disable benchmarking.
Disable TF32 in every path and use highest FP32 matmul precision. Use FP32 master
parameters and parameter-gradient storage; the BF16 paths use autocast with
weight caching disabled. Record actual backend flags rather than inferring them
from a path name. All GPU execution is inside the required project container;
there is no CPU fallback for a GPU diagnostic.

Pin model/configuration hashes, source snapshots, fixture tokens/masks/document
IDs, jitter, counts and loss weights. Check parameter, source and RNG integrity
around the diagnostic. Preserve existing detach/stop-gradient semantics. A
CE-only component disables other loss contributions with zero cotangents rather
than changing model structure or accidentally dropping active branches.

## Initial model bridge: six aggregate gradient cases

Use the canonical sparse loss layout throughout this bridge. Execute CE-only
and combined CE+latent+KL objectives on each of these three paths:

| Path | Ordinary attention | Native RT tile forward/backward | Precision |
| --- | --- | --- | --- |
| FP32 reference | Forced math SDPA | Eager/eager | Full FP32, including RT attention |
| BF16 eager bridge | Forced math SDPA | Eager/eager | Existing BF16 mixed policy, including `attention_precision="mixed"` |
| Current BF16 endpoint | Forced Flash SDPA | Existing Triton tile backends | Existing BF16 mixed policy, including `attention_precision="mixed"` |

This is **six aggregate gradient cases**, each accumulating two physical
fixture records with the original global denominators: **12 model-backward
calls in total**. The prior shorthand “six backwards” referred to the six
objective/path comparisons, not six individual autograd calls. The matrix and
diagnostic scope are unchanged.

Build the bridge by restoring the existing BF16 flags and changing only forced
ordinary SDPA dispatch and RT tile forward/backward backend selection. Keep
recomputation, cast-once, RoPE reuse, KV-only writes, ordinary activation
checkpointing and pointwise settings unchanged between the BF16 bridge and
endpoint. Do not configure full FP32 and simply turn autocast back on: that
would leave RT attention in FP32 and confound the intended BF16 comparison.

Capture forward-state measurements by feedback pass without changing forward
values or the backward graph. Record actual tensor dtypes and the capture point.
For each aggregate case, record objective/per-loss sums and normalized means,
full raw-gradient norm/error/cosine, parameter-group and informative per-tensor
differences, finite/participation checks and bounded memory/time observations.
Compare the BF16 bridge to both endpoints, using exactly the same fixed inputs
and weights. The two virtual inputs contribute to the original global per-term
denominators; they are not two actual DDP ranks.

The bridge separates a combined backend change from the precision change. It
does not by itself distinguish ordinary Flash from native RT Triton. Report
that limitation when describing any backend-associated discrepancy.

## Separate auxiliary-loss cotangent diagnostic

Test latent and KL losses on the same detached hidden states, embeddings,
readout weights and predictor inputs for sparse and prepared layouts. Preserve
real target masks, counts, campaign pass weights and detach semantics. Reuse
one frozen tensor payload within each sparse/prepared comparison; record the
source forward, actual dtypes and hashes. If values are converted for another
precision, identify that conversion rather than claiming bitwise identity
across dtypes. The anchor comes from the bridge's current BF16 Flash/Triton
forward, with its actual hidden/embedding dtypes preserved. Export only the
bounded hidden/embedding/batch payload and checkpoint, readout and predictor
hashes as JSON. Reconstruct the readout from the pinned base source and the
predictor from the same seed/configuration, checking their hashes. Avoid a
large duplicate weight dump. The auxiliary probe needs no additional backbone
forward and must not silently regenerate its anchor under another backend.

The initial loss-level matrix is latent/KL × FP32/BF16 × sparse/prepared:
**eight aggregate loss-gradient cases**, each containing the same two physical
records, for **16 loss-only backward calls**. Measure incoming hidden-state
cotangents directly, alongside predictor/readout/embedding gradients where
those branches are active. Record
finite values, norm ratios, cosine, error norms and the exact selected positions.
This removes backbone propagation from the comparison; it does not train the
predictor or create a separate model.

Following those results, propagate a common detached cotangent through a fixed
backbone only if needed to separate loss-layout rounding from sensitivity in
its propagation. Record the selected state/cotangent and the exact additional
backward count before that follow-up. A common cotangent must be equal at the
chosen injection point; using independently generated approximate cotangents
would not isolate the backbone.

## How results determine the next bounded step

1. If the BF16 eager bridge follows FP32 closely while the current BF16 endpoint
   does not, add **one** crossed-backend condition on the affected objective:
   math SDPA with Triton RT, or Flash SDPA with eager RT. Select it from the
   observed pattern and freeze its controls before execution. This distinguishes
   the two backend changes without starting an all-combinations sweep.
2. If both BF16 paths differ similarly from FP32, inspect activations and
   cotangent scales at selected feedback passes and RT layers, then change one
   precision boundary at a time. Agreement between two BF16 paths is not a
   substitute for addressing their shared FP32 discrepancy.
3. If sparse/prepared auxiliary cotangents already differ at fixed hidden
   values, localize the loss/projection/masking arithmetic before changing the
   backbone. If those cotangents agree, a common-cotangent propagation check can
   isolate sensitivity downstream of that point.
4. Only when necessary, use the smallest feature ablation that separates RT,
   feedback and their interaction. Check a bounded real packed-data fixture
   before generalizing from T16. Neither a full-length sweep nor all eight arms
   is an automatic next step.

Prefer a narrow implementation or precision correction supported by the
evidence. Do not add Q/K normalization, change recurrence placement, alter
objective weights or adopt a new architecture as an unexplained numerical fix.
If a correction is justified, freeze its predicted effect and the affected
acceptance checks before testing it, retain original failures and rerun only
the relevant checks.

## Acceptance language and interruption protection

Operational completion requires correct pinned inputs/configuration, finite
outputs/gradients, intended gradient participation and unchanged parameters,
source files and RNG state. Deterministic same-arithmetic repetitions, where
used, retain exactness requirements. Existing numerical budgets remain attached
to their original scopes; no threshold is relaxed. Cross-precision/backend and
loss-cotangent measurements are descriptive until a justified criterion is
explicitly frozen. Do not invent a BF16 tolerance after viewing results or turn
an operational pass into numerical clearance.

Each GPU diagnostic phase has an external **900-second maximum** and publishes
atomic progress/results after bounded cases. The root agent serializes GPU
launches. Keep code, source snapshots and small evidence on persistent storage;
retain completed or failed evidence in `gs://fast-chunks`, with verified
receipts, at least every 20–30 minutes. These fixed-state diagnostics require
no newly trained checkpoint. An interrupted phase can be rerun from its pinned
inputs; previously completed evidence remains intact. Report any operation
that cannot respect this interruption bound before starting it.

Log graphable comparisons online under W&B entity `taylorbollman`, project
`pretrained-fbt-rt-nextlat`, alongside local reports and GCS receipts. Never
print credentials. Record each run URL and its source/fixture pins in the
progress and results documents.

No production mixture, quality-training campaign, H200 throughput projection,
changed-world-size qualification or model-architecture transition is selected
by this milestone. Its deliverable is a localized explanation or a narrower,
explicitly retained numerical qualification with reproducible evidence.
