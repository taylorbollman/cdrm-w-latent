# Fixed-boundary precision and sensitivity diagnostic

2026-09-29. User approved the continuation after PR41. Start from main
`802a186`, branch `feat/olmo-boundary-precision`. Freeze this protocol and the
new helper/tests before GPU execution. Completed helpers, tests and protocols
from earlier milestones remain unchanged.

## Question

The original isolated NF (NextLat branches present, K4 FBT, no RT) fixture has
60.87% shared-backbone BF16/full-FP32 gradient difference. Promoting only fusion
to FP32 lowers that to 56.16% but worsens final hidden-state agreement on both
records, so that change is not adopted. Determine how much local disagreement
comes from inherited input differences versus precision/backend differences
introduced within selected modules. This is a diagnostic, not a new production
precision policy or a claim that BF16 training cannot work.

## Two full-model anchors

Recompute NF CE under full FP32/math and production BF16/Flash. Use the same
OLMo-1B step60000 checkpoint, initial fusion/predictor weights and buffers,
two physical B2/T16 records, masks/positions/documents, keyed noise, beta=1,
jitter=0.02, K4 weights `(1/2,1/6,1/6,1/6)` and global target denominators.
NextLat branches execute with zero auxiliary cotangents. Preserve the native
ordinary checkpoint settings and FP32 master parameters. No RT executes here.

Reference `.runtime/olmo-recurrence-precision/matrix-01/report.json`, SHA256
`bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412`.
Require exact NF contract, endpoint metrics, forward fingerprints, gradient
summaries and within-pair gradient geometry. Historical full gradient vectors
were not retained; do not claim equality against unavailable vector bytes.
Two aggregate anchor cases contain **four physical full-model backwards**.

## Four observed boundaries and twelve local VJPs

Observe record 0 at transitions entering passes 1 and 3. At each transition:

- Fusion: capture its actual two inputs, including shifted and already jittered
  previous hidden states, and its raw output before eligibility masking.
- Ordinary stack: capture its actual public-call inputs after fusion blending,
  eligibility masking and concatenation, exact keyword arguments and the
  returned final-normalized, padding-masked hidden states.

Use instance/module observation hooks around the original outer FBT forward,
outside the inner checkpointed blocks. Count all eight stack/six fusion calls
across both records; select exactly four sites per precision. Tensor hooks
capture each site's **full actual incoming cotangent**, once. Do not discard
padded entries from the replay computation. Remove all observers on success
and error; their presence must not change either full-model anchor.

For each of the four module/sites, detach the captured inputs and run:

| Case | Arithmetic/backend | Input origin | Common output cotangent |
| --- | --- | --- | --- |
| A | FP32 / math SDPA | FP32 anchor | BF16 anchor at this site |
| B | FP32 / math SDPA | BF16 anchor | Same unchanged tensor |
| C | Production BF16 / Flash SDPA | BF16 anchor | Same unchanged tensor |

Fusion has no attention operation. All actual boundary inputs and cotangents
must be FP32, as observed in the original model; do not silently promote or
round them. Preserve shapes, element strides and captured call settings.
Independent replay leaves need not preserve storage offsets or aliasing; record
that limitation. Parameter values, training flags and checkpoint behavior stay
fixed. Reuse the original module forward, including its normalization formulas.

Measure full output, input-VJP and parameter-VJP relative L2, cosine, norm
ratio and absolute difference for A/B, B/C and A/C. Record input perturbations
too. Report valid-token output summaries separately where appropriate. Use
`autograd.grad`, keep parameter `.grad` fields empty during local replay, and
report truly unused parameters explicitly. The stack's tied embedding/readout
parameter is unused for this detached `inputs_embeds`, `return_logits=False`
call; that is not a missing active-gradient failure. Fusion inputs remain
independent leaves. Release large parameter-VJP references between sites.

Require A's output to reproduce its captured FP32 value exactly, and C's to
reproduce captured BF16 exactly. Require common cotangents, inputs, RNG,
parameters/buffers and source bytes to remain unchanged. Capture and retain
the small actual boundary tensors, masks and layout/call metadata in the
existing JSON/base64 format, with hashes. Do not dump model weights or complete
parameter-gradient vectors. There are **12 local VJPs**, with no additional
model update or training trajectory.

## Interpretation and stopping point

A versus B measures inherited-input sensitivity under fixed FP32 arithmetic.
B versus C measures within-module precision/backend differences at shared
inputs and cotangents. An entire stack still rounds its own internal forward
activations differently; B/C is not a pure backward-kernel error measurement.
Do not add relative error norms or infer additive causal contributions to the
full model from these local comparisons. Sampling two transitions in one record
does not clear every pass, record or the combined RT model.

Stop after these measurements and choose the next bounded action from the
result. Large shared-input local differences motivate inspecting that module;
large inherited-input sensitivity motivates examining recurrence sensitivity
and transition/initialization behavior. No further precision sweep, Q/K change
or architecture change follows automatically. Any candidate remedy will need
its own stated scope and eventual combined-loss/packed-T1024 confirmation.
Existing BF16 qualifications remain open; no acceptance threshold is relaxed.

## Execution and persistence

Root alone launches GPU work in the project container, on device 0, with a
900-second external limit. Configure determinism/CUBLAS before CUDA, disable
TF32, use highest FP32 precision and disable autocast caching. Match recorded
reduction settings. CPU tests run in the explicitly GPU-disabled container.
No optimizer, DDP, CUDA graphs or throughput claim is part of this probe.

Use fresh output directories, immutable source snapshots, atomic per-case
reports and online W&B under `taylorbollman/pretrained-fbt-rt-nextlat`. Retain
reports and boundary tensors in `gs://fast-chunks`; preserve failed attempts.
Commit/push and save evidence at least every 20–30 minutes. Independent review
checks hooks/replay semantics before launch and final evidence afterward.
