# Tiled BF16 validation contract v1

Frozen 2026-09-07 before new confirmation fixtures are executed. The investigation
owner adopts the following prospective engineering screens for the unchanged
`bf16_fp32_state` tiled candidate. This does not claim a pass. Keep the
old [confirmatory criteria](../r3-bf16/confirmatory-criteria.md) and their seven
initialization B64 failures unchanged.

The deployment target is compiled tiled R3 with BF16 dense operations and FP32
recurrent state. The comparison that matters is with the validated tiled FP32
path. Naive FP32 remains an independent implementation check; naive BF16 is a
diagnostic, with no requirement to reproduce its rounding schedule. Preserve
the Stage B architecture, physical B64/T128, normalization, ALiBi, rho 1,
actual answer-only mean CE, optimizer, and recorded runtime policy.

## What the previous maximum budget establishes

The old requirement was, for each gradient tensor,

`maxabs(tiled_bf16 - naive_bf16) <= maxabs(naive_bf16 - naive_fp32) + fp32_floor`.

This was a reasonable conservative screen for an *additional backend
approximation*. It is not a suitable final acceptance condition when tiled
accuracy against FP32 is the target. A candidate closer to FP32 can fail it:
if the reference is 1 and the two approximations are 0.99 and 1.005, their
distance is 0.015 even though the candidate's reference error is only 0.005.
Conversely, agreement with an inaccurate naive BF16 calculation would not
establish correctness.

The old flags justified investigating a possible backward defect. They do not
justify making tiled gradients imitate naive BF16. We do not know what
numerical tolerances the paper's authors imposed, so these screens should be
presented as project engineering choices, not paper-derived requirements.

At the seven flagged B64 coordinates, six mixed pairs straddle FP32 and four
tiled values are closer. Six backend differences equal one BF16 step. Block
3's attention-output weight has smaller tiled maximum error against FP32
(0.0001309 versus 0.0003137); block 11's output weight has larger error
(0.0001424 versus 0.0001075). Every original coordinate and flag remains in
the [retained analysis](../r3-bf16/numerical-analysis.md).

## Retained reference-centered measurements

These numbers are read-only reductions of the saved JSON analyses, not fresh
confirmation. Their reference is **naive FP32**, because a tiled FP32 arm was
not saved for every previous fixture. Fresh confirmation must include tiled
FP32 directly.

| Fixture | Tiled vs FP32 global parameter relative L2 | Naive BF16 vs FP32 global parameter relative L2 | Worst tiled vs FP32 individual gradient relative L2 |
|---|---:|---:|---:|
| Initialization B2 | 0.8907% | 0.9356% | 1.9215% |
| Trained B2 | 1.3304% | 1.3386% | 4.2065% |
| Initialization B64 | 0.8322% | 0.8864% | 1.7452% |
| Trained B64 | 1.3243% | 1.3406% | 1.9239% |

The global parameter norm counts each of the 100 parameter tensors once;
it excludes the two retained input gradients. The individual maximum includes
all 102 gradients. The small trained fixture's worst values are ordinary
block-2 Q/K normalization gradients, not uniquely recurrent weights. At B64,
the largest maximum error divided by that same tensor's FP32 maximum magnitude
is 2.25% at initialization and 2.88% after training.

These observations argue against reusing the old 1.5625% *backend-distance*
ceiling unchanged as a universal mixed-versus-FP32 ceiling. Full conversion
changes forward operands and incoming loss gradients as well as backward
arithmetic. The proposed thresholds below are prospective engineering
guardrails informed by this already-disclosed exploration. They are not
statistical guarantees, numerical-error theorems, or blind thresholds selected
without seeing earlier data.

## Separate two questions

**Local backward correctness:** give the recurrent block the same captured
input and incoming output gradient, then isolate a dense operation with
identical saved operands. For a weight gradient, use the contraction
`dW = sum_t(dY_t.T @ X_t)`. Differences from this reference localize summation,
cast, accumulation, or replay behavior. All intended casts on inputs and
adjoints must be explicit; an ideal unrounded contraction alone is not the
derivative contract of an arbitrary mixed precision execution.

On a bounded fixture, compare the same BF16-representable operands contracted
in FP32 and FP64. If a candidate claims FP32 accumulation with no intermediate
BF16 rounding, require agreement within the existing FP32 numerical screen
(`atol=2e-6`, `rtol=2e-5`, plus its recorded scale-aware checks). If it promises
one final BF16 cast, compare with the correspondingly rounded reference and
record BF16 representable-step distances. Do not require an exact unrounded
FP64 result from a declared BF16 output. If the native accumulation schedule
is retained, record the schedule's error and its source rather than declaring
every deviation from FP64 a defect.

The recurrent probe must also show that every intended earlier write receives
credit, that the matched-input adjoints agree to their declared precision,
and that power-of-two scaling preserves the fixed-forward gradients. Missing
credit, wrong ownership, double accumulation, or a replay mismatch that changes
the declared computation is a correctness failure regardless of small global
norms. Use strict checks on these small local fixtures; do not turn their
thresholds into demands that the whole BF16 model equal an FP32 forward pass.

**End-to-end precision suitability:** the tiled BF16 model and tiled FP32 model
each execute their own full forward pass on identical data and state. This
comparison includes intended precision conversion. Require bounded global and
per-tensor errors, inspect tails, then check the optimizer and a bounded paired
training trajectory. Neither local correctness nor a short stable trajectory
alone establishes this whole claim.

## Frozen full B64 confirmation screens

Let `eps = 2^-7 = 0.0078125`, `g` be a tiled FP32 reference gradient and `e` be
the corresponding tiled BF16 gradient minus `g`. Preserve the old FP32 floor
`f_i = 2e-6 + 2e-5 * abs(g_i)`. All reductions use detached CPU FP64 values.
Zero-reference tensors use these absolute floors, rather than being dropped.

| Quantity | Prospective screen | Purpose |
|---|---|---|
| Every intended parameter and retained input gradient | Present, correct shape, finite FP32 | Reject missing or invalid gradients. |
| Tiled FP32 versus naive FP32 | Existing FP32 regression screens unchanged | Keep the reference implementation validated. |
| Concatenated parameter gradients | `norm(e) <= 2*eps*norm(g) + norm(f)` | Limit overall conversion error to about 1.56%, with the existing absolute floor. |
| Every individual gradient tensor, including both inputs | `norm(e) <= 4*eps*norm(g) + norm(f)` | Avoid hiding a localized problem in the global aggregate; approximately 3.13%. |
| Every individual gradient tensor's maximum | `maxabs(e) <= 8*eps*maxabs(g) + max(f)` | Retain a maximum-error guardrail, now centered on FP32 and scaled by that tensor's maximum; approximately 6.25%. |
| Forward logits | Relative L2 no larger than `2*eps`, with the existing FP32 floor | Bound forward conversion separately. |
| Actual answer-only mean CE | Absolute difference no larger than 0.01 nats | Practical same-state loss check, distinct from gradient correctness. |

The factors 2/4/8 provide progressively more room for global, individual-tensor,
and extreme-coordinate accumulation effects. They deliberately avoid setting
a bespoke allowance for each observed coordinate. They are engineering
tolerances, not a claim that arbitrary deep-network errors are bounded by a
fixed multiple of BF16 epsilon. The maximum condition uses the same tensor's
maximum, not another tensor's scale. Because this can conceal a relative error
at a smaller coordinate, retain the original max/RMS, elementwise, active
support, near-zero, and sign diagnostics, including the worst coordinates.
An unexplained concentration in a recurrent adjoint or a coherent gradient
bias still requires review even if aggregate screens pass.

Apply these full-model acceptance screens to fresh actual B64 fixtures at
initialization and the retained trained checkpoint. Record the same measures
on B2 diagnostics, but do not claim all B2 states meet a B64 clearance: the
already-observed trained B2 case exceeds the proposed individual 3.13% screen.
That fact must be disclosed, with local fixed-operand checks used to determine
whether it reflects conditioning/forward conversion or a backward defect.
The independent small-seed probe is principally a correctness check using
matched operands and cotangents, not an excuse to ignore a failed full-model
comparison.

Freeze the chosen candidate, exact runtime flags, criteria text/hash and fresh
fixture selection before running confirmation. Preserve old counter-1/2001
fixtures as reproducers. Use new counters, selected before examining outcomes,
for confirmation and name their identities. If a fresh gate fails, report it
and investigate; do not repeatedly extend the threshold or resample until a
pass appears. A materially changed candidate needs separately identified fresh
confirmation.

## Optimizer and operational disposition

Use the actual clipping and AdamW configuration with identical starting
parameters and moments, checking initialization and retained nonempty moments.
Record gradient norms, clipping factors, actual parameter deltas, delta cosine
and relative L2, plus signs and disagreement energy inside/outside the old
near-zero mask. Preserve optimizer epsilon and loss scaling.

Initial Adam updates can differ by almost `2*LR` when tiny gradients change
sign; a universal coordinate maximum smaller than this would reject expected
behavior. A practical proposed initial global update guardrail is cosine at
least 0.99. For retained nonempty moments, use global delta relative L2 at most
`2*eps`, with per-tensor and tail diagnostics retained. These are operational
guardrails, not replacements for local derivative correctness. A concentrated
non-near-zero failure must be explained rather than hidden by a passing global
cosine.

After satisfactory numerical disposition, use the planned bounded paired
FP32/BF16 training and exact midpoint recovery check. Require finite states,
correct data/state restoration and no sustained material loss degradation.
Predeclare the training duration and review a last-window paired mean loss
gap above 0.02 nats or development CE degradation above 0.02 nats. A short run
with low accuracy can establish operational stability only. Keep learning
equivalence and speed/memory results separate; no long reproduction of the
paper is required to clear a bounded implementation check.

A sensible PR pause is reached when the discrepancy is causally explained,
the smallest justified patch (possibly no numerical arithmetic patch) has
fresh B64 reference-centered evidence, and the remaining scope is explicit.
It is acceptable to stop there for alignment before broader CDRM experiments.
It is not necessary to repair naive BF16 just to make its tensors match tiled.

## Read-only source records

- [Original criteria](../r3-bf16/confirmatory-criteria.md),
  [numerical analysis](../r3-bf16/numerical-analysis.md), and
  [results](../r3-bf16/results.md).
- Retained analysis directory:
  `.runtime/r3-bf16/20260906T225438Z/analysis/`; the four
  `*-analysis.json` files provide norms and tail measurements.
- Retained per-fixture `report.json` files provide forward logits and actual
  CE. No tensors were re-evaluated on a model, and no archived record was
  modified to prepare this draft.

## Frozen candidate and fixture selection

The numerical model is unchanged from the start of this investigation; its
source hashes and exact runtime are recorded in `contract-freeze.json` in the
new lineage. Use default-on autocast weight caching and BF16 GEMM reduced
precision reduction, FP32 master parameters/state, TF32 off, compiled tiled
helpers, and math SDPA. The criterion is independent of naive BF16 agreement.

Fresh confirmation uses training-stream counters **4096 at initialization** and
**4097 at the retained update-2000 checkpoint**, beyond the original 2000-update
training run. Run all four arms without module/helper observers. A separate
small isolated fixture uses seed **9103**. Numerical diagnostics on previously
used counters remain exploration. The OPS pair is fixed at 100 updates, with
midpoint recovery from update 50; review the last 50 paired training losses
and the fixed development evaluations against the stated 0.02-nat guardrail.
