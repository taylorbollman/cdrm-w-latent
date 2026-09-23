# Native RT efficiency and author-derived backend comparison

Approved 2026-09-23 after PR22. Stage A implementation and bounded GPU checks
are now authorized and underway. Stop at its first review point before the
author-derived comparison in Stages B/C. This performance milestone precedes
the broader graph recovery/accumulation and online-readiness work in V4.

## Objective and fixed model

Remove two identifiable sources of unnecessary work in our current backend,
then determine whether an author-derived RT backend with native RoPE offers a
useful additional advantage. Preserve independent references so neither backend
is assumed correct merely because it is old or author-derived. The deliverable
is a measured choice of backend/components, not a learning-quality comparison.

Original OLMo-1B step60000 (~252B tokens): native D2048, H16/head128, SwiGLU8192
per branch, nonaffine LayerNorm, split-half FP32 RoPE, no Q/K normalization and
tied50304 embedding/readout. Preserve the checkpoint and parameter ownership.
The new ordinary reference is PR22's B64/T512 CE2048 result, 39.19k input tokens/s
with half-target supervision or36.63k with full supervision. Use fresh paired
controls, not those historical rates alone, to attribute future changes.

The author source pin is `a21b42d2bc292edb86ed1b62cee4bcab809a9d21` in the
project-root `recurrent-transformer/` checkout. That checkout's current branch
contains local changes; identify exact source files and retained correctness
repairs rather than treating local HEAD as pristine upstream. Source audit:
https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/olmo/model.py

## Stage A: optimize the current native backend

First capture a bounded current reference profile that separates memory writes,
RoPE/pointwise work, MLP/local backward work and historical attention tiles. Use
device traces and complete-step timers; nested profiler annotations are not an
additive timing decomposition. Keep historical attention kernels and their
rounding policy fixed while testing the following changes separately.

### A1. Reuse RoPE tables

`cdrm/pretrained/olmo.py::_apply_rope` currently constructs inverse frequencies,
phases, sine and cosine on each call. The recurrent forward and reverse local
VJPs invoke it repeatedly. Factor table construction from rotation and reuse
the position-dependent constants.

- Preserve native FP32 split-half rotation, the single input-to-FP32 cast feeding
  both branches, and restoration of the Q/K dtype. No reduced-precision tables,
  different RoPE convention or new fused RoPE kernel.
- For fixed training, prepare immutable tables in `PreparedFBTLayout` before
  CUDA capture. Share them across ordinary/RT layers, FBT passes, activation
  checkpoint replay and custom RT backward. Tables have no learned parameters.
- Include table identity, version, shape, device and dtype in static guards.
  Changed positions/configuration or table mutation/replacement require new
  preparation/capture. Avoid a model-global cache that grows during execution
  or interferes with multiple outstanding layouts.
- Dynamic execution prepares explicit-position constants once per invocation.
  Respect offsets, nonconsecutive positions, padding and prefix key positions;
  no assumption that a one-token call has position zero. Avoid host reads of
  CUDA position values in the execution body.
- Preserve unrotated exported-cache semantics and the exact checkpoint state
  dictionary. Backward must retain valid table lifetimes. Leave the independent
  sequential oracle's existing RoPE implementation unchanged.

This optimization can also improve ordinary OLMo. Measure that benefit separately
so it is not mistaken for an RT-specific speedup.

### A2. Project only K/V for permanent memory writes

Keep the existing packed `[3D,D]` QKV parameter. Use its existing `[D:3D]` row
view where the recurrent writer needs only K and V:

1. Per-token forward permanent writes.
2. Batched backward reconstruction of permanent memory.
3. Reverse per-token local source-gradient calculations.

Input-derived temporary Q/K/V still uses the full projection. Preserve the packed
parameter's optimizer/checkpoint identity, frozen-weight behavior and explicit
autograd gradient returns. The memory branch contributes zero to Q rows and its
K/V gradients accumulate with other uses of the same parameter. Do not introduce
independent copies or hidden writes to parameter `.grad`.

This removes one third of the projection output work at those particular sites,
not one third of total RT cost. A smaller GEMM can choose different reduction
paths; bitwise BF16 equality across implementations is not promised. Update the
analytic operation ledger to reflect removed projection arithmetic.

### A3. Validate and measure the three native arms

Keep separate implementation switches/commits for the milestone:

| Arm | RoPE reuse | Permanent projection |
| --- | --- | --- |
| Current control | Off | Full QKV, Q discarded |
| RoPE only | On | Full QKV, Q discarded |
| Both improvements | On | K/V only |

Use matched weights, data, precision, masks and graph settings. Add a KV-only
arm only if interaction or a regression needs localization. Do not grow this
into a compiler/kernel sweep. Retain simple changes that preserve behavior and
show no meaningful end-to-end regression; report small or absent gains honestly.

Minimum targeted checks reuse the existing suites:

- RoPE output/input-gradient equivalence, including offsets and nonconsecutive
  positions, FP32 constants under autocast, and static table ownership guards.
  Seek exact reuse parity when operands/operation boundaries are unchanged.
- Isolated K/V outputs, source gradients and full packed-weight gradients.
- Tiled-versus-unchanged-scan checks at dyadic boundaries, alpha0/.37/1, masks,
  attached prefixes/exported-cache cotangents, frozen inputs/weights and shared
  calls. Reuse existing coverage instead of repeating unrelated investigations.
- Small FP32 checks and bounded native BF16 checks at intended runtime settings;
  exact same-candidate graph replay and a few changed-input AdamW updates.
- Current 16-layer ordinary, two-selected-RT-layer and RT+FBT+NextLat smoke
  checks; all eight feature switches remain covered by the existing tiny tests.

Freeze gates before runs. Reuse current engineering screens for BF16 changes;
investigate failures without silently relaxing them. This work does not resolve
F4's retained RT+FBT coordinate miss or broader BF16/full-FP32 sensitivity.

**First review point / PR:** independently useful native optimizations, measured
individual contributions, refreshed ordinary/RT timing, numerical evidence,
parameter/FLOP accounting, and documented graph/checkpoint behavior. This is the
recommended first implementation milestone; Stage B follows after review.

## Stage B: add native RoPE to an author-derived comparison backend

Start with fixed-length, unpadded, full-strength RT, without FBT or NextLat.
Adapt the author's sequential reference first, then its tiled forward/backward.
Rotate input-derived Q and temporary K and output-derived permanent K at the
correct absolute position. Values are unchanged; backward propagates through
the rotations. Preserve the author's scheduling/layout/compiled helper strategy
where possible, and record unavoidable adaptations.

Use the same native OLMo weights and block structure as Stage A, including
SwiGLU, disabled Q/K norm, nonaffine LayerNorm, scaling and dropout policy. The
author code supports these ordinary block choices. Adapt packed QKV ownership
without changing parameter values. Use efficient precomputed RoPE constants in
both candidates so the comparison does not penalize either with repeated table
construction. This is an author-derived backend for our model, not a claim to
reproduce the paper's exact trained architecture or published throughput.

The prototype must preserve required local correctness repairs. The author's
nested parameter-gradient accumulation and quadratic attention workspace differ
from our explicit-gradient, bounded-workspace path. A restricted comparison can
record those differences, but cannot silently replace our backend in shared FBT
or distributed execution. Do not bolt on every cache/mask/transition capability
before establishing a performance reason to proceed.

Run a small output/raw-gradient comparison against the unchanged native sequential
FP32 reference and representative BF16 checks. Do not force BF16 bitwise equality
where query prescaling, GEMM grouping or probability rounding differ. If these
differences matter, expose them as separate controlled precision settings rather
than attributing their speed solely to implementation quality.

## Stage C: matched measurement and integration decision

The primary benchmark uses native dimensions, T512 and identical inputs/weights.
No embeddings/CE in isolated block timing; full-model timing includes them.

1. **One RT block:** measure the three native arms and author-derived candidate.
   Small B1 is diagnostic; B32 and B128 are useful operating points. Attempt B512
   if comfortable for both, otherwise an explicit B256 or smaller shared point.
   Record a true physical batch, not an accumulation substitution. Compare at a
   common batch and separately report each backend's comfortably feasible batch.
2. **Stacked recurrence:** confirm the result survives two RT blocks and a
   bounded six-layer all-RT stack using identical native-shaped blocks/weights.
   This stack is a timing fixture, not a pretrained six-layer model or paper
   reproduction. Avoid basing conclusions solely on two recurrent layers inside
   an otherwise ordinary backbone.
3. **Actual integration:** remeasure the original 16-layer model with at least
   RT layers0/15 in the current backend. If the alternate is promising, adapt it
   to this same wrapper and perform one shared FBT+NextLat compatibility check
   before treating it as a replacement. Broader all16/multi-GPU work stays separate.

Report forward, backward, and complete update time; input tokens/s; allocated,
reserved and setup memory; compilation/capture time; kernel counts; parameter
counts and analytic matrix work. Explicitly separate the original materialized
backward versus our bounded-recomputation tradeoff. A same-workspace diagnostic
may help explain a gap, but memory-efficient execution remains a relevant outcome.

Use BF16 mixed with FP32 parameters/gradients/Adam, recorded TF32 and autocast
cache policy, identical objective/clipping/optimizer, CE2048 and KL128 when
applicable. Full-model primary comparisons use full next-token CE; retain one
half-CE bridge only where needed to relate to PR22. Ordinary attention remains
the same deterministic PyTorch Flash backend in both wrappers.

Execution policy must not artificially handicap one backend. Preserve the
author's useful compiled helpers and weight-cast reuse, and record those choices.
Our explicit cast reuse with the global autocast cache disabled is not necessarily
the author's best policy. Different caching mechanisms are acceptable if they
preserve the agreed arithmetic and correctly reread changed weights on graph
replay; otherwise add a labeled diagnostic to separate that effect. Do not claim
the remaining difference comes solely from recurrence scheduling.

Warm up all shapes/compilation before capture and keep setup outside steady
timing. Start with at least ten backward warmups and five timed complete updates;
repeat a close comparison in reversed order before claiming a small difference.
Use CUDA graphs for the primary comparison with matched capture boundaries.
Inspect short eager profiles only for attribution. Restrict T2048 to an optional
follow-up when needed to assess a material memory advantage.

Decision: retain our backend, adopt specific author-derived components, or
advance an alternate backend. A full replacement needs a repeatable stacked/
integrated advantage worth its maintenance and capability cost. Losing speed but
winning memory can still be useful and should be reported as such. A result that
shows comparable performance is also a successful resolution of the uncertainty.

## Scope, records and continuation

No language-quality training, new Q/K normalization, positional scheme change,
FA4 rewrite, Dao CE integration, fused SwiGLU or broad mixed-precision campaign.
Local compiler/fusion/layout experiments are deferred unless a measured residual
bottleneck justifies a separate follow-up. Existing RT precision qualifications
remain visible. This milestone provides no genuine multi-GPU clearance.

For every stage: freeze protocol/source snapshots, log graphable runs online in
`taylorbollman/pretrained-fbt-rt-nextlat`, retain explicit successes/failures and
small evidence in `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/`, and update the
handoff. Reference the existing native checkpoint; few-update profiling weights
are disposable unless needed to reproduce a failure. Notify the user before any
expected multi-hour run; no long learning job is part of the proposed milestone.
