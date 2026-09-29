# Crossed-backend localization

Frozen before the additional GPU run, 2026-09-29. Initial bridge source
`c8575d7`; finalized report SHA256
`39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b`.
This is the conditional crossed-backend step permitted by the initial protocol.
The initial protocol, bridge and auxiliary sources remain unchanged.

## Evidence and question

BF16 math/eager still differs from full FP32 by 81.50% combined gradient L2,
versus 85.96% for production BF16. Production BF16 differs from BF16 math/eager
by 75.94% combined and 79.85% CE-only. Thus the original bridge does not cleanly
select only precision or only backends as the source. Both BF16 paths show
large sensitivity. The ordinary-attention and native-RT backend changes must
be separated before attributing this to one implementation.

The fixed-hidden auxiliary test finds BF16 sparse/prepared hidden-cotangent
differences of 0.1416% latent / 0.2016% KL, versus below 8e-7 in FP32. These
are local loss-arithmetic differences. They do not resolve the larger shared
backbone discrepancy. CE-only is informative here because the two BF16 loss
layouts previously agreed exactly on CE, and it retains the large backend gap.

## One new condition, three CE gradient cases

All three use the original initial NFR T16/B2/two-virtual-input fixture,
production BF16 autocast and mixed native RT attention, original masks,
normalization, feedback noise, source weights and pass policy:

1. Recomputed reference: ordinary math SDPA, native RT eager/eager.
2. Recomputed reference: ordinary Flash SDPA, native RT Triton/Triton.
3. **New condition:** ordinary Flash SDPA, native RT eager/eager.

This is three aggregate CE-only cases, six physical backward calls. Other
loss branches remain enabled with zero cotangents. No combined objective,
optimizer, DDP, capture or training update is added. The two references are
needed because full gradient tensors from the first bridge were intentionally
not retained; summary norms cannot reconstruct vectorwise comparisons.

Verify their checkpoint/recipe/input pins, shared source hashes, scalar metrics,
forward fingerprints and recorded gradient summaries against the finalized
bridge. Retain CPU gradient references within this process only. Compare the
new gradient vector and per-pass states/cotangents with both references.
Preserve all runtime flags except the explicitly selected tile/SDPA backend.
Apply deterministic controls before CUDA; retain TF32-off and FP32 masters.

If the new condition follows math/eager, native RT tile differences dominate
this paired observation. If it follows Flash/Triton, ordinary attention is the
larger contributor. Intermediate results or cancellation remain possible;
do not add gradient-error norms or assign causal percentages from them.
Matching one pair does not establish correctness against FP32.

## Scope and later decision

This is descriptive localization with unchanged health/source/state guards,
not a new tolerance or BF16 acceptance test. Numerical qualifications remain.
The same 900s timeout, W&B and GCS retention rules apply. A selective FP32
attention-boundary test or fixed-cotangent local check may follow only after
this result motivates the precise boundary; record it separately before running.
