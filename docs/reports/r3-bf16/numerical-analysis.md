# Candidate mixed-precision numerical analysis

**The candidate does not meet the frozen numerical gate.** It improves
naive/tiled agreement, and the trained B64 fixture meets the added-error
budgets, but initialization B64 fails seven required maximum-error budgets.
The [authoritative criteria](confirmatory-criteria.md) and all historical flags
remain unchanged. Bounded operational evidence cannot relabel these failures
as numerical passes.

This review reduced saved tensors on CPU only, with no model/GPU execution.
The reusable [analysis helper](../../../.runtime/r3-bf16/20260906T225438Z/analysis/analyze_candidate.py)
records all 102 gradient tensors, dtype checks, exact tolerance reductions,
tail coordinates, and per-parameter/per-block Adam classifications. Its
[summary JSON](../../../.runtime/r3-bf16/20260906T225438Z/analysis/summary.json)
links the four complete analyses and records source/packet/criteria hashes.

## Fixed screens across fixtures

For every tensor, the FP32 allowance is exactly
`f_i = 2e-6 + 2e-5 * abs(g32_i)`. The helper checks the original 0.015625
relative-L2 ceiling between mixed backends and **both** added-error budgets:
`norm(gt-gn) <= norm(gn-g32)+norm(f)` and
`maxabs(gt-gn) <= maxabs(gn-g32)+max(f)`.
Logits receive the analogous RMS budget. Reductions use CPU FP64.

| Fixture | Worst backend gradient relative L2 | Failed added-L2 budgets | Failed added-max budgets | Logit RMS budget | Historical gradient flags: naive/FP32, tiled/naive, tiled/FP32 |
|---|---:|---:|---:|---|---|
| Init B2, exploratory | 0.011101 | 0 | 3 | Pass | 57, 15, 59 |
| Trained B2, exploratory | 0.010532 | 0 | 0 | Pass | 78, 8, 78 |
| Init B64, counter 1 | 0.011133 | 0 | **7** | Pass | 48, 10, 45 |
| Trained B64, counter 2001 | 0.011072 | 0 | 0 | Pass | 74, 28, 73 |

Every intended gradient, parameter, and optimizer-state tensor is finite FP32.
All 102 normalized gradients match exactly at scaling factors 1/32 and 32 in
the two B2 diagnostics; B64 packets do not repeat that test. The init B2 FP32
regression retains zero original gradient flags, with worst relative L2
1.39e-6. These facts support the implementation but do not override the failed
maximum budgets or historical mixed-versus-FP32 flags.

## What improved at initialization

Against the [legacy actual-CE baseline](../../../.runtime/r3-bf16/20260906T225438Z/legacy-ce-init-b2/report.json),
the candidate reduces extra-backend global parameter-gradient relative L2
from 0.006117 to 0.004803, and worst per-tensor relative L2 from 0.021805 to
0.011101. Block 3 Q/K-norm errors fall to approximately 0.0043/0.0044;
attention-output and MLP projection gradients now dominate at 0.0106–0.0111.
The candidate's global error against FP32 is essentially unchanged:
0.008907 versus legacy 0.008850. This is improved backend consistency,
not elimination of ordinary BF16 projection error.

All 15 remaining init B2 between-backend historical flags are max/RMS flags.
Support dilution explains part of two: the embedding gradient has 16.31%
nonzero reference support, and the embedding-output gradient has 78.125%.
Their max-error/active-RMS values are 0.06139 and 0.06206. The other 13 flags
persist with an active-support denominator. All 15 worst coordinates are
outside the frozen near-zero set. For example, block 3 `ff_out[231,800]`
changes from 0.0458984 to 0.0485840, a 5.85% coordinate difference. These
tails cannot all be classified as sparse-support or near-zero artifacts.

## Seven initialization B64 maximum-budget failures

Ratios below divide the added maximum error by the **frozen permitted budget**,
including the exact FP32 floor. “Closer” compares each backend with FP32 at
the worst added-error coordinate, not over the entire tensor.

| Tensor within transformer blocks | Flat coordinate | Budget ratio | Naive mixed | Tiled mixed | Naive FP32 | Tiled closer? |
|---|---:|---:|---:|---:|---:|---|
| 3.attn_out.weight | 65068 | 1.15855 | 0.00958252 | 0.00994873 | 0.00984518 | Yes |
| 4.attn_out.weight | 17717 | 1.15206 | 0.01586914 | 0.01574707 | 0.01584882 | No |
| 4.ff_out.weight | 221440 | 1.00193 | 0.01599121 | 0.01611328 | 0.01607110 | Yes |
| 5.attn_out.weight | 36720 | 1.12921 | 0.01574707 | 0.01586914 | 0.01584104 | Yes |
| 7.ff_out.weight | 243054 | 1.06376 | 0.01599121 | 0.01611328 | 0.01601514 | No |
| 9.ff_out.weight | 92098 | 1.14956 | 0.01733398 | 0.01745605 | 0.01741003 | Yes |
| 11.ff_out.weight | 124764 | 1.11088 | -0.01843262 | -0.01855469 | -0.01841231 | No |

Six coordinates straddle FP32; four tiled values are closer to it. Six added
differences equal one BF16 step at their gradient magnitudes. Block 3's
attention-output difference spans about six steps, yet tiled is closer to
FP32 both there and by the tensor's maximum error: 0.0001309 versus naive
0.0003137. Conversely, block 11's output projection moves farther in the same
direction; its maximum FP32 error grows from 0.0001075 to 0.0001424.
All seven reference coordinates are approximately 3–10 tensor RMS and outside
the near-zero set. These patterns are consistent with differing rounded
gradient reductions; they explain the failure without excusing it.

## Adam effects and remaining localization

At init B2, extra-backend Adam-delta cosine improves from 0.996910 to 0.998441,
and relative L2 falls from 0.07862 to 0.05583. Candidate versus FP32 remains
about 0.99397 cosine and 0.10984 relative L2. Under the fixed mask
`abs(g32) <= 0.015625 * RMS(g32_parameter) + 2e-6`, 92.89% of the extra-backend
delta disagreement energy lies near zero. There are still 617 strict gradient
sign flips outside that set. Init B64 has 95.31% near-zero disagreement energy
and 440 outside-set flips; maximum update differences remain about 2*LR.
Thus first-step sign sensitivity explains most, not all, update disagreement.

With retained trained Adam moments, extra-backend delta relative L2 is
0.000815 at B2 and 0.002008 at B64; cosine is 0.99999967 and 0.99999798.
These are materially smaller update differences, not a universal Adam pass
criterion. The helper retains all clipping, moment, gradient-sign, and delta
statistics without dropping coordinates.

The corrected [isolated write-credit probe](../../../.runtime/r3-bf16/20260906T225438Z/candidate-v2-write-credit/report.json)
uses only block 3, B2/D32/T16, and a final-position cotangent. Mixed naive and
tiled outputs, all 15 earlier projected K/V values, and all 15 projected K/V
gradients match exactly; every earlier write receives nonzero credit. Its only
between-mixed historical gradient flags are `ff_out` and `ff_proj`, while
tiled versus FP32 meets every historical gradient screen in that small probe.

This narrows a useful future investigation to dense parameter-gradient
reduction and accumulation. Naive recurrence differentiates repeated small
projection calls; tiled backward recomputes local adjoints and batches its MLP
parameter-gradient update. Shared autocast weight casts may also change where
low-precision contributions accumulate before reaching FP32 parameter `.grad`.
The retained evidence supports investigating those differences; it does not
prove weight caching is the cause. Exact projected-write gradients weaken a
missing temporal-credit explanation on the isolated fixture, but do not prove
all full-width replay boundaries equivalent.

The independent [recomputation analysis](../../../.runtime/r3-bf16/20260906T225438Z/analysis/confirm-trained-b64-recomputation.json)
also narrows that interpretation. At sampled positions 0, 1, 64, and 127,
both B2 fixtures and init B64 have exact BF16 MLP-boundary inputs and exact
recomputed outputs. Trained B64 has one changed BF16 boundary value among
65,536 sampled coordinates, differing by 5.96e-8, with no changed sampled
output. Sampled permanent K/V values replay exactly. This argues against
material replay-input drift in those samples; it does not establish every
unsampled activation or prove a weight-cache mechanism. Additional retained
attention state is not supported by this evidence.

Keep the candidate's numerical status **failed against the frozen maximum
budget**. A future precision fix would require a newly identified candidate
and fresh independent confirmation. Operational training/recovery/performance
results should be reported separately, without extending the numerical claim.
