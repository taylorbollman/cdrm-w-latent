# F4 RT+FBT roundoff diagnostic

Assessed 2026-09-23. The original BF16 reference-comparison screen remains
**failed and qualified**. The bounded diagnosis supports continuing the
resource measurements under the [continuation decision](continuation-decision.md);
it does not clear BF16-versus-FP32 precision or authorize quality training.

The completed evidence is `f4-rt-fbt-roundoff-02`,
[W&B q6ro7dtw](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/q6ro7dtw),
at runtime `949731b`. `validate_roundoff` independently checked the actual
completed report, native checkpoint pin, exact source/protocol provenance,
five arms, seven comparisons, all 67 parameter gradients and all per-pass
losses. The six diagnostic optimizer updates are counted separately from F4
resource runs. Attempt01 stopped before backward because preparation occurred
outside the selected attention-backend context; its retained failure is a
harness failure with no optimizer updates.

## What failed and what the diagnosis establishes

The fixed case is original OLMo-1B step60000, B8/T512, RT at indices `(0,15)`,
FBT K2 and no NextLat. Both BF16 memory implementations have bitwise-equal
forward losses. They differ only in backward reconstruction/reduction work.

| Same-state comparison | Global gradient relative L2 | Worst tensor relative L2 | Worst error / reference tensor peak |
| --- | ---: | ---: | ---: |
| BF16 fused recompute vs. materialized | 0.01020619 | 0.01380478 | **0.06349206** |
| BF16 eager-history recompute vs. materialized | 0.00857865 | 0.01160284 | 0.04875283 |
| BF16 fused vs. eager-history recompute | 0.00744169 | 0.01052269 | 0.02902758 |
| Full FP32 recompute vs. materialized | 0.00000303684 | 0.00000443138 | 0.00002533063 |

Original limits remain global L2 `1/64`, tensor L2 `1/32`, and coordinate
error/reference peak `1/16`. One tensor, `blocks.11.ff_proj.weight`, misses
the last limit: 6.349206% versus 6.25%, a 1.59% excess relative to the limit.
No threshold changed. The eager-history diagnostic removes that crossing;
it does not identify a semantic defect in the fused kernel.

All three BF16 arms repeat bitwise at fixed state. The fused candidate's
initial, changed-token/overwrite and changed-weight graph checks are bitwise.
Three eager versus three graph Adam updates match metrics, model, moments,
scheduler and counters exactly. Full FP32 materialized/recompute gradients
agree closely, supporting the underlying memory-reconstruction semantics.
FP32 uses eager RT and ordinary math attention, so it does not directly
validate BF16 Triton reductions.

The stored worst-coordinate records confirm the **same tensor, shape
`[16384,2048]`, and flat index `25786967`** in the relevant comparisons:

| Implementation | Gradient at that coordinate |
| --- | ---: |
| BF16 materialized | -0.47265625 |
| BF16 fused recompute | -0.58203125 |
| FP32 materialized | -0.9834583998 |

Fused recompute is closer to FP32 at this particular coordinate, despite
failing its deviation-from-BF16-control budget. This local observation does
not make it the more accurate implementation overall.

## Broader precision qualification

The full FP32 comparison exposes appreciable gradient sensitivity shared by
the original BF16 control:

| BF16 arm vs. FP32 materialized | Global gradient relative L2 | Gradient cosine | BF16 / FP32 gradient norm |
| --- | ---: | ---: | ---: |
| Materialized | 0.17972081 | 0.98376891 | 0.99381086 |
| Fused recompute | 0.18278275 | 0.98318076 | 0.99052476 |
| Eager-history recompute | 0.18008011 | 0.98367899 | 0.99097289 |

Cosines are derived from recorded squared norms, using
`(norm(a)^2 + norm(b)^2 - norm(a-b)^2) / (2 norm(a) norm(b))`.
BF16 norms come from their exact-repeat records; FP32 norms and difference
norms come from the matching comparisons. Directions remain broadly aligned,
but this does not erase the roughly 18% relative discrepancy. These are
descriptive results, not passes of the tighter engineering screen.

All arms are finite. Feedback-pass CE sums differ by approximately 0.125%
between BF16 and FP32; gradients are substantially more sensitive. Full FP32
changes forward states, dense arithmetic and ordinary attention backend as
well as backward arithmetic. It therefore cannot localize that broader
discrepancy to RT tiling, Flash, Q/K scaling or one reduction. The present
evidence does not justify automatically adding Q/K normalization.

Resource-only continuation is reasonable because deterministic execution,
graph reuse, optimizer integration and FP32 memory-path agreement hold, and
the additional fused-versus-materialized difference is small relative to the
broader shared precision sensitivity. Preserve the failed screen in every
RT+FBT resource comparison. A complete timing matrix means resource coverage,
not all-combinations numerical clearance. Before substantive learning, review
whether a small transition-state and clipped-update comparison is warranted;
learning impact remains unmeasured.
