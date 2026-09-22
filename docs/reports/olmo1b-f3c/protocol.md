# F3c: bounded native RT historical backward fusion

2026-09-22. The user explicitly approved continuing after the F3b result.
Reference: F3b merge29408ca. This milestone isolates reverse-pass costs and
prototypes fused historical dK/dV tiles. Quadratic backward storage removal is
a separate change, not an implied result of this prototype. No long quality run.

## Preserved math and execution

Use original native OLMo-1B step60000 (~252B tokens), 16 layers/width2048/H16,
native RoPE/tied readout/non-affine LayerNorm/no QK normalization. Full-model
fixtures select RT layer0. FBT K2 and canonical NextLat/loss coefficients stay.
One H10080GB; BF16 autocast, FP32 weights/gradients/Adam, TF32 off, deterministic
ordinary Flash SDPA, ordinary checkpoints, no autocast weight cache, CUDA graphs.
Both primary arms enable validated F3b forward fusion and local weight-cast reuse.
Only `backward_tile_backend` differs. Eager remains the public default.

The reverse dyadic rectangle has probabilities P, attention cotangents G,
historical values V, queries Q and row dot products D=sum(G*A). Preserve:

- dV = round_BF16(P_BF16^T @ G_BF16), promoted to FP32.
- E = P_FP32 * (round_BF16(G_BF16 @ V_BF16^T)_FP32 - D_FP32).
- dK = round_BF16(E_BF16^T @ Q_BF16)_FP32 / sqrt(head_dim).

The caller adds these outputs to FP32 recurrent adjoints in the same reverse
schedule. Temporary self, prefix gradients, batched query/parameter VJPs, RoPE,
raw exported-KV cotangents, and gradient ownership stay unchanged. Neither helper
writes parameter .grad. Unsupported shapes/precision use the recorded eager
fallback. Every expected fused call must be observed; no all-fused claim from a
flag alone. Initial kernel scope: BF16 query/value, FP32 P/G/D, head widths
16/32/64/128, both rectangle dimensions1..256, strided reads. Full probability
and final error matrices are still materialized.

## Ordered bounded checks

1. CPU extraction/default/cache/static-plan/dispatch contracts plus existing
   native tiled and canonical combined graph-plan tests. Independent CPU
   observer-neutrality check for the profiler.
2. Profile F3b forward+old-backward combined B64/T512: uninstrumented full-update
   median before separate annotated eager and graph device profiles. Distinguish
   complete local writer/finish VJPs, historical adjoints, final batched VJP and
   reconstruction. Annotation ranges overlap children/include gaps; no summing
   with actual device kernels. Source snapshots pin every run.
3. Frozen backward tiles:48 cases across irregular sizes1..256, strided inputs,
   masked/zero probabilities and dynamic range. Compare the existing explicit
   BF16 formulas and independent FP64 reductions with the same BF16 boundaries.
   Twelve tiny native blocks cover alpha0/.37/1, lengths9/17, masked attached
   prefixes, nonuniform positions and raw hidden/exported-KV cotangents. Primary
   forward/cache must stay bitwise equal; include original BF16 and full FP32
   controls, reporting the latter separately.
4. Actual native RT and combined B8/T512: same-state reference/candidate losses,
   gradient participation and tensor/global error; changed-input/weight capture,
   gradient overwrite and three eager versus three graph complete Adam updates.
   Begin smaller if failure localization requires it.
5. Bounded complete-step capacities at combined B64/T512 and RT B128/T512,
   with three eager preparation updates, ten backward warmups, capture and
   three timed graph updates. Report all update counts; no trained checkpoint
   from disposable fixtures. Reuse retained F3b reference with exact source
   identity or collect a fresh reference where useful. Candidate profile verifies
   dispatch and locates remaining costs before deciding the next rewrite.

## Numerical gates declared before GPU candidate testing

Use the established engineering screens: global gradient relative L2<=1/64,
per-tensor relative L2<=1/32, max error/reference tensor max<=1/16. Apply the
same tensor/global criteria to frozen dK/dV pairs versus explicit BF16 and the
FP64-boundary oracle; exact zero references require exact zero. Per-pass loss
relative error<=1/64. Require exact gradient participation. Forward is unchanged
by this backend, so primary initial forward/cache/losses should be bitwise equal;
investigate any difference even if the broad loss screen would pass.

Keep stricter F3 gradient coordinate flags as visible diagnostics for genuinely
changed fused accumulation. Same-candidate eager/graph retains F3 limits:
relative L2<=1e-5 and maxabs<=1e-6+1e-5*reference-max; exactness is recorded
separately. Complete same-candidate Adam/model/moments/scheduler/counters should
match exactly. A fused backward is not required to reproduce cuBLAS bitwise.
These are engineering review triggers, not paper-author published tolerances or
learning-quality guarantees. Do not widen failed GPU thresholds; localize first.

## Durable record and review boundary

Root owns the serial GPU queue in the required container. Use fresh F3c output
paths, W&B under taylorbollman/pretrained-fbt-rt-nextlat and exact per-run source/
protocol snapshots. Retain small evidence in gs://fast-chunks and reference the
existing native weights. Keep failed attempts diagnostic, never overwrite them.
Review after the bounded backward-tile result before removing global probability/
error storage. No claim for all16 RT layers, longer/padded layouts, multi-GPU,
graph resume or quality without its own evidence. The pre-existing profiler
AccumulateGrad stream warning stays visible for future distributed checks.
