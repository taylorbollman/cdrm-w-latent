# F3d: bounded-workspace native RT backward

Prospective protocol, 2026-09-23. User authorized this next functionality and
efficiency milestone. No quality training is authorized by these diagnostics.

## Change and reference

Add opt-in `backward_memory="recompute"`; default `materialized` stays as the
F3c reference. Both primary arms use F3b cast reuse/Triton forward and F3c
Triton historical backward. Recompute stores row maximum, denominator and
temporary-self probability, reconstructs attention in query chunks of 32,
recomputes historical probabilities inside a Triton gradient tile, and forms
final query and prefix adjoints without a full query-by-key P/error tensor.
CPU/FP32/unsupported shapes retain bounded eager fallback. Key-output chunks
keep the full query reduction; query-output chunks keep the full key reduction.
BF16 matrix operands and whole-product rounding, FP32 online state/adjoints,
temporary-self semantics and normal parameter/cache gradient ownership stay.
The forward function and model/checkpoint/loss/parameter count do not change.

## Bounded evidence

1. CPU independent attention/adjoint oracles, full-block gradients, masks,
   attached prefixes, execution/cache signatures, and a backward allocation
   observer with the old quadratic implementation as a positive control.
2. Frozen BF16 reconstruction/history fixtures and tiny native blocks across
   odd lengths, masks, prefixes and alpha0/.37/1. Raw hidden and exported-cache
   cotangents. Additional T65/T129 blocks and historical rectangles257x255,
   513x511,1024x1024,2048x3 at head dimension128 cover long query reductions.
   FP32 is a separate diagnostic.
3. Isolated reconstruction at T512/1024/2048: allocated peak above resident
   inputs, tensor-shape observation and output comparison. Require lower
   candidate peak and no full query-by-key allocation; report scaling ratios
   without a tight allocator-dependent slope threshold.
4. Actual step60000 OLMo-1B checkpoint: RT and combined FBT K2+NextLat at
   B8/T512, selecting only RT layer0. Initial all-loss/all-gradient comparison,
   actual recompute dispatch, then same-candidate eager/CUDA-graph changed-input,
   overwrite, changed-weight and three-versus-three complete AdamW updates.
   Ordinary activation checkpointing and deterministic ordinary Flash remain.
5. Fresh reference/candidate complete graph timings at RT B128/T512 and
   combined B64/T512, followed by a bounded longer-context check at manageable
   batch if preceding checks pass. Three eager preparation updates, ten capture
   warmup backwards, three timed changed-input/weight graph updates. Record
   allocated peak, reserved peak and current reserved separately.

Primary BF16 engineering gates are global gradient relative L2 <=1/64,
per-tensor relative L2 <=1/32 and max absolute delta/reference-max <=1/16.
Reference-zero tensors require exact zero. Forward/cache/loss outputs should
remain bitwise identical when only the backward option changes. Preserve
stricter coordinate/oracle diagnostics even if the primary screen passes.
Same-candidate graph checks keep established stricter tolerances (relative
L2 1e-5, max absolute 1e-6+1e-5*reference-max); full Adam/model/moments/scheduler
and counters must match. Do not widen gates after observing failures.

Every GPU run requires verified container/H100, source/protocol snapshots and
hash rechecks, online W&B, durable JSON and retained small evidence. Preserve
failed attempts. Actual optimizer updates are separate from warmup/backwards;
disposable updated model weights need not be archived. Existing retained native
checkpoint is reused. This session repaired a pre-existing driver/library
mismatch by reloading idle NVIDIA modules (580.173.02 ->580.178.04); fresh
reference timings are needed. One H100 is exposed. No all-layer RT, genuine
multi-GPU, padded native CUDA graphs or long learning claim follows from this.

## Review boundary

Report numerical qualification, isolated/full-model memory, throughput and
actual backend dispatch. Keep reference/default available. Stop expanding the
scope if a meaningful numerical failure remains. Local writer/finish VJP
optimization and discarded permanent-Q projection are separate later work.
