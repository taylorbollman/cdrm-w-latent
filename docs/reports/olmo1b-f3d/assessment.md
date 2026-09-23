# F3d assessment: bounded RT backward attention workspace

F3d passes its bounded functionality and numerical milestone. The opt-in
`backward_memory="recompute"` removes complete probability/error matrices from
the native RT backward, with meaningful memory savings and a small measured
throughput cost. Keep the materialized default/reference available and use the
new path explicitly in the next broader integration/resource checks.
There is no learning-quality claim and no normalization/architecture change.

## Complete-model measurements

Fresh F3c materialized versus recompute control, same H100 80GB, native
OLMo-1B step60000 (~252B tokens), BF16 mixed, ordinary activation checkpointing,
deterministic ordinary Flash, cast reuse and fused historical RT tiles.
Graphs capture forward/loss/backward; wall time includes batch copy/validation,
clipping, AdamW and scheduler. Only layer 0 is RT. Combined means FBT K2
(ordinary bootstrap plus one RT feedback pass) with training-only NextLat.

| Mode / physical batch / length | Input tokens/s, control → candidate | Change | Peak allocated GiB | GiB saved |
| --- | ---: | ---: | ---: | ---: |
| rt, B128/T512 | 26,454 → 26,257 | -0.74% | 42.10 → 38.60 | 3.50 |
| combined, B64/T512 | 10,970 → 10,927 | -0.39% | 40.84 → 39.09 | 1.75 |
| combined, B16/T1024 | 8,637 → 8,579 | -0.67% | 33.16 → 31.29 | 1.87 |

Three-update medians after three eager updates and ten capture warmup backwards
provide a directional comparison, not a randomized performance estimate.
Peak allocated includes setup; reserved peak/current are separately tabulated
in [results](results.md). Memory at a chosen batch is not a tested maximum batch.
Large-batch capacity runs check finite complete updates; their initial complete
gradients were compared at smaller batches below.

## Correctness and arithmetic

The final suite contains 10 successful runs and 90/90 declared gates:
69 frozen/block/memory probe gates,15 native correctness gates and 6 capacity
gates. There are 54 actual optimizer executions,27 eager plus27 graph. Warmups
and backward-only comparisons do not advance optimizer counters.
[428 scoped CPU tests pass](test-results.txt).

| Native initial gradient comparison | Global relative L2 | Maximum tensor relative L2 | Maximum coordinate delta / tensor reference-max |
| --- | ---: | ---: | ---: |
| rt B8/T512 | 0.00089227 | 0.00111361 | 0.00425532 |
| combined B8/T512 | 0.00237615 | 0.00585636 | 0.00755556 |
| combined B2/T1024 | 0.00224863 | 0.00598055 | 0.00744879 |

All initial losses are bitwise unchanged. Actual recompute dispatch is 511 calls
at T512 and 1023 at T1024 per selected feedback/RT invocation. Same-candidate
original/changed-input/gradient-overwrite/changed-weight graph checks are
bitwise exact; three eager versus three graph AdamW updates give exact model,
optimizer moments, scheduler, counters and metrics in all three cases.
The frozen primary gates stayed global 1/64, per-tensor 1/32 and coordinate 1/16;
zero-reference tensors require exact zero. No budget was widened.

The probe covers48 frozen reconstruction cases,4 long history rectangles,
14 raw-cotangent blocks and 3 reconstruction memory sizes. Forward/cache outputs
are bitwise identical in all 14 blocks; 13 also have bitwise gradients, while
T129/P3 has global relative L2 7.92e-7 versus F3c. Maximum tiny-block global
relative L2 versus full FP32 is 0.003489. Long historical reductions reach2048
query rows and head dimension 128; their worst global error versus F3c is 9.62e-5.
Masks, attached-prefix gradients, temporary self, nonunit strides and alpha
0/.37/1 are represented. CPU checks add independent sequential full-block
all-gradient references, frozen inputs/weights and shared tied-weight calls.

22 stricter diagnostic flags in the frozen probe remain recorded, including
long-reduction coordinate differences and one T129 projection comparison.
These do not fail the predeclared primary screens. Native materialized-versus-
recompute gradients are also not generally bitwise; their strict diagnostics
remain in the reports. This is bounded engineering acceptance, not an assertion
of algebraically exact BF16 derivatives or universal long-training equivalence.

## What became linear

Reconstruction at B2/H4/head dimension 64 has incremental allocated peaks:

| Sequence length | Materialized MiB | Recompute MiB |
| --- | ---: | ---: |
| 512 | 29.290 | 3.150 |
| 1,024 | 116.080 | 6.232 |
| 2,048 | 462.160 | 12.396 |

The candidate grows about 2x per length doubling; the control grows about 4x.
These figures measure only attention reconstruction above resident inputs.
Allocation observers detect the old full attention matrices and find none in
the new backward. Device-private Triton scratch is bounded by source inspection;
the observer itself cannot inspect registers/shared memory.

Row maximum/denominator and temporary-self probability are retained. Query
chunks keep the full key reduction; key-output chunks keep the full query
reduction. Fused history tiles accumulate FP32 across all query chunks before
one BF16 product-result rounding. Prefix/self adjoints and autograd ownership
are preserved. Parameters and forward/loss semantics do not change.
[Resource accounting](../../olmo-resource-accounting.md) records the added
no-prefix matrix work:2BD(E+T²) per RT invocation, about 0.103 TFLOPs at B64/T512.
The existing estimator describes materialized backward; apply this correction.

The full model is not claimed to have linear memory. Ordinary/padded attention
masks, cached prefixes and other allocations are separate; current forward
Triton rectangles larger than 256 use eager fallback. AtT1024 the largest
historical forward rectangle therefore falls back, while recompute backward
is fused. This is Triton RT plus ordinary PyTorch Flash, not native FA4 RT.

## Development history and retention

The initial container launch failed because driver libraries had updated to
580.178.04 while kernel580.173.02 remained loaded. After verifying no GPU jobs,
idle modules were reloaded and telemetry/persistence services restored.
The container then verified H100 80GB and working CUDA. Fresh controls avoid
relying on old-driver timing comparisons. Fabric manager reports no NVSwitch
on this single-GPU machine; no two-GPU result is implied.

Probe01 passed on runtime b85337c. Native RT attempt01 then passed its initial
gradient screen but failed CUDA capture: indexed scalar zeroing tried a CPU
copy. Runtime40305d8 replaces it with an equivalent diagonal-view zero.
Native attempt02 and the full final probe pass. Probe01/02 numerical results
and fixture hashes are identical; only allocation-observer operation counts
change. No optimizer update happened in failed attempt01. Its error, W&B run
and exact earlier source remain retained rather than silently overwritten.

The final [summary](summary.json) selects10 runs/90 gates. The archive includes
12 runs: those10 plus the superseded passing probe and failed native attempt.
The latter two add no optimizer updates and are not extra final-suite evidence.
Run-local snapshots preserve both source versions. Current reporting/retention
code is a separate overlay. Original pinned native weights are referenced at
their existing GCS object; no disposable trained weights are uploaded.
See the storage receipt alongside this report for verified object hashes.

An existing AccumulateGrad stream-mismatch warning remains visible in captured
runs, as in F3c. Exact measured graph/update parity passes, but future DDP setup
must review stream ownership; single-GPU success does not settle that question.

## Next boundary

Use the recompute option for bounded additional-layer/longer-context integration
and finish the broader feature/resource cards. Actual full-model F3d coverage
selects one RT layer; multiple/all RT layers, native T2048 complete updates,
padded graphs, graph recovery/accumulation and genuine multi-GPU checks remain.
Local writer/finish VJPs and discarded permanent-Q projection remain separate
performance opportunities. Preserve native Q/K math. No quality run is queued.
