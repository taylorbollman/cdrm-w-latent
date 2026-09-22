# F3b: bounded native RT tile fusion

2026-09-22. The user authorized the remaining RT execution work after PR15,
including repairing FA4/CuTE dependencies if needed. This milestone profiles
the validated graph path and implements a bounded historical-tile prototype.
No long learning comparison or multi-GPU claim is included.

## Reference and scope

Keep original OLMo-1B step60000 (~252B tokens), native tied readout/RoPE,
non-affine LayerNorm and no Q/K normalization. Reuse the pinned checkpoint and
tokenizer. The F3 reference is merge e86a818. Standard runs select RT layer0;
FBT K2 and NextLat semantics/weights remain canonical. Use one H100 80GB,
BF16 autocast with FP32 parameters/gradients/Adam, TF32 off, deterministic
ordinary Flash, disabled autocast weight caching, ordinary checkpointing and
CUDA graphs. Input staging/clipping/Adam remain outside capture.

First profile combined B64/T512: uninstrumented full-step timings, then separate
observer-only eager phase annotations and graph device operators. Profiles
must not be presented as uninstrumented throughput measurements. Add a RT-only
profile if it resolves a remaining attribution question. Dense FLOP estimates
are an explanatory ledger with explicit recomputation and coverage assumptions.

## Prototype and acceptance

Historical tiles combine FP32 maximum, denominator and unnormalized numerator.
The temporary-self contribution, RoPE placement, dyadic schedule and permanent
memory write remain unchanged. A BF16 Triton kernel may fuse both matrix
products with online-softmax plumbing, preserving explicit BF16 conversion
boundaries before FP32 score scaling and after PV. Generic FA4 normalized
output/LSE uses different rounding; it is a separately named numerical candidate
if investigated, never silently substituted. Verify the actual installed FA4
runtime with a bounded GPU call before dismissing it on import compatibility.

Before interpreting speed, check:

1. Frozen tile operands, widths1/3/8/17/32/64/128/256, unequal target widths,
   noncontiguous strides, masked/empty history, nonzero accumulators and
   large score shifts. Compare against explicit current BF16 math and a local
   FP64 oracle on the same rounded operands. Report numerator and normalized
   output separately; denominator/max remain FP32. No causal mask is added
   inside an already historical rectangle.
2. Block forward/cache and raw output/exported-KV cotangent gradients at
   alpha0/.37/1, including a masked attached prefix and nonuniform positions.
   Same-state full FP32 is the mathematical reference; eager BF16 is a control.
3. Candidate graph/eager replay with changed tokens and weights, gradient
   overwrite/ownership and complete Adam updates, including an actual native
   checkpoint representative of the runtime settings.
4. Separate helper/block/full-step timing. Keep the original implementation
   available and report unsupported configurations explicitly. Do not widen
   a failing tolerance or promote a slower/unvalidated candidate to default.

Rounding-sensitive candidate comparisons use existing historical engineering
screens: global gradient relativeL2 1/64, per-tensor relativeL2 1/32 and max
error/reference-max 1/16. These are review triggers, not author-published
tolerances or claims about learning quality. Tile normalized-output relativeL2
screen is 1/64 and max-error/reference-max is 1/16; inspect failures locally.
Native candidate per-pass loss relative errors must also be <=1/64, with
zero-reference losses requiring an exact zero and gradient participation equal.
FP32 max/denominator relative errors should remain below1e-5 on finite entries.
This strict state screen compares candidate with the current eager arithmetic.
Before GPU testing, CPU frozen-operand checks showed that FP64 versus FP32
matmul reductions can land on opposite BF16 rounding ties even in the unchanged
control. Independent-oracle state screens retain the same limits as diagnostics;
candidate numerator/output and finite-pattern oracle screens remain acceptance
requirements. This does not relax candidate-versus-eager state thresholds.
Same-candidate graph/eager checks retain F3's stricter relativeL2<=1e-5 and
maxabs<=1e-6+1e-5*reference-max. Record exactness separately. A new fused
accumulation implementation is not required to reproduce cuBLAS bitwise.

## Durable operation

Use fresh F3b runtime directories, exact per-run source snapshots and W&B under
taylorbollman/pretrained-fbt-rt-nextlat. Root owns the serial GPU queue. Retain
small reports/tests/profiles/source in gs://fast-chunks, referencing the existing
native checkpoint. Keep failed prototypes labeled. Broader fused backward,
quadratic-memory removal, all-layer RT and multi-GPU remain open unless actually
implemented and checked. Stop at a useful reviewed kernel milestone before an
unbounded rewrite.
