# F4 RT+FBT boundary diagnosis

Added after the frozen F4 initial RT+FBT B8/T512 screen failed on2026-09-23.
The original report and6.25% coordinate budget are retained unchanged. Only
layer11.ff_proj exceeds that budget (6.3492%); globalL2 is1.0206% and its
tensorL2 is0.8389%. Exact forward/ownership/dispatch checks pass.

Use the same pinned weights, seed, B8/T512 token fixture, RT layers0/15, K2,
CE-only objective and unchanged model code. No optimization proposal or
retrospective tolerance change is authorized by this diagnostic.

1. Repeat BF16 materialized and recompute backwards at identical state/input.
   Require each repeat to be bitwise deterministic. Retain the original
   cross-implementation error screen, including any failed tensor.
2. Also run BF16 recompute with only historical backward switched to eager,
   preserving the fused forward. Compare to both primary arms to distinguish
   fused historical reduction from the remaining reconstruction changes.
3. At this same shape, check candidate initial/changed-input/overwrite graph
   replay and three eager versus three graph complete Adam updates, followed
   by changed-weight replay. Require bitwise graph and update parity. This
   validates the candidate execution path independently of the failed
   materialized-versus-recompute numerical screen; it does not erase it.
4. Run full FP32 materialized and recompute backwards with ordinary math SDPA,
   FP32 RT attention and TF32 off. Record loss/global/per-tensor errors for
   FP32 implementations and for both BF16 arms versus FP32 materialized.
   FP32 uses eager fallback kernels and cannot itself clear BF16 Triton.
   These comparisons are diagnostic, not a newly invented acceptance budget.
5. Inspect results before resuming resource measurements. Any resumed
   RT+FBT card must prominently retain the failed numerical-screen qualification.
   If a material discrepancy appears, stop dependent optimized RT+FBT work.

Store scalar comparisons and exact sources/protocol/checkpoint/input identity;
do not retain the large transient full-gradient copies or disposable updates.
W&B uses the same F4 group, separately named roundoff diagnostic. No quality
training, Q/K change or model/kernel rewrite.
