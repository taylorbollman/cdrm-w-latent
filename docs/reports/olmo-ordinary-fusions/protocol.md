# Ordinary OLMo fused RoPE and optimizer: prospective protocol

2026-09-24. User authorized a further bounded ordinary-model efficiency pass,
including Dao and Transformer Engine candidates and the optimizer setting.
This follows PR26. Preserve native checkpoint/model ownership, production
defaults, the existing RT qualifications, and all failed attempts. No quality
training, RT backend change or broad module migration.

## Fixed reference

Actual original OLMo-1B step60000 (~252B tokens),16 layers,D2048,H16/head128,
SwiGLU8192 per branch, tied50304 vocabulary, native nonaffine LayerNorm and
FP32 split-half RoPE/residuals, no Q/K normalization. Ordinary only. Retain
frozen unused fusion weights separately in parameter accounting.

PR26 improved conservative reference: PyTorch Flash SDPA, rounded compiled
ordinary SwiGLU, all-layer checkpointing, reused native RoPE tables, full CE
chunks2048/KL128 inactive. Same fixed repeated-text fixtures/rotations and full
CE masks for every update. BF16 mixed, FP32 parameters/gradients/Adam, TF32 off,
autocast weight cache off, deterministic execution, seed20260922. CUDA graph
captures forward/loss/backward; clipping, Adam and scheduler remain external.

## Candidates and compatibility

1. Installed Dao fused split-half RoPE, out of place. Keep native FP32 tables
   and cast projected Q/K to FP32 before the installed kernel, restoring the
   projection dtype afterwards. Prepare compact contiguous tables and offsets
   once from actual native positional coordinates; do not regenerate with Dao's
   table module or silently downcast tables. Native RT calls remain unchanged.
   The installed kernel can fuse FP32 multiply/add; do not promise bitwise math.
   First scope is cache-free full-sequence ordinary execution; reject unsupported
   prefix/export-cache paths explicitly. No dependency reinstall is needed.
2. Native PyTorch AdamW `fused=True, foreach=False`, explicit opt-in. Existing
   `fused=None, foreach=False` stays default. Same LR1e-5, betas(.9,.95), epsilon1e-8,
   matrix weight decay0.1, two-update warmup, max-norm1 clipping. Keep clipping
   unchanged. FP32 moments/master parameters; CUDA step scalar is an intentional
   fused-optimizer difference. It is not an exact historical-resume identity.
3. Audit Dao SwiGLU/CE and Transformer Engine modules against native parameter
   ownership/precision. Integrate additional candidates only if they have a small,
   clearly compatible adaptation and a credible measured benefit. Broad TE module
   replacement, FP8, lower-precision residuals/norms and new affine parameters are
   outside this bounded milestone.

## Correctness and measurements

Focused CPU tests cover new option guards, untouched defaults/RT paths, static
metadata/storage invalidation, arbitrary positional-coordinate table adaptation,
parameter/optimizer ownership and explicit optimizer-factory injection in parity.
GPU execution only inside the required container after confirming H100 access.

Actual-checkpoint comparisons: B8/T512 each candidate and useful combination;
RoPE also B2/T2048. Preserve PR26 budgets: global raw-gradient L2<=1/64;
per-tensor L2<=1/32 and max/reference-peak<=1/16; output L2<=1/64/max<=1/16;
relative loss<=1e-5. No post-result threshold change. Optimizer-only changes
must leave initial forward/loss/raw gradients bitwise. Finite numerical-only
misses can finish operational diagnosis but remain failed; subsequent timings
are exploratory. Structural, dispatch and operational failures stop the queue.

Each candidate needs exact own eager/graph initial/changed-input/repeated-overwrite
and changed-weight checks; three eager versus three graph full Adam updates must
match weights, optimizer state, scheduler/counters and metrics. The parity helper
must receive the selected optimizer factory explicitly, and record actual flags.
For fused versus scalar Adam, add a fixed-gradient comparison before claiming
optimizer compatibility: compare moments, step/LR values, parameters and update
deltas, allowing explicit bounded FP32 implementation rounding, not bitwise
cross-implementation identity. Record prospective numerical budgets before it runs.
Fixed-gradient budgets: FP32 moment and final-weight global/per-tensor relative
L2 and maximum/reference-peak<=1e-6; cumulative update-delta global relative
L2<=1e-3, with maximum delta differences descriptive. Step/LR/clipping/ownership
values must match exactly. Compare scalar step values despite CPU/CUDA placement.
Restore initial parameters and raw gradients in the original storage afterwards.

Capacity: B64/T512 reference, Dao RoPE, fused Adam, useful combination; secondary
B16/T2048 matched reference/combination if useful. Keep physical batch/checkpointing
fixed to isolate effects. Three actual preparation updates, ten backward warmups,
five synchronized complete timed updates and three forward/loss/backward timings.
Repeat chosen baseline/candidate in reversed order. Target eight capacity runs,
not a Cartesian sweep. Track incremental physical updates and setup/steady memory.

Separate untimed profiles include graph backward and a canonical full-step trace
with optimizer/clipping device attribution. Profile data are not timed samples;
do not call full-step-minus-backward timing pure optimizer cost. Analytic matrix
FLOPs and parameter counts stay unchanged; fused kernels remove excluded work.
The full-step profile performs one additional optimizer update (nine physical
updates in profiled capacity runs, eight otherwise), with its input hashes,
counters and post-profile health recorded separately.

## Evidence and completion

Freeze runtime sources/protocol before GPU measurements. Source changes need new
revision/run directories. Snapshot imported Dao/TE source files actually used,
package identities and compiler settings. Log W&B online under`taylorbollman`,
project`pretrained-fbt-rt-nextlat`, group`olmo-ordinary-fusions`.
Retain explicit successes/failures, source/protocol/dependency snapshots, reports,
logs, profiles and plots in`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-ordinary-fusions/`
with downloaded hash verification and the original checkpoint reference. No
profiling-update weights need saving. Finish with reviewed code, a measured
configuration recommendation and updated handoff, then stop for the next direction.
