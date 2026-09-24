# Ordinary OLMo efficiency: prospective protocol

Authorized2026-09-24 following the isolated FA4 investigation. This precedes
further RT-backend decisions and graph recovery/accumulation. No quality
training. Keep failed attempts and all existing RT numerical qualifications.

## Fixed model and objective

Original OLMo-1B step60000 (~252B tokens),16layers,D2048,H16,head128,
SwiGLU8192 per branch, native nonaffine LayerNorm epsilon1e-5, native RoPE,
no Q/K normalization, tied50304-token vocabulary. Strict native checkpoint
loading; no parameter conversion/addition. RT, FBT and NextLat disabled.
The common wrapper retains frozen unused fusion weights, counted separately.

Full next-token CE on the existing real-text fixture/token rotations, one
unpadded independent document per row. CE chunks2048, KL128 inactive. Full CE
masks must persist in changed-batch and optimizer-parity checks. Seed20260922.
One physical microbatch per update, no accumulation. BF16 autocast with FP32
parameters/residuals/gradients/Adam; TF32 off, deterministic algorithms, no
global autocast weight cache. Reuse native RoPE tables in every arm. Graphs
capture forward/loss/backward; clipping/AdamW/scheduler remain outside.

## Independent options

- `ordinary_attention_backend`: default`sdpa`, opt-in`fa4`; pinned installed
  FA4b20, deterministic backward. Initially dense unpadded full-sequence causal
  CUDA BF16/FP16 only. Reject unsupported masks/cache/math-reference conflicts;
  no silent fallback and no change to RT attention.
- `ordinary_pointwise_backend`: default`eager`, opt-in`compiled` for ordinary
  SwiGLU pure elementwise math. Preserve native split order/output dtype.
- `ordinary_checkpoint_layers`: optional tuple, `None` preserves previous
  boolean all/none behavior. Test all, alternating0/2/.../14, and empty/none.

Preserve checkpoint keys/ownership/default behavior. Static metadata must
capture options and reject stale layouts. Import CuTe after determinism/device
setup. Warm forward/backward before capture; record compilation/fallback
state and actual FA4 package/source identity. No whole-model compiler rewrite.

## Correctness

Focused CPU coverage: ownership/state keys, ordinary/RT isolation, stale static
metadata, options and FA4 layout/causal guards, selective checkpoint and pure
helper forward/gradient behavior. Run relevant existing regressions.

Actual-checkpoint primary checks: B8/T512 each individual candidate versus
SDPA/all-checkpoint/eager control, FA4 additionally B2/T2048; selected stacked
candidate gets its own B8/T512 check. B1/T32 available for startup diagnosis.
Full CE throughout. Checkpoint-only comparisons require bitwise outputs,
losses and raw gradients. FA4/compiled retain existing prospective budgets:
global gradient relative L2<=1/64, each tensor L2<=1/32 and max error/reference
peak<=1/16; output L2<=1/64 and max<=1/16; loss relative error<=1e-5.
Compare before clipping. No budget changes after observing a miss.

Each candidate separately requires exact own eager/graph losses and gradients
initially, with changed tokens/repeated overwrite, and after weight changes.
Three eager versus three graph AdamW updates from restored state must match
model, moments, schedule, counters and metrics, with finite participating
gradients and real weight changes. Full CE for every update. Track physical
updates incrementally on failure. Failed numerical screens remain failed even
if operational checks pass; any useful subsequent timing must be exploratory.

## Performance schedule

One confirmed container H10080GB, no competing GPU work. Fresh process per
run; three preparation updates, ten backward warmups, capture, then five
synchronized complete timed updates. Report medians, input/target tokens/s,
setup and steady allocated/reserved memory. Full steps include copies,
clipping/optimizer/scheduling; exclude compilation/setup/logging/profiling/data
reading. No sustained-learning claim.

1. Refresh native16-layer control B64/T512 with an untimed device-event trace.
2. Matched B64/T512 individual FA4 and compiled-SwiGLU arms; profile a useful
   candidate separately for attribution.
3. All versus alternating checkpointing B64; no-checkpoint B32 with matched
   all-checkpoint B32 control. B64/no-checkpoint only if memory is comfortable.
4. B16/T2048 control/FA4 and control trace; reduce both to B8 if needed.
5. If individual changes pass/show value, verify/time a selected combination;
   repeat selected matched baseline/candidate in reversed order. Target around
   fifteen capacity runs or fewer, not a Cartesian sweep.

Batch-size alternatives compare throughput/memory only; do not infer learning
benefits without matched effective batch/exposure. Accumulation is later work.
Profiles are separate replay traces, not timed samples. Device-event sums are
not full-step wall time. Classify GEMMs, Flash/CuTe, pointwise/norm, copies and
fills transparently; do not assume fills are avoidable. Analytic matrix FLOPs
must reflect actual checkpointed block counts and stay distinct from measured
hardware work. Runtime/memory concerns may narrow this bounded schedule.

## Evidence and completion

Online W&B entity`taylorbollman`, project`pretrained-fbt-rt-nextlat`,
group`olmo-ordinary-efficiency`. Freeze source/protocol before GPU execution;
save hash maps and snapshots. Runtime changes require new revision/directory.
Retain failures, reports/traces/plots/sources and existing checkpoint reference
in `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-ordinary-efficiency/`,
with downloaded hash verification. Disposable few-update states need not be
saved; the initial pretrained checkpoint is already retained.

Finish with reviewed code/tests, measured tradeoffs and a recommended explicit
ordinary configuration. New options remain default-off. No automatic quality
run, normalization change, RT-backend decision, broad loss rewrite or multi-GPU
claim. Future loss/cast/optimizer work follows the measured profile.
