# Native RT large-batch integration: prospective protocol

2026-09-24. User authorized the native RT large-batch plan and a conditional
FA4 memory comparison near capacity. No quality-training campaign. Native RT
is selected; historical backend/precision qualifications remain open.

## Fixed setup

Original OLMo-1B step60000 (~252B tokens),16 layers,D2048,H16/head128,
SwiGLU8192 per branch,tied50304 vocabulary,native nonaffine LayerNorm and
FP32 split-half RoPE/residuals,no Q/K normalization. Selected RT layers0/15.
RT-only or combined K2 FBT+NextLat; K2 bootstrap remains ordinary. Real-text
repeated fixtures with recorded token rotations and full valid CE on every
comparison/preparation/parity/timed update. CE2048/KL128, seed20260922.

All arms: native RT, reused RoPE tables and weight casts,K/V-only permanent
writes,Triton historical forward/backward tiles,recompute backward. All ordinary
layers checkpointed. BF16 autocast with FP32 parameters/gradients/Adam moments,
TF32 off,autocast weight cache off,deterministic execution. CUDA graphs capture
forward/loss/backward; input validation,clipping,Adam and scheduler outside.
One physical per-device batch per optimizer update; no accumulation/multi-GPU.

Control: ordinary native RoPE,eager SwiGLU,PyTorch Flash SDPA,scalar Adam.
Optimized: ordinary Dao native-FP32 RoPE,rounded compiled SwiGLU,fused AdamW.
Conditional FA4: optimized arm with ordinary attention alone replaced by FA4;
native RT remains Triton. Same LR1e-5,betas(.9,.95),epsilon1e-8,matrix decay0.1,
two-update warmup,max-norm1 clipping. These are execution probes, not learned
checkpoints to resume. Record optimizer identity and external dependency sources.

## Gates

Primary integration B8/T512, optimized versus control, separately RT/combined.
Optional B1/T32 smoke for harness failures. FA4 compared against optimized.
Preserve existing budgets: global raw-gradient L2<=1/64; per-tensor L2<=1/32
and max/reference-peak<=1/16; output L2<=1/64/max<=1/16; relative loss<=1e-5.
Zero-reference rules remain unchanged. No widening after observing results.
Finite numeric-only misses may complete operational diagnosis but remain failed;
timings of such a candidate are explicitly qualified. Ownership/nonfinite/graph/
optimizer/dispatch failures stop dependent runs pending a bounded fix.

Every shape requires own exact eager/graph losses and all parameter gradients,
changed tokens, repeated overwrite and changed weights. Capacity reference
gradients live on CPU, copied tensor by tensor, outside timing; no model-sized
GPU reference clone accompanies the graph. Small-batch integration additionally
requires three eager versus three graph full optimizer updates with exact
weights,optimizer state,scheduler,counters and metrics. Existing fixed-gradient
scalar/fused Adam qualification is reused, not rerun without a new concern.

## Capacity and memory

T512 physical B64/128 initially; adapt through B192/256/384/512 as memory allows.
Bounded refinements B32/96/160/224/320/448 are allowed to find useful headroom.
Fresh process per shape/arm. Three actual preparation updates, ten backward
warmups, five synchronized complete timed updates, three backward-only samples.
Repeat chosen point and smaller neighbor in reverse order; primary integration
arm comparisons have matched batches/objectives. No maximum-batch optimization
to the last GiB. Target at least ~8GiB device headroom when recommending a
comfortable point, accounting for setup as well as steady execution; explicitly
label any tighter shape as a capacity probe rather than recommendation.

Record absolute allocated/reserved peaks for load,prepared initialization,
dispatch,optimizer warmup,capture preparation,capture,validation and timing,
plus sampled device free/used/total at boundaries. Reset allocator peaks per
phase, preserving all phase records. Aggregate setup peak is the maximum across
setup phases, not a difference of snapshots. Steady timing peaks are separate.
Keep OOM stage and partial physical-update counts even if capture fails.

The optional release_transient_cache flag performs GC/unused allocator-cache
release only after warmup stream synchronization and before capture. Defaults
preserve behavior; live gradient/graph/model storage must remain valid. Compare
at a common shape before crediting capacity improvements to this setup change.
If larger physical batches fail, compare FA4 with optimized SDPA at the same
shape and at the next failed boundary, after FA4 integration checks. Retain
failures and do not presume FA4 saves memory already avoided by Flash SDPA.

Full-model timings include input handling,graph forward/loss/backward,clipping,
Adam and scheduler. Log input tokens/s and supervised CE targets/s separately.
Analytic matrix-FLOP ledgers exclude elementwise work/kernel padding and are
not hardware utilization measurements. Parameter ledgers distinguish resident,
active and deployable. No learning-efficiency inference from these timings.

Separate optional profiles run after timing: one captured backward plus one
complete optimizer update. Nonprofiled capacity has8physical updates,profiled9;
correctness has6. Exclude overlapping GPU annotation intervals in derived kernel
inventories, retaining raw traces. Compile the sequential RT finish/writer only
as a separately frozen follow-up if a useful profile supports it; no broad
activation-library sweep or long-recurrence unrolling in the capacity run.

## Evidence

Freeze source and protocol before runs. Persist report progress per phase/update,
log online W&B to taylorbollman/pretrained-fbt-rt-nextlat group olmo-rt-large-batch,
and retain source snapshots,dependency sources,raw reports/logs/traces/plots,
checkpoint provenance and failed attempts to gs://fast-chunks. Runtime changes
require new hashes and a fresh cohort; never rewrite old results. Review point:
integration gates, batch curves, explanation of any below512 limit, comfortable
operating points, bounded bottleneck profile and remaining limitations.
