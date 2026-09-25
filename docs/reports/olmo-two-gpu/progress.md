# Two-H100 progress and interruption handoff

Updated 2026-09-25. Active branch `feat/olmo-two-gpu`, base `25bc29d` (PR29).
User authorized the distributed plan and progressive checkpoint/evidence
retention. No quality-training campaign is queued. Main plan:
`docs/native-rt-single-to-two-gpu-plan.md`; frozen first-stage protocol is beside
this file. GPU work runs only in the project container. Inspect live processes
and the latest per-rank JSON before restarting any interrupted stage.

## Verified results

- Hardware: two H100 80GB, NV18 link, bidirectional peer access; PyTorch
  2.13.0a0+8145d630e8.nv26.06, CUDA13.3, NCCL2.30.5. Genuine NCCL sums pass
  through256MiB. Median25MiB all-reduce0.133ms;256MiB0.918ms. These isolated
  payload timings do not establish training overlap. `nccl-01`, W&B `cc0xn9sa`.
- Tiny eager: eight FP32 feature combinations, three accumulated updates each,
  genuine NCCL/DDP. Raw gradients and complete Adam states meet unchanged
  canonical-reference budgets; replicas match exactly. Includes predictor use
  in an earlier rank-zero no_sync microbatch but not the final synchronized
  microbatch, and a globally inactive predictor on update3. `tiny-eager-01`,
  W&B `ngrrexb6`.
- Actual eager B1/rank/T512 with two accumulation microbatches: ordinary and RT
  pass two updates; gradient relative L2 ordinary1.00e-8/5.57e-9 and
  RT3.73e-9/6.48e-9. Combined update1 passes; update2 has a retained strict
  complete-update qualification described below. `actual-eager-01`, `n4wmj1r4`.
- Actual combined anchored follow-up passes both fixed-state updates, unchanged
  budgets. Gradient relative L2:2.343016713e-9 and3.659744905e-9. Canonical model,
  Adam, scheduler and counters are restored between comparisons. This resolves
  the fixed-state distributed-update question, not independent trajectory
  equivalence. `combined-anchored-01`, W&B `jkmiz5jt`.
- Tiny eager recovery passes24checks, including exact next local RNG draws,
  data cursor, gradients, complete state and next update. Four physical updates
  per rank, logical endpoint3. `tiny-recovery-01`, W&B `zlmvso6v`.
- Tiny actual-DDP CUDA graph correctness passes initial raw parity and two
  changed-input complete Adam update comparisons, with exact replicas. Seven
  physical updates per rank, logical endpoint5. `tiny-graph-03`, `z3hce07b`.
- Tiny graph reconstruction recovery passes42checks. Both reference and restored
  branches rebuild graphs; next RNG draws, gradients, loss, full state and cursor
  are exact. Six physical updates per rank, logical endpoint5. Graph pools are
  released before checkpointing. This is not a live-graph checkpoint or a new
  torchrun/process-group restart. `tiny-graph-recovery-02`, `oqrupdlt`.
- Tiny ZeRO-1 passes13checks/rank: two fixed-gradient Adam comparisons to fully
  replicated Adam are exact, shard ownership/state bytes are verified, then a
  consolidated checkpoint reproduces the third update exactly. Four distributed
  updates plus two reference Adam steps per rank. `tiny-zero1-01`, `094wai23`.
- Actual RT eager recovery passes24checks, exact next update after model/DDP/
  Adam reconstruction, including per-rank RNG/cursor. Four physical updates per
  rank, logical endpoint3. `rt-recovery-01`, W&B `02tpiy0e`. Its14,154,909,669-byte
  state checkpoint is verified in GCS; local state was removed after rehashing
  to leave room for the next checkpoint. Manifest and cleanup receipt remain.
- Actual RT CUDA graph correctness passes all four compound checks on both
  ranks: initial raw parity/replicas and two changed-input complete Adam updates.
  Seven physical updates/rank, logical endpoint5. `rt-graph-02`, `a4sdnn1o`.

All actual cases use the original OLMo-1B step60000 checkpoint, native RT at
indices0/15, BF16 mixed with FP32 weights/gradients/moments, native RoPE,
compiled rounded ordinary SwiGLU, fused Adam and ordinary activation
checkpointing. Combined means K2 FBT plus NextLat. Original input tokens count
once, regardless of feedback passes. No Q/K normalization change.

## Qualifications and retained failures

`actual-eager-01` remains **failed**. Combined independent update2 passes loss,
counts, raw-gradient budgets (global L2 1.3509761003e-5) and exact inter-rank
replicas, but20 parameter and7 moment tensors miss strict elementwise update
budgets. Maximum parameter absolute difference1.383945346e-6; maximum parameter
tensor relative L2 9.0963e-8. The anchored result supports reduction-order/BF16
trajectory divergence as an explanation; it does not erase this failure or
clear the older native-versus-author BF16 qualification.

Retained failed attempts: `tiny-graph-01` used an empty DDP forward argument list
and hit installed PyTorch's moved_inputs[0] path; fixed by passing the static
token tensor explicitly. `tiny-graph-02` restored unchanged persistent buffers
and advanced their versions; fixed by restoring parameters only while checking
fixed-buffer equality/storage/version. This failed attempt executed four Adam updates per rank before the restoration error. `tiny-graph-recovery-01` omitted the
dependency recorder output directory; corrected call. `rt-graph-01` launcher
omitted required NCCL async-error settings and was rejected before process-group
initialization. None is hidden by a passing retry; no numerical budget changed.

## Active queue

`combined-graph-01` full-model B1/T512 correctness passed all four compound checks on both ranks (W&B `qldwb8so`); its evidence is verified in GCS. `combined-graph-recovery-01` passes42checks (W&B `xq1w6i0m`), with six physical updates/rank and logical endpoint5; checkpoint retention is running. The updated terminal-check harness passes `tiny-capacity-01` (`wgvh2kza`) and `tiny-zero1-graph-01` (`jeubp3ph`), eight updates/rank each. Source `340fe25` includes those checks. Next:

1. Finish retaining actual combined graph reconstruction checkpoint.
2. Actual combined ZeRO-1 fixed-gradient Adam and recovery; retain checkpoint.
3. Matched single-GPU B128 versus two-GPU B64/rank scaling for RT and combined.
4. Comfortable larger local batches (RT128/192, combined128), optionally validated
   bucket views. Report setup/steady per-rank memory, aggregate tokens/sec,
   GPU-seconds/token, actual parameter/state bytes and analytic matrix FLOPs.
5. Tiny then actual ZeRO-1 graph/capacity integration. ZeRO-2 only if useful
   memory needs justify the larger integration change. No offload/Stage3/FSDP.

Root owns GPU execution. Agent work is isolated; source snapshots are frozen
for running probes. Do not change runtime sources while a probe validates hashes.
Full checks include expensive CPU hashing/comparison; their wall time is not
training throughput. Timed capacity probes exclude those checks.

## Persistence and tests

Verified stage receipts are in `.runtime/olmo-two-gpu/retention/`; prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T172000Z/`.
All completed stages through `combined-graph-01`, including failed attempts,
are retained. Source through `d88be56` is pushed. Original pretrained artifacts are retained by their existing
O1 manifest/storage receipt, not repeatedly uploaded.

Main disk has about19GiB free after cleaning the GCS-verified generated RT
checkpoint. Each actual full-state checkpoint is13–14GiB. Keep one at a time,
verify GCS before local cleanup, retain local manifest/receipt. Never treat local
SSD as durable. For GCS container calls use `env -u GOOGLE_APPLICATION_CREDENTIALS`
to select mounted ADC; inherited custom ADC path is stale. Never print secrets.

Final capacity integration CPU scope:77passed (replicated graph, single-reference, ZeRO1 graph and ZeRO1 runtime tests). Earlier focused CPU scope:91passed (graph runtime/harness, anchored validation,
graph recovery and ZeRO1). Earlier scopes:129 integrated,58 static-input fix,
44 eager recovery/checkpoint,24 retention,13 single-reference,64 eager and32
checkpoint. These scopes overlap: do not sum them into a distinct total.
