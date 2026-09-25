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

Completed additions since the initial ledger above:

- Actual combined graph correctness `combined-graph-01` passes four compound
  checks on both ranks (`qldwb8so`).
- Actual combined graph reconstruction recovery `combined-graph-recovery-01`
  passes42checks, six physical updates/rank, logical endpoint5 (`xq1w6i0m`).
- Actual combined ZeRO1 `combined-zero1-01` passes13checks/rank, four distributed
  updates plus two reference Adam steps/rank (`2t6fk853`). Native consolidation
  exposed a slow per-byte sender conversion. Its full checkpoint is retained.
- Buffer-based transport fix at `bfa3649` preserves the native wire/format and
  optimizer math. Root54focused CPU tests, tiny NCCL recovery and full-model
  `combined-zero1-buffer-01` recovery all pass. Full checkpoint save83.4seconds
  includes consolidation/disk writing/hashing. Both checkpoints are verified in
  GCS and local large duplicates cleaned only after receipt/hash verification.
  See `zero1-consolidation-transport.md` for measured versus extrapolated scope.
- Actual combined ZeRO1 graph `combined-zero1-graph-01` passes initial/terminal
  exact raw checks plus parameter-replica and local-moment ownership/health
  checks. This B1 integration row is not a representative throughput result.
- Matched RT globalB128: single B12828,000.26tokens/s (`eezfn9r8`) versus two
  GPUs B64/rank47,384.11tokens/s (`ahbhvxmx`), a1.6923x speedup. Both pass
  initial/terminal raw graph checks. GPU-microseconds/input-token35.714→42.208.
- Single combined B128 passes at12,361.82tokens/s,5.30148seconds/update.
  `ddp-combined-b64-01` is running. No combined scaling conclusion yet.

Source `6921c53` is pushed; runtime/code change is `bfa3649`. Active exec queue
(session17974) runs combined Zero1 graph, single combinedB128, then DDP
combinedB64 sequentially; only the last is still active. After it:

1. Retain completed combined scaling evidence.
2. DDP useful physical batches: RT128/192 per rank, combined128 per rank.
3. ZeRO1 RT192 and combined128 per rank, matching the replicated candidates.
   Consider one RT256 boundary probe only if useful; no bucket-view adoption
   without separate checking. Report per-rank setup/steady memory, aggregate
   tokens/sec, GPU-seconds/token, parameter/state bytes and matrix-FLOP estimates.
4. Final inventory/report/PR. ZeRO2 remains conditional on a demonstrated need;
   no offload/Stage3/FSDP or quality run is queued.

Host standard-library audit: `python3 .runtime/olmo-two-gpu/audit/inventory.py`.
It writes only audit artifacts, excludes running stages from final counts,
rehashes source snapshots and retained archives, and labels B1 integration
versus capacity measurements. Re-run after final retention. Do not sum
compound checks across scopes or count replicated checks as independent cases.

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
