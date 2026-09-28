# Combined T1024 interruption handoff

**Complete, 2026-09-28. Read results.md first.** No GPU job, quality training,
SFT, FA4 experiment or optimization is queued. GPU verified idle.

The user is considering context 1024 for an early RT value screen, possibly
~500M continuation tokens plus SFT. That is planning context only. Current
milestone measured unchanged K2 FBT + native RT0/15 + NextLat with Flash SDPA,
native ordinary RoPE and the established mixed-precision/checkpoint/graph path.

## Result

- B32/T1024: 9,823.08 input tokens/s, 41.123 GiB reserved, 36.545 GiB sampled free.
- Repeated B64/T1024: 11,092.06/s, 68.445 GiB reserved, 9.195 GiB sampled free.
- Saved B128/T512: 12,361.82/s. T1024 retains 89.73% throughput: 10.27% deficit,
  or 11.45% more time per token. Saved B32/T2048 is 9,184.55/s (25.70% deficit).
- All three equal-token rows process 65,536 input tokens/update. T1024 B64 is
  20.77% faster than T2048; its repeats differ only 0.052%.
- 500M input tokens imply ~12.52 compute-only hours at B64, with this fixture's
  full CE/latent and response-half KL supervision; not including SFT/overheads.

Three new stages pass 15/15 gates and 24 updates. Exact own initial/terminal
loss/raw-gradient graph checks. Runtime dcbde27, report helper46c03f0; 160 scoped
CPU tests. All 489 new source snapshots verify; all 37 pretrained files and
recorded packages match both retained contexts. Historical comparisons were
re-audited, not rerun. Existing independent BF16 qualifications remain.

Actual T1024 forward history: 2,046 tiles across RT0/15, 2,044 Triton plus two
existing eager512x512 tiles. All 2,046 historical backward calls use Triton.
No production arithmetic, kernel, Q/K normalization or parameter change.

## Evidence and next action

All three stages and closeout retained under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T181800Z/`.
The legacy prefix is used by the retention helper; this is a one-GPU milestone.
See storage-receipt.md for verified hashes. Local evidence remains under
`.runtime/olmo-combined-t1024/`. Original O1 weights are retained by reference;
no full checkpoint is needed for disposable timing updates.

Recommend T1024 for the initial directional screen, B64 for the unchanged
configuration and B32 when headroom matters. Next review should select the
actual continuation/SFT data, objective masks, held-out metrics and matched
FBT+NextLat control without RT. Benchmark that final data/objective configuration
briefly before a long run. No automatic launch of 500M tokens is authorized by
this benchmark. See screening-notes.md for intent and qualifications.
