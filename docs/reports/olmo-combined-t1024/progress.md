# Combined T1024 interruption handoff

2026-09-28: running. Read protocol.md. User set FA4 aside and is
considering T1024 for a future early RT value screen (~500M tokens plus SFT).
Current work is the throughput/memory benchmark only, not that training run.

Preserve K2 FBT, native RT0/15, NextLat both losses, native ordinary RoPE and
Flash SDPA. Reuse historical T512/B128 and T2048/B32 results. Planned new runs:
B32/T1024, B64/T1024 if headroom permits, and a fresh repeat of the selected
batch. Each run includes the existing five operational gates and eight updates.

Root owns GPU execution. Container preflight passed: one idle H10080GB, no GPU
processes. Persistent disk has ~14 GiB free; no new full checkpoint needed.
Agents combined_t1024_harness and combined_t1024_audit own isolated implementation
and report-helper changes. Source/protocol must freeze before starting GPU work.
Runtime dcbde27 is frozen and pushed; 137 runtime CPU tests passed.
B32/T1024 stage sdpa-b32-01 started at 18:18 UTC, W&B run o8ui9kw7. Retain every completed
stage before proceeding; prior BF16 qualifications remain.

B32 complete: all five gates/eight updates pass. 9,823.08 input tokens/s,
3.33582 s/update, 41.123 GiB reserved and 36.545 GiB sampled free. B64 next.
