# Combined T2048 benchmark interruption handoff

2026-09-28: authorized and in preparation. Latest user instruction: **SDPA only,
then review; no FA4 run**. Read protocol.md. Historical T512 combined B128 is
12,361.82inputtokens/s. New model keeps K2, RT0/15, both NextLat losses, native
ordinary RoPE and accepted native optimizations. B2 operational, B16/B32 capacity
and one selected repeat planned; no new T512 or quality training.

GPU preflight passed: single idle H10080GB inside project Docker. Harness work
is isolated with agent combined_long_harness; root owns all GPU execution.
No GPU stage for this milestone has started yet. Every finished stage should
be retained before proceeding. Source/protocol edits must stop before runtime
freeze and GPU queue. Prior numerical qualifications remain open.
