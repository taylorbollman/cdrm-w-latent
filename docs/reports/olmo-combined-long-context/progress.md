# Combined T2048 benchmark interruption handoff

2026-09-28: authorized and running. Latest user instruction: **SDPA only,
then review; no FA4 run**. Read protocol.md. Historical T512 combined B128 is
12,361.82inputtokens/s. New model keeps K2, RT0/15, both NextLat losses, native
ordinary RoPE and accepted native optimizations. B2 operational, B16/B32 capacity
and one selected repeat planned; no new T512 or quality training.

GPU preflight passed: single idle H10080GB inside project Docker. Harness work
is isolated with agent combined_long_harness; root owns all GPU execution.
Runtime79e850a is frozen/pushed;122CPUtests passed. B2 operational stage
sdpa-b2-check-01 is running from16:59UTC, W&B run x8hkueyb.
No FA4 or larger-batch stage has started yet. Every finished stage should
be retained before proceeding. Source/protocol edits must stop before runtime
freeze and GPU queue. Prior numerical qualifications remain open.

17:06UTC: B2check complete, all5gates/8updates pass; exact initial/terminal
eager/graph parity. GCS verified under20260928T165900Z/sdpa-b2-check-01.
B16capacity launched; sources unchanged, reporthelper5f6c6ac adds15CPUtests
and is outside runtime pin selection. B32depends on measuredheadroom.

Audit: all37pretrained runtime sources and recordedpackageversions match the
savedcombinedT512report. NewB2 actualdispatch:4,094forwardhistorytiles,
4,088Triton/6eager; all4,094historicalbackwardcallsTriton. B16 W&B0gunb0o4.

17:14UTC: B16complete,5/5gates/8updates pass;7,501.52inputtok/s,
4.36818s/update,38.918GiBsetupallocated/41.227reserved;40.986GiB
steadyreserved and35.977GiBsampledfree. Proceed B32 for equal-tokenreference.
