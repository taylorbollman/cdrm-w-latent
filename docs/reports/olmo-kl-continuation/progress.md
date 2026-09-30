# Progress and recovery

2026-09-30: user approved paired NF update32→64 KL1/.1 continuation. Branch
`feat/olmo-kl-paired-continuation`, based on main a2ea5c4 / merged PR53.
Original 200-source production runtime remains unchanged. New helpers explicitly
fork only objective metadata after exact saved-Adam restore. Protocol in this
folder is the execution authority; no automatic extension beyond64.

Implementation review covered independent parent boundary equality, original
strict configuration loading, explicit child identity, scheduler/data continuity
and graph construction after the transition. The launcher retains a dependency
on the original parent files even for child resume; metadata records it.

Initial focused CPU validation: 40 tests pass (7.50s). Independent auditor tests
and tiny two-GPU unchanged-control/cloud-restart checks are pending. No native
continuation launched yet. Runtime/evidence: `.runtime/olmo-kl-continuation`.
SSDs: `/mnt/localssd/cdrm-checkpoints/kl-continuation`; parent NF32 stays under
`adaptation-pilot/native-nf12-first32-01/update-000032`.

Host launchers must be detached with a new session. Inspect live container work,
report.json, *-launch.json and publication receipts before recovery/relaunch.
A running/stale host report alone is not evidence the GPU run stopped. Commands
and source snapshots are retained per attempt; never overwrite failed attempts.
