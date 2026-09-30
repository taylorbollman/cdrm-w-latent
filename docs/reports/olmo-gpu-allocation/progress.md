# Allocation readiness progress and restart notes

2026-09-30, branch `feat/olmo-gpu-allocation`. User authorized the first milestone
in plan.md, including the H200 capacity assessment. No quality-training
continuation beyond NFR128 is part of this work.

- Plan, capacity evidence and isolated host launcher committed/pushed50660d5.
- New benchmark and independent acceptance committed/pushed5c9fa46.
- Historical B/NFR source inventories:422entries checked, zero mismatches.
- CPU:18benchmark tests and4independent acceptance tests pass. Actual CPU
  imports of B32/NFR128 preserve parameter/Adam ownership, populated moments,
  saved LR and actual loss coefficients.
- GPU: tiny one- and two-rank graph B/NFR acceptance both pass. Each condition
  executes three full updates against an independent global objective with
  changing unequal partitions, masks, true document boundaries and dummy rows.
  Reports `.runtime/olmo-gpu-allocation/tiny-graph{1,2}-01/report.json`.
  This is same-precision FP32 operational acceptance, not general BF16 clearance.
- Native Bpair cell launched20:32UTC, W&Bp24fz4yu. Inspect live report and
  supervisor before relaunching anything; no automatic restart is configured.

Runtime root: `.runtime/olmo-gpu-allocation`. `origins.json` records original
report/checkpoint identities, configuration and frozen source hashes. Original
B32 and NFR128 production checkpoints remain read-only. New cells are disposable
performance clones; only their small evidence is retained, not full new weights.

Planned cell order: Bpair, Bsingles, NFRpair, NFRsingles. Each uses two full
optimizer warmup updates and four measured updates. Native Bphysical32/rank,
NFR12/rank; T1024/effective512 real rows. Both singles use a shared start gate
after warmup. Core arithmetic/DDP/graphs remain unchanged. Check `supervisor.json`,
per-job `report.json`, logs and live GPU state before proceeding after interruption.

The host supervisor command is:

```
python3 scripts/olmo_allocation_queue.py --arm B --layout pair \
  --origins .runtime/olmo-gpu-allocation/origins.json \
  --output-dir .runtime/olmo-gpu-allocation/b-pair-01
```

Use new output paths for new cells. Host supervisor executes all GPU work via
the required project Docker launcher, sets per-job GPU visibility, separate
torchrun rendezvous, validates idle assigned devices and captures container IDs
for bounded cleanup. Do not run training directly in the host shell.

Later milestone remains full production checkpoint topology migration/restart.
This versioned performance entrypoint does not relax the historical loader's
same-world-size restriction or extend the finite128-update training plan.

20:42UTC update: first Bpair01 finished finite math and W&B sync but blocked
in process-group teardown while captured NCCL graphs remained live. Stopped
only its owned container; supervisor correctly records failure. Evidence is
retained under `allocation-b-pair-teardown-01` in the declared GCS prefix.
New wrapper explicitly releases graphs/reducer before group destruction and
marks completion only afterward; core math unchanged. Fix committedcdc4636.
Primary Bpair02 and Bsingles01 both complete and exit0. Joint-window rates
70,318 and71,590 inputs/s respectively (singles+1.81%, directional only),
99.53%common overlap over the singles makespan; pair gives1.96x the mean
individual single-job rate. All use35.10GiBallocated/42.39GiBreserved perGPU.
Local independently validated Bsummary is `b-summary-01/report.json`.

Native NFRpair01 is running, W&Bzbbf26i3. Detached bounded suite PID1942554
will then launch NFRsingles01 and stop. Its report is `suite.json`; write
`.runtime/olmo-gpu-allocation/STOP` to stop between cells. Do not launch a
duplicate. Remaining suite producer and commands are retained under the runtime
root. Final focusedCPUtest invocation has42passed in23.05s; log there.
Summary helper committed0f4dd98. All code is pushed; full checkpoint migration
and eight-rank/H200 measurements remain outside this first milestone.

21:14UTC update: NFRpair01 completed and exited cleanly, W&Bzbbf26i3 synced.
Measured-window throughput3,552.78inputs/s; peakallocated42.80GiB and reserved
59.03GiB/rank. Its76-member evidence archive is cloud-verified. NFRsingles01
is the only active cell, with W&Bsingle0ydqeqv2y/single1ube3ynpa; the bounded
suite will stop after it. No scientific continuation beyond128 has occurred.
DraftPR57 is open. Final allocation summary and production migration plan are
being prepared without changing frozen benchmark/runtime sources.
