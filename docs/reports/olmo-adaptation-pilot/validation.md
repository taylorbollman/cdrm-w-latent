# Validation and reproducibility

No runtime/model/test source changed in this milestone. The accepted PR51
runtime85e5f78 remains pinned to200 files (canonical inventory digest
`d84006ede4e5ad488880da2af387355ca82fc76d186cf87ba16cb007d2279fc5`).
Prior runtime unit/gradient/restart tests were not rerun for this execution-only
cohort; they remain prior evidence, not new results.

The new CPU-only adoption report verifies the copied declarations byte-for-byte,
all128 logical update memberships/counts/order across B/NF/NFR, and allthree
native preflights. All96 actual optimizer updates and six fixed-panel evaluations
completed. Training used two GPUs only inside the required container. CPU-only
analysis/retention children explicitly disable GPU passthrough where applicable.

The independent JSON summary checks:

- Complete fresh32-update histories at the authorized segment boundary; the
  finite declaration remains128, with no automatic continuation.
- Current and snapshotted200 runtime pins, declaration/resolved hashes,
  execution recipe, physical allocation and actual global loss denominators.
- Same ordered524,288 inputs/update, exact cursors and counters, actual LR used
  and next LR, finite raw norms/objectives, clipping telemetry and ownership.
- Exact named parameter SHA/shape/dtype across both replicas and NF/NFR at
  origin:71 tensors. This comparison excludes buffers, flags, RNG and Adam.
- Same fixed development membership at16/32 and per-pass counts. Lean insertion
  preservation checks tensor ownership/storage/version, zero gradients,
  runtime/modes/RNG and cursor/graph metadata. This is not a fresh full-byte
  model/Adam equivalence claim.
- Every checkpoint job, worker result, receipt, publication, journal and final
  cloud32 authority. Existing generation/hash/readback receipts are validated;
  the summary does not download all state payloads again.
- Final synced native W&B runs and no outstanding retention worker.

Summary source: `.runtime/olmo-adaptation-pilot/summarize_cohort.py`, SHA
`309de6a85cc95f9c28b5d98716a158f443a8ebb03ebe2b96e2b41a5770a24e85`.
It passed first for completed B, then B+NF, then the full cohort. These are
incremental metadata summaries, not repeated GPU tests. Full reportSHA:
`ffb6e4eae228e4ae8e57394bc117a90ce2918eab3d3adb7820f64084d3e1178c`.

The CPU-only W&B chart logger passed and synced
[p7vk0qz0](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/p7vk0qz0).
Its sourceSHA is`20ea119dd004dd903fd27e27d6e4e64b19d076f7fe3a4769f01360f4a186dd52`,
reportSHA`f8853c62729f88bbb9af40ee6b98f9b3b3ea458c093f29d36c1766fe53d10935`.
It leaves original run metadata untouched and distinguishes disabled auxiliary
losses from measured zero. It adds no training updates.

Recovery queue02 preserves and validates13 original authority snapshots. B's
original launcher exit0 is recorded; NF's launcher exit is unknown after the
host scheduler interruption, while its final executor/W&B/cloud authorities
establish completion. Only unstarted NFR was launched, with exit0. queue01
remains preserved/stale. There was no training restart or repeated update.

The storage inventory uses immutable receipt metadata and neutral snapshot
filenames; no live state payload is read or reverified during its construction.
Its limitations and later receipts are in [storage-receipt.md](storage-receipt.md).
Python helper syntax/metadata checks and `git diff --check` pass. Independent
read-only review checks numerical/accounting claims against the summary.

Execution success does not clear all historical BF16 differences or demonstrate
useful refinement. The same large later-pass CE deficit exists in NF without RT.
See [results](results.md), [assessment guide](assessment-guide.md), and
[next steps](next-steps.md) for interpretation and the explicit review stop.
