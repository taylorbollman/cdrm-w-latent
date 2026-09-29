# Progress and interruption handoff

2026-09-29: work in progress on `feat/olmo-pilot-data`, starting from `b9b0cb4`.
The user authorized proceeding with the bounded pilot-data milestone. No GPU
work or training is part of the current stage.

- Plan/recipe: 30 CPU tests passed. All 37 selected upstream objects passed HEAD
  inspection. The earlier readiness metadata yields exactly 12,512 exclusions.
- Source plan: `.runtime/olmo-pilot-data/declaration-01/source-plan.json`, SHA256
  `5954f4480268e50c9041ffc06651a6539efe93ec54f6eb36a2977a4d5e2a9707`.
- Canonical recipe SHA256:
  `d826a3d005c3dfa56b8f1dbe882cd91f84d0d41922cac97563ec640cb5f93ba6`.
- Streaming generation-pinned stage restore: 16 CPU tests passed. Actual cloud
  restore is pending.
- Preparation frozen at `3f26dac`: 94 tests in the affected CPU scope pass.
  The actual plan passed preflight. Sources and tests are immutable because their
  hashes are in each extraction authority. All acquisitions use the same plan.
- Ordered reader frozen at `dff5d18`: 66 affected CPU tests pass. Independent
  original-corpus interval/count auditor at `6641cea`; 24 combined audit/restore
  tests pass. Raw/token auditor frozen at `129efdf`: 23 new tests pass (53 with
  the overlapping planner tests). Scope counts overlap; do not sum them blindly.
- Acquisition and a separate exact-generation raw recovery queue are running.
  At 18:15 UTC, over20/37 sources are retained. Logs are
  `acquisition-queue-01.log` and `raw-restore-queue-01.log`. Source receipts and
  seals are in `acquisition-01/receipts`. First source actual cloud restore
  passed33,864,700bytes/3objects in2.54seconds.
- The first four-shard preparation call is queued behind complete acquisition;
  see `preparation-first-01.log`. It stops after that bounded call. Continue with
  a first-four-shard cloud restore and fresh-process partial preparation replay,
  then full original preparation, ordered build and independent audits.
- Source declaration, pins, recipe, exclusion list and source snapshots are
  retained with readback at
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T174435Z/pilot-data-declaration`.
  Receipt: `.runtime/olmo-pilot-data/retention/declaration.json`.
- Separate streaming ordered-index retainer is complete/frozen at `db33a31`.
  It handles the exact44-file suite up to8GiB without changing old128MiB evidence
  limits or disguising SQLite file types. Its affected54CPUtests passed.

At18:30UTC, all37sources are acquired/retained/restored. Actual totals:
584,894rawdocuments,310,670,964candidate tokens,637,272,434compressed bytes and
2,007,717,910rawJSONLbytes. Raw recovery verified111objects/2,086,379,363bytes;
summed transfer time96.71s, excluding waiting/orchestration. Evidence retained
under stages `pilot-data-acquisition` and `pilot-data-raw-recovery` in the same
small-evidence root; receipts are in `.runtime/olmo-pilot-data/retention/`.

The final combined new-code suite passed161distinctCPUtests in57.64s. See
[test-ledger.md](test-ledger.md); prior scoped counts overlap.
Cloud-restored first4token shards followed by fresh-process next4 matches the
original8-shard preparation exactly:25files and complete summary.
`.runtime/olmo-pilot-data/partial-recovery-comparison.json` is the comparison.
Current original preparation has12shards/~100.9Mtokens; remaining bounded calls
are active. All current code is saved/pushed. Queued follow-ons wait for complete
preparation: raw/token audit, ordered build and restoration of remaining shards.
Complete ordered-suite quota/audit/retention acceptance is still pending.

Evidence: `.runtime/olmo-pilot-data/`. Proposed SSD root:
`/mnt/localssd/cdrm-data/olmo-dolma-v1_5-pilot-20260929`.
Every source/shard must be retained before more work. Keep old sources frozen;
new source files are versioned separately. Do not infer a quality campaign from
this data-preparation work. See [protocol.md](protocol.md) and preceding
[next steps](../olmo-campaign-storage/next-steps.md).
