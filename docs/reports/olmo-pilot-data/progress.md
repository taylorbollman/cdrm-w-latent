# Progress and interruption handoff

2026-09-29: **pilot-data milestone complete**, PR49 on `feat/olmo-pilot-data`,
starting from `b9b0cb4`. No GPU work or model training was performed. All bounded
data jobs have finished; do not restart acquisition or preparation. The next
milestone is runner integration, described in [next-steps.md](next-steps.md).

## Completed acceptance

- All 37 declared sources acquired and retained without substitutions or quota
  changes: 584,894 raw documents and 310,670,964 candidate tokens.
- Global exact deduplication: 584,851 documents / 310,669,141 tokens, 43 duplicate
  rows removed. Whole-document splits and readiness exclusions are frozen.
- All 21 ordered T1024 panels built. Training contains 131,072 full chunks /
  134,217,728 inputs; main and source held-out views overlap as documented.
- Final integrated new-code CPU suite: **161 distinct tests passed**. Earlier
  scoped runs overlap; see [test ledger](test-ledger.md).
- All metadata and 592 sampled complete-document retokenizations pass.
- All 111 raw objects restored from pinned GCS generations. Four token shards
  restored and next four prepared in a fresh process equal the original eight
  shards exactly (25 files plus summary). Remaining 29 shards restored and full
  recovered corpus verification passes.
- All 44 ordered-index objects restored exactly. Independent audit on the
  recovered corpus/indexes passes: 134,272 row entries, 336 literal sampled
  chunks, two-rank masks, cursor restoration, finite exhaustion and membership.
- W&B data-audit run `qyp83axd` is synced. No model outcomes were evaluated.

The immutable source plan is
`.runtime/olmo-pilot-data/declaration-01/source-plan.json`, SHA256
`5954f4480268e50c9041ffc06651a6539efe93ec54f6eb36a2977a4d5e2a9707`.
The corpus manifest is
`f3206c360dd412ccbaf08c65b30a7ba7f1b30b56db7ffbb9559522831c797e06`;
ordered suite manifest is
`080402225e24350cc377b720bdf39a55b87e71e4a6af5fea3cda7460eb169215`.
All other authoritative hashes are in [storage-receipt.md](storage-receipt.md).

## Locations and source freeze

- Persistent evidence: `.runtime/olmo-pilot-data/`, including acquisition and
  preparation receipts, audit reports, recovered-data records and test logs.
- Original SSD root: `/mnt/localssd/cdrm-data/olmo-dolma-v1_5-pilot-20260929`.
  Separate recovered root appends `-recovery`.
- Data GCS root:
  `gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/pilot-20260929-v1`.
- Evidence GCS root:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T174435Z`.
  Every stage uses a distinct `pilot-data-` name; receipts are local under
  `retention/` and the final closeout preserves their catalog.

Runtime helpers reached their final version at `db33a31`; subsequent edits are
only documentation. Individual source snapshots and helper pins are retained
with each stage; do not equate the latest Git commit with every earlier stage's
source identity. The original helpers, model/vendor sources and tests remain
unchanged. Add new versioned adapters for ordered execution rather than changing
historical acceptance contracts or migrating old cursors silently.

## Interpretation and next action

Read [results.md](results.md) and [coverage-assessment.md](coverage-assessment.md).
Books have small selected-document counts and source/main panels overlap.
This is a conditional sample of selected object prefixes, not original OLMo
stream reconstruction; unknown pretraining exposure and near duplicates remain.
Confirmation membership checks do not authorize outcome-driven selection.

The ordered reader is accepted at the data/CPU level, not yet in the distributed
executor. Next: new execution/evaluation adapters, tiny two-GPU graph/evaluation/
cloud-restart acceptance, then native B/NFR T1024 physical batch and evaluation
cost measurements. Startup exposure, BF16 qualifications and training-budget
choices remain explicit. No quality campaign or next data round is queued.

## Closeout metadata

PR: https://github.com/taylorbollman/cdrm-w-latent/pull/49.
Code and report commit `496edaf` is preserved in the verified `pilot-data-closeout`
cloud stage, including source snapshots, receipt catalog, operational scripts and
all test logs. The receipt is `.runtime/olmo-pilot-data/retention/closeout.json`.
An independent closeout review verified report counts, hashes, links, packing,
recovery provenance and source isolation without finding a blocker. Final merge
metadata follows after publication.


PR49 merged successfully on 2026-09-29 as
`c433406649c78656a7d820e9f02ba815c30aed08`; final PR head was `0ef18b0`.
Local checkout is `main`. Closeout receipt SHA256:
`235ae69a7860f22d02b45c7662d513e689e3f7244d19fa7819b582ad61e7e141`.
The post-merge record and final documentation are retained separately under
`pilot-data-publication`, with local receipt `retention/publication.json`.
There are no active or queued jobs from this milestone.
