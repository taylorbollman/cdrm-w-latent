# Pilot data test ledger

The final integrated CPU run passed **161 distinct new tests in 57.64 seconds**.
This is a data-preparation and recovery implementation result. It does not by
itself certify the actual 37-source corpus, its cloud recovery, or model training.

The run used the project container with GPU passthrough disabled. The log is
`.runtime/olmo-pilot-data/cpu-final-all-new-01.log`, SHA256
`c1ba832a80ea262d163473fbffe865860f56c8aa54b192b8a302580be11a0095`.
Its one warning is the existing Google API/grpc dependency notice. No GPU test
or live transfer is represented by these unit tests.

## Final integrated scope

The 161 tests cover these seven new files in one collection:

| Test file | Meaningful checks |
| --- | --- |
| `tests/test_pilot_data_plan.py` | Pinned source inventory and recipe; deterministic URL/document/chunk ordering; exact integer mixture quotas; content-hash splits and readiness exclusions. |
| `tests/test_pilot_data_prepare.py` | Actual gzip/JSONL extraction with injected network/cloud clients; whole-document bounds; source-plan/ETag authority; atomic source commits; failed-attempt accounting; global deduplication; 90/5/5 splits; byte-identical resumed preparation; retention before further preparation. |
| `tests/test_pilot_data_restore.py` | Exact-generation, size/SHA256/MD5 recovery; publication marker written last; malformed receipts, corruption, interruption, overflow, changed receipts and unsafe destinations rejected. |
| `tests/test_pilot_ordered_data.py` | Literal tokens and true document intervals after chunk shuffling; T1024 loss masks; source-order authority for first-occurrence deduplication; quotas, exclusions and panel overlap; byte-identical rebuild/relocation; pure peeks, rank partitioning, cursor recovery and finite exhaustion; bounded late-stream SQL lookup. |
| `tests/test_pilot_ordered_audit.py` | Independent interval/count arithmetic, original-corpus metadata comparisons, exact panel inventory, per-source selected membership, sampled literal chunks and CPU rank masks; forged catalog offsets and missing panels rejected. |
| `tests/test_pilot_data_audit.py` | Independent complete-document retokenization, terminal and embedded EOS, text/token/content hashes, sampled raw-to-original-line mapping, full metadata duplicate/split accounting and per-source raw EOF coverage. |
| `tests/test_pilot_index_retain.py` | Complete 44-object ordered-suite retention/recovery with injected cloud clients; exact generations and marker ordering; retries of already uploaded objects; unsafe paths, links, conflicting metadata, mutation and partial recovery rejected; bounded streaming above the old 128 MiB evidence limit. |

The streaming-size test hashes 129 MiB through a counting sink while reusing a
1 MiB block. It is not evidence of an actual 129 MiB SQLite upload. Likewise,
fake-cloud round trips exercise byte and publication contracts without making
an actual GCS reliability or performance claim.

## Earlier focused runs

These runs overlap the final collection and one another. **Do not add their
counts to 161.** Runs that include unchanged predecessor tests are explicitly
identified below.

| Log | Completed result | Scope and overlap |
| --- | --- | --- |
| `.runtime/olmo-pilot-data/cpu-plan-01.log` | 30 passed, 0.81 s | Planner scope, included in the final collection. |
| `.runtime/olmo-pilot-data/cpu-prepare-final-03.log` | 94 passed, 3.76 s | Final preparation tests plus unchanged extraction, document-shard and document-retention tests. |
| `.runtime/olmo-pilot-data/cpu-restore-01.log` | 16 passed, 0.13 s | New stage-restoration scope, included in the final collection. |
| `.runtime/olmo-pilot-ordered-data-cpu-03.log` | 66 passed, 12.66 s | Ordered-reader tests with overlapping planner and unchanged packed-reader tests. |
| `.runtime/olmo-pilot-data/cpu-root-audits-final-03.log` | 24 passed, 3.99 s | Ordered auditor plus stage-restoration tests. |
| `.runtime/olmo-pilot-data/cpu-ordered-audit-final-04.log` | 8 passed, 3.34 s | Ordered auditor alone after the original-corpus and panel-coverage checks. |
| `.runtime/olmo-pilot-data/cpu-audit-03.log` | 53 passed, 4.55 s | 23 raw/token-audit tests plus the same 30 planner tests. |
| `.runtime/olmo-pilot-index-retain-cpu-03.log` | 54 passed, 42.45 s | Index-retention tests with overlapping ordered-reader tests. |

Earlier logs remain available rather than being overwritten. In particular,
`.runtime/olmo-pilot-ordered-data-cpu-01.log` recorded 1 failure and 11 passes:
the T1024 test fixture supplied too few held-out tokens for its own declared
quota. The fixture was expanded; the builder's insufficient-data rejection
remained. `.runtime/olmo-pilot-data/cpu-ordered-audit-01.log` recorded 1 failure
and 7 passes from an incorrect `dev` lookup when calculating main/source panel
overlap; the auditor now uses the explicit `dev-main`/`confirmation-main` keys.
Both scopes pass in the final integrated run.

The actual immutable source plan also passed the preparation loader's CPU
preflight: 37 sources, the fixed native tokenizer and the declared recipe.
Evidence is `.runtime/olmo-pilot-data/actual-plan-preflight-01.log`, SHA256
`e6297d77eb4661198caf3d3f1b1ae2734d970959fd9645ee79860a59ae297cdf`.
This is a metadata-authority check, not an acquisition or corpus result.

## Source isolation

A read-only `git diff b9b0cb4 --name-status` check at
`db33a318fde178de27e8d9f45f765bdfe613449f` showed additions only: one new data
recipe, seven new scripts, seven new test files and four report documents.
There were no modifications or deletions of pre-existing files. In particular,
`cdrm/`, vendor code, the historical extraction/retention helpers and their
pre-existing tests were unchanged. This ledger is a subsequent documentation
addition; it does not change that runtime-source scope.

The closeout recheck at `d1654eceaa2ac6b3a689d64e1f7b43c95983312e` also
shows a documentation-only update to the existing `AGENTS.md` handoff. Existing
runtime, core, vendor and test files remain unchanged from `b9b0cb4`.

The new ordered reader has its own schema. Reusing established batching and
cursor methods does not mean that historical campaign executors accept the new
ordered index. That integration is a separate milestone.

## Completed actual-data acceptance

The following stages completed on the actual declared corpus. These are
separate results from the 161 unit tests. Report paths below are relative to
`.runtime/olmo-pilot-data/`; this ledger update checked the local finalized
reports and receipts without repeating downloads or tokenization.

| Acceptance stage | Completed result and evidence |
| --- | --- |
| Acquisition and raw retention | All 37 sources completed with verified receipts: 584,894 raw documents, 310,670,964 candidate tokens, 637,272,434 compressed bytes read and 2,007,717,910 retained raw JSONL bytes, within the declared successful-acquisition bounds. `acquisition-01/acquisition-progress.json`, its 37 raw receipts, and `raw-token-audit-01/report.json`. |
| Incremental tokenization and global exact deduplication | Completed 37 token shards containing 584,851 unique documents and 310,669,141 tokens; 43 duplicate rows removed. `prepare-call-10-summary.json` and `preparation-01/preparation-progress.json`. |
| Raw cloud recovery | All 37 sources restored from exact generations: 111 objects and 2,086,379,363 bytes, including raw JSONL, original-line maps and manifests. `raw-recovery-summary-01/report.json`. |
| Partial preparation recovery and continuation | Cloud-restored configuration and first four token shards, then regenerated the next four in a fresh process. All 25 compared files and the preparation summaries matched the original eight-shard preparation exactly. `partial-recovery-comparison.json`. |
| Whole-document retokenization | All 592 samples passed: 16 unique documents per source, including text/token/content hashes, complete token payloads, terminal/embedded EOS and original source-line mapping. Full metadata and per-source EOF counts also passed; exact-content split intersections were zero. `raw-token-audit-01/report.json`. |
| Ordered suite and independent counts | All 21 panels built with the declared quotas/reserves and exclusion policy. The independent auditor counted all 134,272 selected chunk rows across panels and verified original-corpus intervals, selected document/source memberships and aggregate per-loss counts. `ordered-build-01/report.json` and `ordered-audit-01/report.json`. |
| Literal ordered chunks, masks and cursors | All 336 literal samples passed: 16 chunks per panel. Two-rank CPU mask/count checks, committed-cursor restoration and finite exhaustion passed for all 21 panels. `ordered-audit-01/report.json`. |
| Complete tokenized-corpus recovery | Restored the remaining 29 token shards after the four restored and four exactly regenerated shards. Full semantic corpus verification passed, with identical original/recovered root manifest SHA256. `complete-corpus-recovery.json`. |
| Ordered-index recovery | All 44 index/manifest objects restored from exact generations, totaling 792,408,198 bytes, with the original suite identity and manifest hash. `index-recovery-01/report.json`. |

The 134,272 counted rows include intentional main/source-panel overlap; they are
not 134,272 independent observations. The audit records this overlap explicitly.
Its literal token reads cover 336 chunks, while interval/count checks cover every
selected row. Similarly, 592 documents were independently retokenized, not all
584,851 unique documents. The candidate corpus contains 2,119 earlier-readiness
identities; the ordered selection excludes them rather than rewriting the
candidate corpus.

The partial-continuation check establishes exact regeneration of four shards
from the restored boundary, not a second complete retokenization of the corpus.
The later full-corpus check validates the recovered tokenized artifacts and
their semantic metadata. Index recovery establishes exact bytes and identity;
it does not exercise a training executor or add a second full panel evaluation.

Key finalized report SHA256 values:

| Report | SHA256 |
| --- | --- |
| `raw-token-audit-01/report.json` | `b6053115ea9cf5101fc0cd29cd82f56af16bf6644dc64bd103c77b2505902484` |
| `ordered-audit-01/report.json` | `414fd5ab28b852757eb63e973ed24757a8b8e50360c602ce858c7102673db990` |
| `partial-recovery-comparison.json` | `939def46297ebedafbfb8acc040fc7fc954ea63815d8dabb3f220ab2388ed110` |
| `complete-corpus-recovery.json` | `259d22548ff31c8a62361eba4fe66b2bc46b6bc80e512597825eb6bec5d6ee36` |
| `index-recovery-01/report.json` | `70d4a2abc5aa141783bb0ef70d25dd639f2aa4a496497247d1bcaa6d9f31214c` |

No model training, GPU execution, outcome-based confirmation-set use,
near-duplicate filtering, reconstruction of the original OLMo training mixture,
or claim of unknown pretraining-exposure removal is established here. See the
[protocol](protocol.md) for the declared sampling and packing qualifications.
