# Dolma pilot data: preparation and recovery passed

The declared T1024 training panel now contains **134,217,728 input tokens**.
All 21 training, development and confirmation panels meet the original quotas,
and independent audits pass on the recovered data and indexes. No quota was
reduced and no source was substituted to obtain this result.

This is data readiness, not a learning experiment: there were **zero optimizer
updates**, no GPU execution and no model-outcome evaluation. The previous BF16
qualifications remain unchanged. The ordered reader still needs integration
with the accepted distributed runner before training can use this stream.

Implementation: [PR49](https://github.com/taylorbollman/cdrm-w-latent/pull/49).
Data allocation and coverage charts: [W&B audit run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/qyp83axd), synced.

## Prepared data

| Quantity | Actual result |
| --- | ---: |
| Selected upstream objects / strata | 37 / 9 |
| Complete raw documents | 584,894 |
| Candidate tokens before exact deduplication | 310,670,964 |
| Unique documents / stored tokens | 584,851 / 310,669,141 |
| Duplicate rows removed | 43 |
| Training chunks at T1024 | 131,072 |
| Training CE targets | 134,086,656 |
| Training latent pairs / KL triples | 133,848,048 / 133,478,846 |
| Development / confirmation main panels | 1,024 chunks each |
| Dedicated source panels | Nine per held-out split, 64 chunks each |
| Actual selected unique documents: train / dev / confirmation | 238,861 / 2,511 / 2,282 |

The selected-document counts are unions within each split, including the source
panels. Main and source panels share 376 chunks per held-out split; each split
therefore has 1,224 distinct chunks, not 1,600 independent observations.
All cross-split and readiness-exclusion intersections are zero. Of 12,512
declared readiness identities, 2,119 occur in the candidate pool and none enter
the selected panels.

Packing preserves actual document identities. CE crosses document boundaries
within a row, while latent pairs and KL triples respect those boundaries.
The training panel has 238,608 cross-document CE targets, zero padding and no
cross-row prediction target. RT, attention and FBT remain continuous within a
row and reset between rows. See the [protocol](protocol.md) for exact semantics.

## Validation and recovery

| Check | Result |
| --- | --- |
| Integrated new-code CPU suite | 161 distinct tests passed in 57.64 s |
| Raw/token audit | All metadata accounted for; 592 complete documents independently retokenized, 16 per source |
| Ordered audit on recovered files | All 134,272 selected row entries counted; 336 literal chunks independently reconstructed, 16 per panel |
| CPU rank and reader state | Uneven two-rank allocation, all loss masks, committed cursor restoration and finite exhaustion passed |
| Raw GCS recovery | 111 objects / 2,086,379,363 bytes recovered at exact generations |
| Partial tokenization recovery | Four shards restored, next four regenerated in a fresh process; all 25 files and the complete eight-shard summary equal the original |
| Complete corpus recovery | Remaining 29 shards restored; complete semantic verification and corpus-manifest equality passed |
| Ordered-index recovery | All 44 objects / 792,408,198 bytes recovered; independent ordered audit then passed |

The 134,272 count is panel row entries and includes the documented within-split
overlap. The independent payload checks are bounded samples; this is not a
claim that every document was independently retokenized. See the
[test ledger](test-ledger.md) for scopes and retained early failures.

Large indexes required a new streaming retention helper: the catalog is
593,391,616 bytes and the train index 188,342,272 bytes. Historical evidence
limits and helpers remain unchanged. Actual index restore took 39.48 s; full
corpus semantic verification took 31.36 s. These are data operations, not
training-throughput measurements.

## Interpretation and next milestone

This is a bounded sample of selected Dolma object prefixes with a declared
weighted mixture. Common Crawl stratum weights use object counts as a proxy;
the result is neither a uniform Dolma sample nor the original OLMo training
stream. Exact duplicates are removed; near duplicates and unknown pretraining
exposure are not. Confirmation membership was audited without model outcomes.

Books have sparse representation under the declared mixture: training includes
only three books, and each main held-out panel has only two book chunks.
Dedicated book panels help expose source-specific behavior but remain small.
The [coverage assessment](coverage-assessment.md) distinguishes reserved from
actually represented documents and explains the limits of source-level claims.

Next, connect this ordered corpus and named development panels to the unchanged
SSD training engine, then verify a tiny two-GPU graph/evaluation/restart cycle.
After that, measure native base and combined NFR capacity at T1024, including
physical batch size, throughput, evaluation cost and memory. The detailed
[next steps](next-steps.md) retain the startup and precision qualifications.
The 134M-token panel is available capacity, not a frozen training budget.

All data and evidence are retained in GCS; see the
[storage receipt](storage-receipt.md) and [interruption handoff](progress.md).
