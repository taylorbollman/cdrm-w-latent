# Reusable Dolma document preparation

2026-09-28. **CPU preparation and recovery checks passed.** Complete documents
are now available as compact, verified token shards, independently of future
packing or model execution. No GPU run, model update or production training
corpus was created. Larger boot storage was not needed for this milestone.

Implementation `253f5ac`, audit `90eda48`. Read [usage](usage.md),
[source/OLMo audit](olmo-source-audit.md), [progress](progress.md), and
[durable receipts](storage-receipt.md).

## OLMo target shifting is not an EOS-aware packing offset

The user's cited trainer matches logits at positions0–2046 with tokens1–2047.
Internal EOS is a normal target, as is the following document's first token
when both fall inside the chunk. The final logit does not predict into the
next chunk. The corresponding loader reads consecutive fixed-size chunks;
neither operation deliberately places EOS at the penultimate position.
See the pinned [trainer](https://github.com/allenai/OLMo/blob/ae84d479fa5775b1935b50b2120e0b514313ce18/olmo/train.py#L608-L641)
and [loader](https://github.com/allenai/OLMo/blob/ae84d479fa5775b1935b50b2120e0b514313ce18/olmo/data/memmap_dataset.py#L151-L165).

No offset or overlap was introduced into the new storage format. We preserve
whole documents and their offsets. Future 1024-token training chunks will be a
versioned derivative, with explicit CE, NextLat and FBT/RT boundary policies.
The existing model adapters still reject multiple documents in one row.

## Bounded actual data

Pinned Dolma **v1_5**, HF revision
`7f48140530a023e9ea4c5cfb141160922727d4d3`, and the native tokenizer from our
OLMo-1B checkpoint. Selected one source object per broad family by a fixed hash
rule; streamed complete-record prefixes until approximately1M tokens per source.
This is a **source-coverage fixture**, not the final mixture, a random sample,
or a reconstruction of the checkpoint's original training order. In particular,
the chosen Common Crawl object covers only one tail stratum.

| Source | Unique documents | Stored tokens including EOS |
| --- | ---: | ---: |
| Books | 14 | 1,046,266 |
| C4 | 2,113 | 1,000,306 |
| Common Crawl | 2,609 | 1,000,010 |
| peS2o | 174 | 1,006,856 |
| Reddit | 4,944 | 1,000,057 |
| The Stack | 555 | 1,000,644 |
| Wiki | 2,103 | 1,000,091 |
| **Total** | **12,512** | **7,054,230** |

Input contained12,514 retained raw documents. Two exact normalized-token
duplicates (80 tokens) were audited and excluded from stored token payloads.
Extraction separately skipped725 empty/whitespace-only Common Crawl records
and14 Reddit records, recording original source lines and hashes. Nonstring
text and malformed records fail; no document was cut to fit a token budget.

Documents range from4 to**215,058 tokens**; median181. Long documents remain
complete. The format uses uint16 little-endian token arrays plus JSONL document
offsets, IDs, source lines, raw-text/token hashes and split assignments. Actual
document offsets are authoritative; literal embedded EOS remains content.
No embedded EOS occurred in this particular slice; CPU fixtures test it.

Twenty-eight atomic shards were produced. Shard targets are250k tokens/1024 raw
rows; a complete long document may cross the token target. Payload token arrays
occupy14,108,460 bytes; the complete prepared directory is21,016,338 bytes.
Raw fragments plus extraction manifests/maps occupy37,828,621 bytes. Including
recovery/reference copies and scratch, SSD use remained under100MiB. Boot disk
still has approximately12GiB free; a 1TB boot disk remains a sensible general
upgrade but is not required to hold this preparation workload.

## Split and provenance contract

Seed20260928 and explicit98/1/1 document-hash buckets produce:

| Split | Unique documents | Tokens |
| --- | ---: | ---: |
| Train | 12,283 | 6,947,277 |
| Development | 104 | 49,427 |
| Confirmation | 125 | 57,526 |

The identity is a hash of normalized content token bytes, excluding one terminal
EOS. Token-equivalent/Unicode-normalized duplicate text cannot cross splits.
Conflicting content under the same source/document ID is rejected. These are
document allocations, not guaranteed token percentages. Exact deduplication is
not fuzzy decontamination, and original-checkpoint exposure is unknown.

Each upstream URL, release pin, selected HTTP ETag, extraction bounds and source
line mapping is retained. The source object was intentionally not downloaded in
full; multipart ETags are not SHA256 checksums. We verify and retain the exact
raw fragments used, without claiming a complete-upstream-object hash.

The initial extractor stopped on an empty Common Crawl record. A new v2 attempt
added the explicit audited skip policy above and completed all sources. The
earlier development log is retained. No failed partial was fed to tokenization.

## Validation and recovery

**124 CPU tests passed** in3.80s, including existing data/ingest contracts and
new tokenization, retention and extraction cases. Coverage includes hash/config
changes, corruption, interruption, literal/terminal EOS, Unicode equivalents,
long documents, duplicate identities, publication ordering and cloud retries.

Actual-data acceptance:

1. Prepare two shards:10 documents,638,155 tokens. Publish config and both shards
   to GCS and verify size, MD5, SHA metadata and generation-pinned SHA readback.
2. Download the seven required config/shard files into a new empty directory;
   verify their bytes and recover cursor source0/line11.
3. Start a new preparation process and finish from that cursor. Separately
   prepare the complete dataset uninterrupted in another process/directory.
4. **All86 committed files match byte for byte**, including config, manifests,
   token payloads and metadata. This includes all28 shard hashes and the final
   cursor; no source rows were lost or repeated on resume.
5. Independently retokenize all12,514 retained raw records using the native
   tokenizer wrapper; compare IDs, raw-text hashes, token hashes and exact
   stored token bytes. All pass.

Final corpus manifest SHA256:
`f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76`.
[W&B data audit](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/abscmr9h)
is synced. This is a data-preparation restart check, **not** the pending
distributed model/Adam/DDP/CUDA-graph checkpoint restart.

## Implications and next work

Padding every isolated window to1024 would use only42.2% of its token slots in
this equal-source slice, including one-token window overlaps. That is a static
length calculation, not a GPU throughput measurement or a prediction for the
eventual mixture; length bucketing would change it. It reinforces the value of
qualifying an efficient packing policy rather than discarding short documents.

Reusable document preparation is complete for this slice. Next validate the
chosen concatenated-stream semantics across B/N/F/R and combinations before
production packing; retain the separate right-padding route for other workloads.
Real two-GPU accumulated updates and fresh-process training restart remain the
next hardware milestone. Final production source mixture/strata, near-duplicate
and evaluation exclusions, deterministic sample order, and the larger corpus
manifest remain to be frozen before campaign training. Do not use this coverage
fixture as the campaign dataset merely because it is ready.
