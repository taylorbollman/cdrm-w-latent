# Bounded Dolma pilot data

This milestone prepares data and checks recovery and packing. It authorizes no
model training budget or quality claim. The new recipe is
[`dolma-v1_5-pilot-v1.json`](../../../configs/data/dolma-v1_5-pilot-v1.json).
Historical readers, model code, checkpoints and acceptance helpers stay frozen.

The selected Dolma v1_5 inventory has 3,221 objects. A namespaced SHA256 ordering
selects 37 objects across books, C4, Common Crawl head/middle/tail, peS2o, Reddit,
Stack and Wikipedia. Each acquisition stops after the complete document that
reaches 8 Mi tokens. Hard bounds are 10 Mi tokens, 110 MiB compressed and 221 MiB
retained raw bytes per object, with 900 seconds per extraction attempt. Global
successful-acquisition caps are 384 Mi tokens, 4 GiB compressed and 8 GiB raw.
These caps do not describe cumulative traffic across failed/retried attempts.
There is no automatic retry, source replacement, document truncation or repeated
training chunk used to meet an unavailable quota.

Source HEAD metadata pins the upstream ETag and complete object size; it does
not verify the complete upstream gzip SHA. Exact retained raw prefixes are
separately hashed and generation-pinned in GCS. Tokenized source identities name
those retained prefixes, with an explicit mapping back to the acquisition plan.
Use the pinned OLMo tokenizer, retain complete documents and their identities,
append one terminal EOS when needed and deduplicate exact content-token hashes.
Split whole documents by content hash, independently of order: 90/5/5 with seed
20260929. Exclude all 12,512 identities in the earlier readiness corpus from the
selected pilot panels. Near-duplicate filtering is not implemented, and original
pretraining exposure is unknown.

Family weights are the declared provisional weights from the preceding data
plan. Common Crawl head/middle/tail weights use object counts as an explicit
proxy, not measured token mass. This is a conditional sample of selected object
prefixes, not a reconstruction of OLMo's original stream or a uniform Dolma sample.

The train panel contains 131,072 full T1024 chunks (134,217,728 inputs). Each of
dev and confirmation has a 1,024-chunk weighted main panel and nine 64-chunk
source panels. Whole-document reserves must satisfy both token and document
minimums. Actual evaluated document counts can differ from reserve counts.
Main/source overlap is recorded and is not independent replication. Confirmation
membership is auditable; model outcomes remain unopened. Hash-order documents,
form full per-stratum chunks, hash-order chunks and interleave using exact integer
quotas. Unselected tails remain in the underlying corpus. Future rounds append
new frozen streams without reshuffling this one.

The row policy remains continuous-stream-v1: attention/RT/FBT continue across
documents within a row, CE predicts ordinary internal EOS and cross-document
targets, NextLat latent/KL supervision respects actual document boundaries, and
all model state resets between rows. No extra EOS-aware offset or cross-row CE
target is introduced. The new ordered reader has its own authenticated schema;
the old canonical-order reader cannot be silently reused for shuffled indexes.

Acquisition runs in the GPU-disabled project container. Each committed source
is uploaded and independently read back before proceeding. Tokenization runs in
at most four 8 Mi-token shards per invocation, retaining committed shards before
further preparation. Small authorities, receipts and progress live under the
persistent project; large raw/token/index files live on SSD and in
`gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/pilot-20260929-v1`.

Acceptance covers literal token reconstruction and per-loss counts, source
coverage and quotas, exact split/exclusion separation, actual panel overlap,
cursor/exhaustion/rank partitioning, independent sampled retokenization and a
fresh-process generation-pinned cloud recovery. A byte-recovery result alone is
not a semantic corpus/reader acceptance result. Model capacity and evaluation
allocation are the following milestone, with previous BF16 qualifications intact.
