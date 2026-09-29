# Provisional Dolma pilot data plan

2026-09-29. **Proposal only: no download, tokenization, new data selection or training has run.** Storage and recovery are the current implementation milestone. This note defines a bounded next data milestone; it does not authorize the broader 500M-token campaign, choose its startup state or fix its hardware/batch schedule.

## What is available now

The verified readiness corpus is at `/mnt/localssd/cdrm-data/olmo-dolma-v1_5-readiness-20260928/tokenized`, with manifest SHA256 `f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76`. It contains 12,512 unique documents and 7,054,230 tokens in 28 shards; the prepared directory occupies 21,016,338 bytes. Its raw extracts and tokenized artifacts are already retained under `gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/readiness-20260928/`.

The sources are one complete-record prefix per family, approximately 1M tokens each, ordered books, C4, Common Crawl, peS2o, Reddit, Stack, wiki. Only one Common Crawl **tail** object was sampled. This is a coverage fixture, not a representative continuation mixture.

| Existing split | Documents | Tokens | Important limitation |
| --- | ---: | ---: | --- |
| Training | 12,283 | 6,947,277 | Source blocks dominate early prefixes |
| Development | 104 | 49,427 | No books; Stack has only 2 documents / 295 tokens |
| Confirmation | 125 | 57,526 | No books; too small for a dependable source breakdown |

All 14 books landed in training. The evaluated 3,072-token packed dev prefix contains just two C4 documents. Development has also been used repeatedly for numerical diagnostics. Keep all these artifacts for regression tests; do not rename them as representative pilot data or pristine confirmation data. These counts were checked directly against the existing document metadata.

## Pinned source and proposed size

Continue with **Dolma v1_5**, repository revision `7f48140530a023e9ea4c5cfb141160922727d4d3`. The locally saved 3,221-object URL list has SHA256 `0b660ad1cd93a840d759a4efa82f800a0580fd693de6da64dba484aa55d21efc`; see [source pins](../olmo-document-shards/source-pins.json) and the [pinned official list](https://huggingface.co/datasets/allenai/dolma/blob/7f48140530a023e9ea4c5cfb141160922727d4d3/urls/v1_5.txt). Do not substitute v1_7 or interpret `v1_5-sample` as a small download.

Prepare a first immutable training stream of **134,217,728 valid input tokens**: 131,072 full T1024 chunks. This is a data-capacity proposal, not a training commitment. It accommodates 256 updates at the provisional 524,288-token effective batch and can be extended without cycling or changing its existing prefix. It does not yet cover a 500M-token campaign.

Use the native pinned tokenizer: original OLMo-1B revision `81b71efbce6f4dada57c94860301af4298bcd351`, tokenizer SHA256 `9ad33b4b39a9f83973c3f8c42a01948dd5b877a28ac9a5356956c4ff4ed0b714`, EOS 50279. Preserve complete documents, existing embedded EOS and the existing append-terminal-EOS policy; no BOS or tokenizer truncation.

The following **chosen pilot weights** normalize the rounded token quantities in [OLMo Table 3](https://arxiv.org/html/2402.00838v3#S2.T3). They provide broad source coverage close to the published corpus composition, not a reconstruction of the selected checkpoint's first 252B tokens.

| Family | Exact relative weight | Approximate share | Initial upstream objects |
| --- | ---: | ---: | ---: |
| Common Crawl | 20,060 | 75.2% | 4 head + 4 middle + 8 tail |
| Stack | 3,420 | 12.8% | 4 |
| C4 | 1,740 | 6.5% | 4 |
| Reddit | 800 | 3.0% | 4 |
| peS2o | 570 | 2.1% | 4 |
| Books | 52 | 0.2% | all 3 |
| Wiki | 37 | 0.14% | both 2 |

For Common Crawl, provisionally divide its quota in the fixed ratio **611:773:1493** among head/middle/tail, matching the pinned object's directory counts. This is an explicit proxy: object counts are **not verified token-mass weights**. Exact historical stratum proportions remain unknown. Record this uncertainty rather than inferring a historical mixture from the filenames.

Choose URLs within each stratum by ascending `SHA256("cdrm-dolma-pilot-v1-url" + NUL + URL)`, using the pinned list. Scan a complete-record prefix of about **8,388,608 candidate tokens per selected object**, yielding about 310M candidate tokens across 37 objects before whole-document overshoot. This expands object/stratum coverage substantially but still samples within bounded prefixes: it is not uniform sampling of all Dolma documents. Hash-ranking documents inside these pools does not remove that limitation.

Proposed hard preparation limits are **384Mi candidate tokens, 4GiB received compressed bytes and 8GiB retained raw bytes**, stopping at the first bound. Oversized/malformed records fail with provenance; empty-text handling remains explicit and audited. If quotas or minimum held-out coverage are unmet, publish an **incomplete** preparation report and revise the plan explicitly; do not substitute sources, silently reduce quotas or repeat examples. Remote compressed bytes and retained decoded bytes are separately counted. These are resource bounds, not predictions of actual download size.

## Membership before windows or model outcomes

1. Globally identify exact duplicate content using the existing `content_token_sha256`: normalized native uint16LE tokens with exactly one terminal EOS excluded. Deduplicate across every source/object in this new artifact. Preserve original URL, object metadata, source line, ID, raw-text hash, token hash and duplicate mapping. Reject conflicting source/document identities.
2. Exclude **all 12,512 content identities from the existing readiness corpus** from the new selected training, dev and confirmation streams. This avoids importing its known adaptation/diagnostic exposure into the new comparison. Pin the exclusion list and its source manifest. Unknown exposure during original OLMo pretraining remains unknown.
3. Use a new declared `SplitPolicy(seed=20260929, weights=((train,90),(dev,5),(confirmation,5)))` on the content identity. This assigns documents, not exact token percentages. Identical content always has the same membership, independently of source, process, rank, sampling order or later pool extension. The policy is separate from the old 98/1/1 readiness split.
4. Within each source stratum and split, rank eligible documents by a separately namespaced deterministic hash of their content identity, with a stable lexical tie-break. Freeze the selected membership and order manifest before any new training. All windows of a document remain in its assigned split.
5. Reserve at least **131,072 tokens per stratum per held-out split**, or its main-panel token quota below if larger, with at least 32 documents per stratum except books, where the provisional minimum is 8. Hold complete documents, so reserve sizes may overshoot. Shortfalls stop preparation rather than reassigning training documents to satisfy the evaluation quota.

This establishes exact-content/document disjointness for this continuation. It does **not** claim near-duplicate-family separation, benchmark decontamination or unseen content relative to pretraining. Record `near_duplicate_policy: not_run` unless a separately reviewed family/exclusion method is added. The broader plan's family-level protection remains a qualification for stronger quality claims, not an implied feature of token hashing.

## Packing, training order and evaluation membership

For each split/stratum, concatenate the selected complete documents in their frozen hash order and make nonoverlapping T1024 chunks. Preserve true document intervals, including continuations of a document across chunks. The accepted `continuous-stream-v1` semantics remain: CE includes valid within-chunk EOS transitions; latent and KL exclusions use true document boundaries; attention/RT/FBT can cross documents within a chunk; state resets between chunks. There is no overlapping context or EOS-aware offset.

Shuffle **full chunks** within each stratum using a fixed namespaced hash of their immutable chunk keys. Allocate the 131,072 training chunks to the nine strata by largest-remainder rounding of the rational weights above, with lexical tie-breaking. Interleave the per-stratum queues by deterministic weighted token deficit, again with a declared tie-break. This prevents a long initial books/source block and distributes long-document chunks through the stream. All full chunks contribute 1024 inputs and 1023 CE targets; latent/KL denominators still depend on their actual boundaries.

Freeze a mixture-weighted **1,048,576-token main panel** independently for tuning and confirmation, using the same integer quota rule. Also retain a **65,536-token panel per stratum** for source diagnostics, drawn only from that panel's held-out document pool. Declare any overlap between a split's main/source panels; never treat them as independent replications. Small book/wiki weights provide few main-panel chunks, so source panels and actual document counts matter. The full set of reserved documents, evaluated chunks and their exact CE/latent/KL counts must all be reported separately.

Choose a smaller immutable online tuning subset and its cadence after measuring evaluation cost; do not use a convenient canonical prefix with undisclosed source coverage. The current evaluator's aggregate nats/target are actual target-weighted means. A balanced source panel is a different estimand: report its per-source means, and any mixture-reweighted result with explicit weights, rather than presenting its unweighted aggregate as the main mixture loss. Confirmation membership can be inspected for integrity now; its model outcomes stay sealed until a comparison is selected.

Extension rounds append newly selected, deduplicated chunks under a new pinned pool/round manifest. They never reshuffle or replace an already frozen prefix. Carry the global exclusion/split authority forward, record unused reserve versus exposed tokens, and stop on finite-stream exhaustion. This supports later common-token and compute-matched branches without silently cycling data.

## Smallest implementation boundary after storage readiness

The frozen `prepare_document_shards` already supplies native tokenization, global exact deduplication, hash-based splits, atomic shards and recovery. Its `SplitPolicy` can express the proposed new ratios unchanged. Reuse those semantics and formats where they fit. The frozen extractor handles only atomic per-object prefixes; a new orchestration layer must pin all 37 objects and checkpoint bounded extraction segments so interrupted work is recoverable within the requested 20–30-minute interval. Do not edit historical scripts or reinterpret their manifests.

The frozen `build_packed_index` and `PackedCampaignData` authenticate **canonical shard/record order** and a fixed schema. They do not support this weighted chunk selection/order. Implement a **new versioned selection/index and reader route**, sharing proven loss-mask/count and logical-update concepts, with explicit source-manifest and ordered-chunk pins. Bind it through a new manifest/resolver version and the execution identity; do not patch an old SQLite index or monkeypatch a frozen reader. This is a deliberate new data lineage, so old training checkpoints cannot silently resume onto it.

Suggested new files are `scripts/olmo_pilot_data_plan.py` (pure plan/quotas), `scripts/olmo_pilot_data_prepare.py` (bounded acquisition/tokenization orchestration), and a separately versioned ordered-data adapter with focused tests. Exact file/API boundaries can follow implementation review. Keep experiment decisions in a pinned JSON recipe; absolute SSD paths are relocation details, not corpus identity.

Before any GPU consumption, check deterministic replanning/resume, full split intersections/exclusions, duplicate conflicts, all nine source quotas, whole-document and literal-EOS handling, chunk union/no overlap, mixture order, exact per-loss counts, finite exhaustion, and global update/rank allocation. Independently reconstruct a small stream from document metadata and compare tokens/masks/cursor. Publish generation-pinned raw/token/index manifests and verify a bounded restore at a different local path.

The selected 128Mi training tokens alone need **256MiB of uint16 payloads**; candidate/held-out tokens, raw JSON, metadata, duplicate records and temporary recovery files add to that. Stage these on SSD and publish completed artifacts to `gs://fast-chunks`; keep compact plans/receipts/docs on persistent storage. Reserve the declared preparation working set separately from the much larger model/Adam checkpoints. No data download should compete with checkpoint durability on the currently constrained boot disk.

Remaining choices before implementation are finalizing these provisional weights/stratum proxy and bounds, the exact online tuning panel/cadence, whether stronger near-duplicate protection is required now, and the storage working-set allocation. Model startup, pilot training exposure, hardware, physical batch/accumulation and the later 500M/SFT decision remain separate.
