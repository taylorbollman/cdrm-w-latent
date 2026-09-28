# Durable data and evidence

2026-09-28. Root namespace:

`gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/readiness-20260928/`

All raw extracts and the finished tokenized corpus are published and verified.
Each stage is immutable, with `manifest.json` published last. Every object was
checked for server size, MD5, SHA256 metadata and a generation-pinned streamed
SHA256 readback. Local files were not deleted.

| Stage | Objects | Published manifest SHA256 |
| --- | ---: | --- |
| `raw-books` | 3 | `8951169f12015e425cd11349468e8883e70a468bd80623edfd2891a0ec710389` |
| `raw-c4` | 3 | `626a649e4304c8fb65c3324560d9dc8d05e210031e1a9370b3cf59bd3553422f` |
| `raw-common_crawl` | 3 | `e9883cfbee0e0a8f05176f6c806fd84e6ef98df55c618ff234710a1b69737336` |
| `raw-pes2o` | 3 | `4a9f3a6ff056b4950def15d6d5b2888a16167f5b821ce9f564f3d18cd3155473` |
| `raw-reddit` | 3 | `4cbbfec856dd6eacc1fc3d6407fde1eef9759ac1365cb22e0c16be1f62285f1a` |
| `raw-stack` | 3 | `e3adbfedf6c2573068e2545a20e7b3083f4317a48a91d21a2a96d6256d056ace` |
| `raw-wiki` | 3 | `3f2933a6b84e16a3da4d9b7fd9f0e61aed2a0c91d8d18c66ebe24a4bae3e417d` |
| `token-config` | 2 | `910f37c708f95d23daf20b02de8795cb9c01955648cc8a693a85ed1f458f49df` |
| `token-shard-000000` | 3 | `0346dceaa1a6a9d9ec7b56ad737b0d26438631b2a84525374ee20d1323490269` |
| `token-shard-000001` | 3 | `e5866dd26b0474fbef5a2ffd3feaa70b3b6e2a57110164f21af7bea0dbcee70f` |
| `tokenized` | 86 | `f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76` |

These data stages contain **115 verified objects, 60,131,300 bytes**, including
the separately retained partial preparation used for the recovery rehearsal.
The canonical final corpus is `tokenized/`; `token-config/` and the first two
`token-shard-*` stages preserve the tested partial-resume state.

Each `raw-<source>/` retains exact extracted JSONL bytes, original source-line
mapping and provenance manifest. Upstream full gzip objects were not completely
downloaded or SHA256 verified. This receipt certifies our retained fragments.

The first seven config/shard files were actually downloaded into a fresh SSD
directory and verified; new-process continuation produced all86 files exactly
matching uninterrupted preparation. See `cloud-restore.json` in the evidence.

`evidence/` stores the audit report, source snapshots, metadata, scripts, tests,
receipts and logs. Its final publication hash is recorded below after retention.
Local receipts are `.runtime/olmo-document-shards/retention/*.json`.

After VM/SSD loss, recover the canonical corpus using the immutable
`tokenized/manifest.json`, its configuration and listed shard manifests/payloads;
verify them with `verify_document_shards` before use. The source fragments are
available independently for retokenization. These files are not packed model
inputs and are not a model/optimizer checkpoint.

Final evidence: **60 verified objects**. Publication marker SHA256:
`6b8a1d5c34412f380c417ac866aea3ff1598698be94720d5a16b144b3cf36f2d`. GCS generation `1790633534742079`.

The evidence archive directory contains the pre-upload version of this receipt;
this final Git record adds the evidence marker after upload without mutating
the already published evidence. Total retained objects across all stages:175.
