# OLMo chunk boundaries and Dolma source audit

Audited 2026-09-28. This is a source/metadata audit for bounded CPU data
preparation, not qualification of packed RT/FBT/NextLat training.

## The cited OLMo shift is not an EOS-aware packing offset

The user-cited revision resolves to
`ae84d479fa5775b1935b50b2120e0b514313ce18`.
[Trainer lines 608–641](https://github.com/allenai/OLMo/blob/ae84d479fa5775b1935b50b2120e0b514313ce18/olmo/train.py#L608-L641)
clone the input IDs, apply any explicit validity masks, select labels from
positions 1 onward, and pair them with logits excluding the last position.
There is no EOS-dependent offset or EOS-specific loss exclusion in that code.
Thus the quoted loss behavior is correct, but it does not make chunks end in
EOS followed by one token more often.

[Memmap lines 151–165](https://github.com/allenai/OLMo/blob/ae84d479fa5775b1935b50b2120e0b514313ce18/olmo/data/memmap_dataset.py#L151-L165)
read at `index * chunk_size` and count complete chunks using integer division.
These are nonoverlapping, fixed-stride chunks; incomplete shard tails are
discarded. The loader does not inspect EOS to choose chunk starts.
[OLMo §3.3](https://arxiv.org/html/2402.00838v3#S3.SS3) describes appending EOS to
each document, concatenating documents and cutting consecutive 2,048-token
instances. This supports the interpretation of the cited code, without claiming
that this later code revision is the exact historical training executable.

For a four-token example `[a, b, EOS, c]`, the supervised pairs are
`a→b`, `b→EOS`, and `EOS→c`. The final `c` has no target beyond this chunk.
If EOS is the final input token instead, the model is trained to predict EOS
from the preceding token, but this chunk does not train EOS→the next document.
At our proposed context length 1,024, a complete input chunk therefore has 1,023
CE targets under this policy. Input-token and supervised-target counters remain
distinct. This audit concerns alignment, not a claim that every historical
optimizer/loss normalization detail is reproduced.

## Source pins and access

Pinned dataset repository: `allenai/dolma`, revision
`7f48140530a023e9ea4c5cfb141160922727d4d3`.
The [pinned dataset card](https://huggingface.co/datasets/allenai/dolma/blob/7f48140530a023e9ea4c5cfb141160922727d4d3/README.md)
identifies `v1_5` as the OLMo-1B release. `v1_5-sample` is an approximately
1.9-trillion-token selection for OLMo-7B, not a small exploratory download.
The currently displayed statistics concern v1.6 and v1.7; neither should be
silently used as an exact v1.5 source-mixture specification.

[The pinned v1.5 URL list](https://huggingface.co/datasets/allenai/dolma/blob/7f48140530a023e9ea4c5cfb141160922727d4d3/urls/v1_5.txt)
contains 3,221 objects under `https://olmo-data.org/dolma-v1_5r1/`:

| Broad source | URL directories | Number of objects |
| --- | --- | ---: |
| Common Crawl | `cc_en_head`, `cc_en_middle`, `cc_en_tail` | 611 + 773 + 1,493 |
| The Stack | `stack` | 149 |
| C4 | `c4` | 86 |
| Reddit | `reddit` | 78 |
| peS2o | `pes2o` | 26 |
| Project Gutenberg | `books` | 3 |
| Wikipedia/Wikibooks | `wiki` | 2 |

These are object counts, not document/token mixture weights. v1.7 has a
different source inventory, including RefinedWeb, additional mathematical/code
material and instruction data. It is not an interchangeable source list.

Metadata snapshots are under
`.runtime/olmo-document-shards/source-metadata/`. The compact, committed
[source-pins.json](source-pins.json) records URL-list SHA256 hashes, selected
URLs, response sizes/ETags and the audited OLMo code hashes.

Initial Python-default-User-Agent HEAD requests returned HTTP 403. Sending
`User-Agent: curl/8.0` succeeded for all seven selected official URLs. HEAD
with `Range: bytes=0-99` returned HTTP 206 and full object sizes in
`Content-Range`; one bounded GET check ignored Range and returned HTTP 200.
The downloader must limit bytes/records itself and close the response when done.
No authentication bypass or alternate corpus is required. Multipart ETags are
not SHA256 hashes, and this audit has not downloaded or verified entire objects.

## Bounded coverage slice

Use approximately one million native-tokenizer tokens per broad source, ending
only after a complete document. Select one URL per source by minimum
`SHA256(seed + NUL + URL)`, with seed
`cdrm-dolma-v1_5-coverage-20260928`, then retain a bounded prefix of complete
records. This is deterministic source coverage for plumbing, not a representative
production sample. One Common Crawl tail shard does not cover all of Common
Crawl's quality/source strata.

| Source | Selected object basename | Full remote bytes |
| --- | --- | ---: |
| Books | `books-0001.json.gz` | 3,230,166,910 |
| C4 | `c4-0021.json.gz` | 4,047,664,975 |
| Common Crawl | `cc_en_tail-1458.json.gz` | 966,791,278 |
| peS2o | `pes2o_v2-0025.json.gz` | 477,949,183 |
| Reddit | `reddit-v5-dedupe-pii-nsfw-toxic-0044.json.gz` | 2,127,043,706 |
| The Stack | `stack-v4-train-0021.json.gz` | 1,733,168,813 |
| Wiki | `en_simple_wiki_v0-0001.json.gz` | 2,204,313,187 |

Do not download those complete objects for this task. Stream and preserve a
complete-record fragment, pin its own exact hash, preserve source URL/ETag and
record ordinals, and explicitly distinguish that verified fragment from an
unverified full remote object. Retain raw fragments, token arrays and manifests
in GCS as each source completes. This slice should require far less space than
even one full remote object; a boot-disk expansion is not a prerequisite.

Use the project's pinned tokenizer from `cdrm/pretrained/olmo_artifacts.py`:
OLMo-1B revision `81b71efbce6f4dada57c94860301af4298bcd351`, tokenizer SHA256
`9ad33b4b39a9f83973c3f8c42a01948dd5b877a28ac9a5356956c4ff4ed0b714`, vocabulary
50,280 with EOS 50,279. Append one document delimiter explicitly; do not add
BOS, truncate documents, or discard their boundaries. Establish train/dev/test
membership before any future concatenation, with exact-content duplicates in
the same split. Original-checkpoint exposure remains unknown.

## Production mixture remains a decision point

The OLMo paper's [Table 3](https://arxiv.org/html/2402.00838v3#S2.T3) gives a
rough original-release composition. Normalizing its rounded token counts gives
approximately 75.2% Common Crawl, 12.8% Stack, 6.5% C4, 3.0% Reddit, 2.1% peS2o,
0.2% books and 0.14% wiki. These are a reasonable provisional continuation
mixture, but are not a verified reconstruction of the selected checkpoint's
first 252B tokens. The current v1.6/v1.7 card and an older card predating Reddit
do not resolve exact v1.5 token proportions. Freeze an explicitly chosen
mixture, source-stratum sampling and deterministic document order before scaling
to the campaign; do not label equal-source coverage as the campaign mixture.

Tokenization is reusable now. Concatenating into production model inputs must
wait for the agreed CE/NextLat/FBT/RT boundary policy and packed-model checks.
That future change should not reinterpret already saved single-document runs.
