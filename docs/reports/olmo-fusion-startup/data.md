# Fusion startup data and probe controls

2026-09-29. CPU preparation passed in 3.47 seconds. This stage reuses the
previously prepared Dolma v1_5 readiness coverage slice; it downloads and
tokenizes nothing. The slice contains 28 shards, 12,512 unique documents and
7,054,230 tokens: 12,283 training, 104 development and 125 confirmation documents.
The confirmation split remains unused.

This is a seven-source coverage fixture, **not a representative production
mixture or a quality-evaluation corpus**. Shuffling documents mixes the existing
sources without correcting their sampling or exposure proportions. The planned
128 updates are a numerical startup experiment, not a language-model result.

## Schedule and exposure

The prepared schedule uses isolated T128 rows, physical B8, document-shuffle seed
20260929 and exactly 8,192 supervised CE targets per logical update. Long
documents use stride 127, retaining one preceding context token. Document
boundaries come from shard metadata; an internal EOS remains an ordinary token.
There is no packing, fabricated window EOS or cross-document target.

When an update boundary cuts a window, both updates see that complete context
with complementary target masks. Thus targets are unique while some input
context repeats. A lookup is pure; only the trainer commits a completed-update
cursor. Cursors bind the schedule manifest and next logical update. Noise is
keyed by window occurrence, pass and logical update, independent of physical
partition and the global RNG.

| First 128 logical updates | Count |
| --- | ---: |
| Supervised CE targets | 1,048,576 |
| Presented valid input tokens, including repeated context | 1,073,565 |
| Physical microbatches | 1,219 |
| Window presentations, including split-window repeats | 9,235 |
| Padding positions | 174,691 |

The `documents` total in the preparation report denotes window presentations,
not distinct documents. Eligible latent pairs are 1,048,576 and KL triples
1,039,468; their losses are omitted from this CE-only warmup. The complete
schedule offers 846 updates from 6,934,994 available targets, with 4,562 final
targets left unused rather than cycling. Only the first 128 updates are
authorized by the initial protocol.

| Source | First-128-update CE targets |
| --- | ---: |
| Books | 268,666 |
| Common Crawl | 152,519 |
| peS2o | 146,014 |
| C4 | 140,565 |
| Reddit | 128,282 |
| Wikipedia | 108,543 |
| Stack | 103,987 |

## Fresh numerical fixture

Four distinct development documents, disjoint from training documents, provide
real prefixes of lengths `(16, 5)` and `(6, 2)`: two physical B2/T16 records,
29 valid inputs, 25 CE targets, 25 eligible latent pairs and 21 KL triples. These
match the old numerical fixture's counts but use different content. No synthetic
EOS is appended to a prefix. This tiny fixture is a second numerical observation,
not a generalization estimate.

The standalone JSON records source, document, text, full-token and slice-token
identities, masks, exact tensor hashes and deterministic noise keys. Its loader
validates those contracts and regenerates the pinned noise without loading the
training corpus. The expected whole-file SHA authenticates the preparation-time
train/development disjointness check; the loader does not independently reread
all training documents. Token-content metadata hashes use canonical JSON token
lists, while source shard hashes separately authenticate stored bytes.

## Evidence and validation

Artifacts are under `.runtime/olmo-fusion-startup/data-01/`:

| Artifact | SHA256 |
| --- | --- |
| Original prepared corpus manifest | `f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76` |
| Startup `manifest.json` | `2316978559b8db357c8d4adf706e41c94f809922171e8fb0ba50dc17ad66cbc0` |
| `fresh_fixture.json` (8,370 bytes) | `5e55ee7bab67bcffb9fb01de48b9a971bae6ecbee59b8b7ebdf189c52600918f` |
| Preparation `report.json` | `bbdb07a8463f64379460e4782bf800a3f7a05d1fa608b3329b97581a6f6d42af` |

Preparation includes the reusable script and seven pinned source snapshots. All
seven current-source and snapshot hashes matched the report on review. The data
helper and focused tests are frozen at SHA256
`882d3655bcd03e2a12c60e4e615277e716f2ea245d8d53bcdcd516b05b9b1e6d`
and `43df86917dac70375064ce476a52ff5687ee0e54d5325e838799230d27df0033`.
The verified upload receipt is
`.runtime/olmo-fusion-startup/retention/data-01.json`; it lists the archive and
manifest generations under the existing `olmo-two-gpu/20260929T075900Z/startup-data-01`
storage namespace. That namespace does not imply distributed execution here.

The focused data tests passed **14/14** in a GPU-disabled container. They check
target coverage against independent document positions, complementary masks,
padding and EOS handling, pure cursor recovery, partition-independent noise,
fresh-fixture round trips, corruption rejection, relocation, corpus mutation
during initialization and split disjointness. The final combined CPU suite passed
**39/39 in 3.84 seconds**, recorded in `cpu-final-01.log`.

Read-only review of the new precision probe found no blocking state or comparison
issue. Probes require the full trainable NF diagnostic contract even though
warmup training freezes the backbone. Startup import restores only complete
fusion weights and preserves modes, trainability and RNG; frozen backbone and
predictor must match exactly. Each BF16 case uses its own saved state's FP32
reference, with unchanged inputs and noise. The old cold anchor and corresponding
backbone's first-pass identity remain explicit checks. No optimizer is created
by the probe, and operational success is not BF16 production clearance.
