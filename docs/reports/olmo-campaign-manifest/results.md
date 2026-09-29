# CPU campaign manifest draft: actual readiness resolution

2026-09-29. The new resolver successfully validated a pinned, unlaunched example
for **B/N/F/R/NF/NR/FR/NFR** against the actual retained OLMo-1B artifacts,
tokenized readiness corpus, T1024 train index and parameter-ownership ledger.
Resolution took **4.44 seconds** in the GPU-disabled project container.
No model, optimizer or scheduler was constructed; no training, evaluation,
GPU execution, download, cloud publication or W&B run occurred.

The status is `cpu_plan_validated_not_authorized`, with both
`launch_authorized=false` and `numerical_clearance=false`. This example does
not select a production configuration, certify the declared batches fit, or
clear the BF16 qualifications. It declares original pretrained weights with
fresh Adam; adapted startup is explicitly unsupported.

The example describes three logical updates of 16,384 valid inputs each,
shared across the eight arms. At the declared two ranks and B8 per rank, each
update has one physical microbatch per rank. These are plan rows, not updates
that were executed. The shared exposure is **49,152 valid inputs in 48 chunks**,
with no overshoot, dummy rows or padding. The source cursor would move from
chunk 0/update 0 to chunk 48/update 3. The real index contains 6,947,277 valid
tokens in 6,785 chunks; no cycling or corpus extension is needed for this draft.

| Objective positions, summed over the three updates | Supervised | Dense prepared capacity per pass |
| --- | ---: | ---: |
| CE | 49,104 | 49,104 |
| NextLat latent pairs, when enabled | 49,103 | 49,104 |
| NextLat KL triples, when enabled | 49,054 | 49,056 |

The single internal document boundary remains a CE target and excludes one
latent pair and two KL triples. FBT arms execute four passes, so loss-matrix
capacity and pass-token work are multiplied by four. All active weights are
declared trainable; this is not fusion-only startup accounting.

| Arms | Trainable parameters | Registered resident parameters |
| --- | ---: | ---: |
| B / R | 1,176,764,416 | 1,185,153,024 |
| N / NR | 1,259,491,328 | 1,267,879,936 |
| F / FR | 1,185,153,024 | 1,185,153,024 |
| NF / NFR | 1,267,879,936 | 1,267,879,936 |

These counts reuse the retained actual CPU module inventory, cross-checked
against architecture formulas. Tied embedding/readout weights count once.
The 8,388,608-parameter fusion wrapper remains resident but frozen in arms
without FBT. NextLat adds 82,726,912 training-only parameters. RT adds no
parameters to this architecture. The readable card records matrix-work bounds
for every arm; they exclude pointwise work, communication, optimizer and graph
setup, and do not estimate speed, peak VRAM or measured hardware FLOPs.

The canonical prefix happens to span just the first **two books documents**,
with one completed document and a prefix of the next. This is useful for
verifying order and boundary accounting, but is not a representative production
data mixture. Evaluation is explicitly deferred. Production data order/mixture,
startup, numerical policy, topology/physical batches and training/evaluation
budgets still need review before the later common training entrypoint.

Validation comprised **27 focused CPU tests** (3.78 seconds), independent code
reviews by both the data and runner reviewers, and **51 post-resolution checks**.
The latter independently checked counts against the SQLite document intervals,
all arm allocations/parameter groups, component sums, non-authorization fields,
manifest identities and all **52 live/source-snapshot file pins**. Runtime checks
also confirmed unchanged CPU RNG, unchanged sources/model manifest and no CUDA
initialization. No frozen core or previous experiment source was changed.

Evidence, relative to the repository root:

- `.runtime/olmo-campaign-manifest/cpu-tests-02.log`
- `.runtime/olmo-campaign-manifest/actual-01/manifest.json`
- `.runtime/olmo-campaign-manifest/actual-01/resolved/resolved.json`
- `.runtime/olmo-campaign-manifest/actual-01/resolved/plan-card.md`
- `.runtime/olmo-campaign-manifest/actual-01/resolved/source-snapshot/`
- `.runtime/olmo-campaign-manifest/actual-01/execution.json`
- `.runtime/olmo-campaign-manifest/actual-01/independent-audit.json`

Exact manifest SHA256:
`458d449eb72e2daea3682999b7e094b3d79bb0564cd9019249e911cc05069ca1`.
Resolved JSON SHA256:
`52f8d376ef99b15b5a61de0aea2acddf76f1a2cc71410dc5f54e54daecbe9238`.
Independent audit SHA256:
`1493b666846f8e914b3486296bd436c6abf215c302f19e1f9127b67881f8c044`.

See [protocol](protocol.md) and [usage](usage.md). The remaining integration
step is a separately reviewed all-arm training entrypoint that composes the
existing data, loss, graph, logging and recovery primitives. This resolver is
the inspectable configuration boundary for that future work, not the launcher.
