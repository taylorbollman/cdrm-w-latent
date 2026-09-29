# Ordered pilot execution readiness

The representative ordered Dolma stream now runs through the shared two-GPU
training engine, with named development evaluation and exact completed-boundary
recovery in the tiny integration fixture. Native T1024 B32 and NFR12 capacity
runs are complete; the larger base batch is the final pending check. This
milestone does not start a matched learning comparison.

## What changed

New declaration, resolution and execution identities bind the corpus, ordered
train/dev panels, acquisition/source authorities, finite token schedule, model
recipe and runtime sources. New reader/evaluation adapters connect those
identities to the existing SSD engine. Narrow storage and restore adapters
recognize the new identity while retaining the existing publication, byte
verification, ownership, journaling and pruning behavior.

The model, objectives, attention kernels, optimizer, CUDA-graph implementation
and earlier test files are unchanged. Native construction uses the existing
accepted constructor. The runtime was frozen at `be74dde`: 192 source pins,
canonical inventory SHA256
`4d6f9fd0d3e61e45e13236d2c2804913b9a4790275b3e71c37657df2da83a7cf`.

Packing remains continuous within each T1024 row. CE includes internal document
transitions; NextLat regression pairs and KL triples respect actual document
identity. Model state resets between rows. Named dev prefixes have independent
physical evaluation batches, common FP32/no-jitter evaluation, all trained
passes and separate loss denominators. Main/source overlap is not pooled as
independent evidence. Confirmation outcomes were not used.

## Integration and recovery

The new runtime passed **149 distinct CPU tests**. A separate revised auditor
passed **27 additional tests**. The [test ledger](test-ledger.md) records scope
and early failures without adding overlapping test collections.

The actual two-H100 tiny NFR fixture uses the native tokenizer, synthetic text,
T16, physical batch 2 per GPU and five packed rows per update. It exercises
cross-document masks, unequal rank allocation, dummy rows and accumulation.

| Actual comparison | Audit checks | Result |
| --- | ---: | --- |
| Insert named evaluation after update 2 versus uninterrupted reference | 2,063 | Exact inputs, raw gradients, updates and full boundaries on both ranks |
| Restore cloud checkpoint 2 in a new process, then run update 3 | 2,136 | Exact continuation and final model/Adam/schedule/RNG/cursor state |
| Restore checkpoint 2 and evaluate without advancing training | 1,914 | Exact repeated evaluation and preserved state; no graph preparation, optimizer update or new checkpoint |

The cloud restore downloaded and verified 20,150,587 bytes in 1.92 seconds.
Its manifest is pinned to
`02ee7dc9e2aa6975c406a557bcc51bd820f45d9c8f08d84bef457be851b193f1`.
The fixture's upstream acquisition metadata is simulated; it is not evidence
of real Dolma acquisition. PR49 supplies the real data provenance.

The original JSON auditor incorrectly required elapsed evaluation time to match
between processes and did not handle an empty terminal-update map. Both failed
audits are retained. The separate v2 auditor excludes only elapsed time from
cross-process equality, still requires finite nonnegative timing, and explicitly
handles validated zero-update segments. Meaningful state/evaluation fields
remain exact. No GPU evidence or frozen runtime source was changed to pass.

These are exact same-lineage integration checks, not BF16-versus-FP32 equivalence
or a new native-model recovery claim. Earlier native recovery evidence remains
separately scoped to its own tested layouts.

## Native capacity

All candidates use two H100 80GB GPUs,
T1024, BF16 mixed precision, prepared CUDA graphs, activation checkpointing,
fused AdamW and the unchanged accepted attention backends. Ordinary attention
uses Flash SDPA; RT uses native Triton tiles with recomputation. NFR uses K4
FBT, RT in layers 0 and 15 on every pass, and NextLat regression plus KL.

Each candidate has eight updates; timing uses updates 4–8. Checkpoints 0, 4
and 8 are verified in GCS before keep-two local retention. Final evaluation
uses 5,120 dev-main inputs and 2,048 books inputs at FP32 batch 1 per GPU.
These are functionality/cost prefixes, not representative quality evaluation.

| Configuration | Physical batch/GPU | Global inputs/update | Compute-region inputs/s | Including recorded materialization | Maximum sampled reserved/GPU | Minimum sampled free/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary base B32 | 32 | 65,536 | 69,494 | 67,279 | 42.40 GiB | 35.21 GiB |
| Combined NFR12 | 12 | 24,576 | 3,759 | 3,578 | 59.06 GiB | 12.78 GiB |

Rates use the sum of inputs divided by the sum of each update's slowest-rank
recorded regions. The compute region includes captured forward/loss/backward,
DDP, optimizer work and cursor coordination. The second rate adds host data
materialization. Neither includes all health checks, logging, coordination,
checkpointing or evaluation; neither is complete training throughput. Memory
is sampled during capture and updates. Peak counters are cumulative; no
post-evaluation memory sample or continuous free-memory minimum is inferred.

B32 preparation took 32.26 seconds and its final two-panel FP32 evaluation
2.07 seconds. NFR12 took 442.81 seconds for preparation and 30.47 seconds for
evaluation. Both finished eight finite updates and passed declared preparation
and evaluation preservation checks. No smaller NFR fallback was needed.

Checkpoint-heavy stage elapsed times were 17.49 minutes for B32 and 26.89 minutes
for NFR12. Those short-stage averages are not production throughput estimates.
Full-state checkpointing is a substantial synchronous cost; see
[checkpoint-cost.md](checkpoint-cost.md) before choosing pilot cadence.

The base starts from original OLMo weights with fresh Adam. NFR imports only
the accepted fusion128 weights and uses fresh all-active Adam. That adaptation
previously saw 1,073,565 inputs and 1,048,576 CE targets; it is separate from
new capacity exposure. Different effective batches and startup histories make
these capacity runs unsuitable for a learning-quality comparison.

Eight combined updates are finite, with substantial clipping throughout: raw
gradient norm falls from 222.53 to 24.14. B32's norms range from 1.01 to
1.48, also exceeding the configured limit of 1.0. These observations neither
establish long-run stability nor identify a new precision defect. No additional
per-loss gradient attribution was performed in this capacity milestone.

The small common-FP32 dev-main prefix has combined per-pass CE of approximately
2.993, 7.654, 7.652 and 7.631 nats/target after update 8. Books gives 3.480,
7.411, 7.323 and 7.337. Later passes are currently worse than the first pass;
successful execution is not successful refinement. The base dev-main/books CE
is 2.424/2.518 after its different exposure and original startup. Do not treat
these tiny prefixes and unmatched runs as an architectural comparison. The
next matched adaptation pilot should monitor per-pass CE and clipping explicitly.

## Parameters and analytic work

| Configuration | Active training parameters | Registered resident parameters | Deployable inference parameters |
| --- | ---: | ---: | ---: |
| Ordinary B | 1,176,764,416 | 1,185,153,024 | 1,176,764,416 |
| NFR | 1,267,879,936 | 1,267,879,936 | 1,185,153,024 |

The common base wrapper registers 8,388,608 dormant fusion parameters; they are
not executed or optimizer-owned in B. NFR adds that fusion plus an 82,726,912
parameter NextLat predictor, which is training-only. RT adds no parameters.

The existing resource cards estimate **0.599–0.656 quadrillion matrix FLOPs**
per B32 update and **1.123–1.198 quadrillion** per NFR12 update. These updates
have different input counts. The estimates include declared checkpoint and
attention reconstruction bounds; they exclude pointwise operations, optimizer,
communication and other overhead. They are arithmetic accounting, not measured
hardware FLOPs or utilization. Exact component ledgers accompany the capacity
summary.

## Next decision

Use completed physical-capacity measurements to prepare a costed, matched short
pilot declaration: startup ancestry, arms, common effective batch, finite token
ceiling, initial stop point and fixed dev monitoring. The proposal and remaining
qualifications are in [next-steps.md](next-steps.md) and
[readiness-map.md](readiness-map.md). Existing BF16 trajectory qualifications,
limited books coverage and untested H200/cross-topology behavior remain visible.
