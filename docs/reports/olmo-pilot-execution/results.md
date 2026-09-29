# Ordered pilot execution readiness

The representative ordered Dolma stream now runs through the shared two-GPU
training engine, with named development evaluation and exact completed-boundary
recovery in the tiny integration fixture. Native T1024 capacity measurements are
in progress; their results will be added below before closeout. This milestone
does not start a matched learning comparison.

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

Pending completed measurements. All candidates use two H100 80GB GPUs,
T1024, BF16 mixed precision, prepared CUDA graphs, activation checkpointing,
fused AdamW and the unchanged accepted attention backends. Ordinary attention
uses Flash SDPA; RT uses native Triton tiles with recomputation. NFR uses K4
FBT, RT in layers 0 and 15 on every pass, and NextLat regression plus KL.

Each candidate has eight updates; timing uses updates 4–8. Checkpoints 0, 4
and 8 are verified in GCS before keep-two local retention. Final evaluation
uses 5,120 dev-main inputs and 2,048 books inputs at FP32 batch 1 per GPU.
These are functionality/cost prefixes, not representative quality evaluation.

The base starts from original OLMo weights with fresh Adam. NFR imports only
the accepted fusion128 weights and uses fresh all-active Adam. That adaptation
previously saw 1,073,565 inputs and 1,048,576 CE targets; it is separate from
new capacity exposure. Different effective batches and startup histories make
these capacity runs unsuitable for a learning-quality comparison.

## Next decision

Use completed physical-capacity measurements to prepare a costed, matched short
pilot declaration: startup ancestry, arms, common effective batch, finite token
ceiling, initial stop point and fixed dev monitoring. The proposal and remaining
qualifications are in [next-steps.md](next-steps.md) and
[readiness-map.md](readiness-map.md). Existing BF16 trajectory qualifications,
limited books coverage and untested H200/cross-topology behavior remain visible.
