# Campaign training on two H100s

2026-09-29. **Distributed execution and full pretrained cloud restart pass;
T1024 resource calibration is being completed.** One independent
BF16 numerical qualification remains visible below. This is readiness work,
not a quality-training experiment.

Read [protocol](protocol.md), [usage](usage.md), [progress](progress.md) and
[numerical localization](qualification-plan.md). Core runtime `79fc75b`;
checkpoint metadata fix `0c77166`; reference isolation `61d6d2b`;
FP32 diagnostic `28d6a93`. Legacy runners/model defaults remain unchanged.

## What works

The new opt-in runner uses two CUDA graphs: local/no-sync forward/backward for
the first M-1 microbatches and synchronized DDP/NCCL forward/backward for the
last. It clears gradients once per update, normalizes each objective by its own
global target count, clips once and performs one fused Adam update. Tokens,
right-padding, masks, jitter and accumulation length can change while physical
shape and parameter participation remain fixed. A rank or final slot can be
entirely empty. Optimizer and token clocks stay outside capture.

Tiny FP32 tests cover all8arms with M1/2/3; actual pretrained checks cover
ordinary B and combined NFR. NFR is K4 FBT, native RT0/15 on every pass and both
NextLat losses. The pretrained model is OLMo-1B step60000/~252B source tokens,
revision `81b71efbce6f4dada57c94860301af4298bcd351`. NFR has1,267,879,936 trainable
parameters. Checks use BF16 mixed with FP32 masters/gradients/Adam, TF32off,
ordinary forced Flash SDPA, native Triton RT recomputation, ordinary activation
checkpointing and native reused RoPE. No FA4 or torch.compile.

| Check | Result |
| --- | --- |
| CPU regression suite | 445 tests pass; earlier scopes overlap |
| NCCL sums,4bytes through256MiB | All5 sizes exact |
| Tiny eager and graph,8arms each | 88 gates each pass |
| Tiny max raw-gradient relativeL2 vs independent canonical | 3.69222e-7 |
| Tiny max3-update parameter-update relativeL2 | 5.78202e-6 |
| Pretrained ordinary versus independent canonical | All11 gates pass |
| Pretrained NFR versus prepared local eager | All11 operational gates pass |
| Pretrained B/NFR graph versus prepared local eager | All22 operational gates pass |
| NFR max raw-gradient relativeL2,3updates | 4.95627e-9 |
| NFR parameter-update / Adam-moment relativeL2 | 1.49771e-8 /3.49359e-9 |
| Rank model/Adam/counter agreement | Exact |
| Corrected tiny GCS-restored fresh-process restart | Next update bitwise exact |
| Pretrained NFR GCS-restored fresh-process restart | Next update bitwise exact on both ranks |

Tests inspect raw gradients before clipping. Fixtures change masks and global
denominators, include a wholly empty rank, and include empty final synchronization
slots. Preparation/capture does not consume RNG, optimizer updates or token
clock. No long-run stability or learning-rate conclusion follows from these
disposable updates. Short B2/T16 checks are not capacity measurements.

## Retained numerical qualification

The independent compact selected-position reference and dense masked campaign
loss path differ in pretrained NFR BF16 gradients by **3.40224% relativeL2** at
initialization. Losses pass their budgets; CE is exact, and normalized objective
difference is4.19435e-6. The original distributed comparison remains FAILED.

The exact same discrepancy appears locally before DDP. Distributed eager and
graph execution agree with the prepared local path to the tiny errors above.
A separate full-FP32 check at the actual pretrained weights passes, with
sparse/dense gradient relativeL2 **7.38198e-7** and objective difference4.83649e-7.
It uses math SDPA/eager native RT rather than BF16 Flash/Triton.

This supports a BF16 arithmetic explanation rather than a distributed reduction
bug in this fixture. It neither proves harmlessness for learning nor identifies
a single offending operation. No numerical threshold was relaxed and no model
math/QK normalization changed. Prior native/author and BF16 qualifications also
remain. Prepared operational passes are reported separately from independent
numerical acceptance; the latter is still failed for NFR BF16.

## Interruption recovery

The first tiny restart caught and fixed a serialization bug: TorchVersion is a
string subclass, and metadata normalization retained its Python class, which
the safe loader rejected. Metadata now canonicalizes allowed scalar and string
key subclasses to builtins; `weights_only=True` is retained. The failed attempt
and checkpoint are preserved. The corrected tiny checkpoint was uploaded,
downloaded to a fresh directory, hash-verified and resumed in new processes.

Full pretrained NFR write phase passes and writes a15,214,756,865byte canonical
model/Adam checkpoint after one accumulated update. It then records the next
update on the original live graph. The checkpoint was uploaded, downloaded into
a fresh directory and fully SHA256-verified. New torchrun processes restore it
before rebuilding DDP/graphs. The next raw gradients, complete Adam/model state,
loss metrics, RNG draws/states, scheduler/counters and data cursor are all
bitwise identical on both ranks to uninterrupted continuation. Write/resume
phases take187.75s/129.57s excluding cloud transfer. This is B2/T16 functionality;
same-world-size/runtime/hardware, not a T1024 capacity or real-loader cursor claim.

The previous tokenized document corpus was restored from GCS without repeating
preparation: all86 files hash-verified,28shards,12,512documents,7,054,230tokens.
It remains a source-coverage fixture, not a production mixture or packing test.

## Resource calibration and next scope

T1024 NFR K4 resource checks follow full-model restart acceptance. They
will use one candidate per process, separate graph pools, full-valid isolated
documents, full CE/auxiliary masks and two accumulated physical batches.
This workload differs from older K2/partial-KL benchmarks; do not attribute its
throughput difference solely to two-GPU scaling.

Packed multidocument rows remain rejected. Before production training, qualify
the agreed concatenation/EOS policy separately for CE, NextLat, feedback and
temporal recurrence; then integrate real-data cursors and calibrate the actual
loader/batch/schedule. Same-world-size recovery here does not qualify changing
rank count or H200 execution. No long quality run has been launched.
