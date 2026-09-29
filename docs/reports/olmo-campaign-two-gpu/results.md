# Campaign training on two H100s

2026-09-29. **Distributed execution, full pretrained cloud restart and bounded
T1024 resource calibration are complete.** One independent
BF16 numerical qualification remains visible below. This is readiness work,
not a quality-training experiment.

Read [protocol](protocol.md), [usage](usage.md), [test ledger](test-ledger.md),
[storage receipt](storage-receipt.md), [progress](progress.md) and
[numerical localization](qualification-plan.md) and [next steps](next-steps.md).
Core runtime `79fc75b`;
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

Tiny FP32 tests cover all eight arms with one, two and three microbatches; actual pretrained checks cover
ordinary B and combined NFR. NFR is K4 FBT, native RT0/15 on every pass and both
NextLat losses. The pretrained model is OLMo-1B step 60,000/~252B source tokens,
revision `81b71efbce6f4dada57c94860301af4298bcd351`. NFR has 1,267,879,936 trainable
parameters. Checks use BF16 mixed with FP32 masters/gradients/Adam, TF32 off,
ordinary forced Flash SDPA, native Triton RT recomputation, ordinary activation
checkpointing and native reused RoPE. No FA4 or torch.compile.

| Check | Result |
| --- | --- |
| CPU regression suite | 445 tests pass; final 21 focused capacity checks also pass (overlapping scope) |
| NCCL sums, 4 bytes through 256 MiB | All five sizes exact |
| Tiny eager and graph, eight arms each | 88 gates each pass |
| Tiny max raw-gradient relative L2 vs independent canonical | 3.69222e-7 |
| Tiny max three-update parameter-update relative L2 | 5.78202e-6 |
| Pretrained ordinary versus independent canonical | All 11 gates pass |
| Pretrained NFR versus prepared local eager | All 11 operational gates pass |
| Pretrained B/NFR graph versus prepared local eager | All 22 operational gates pass |
| NFR max raw-gradient relative L2, three updates | 4.95627e-9 |
| NFR parameter-update / Adam-moment relative L2 | 1.49771e-8 / 3.49359e-9 |
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
loss path differ in pretrained NFR BF16 gradients by **3.40224% relative L2** at
initialization. Losses pass their budgets; CE is exact, and normalized objective
difference is 4.19435e-6. The original distributed comparison remains FAILED.

The exact same discrepancy appears locally before DDP. Distributed eager and
graph execution agree with the prepared local path to the tiny errors above.
A separate full-FP32 check at the actual pretrained weights passes, with
sparse/dense gradient relative L2 **7.38198e-7** and objective difference 4.83649e-7.
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

Full pretrained NFR write phase passes and writes a 15,214,756,865-byte canonical
model/Adam checkpoint after one accumulated update. It then records the next
update on the original live graph. The checkpoint was uploaded, downloaded into
a fresh directory and fully SHA256-verified. New torchrun processes restore it
before rebuilding DDP/graphs. The next raw gradients, complete Adam/model state,
loss metrics, RNG draws/states, scheduler/counters and data cursor are all
bitwise identical on both ranks to uninterrupted continuation. Write/resume
phases take 187.75 s / 129.57 s excluding cloud transfer. This is B2/T16 functionality;
same-world-size/runtime/hardware, not a T1024 capacity or real-loader cursor claim.

The previous tokenized document corpus was restored from GCS without repeating
preparation: all 86 files hash-verified, 28 shards, 12,512 documents, 7,054,230 tokens.
It remains a source-coverage fixture, not a production mixture or packing test.

## Resource calibration and next scope

T1024 NFR K4 resource checks follow full-model restart acceptance. They
use one candidate per process, separate graph pools, full-valid isolated
documents, full CE/auxiliary masks and two accumulated physical batches.
This workload differs from the [older one-GPU T1024 benchmark](../olmo-combined-t1024/results.md);
these rates do not measure two-GPU scaling of that benchmark:

| Work per input / execution choice | Earlier benchmark | Current campaign probe |
| --- | --- | --- |
| FBT passes | 2 | 4 |
| RT on passes | Ordinary bootstrap, then RT0/15 on one pass | RT0/15 on all four passes |
| Recurrent-layer executions per input | 2 | 8 |
| NextLat KL mask | Response half | All valid same-document triples |
| Ordinary SwiGLU | Compiled | Eager |
| Layout / noise | Fixed layout, no feedback noise | Prepared masks/counts and keyed 0.02 feedback jitter |
| Parallel execution | One GPU, one graph | Two GPUs, local/sync graphs, accumulation and NCCL |

The earlier physical batch was 64 on one GPU; a smaller physical batch per GPU
also changes RT utilization. Accumulation increases the logical update batch,
not RT's physical kernel batch. CE/pass weights differ too. No matched ablation
attributes a measured fraction of the speed difference to any one change.

| Physical batch per GPU | Accumulation | Global sequences/update | Global input tokens/s | Peak reserved/GPU | Final sampled free/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 2 | 32 | **3,428.23** | 50.35 GiB | 22.97 GiB |
| **12** | **2** | **48** | **4,311.01** | **59.06 GiB** | **14.23 GiB** |
| 16 | 2 | 64 | **4,969.84** | 69.97 GiB | 3.30 GiB |

The [B8 run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/kkmsgtfg)
passes all 12 stages, including three eager optimizer updates, capture with
9.45 GiB/GPU of actual Adam moments resident, replay priming without clock
advancement, and five measured graph updates. Median measured update time is
9.564 s; the entire attempt took 524.64 s. The timing includes CPU preflight,
refills, CUDA graph replay, NCCL, clipping, Adam and scheduling. It excludes
fixture construction, logging and extra post-update health scans. Tokens count
each valid input once, rather than multiplying by the four feedback passes.
These are short resource measurements, not a long-run stability assessment.

The [B16 run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/cbl8xn7l)
also passes all 12 stages; median update time 13.187 s, whole attempt 575.49 s.
Its 3.30 GiB free memory is too tight for a comfortable recommendation. A
prospective protocol amendment added one otherwise identical B12 run and skipped
B32; the original B8/B16 evidence and source snapshots remain unchanged.

The [B12 run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8yx9sar0)
passes all 12 stages in 542.40 s, with median update time 11.400 s. **Recommend
B12 per GPU as the starting point for the next integration checks**, retaining
B8 as a fallback. B12 is about 26% faster than B8 in these short measurements
and leaves 14.23 GiB sampled free. This is not a maximum-batch claim; B32 was
skipped, not observed to fail. All three probes together complete 36 stages
and 24 optimizer updates. No capacity attempt encountered OOM.

The tested logical update is only two microbatches per rank. Final training
accumulation, actual-loader timing and cold T1024 graph reconstruction with a
restored optimizer need their own bounded acceptance; see the next-step plan.

Packed multidocument rows remain rejected. Before production training, qualify
the agreed concatenation/EOS policy separately for CE, NextLat, feedback and
temporal recurrence; then integrate real-data cursors and calibrate the actual
loader/batch/schedule. Same-world-size recovery here does not qualify changing
rank count or H200 execution. No long quality run has been launched.

## Evidence closeout

All 1,120 report/source-snapshot pairs verify. The 19 completed stage receipts
cover 44 cloud objects, including failed diagnostic attempts and the retained
checkpoint objects. Full-checkpoint SHA readback is independently recorded in
the restore evidence. Both GPUs are idle with no compute processes after the
last probe. Final documentation, CPU logs, receipt inventory and audit are
retained separately under `campaign-closeout`; its exact receipt is linked in
the [storage record](storage-receipt.md).
