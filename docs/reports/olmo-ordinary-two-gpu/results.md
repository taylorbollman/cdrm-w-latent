# Ordinary OLMo throughput on two H100s

2026-09-25. All eight stages and their evidence retention are complete. The
recommended ordinary baseline is **DDP B64 per GPU, global batch 128**, using
Dao RoPE: approximately **85,200 input tokens/s** on two H100s, with **44.9 GiB
sampled free memory per GPU**. The matched single-GPU reference reaches 44,041
tokens/s, giving **1.93× two-GPU scaling**.

[Throughput plot](ordinary-throughput.pdf), [memory plot](ordinary-memory.pdf),
[measurement CSV](ordinary-performance.csv) and [storage receipt](storage-receipt.md).

The two fresh-process B64 measurements are **85,152 and 85,274 input tokens/s**,
a 0.14% spread. Increasing the physical batch to B128 or B192 adds only about
**0.7% or 1.3% throughput** relative to the pooled B64 rate, while
reducing sampled free memory from **44.9 GiB to 35.9 or 25.8 GiB per GPU**.
This is a useful throughput plateau, so there is no reason to seek an OOM or
test B256 merely to fill memory. ZeRO-1 is also deferred: DDP already leaves
substantial room at the recommended operating point.

## Model and execution

These are the **ordinary base model**, with no RT, FBT feedback or NextLat
execution. The checkpoint is the original OLMo-1B step 60,000, approximately
252B pretraining tokens: 16 layers, width 2,048, 16 heads of width 128,
SwiGLU width 8,192 per branch, tied 50,304-token embedding/readout, native
normalization and no added Q/K normalization.

The active backbone has **1,176,764,416 parameters**. The shared harness retains
a frozen, unused 8,388,608-parameter fusion module, occupying **32 MiB** in FP32.
It is not executed, does not receive gradients or Adam state, and is reported
separately: total resident registered parameters are 1,185,153,024. The NextLat
predictor is absent. Reported mode and resource cards confirm zero RT layers,
16 ordinary block calls, one executed pass and zero auxiliary-loss targets;
the fixture's unused `case.passes=2` template field does not mean two passes run.

All rows use T512 and full valid next-token CE: 511 target positions per
512-token sequence, position chunks of 2,048. BF16 mixed execution retains FP32
parameters, gradients, residuals, normalization and Adam state. TF32 and
autocast weight caching are off. All ordinary layers use activation checkpointing.
The selected path combines:

- **Dao native-FP32 RoPE**, with an explicitly labeled native-RoPE control.
- **Rounded compiled SwiGLU**, preserving the accepted intermediate rounding.
- **Fused AdamW** and deterministic **PyTorch Flash SDPA**; FA4 is not enabled.
- **CUDA graphs** capturing actual forward/loss/backward and DDP/NCCL reduction.
  Clipping, Adam, scheduler, and coordinated health/metric checks remain outside.

DDP keeps a model/gradient/optimizer replica on each GPU. Gradient bucket views
are off. The graph accepts new token values and changed weights while fixing
batch shape, masks, document layout, objective counts and parameter participation.
See [execution-options.md](execution-options.md) for bucket views and changing
graph layouts; neither option is needed or newly adopted for this fixed baseline.

## Completed measurements

Hardware and software match the preceding milestone: two H100 80GB HBM3 GPUs
with NV18 connectivity; PyTorch `2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3,
NCCL 2.30.5. Each stage uses a fresh process. Capacity rows have three Adam
preparation updates, 11 DDP warmup backwards before capture, and five timed
complete updates. There is one physical microbatch per rank and no accumulation.

| Ordinary RoPE | Local / global batch | Aggregate tokens/s | Seconds/update | GPU-microseconds/input token | Setup peak allocated GiB/GPU | Setup peak reserved GiB/GPU | Steady reserved GiB/GPU | Sampled free GiB/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native control | 64 / 128 | 78,183.60 | 0.83823 | 25.581 | 30.591 | 53.318 | 33.412 | 44.217 |
| Dao | 32 / 64 | 80,279.48 | 0.40817 | 24.913 | 26.583 | 40.951 | 28.498 | 49.133 |
| Dao | 64 / 128 | **85,151.90** | **0.76964** | **23.487** | 30.606 | 53.039 | 32.725 | **44.906** |
| Dao, fresh-process repeat | 64 / 128 | **85,274.14** | **0.76853** | **23.454** | 30.606 | 53.039 | 32.725 | **44.906** |
| Dao | 128 / 256 | 85,804.03 | 1.52757 | 23.309 | 38.658 | 76.734 | 41.703 | 35.926 |
| Dao | 192 / 384 | 86,282.50 | 2.27865 | 23.180 | 46.708 | 76.666 | 51.803 | 25.824 |

Memory columns use the larger reservation/allocated peak or smaller sampled
free value across ranks. Dao improves the matched B64 native control by **8.9%**
in this pair. Its current reservation is about 0.69 GiB lower; the main measured
benefit is speed. Both B64 runs have identical reported memory figures. Their
122.24 tokens/s range is 0.1435% of their mean; their pooled rate, using total
tokens divided by total time, is **85,212.98 tokens/s**. The B64/B128/B192 curve
is nearly flat. These short repeats support a stable direction, not a statistical
guarantee of long-run throughput.

B64 is 6.1–6.2% faster than B32. B32 saves only another 4.23 GiB of sampled
headroom, which is unnecessary for the present ordinary baseline. It remains
a reasonable smaller batch if a future addition needs that space. There is no
gradient accumulation in any row.

Setup reservation is not live activation size. The B128/B192 runs transiently
reserve about 76.7 GiB during warmup, then release unused cache before capture.
Conversely, graph replay reuses its pool, so the roughly 22 GiB shown as
allocated during replay does not represent its entire footprint. Consider
setup allocated and reserved peaks, steady reservation and sampled device free
memory together. Free memory is sampled at phase boundaries, not continuously;
these values do not establish a maximum supported batch.

### Matched single- versus two-GPU scaling

Both rows use Dao RoPE, the same global batch of 128 and the same global
examples. The single-GPU fixture concatenates the two rank fixtures.

| Execution | Physical batch | Aggregate tokens/s | GPU-microseconds/input token |
| --- | ---: | ---: | ---: |
| One GPU | 128 | 44,041.29 | 22.706 |
| Two GPUs, pooled B64 repeats | 64 each | 85,212.98 | 23.471 |

The two individual DDP scaling factors are 1.9335× and 1.9362×; the pooled
factor is **1.9348×**, about 96.7% of ideal 2× scaling. Total GPU time per input
token increases only **3.3–3.4%**. The single-GPU row takes 1.48806 seconds/update,
with setup peak allocated/reserved memory of 34.271/37.518 GiB, steady reservation
37.518 GiB and sampled free memory 40.912 GiB. These are matched systems
measurements; own eager/graph checks do not assert bitwise equivalence between
different physical batch shapes.

**B256 and ZeRO-1 were skipped for this sweep:** the observed plateau and
existing DDP headroom answer the
question of a useful batch without maximizing capacity. Larger global optimizer
batches can still be chosen for a learning experiment, but that is a separate
decision; batch-size changes are not learning-equivalent merely because their
tokens/sec are similar.

ZeRO-2 also remains deferred, **not ruled out as a possible throughput
optimization**. It changes gradient communication/storage and needs its own
graph/update/recovery checks. There is no demonstrated memory or throughput
need for that additional change before using this ordinary baseline.
See [ZeRO-2 and NextLat clarification](zero2-nextlat-clarifications.md) for the
accumulation/reduction tradeoff and the active auxiliary objective outside this
ordinary-only benchmark.

## Correctness and what the timings include

The actual Dao B1/GPU graph check passes initial exact raw eager/graph loss and
gradient parity, exact replicas, and **two changed-input complete Adam update
comparisons**. Each completed capacity row also passes initial and changed-weight
terminal raw parity plus initial/final replica checks. No tolerance was relaxed,
and all eight runs in this ordinary-only sweep pass. These are own
execution checks, not a new FP32-reference or long-training precision campaign.
They do not clear older RT/native-author or other precision qualifications.

The integrated CPU scope records **93 passed, 67 warnings in 3.26 seconds**, in
`.runtime/olmo-ordinary-two-gpu/logs/integrated-cpu-tests.log`. The warnings are
installed PyTorch JIT deprecations. Earlier overlapping suites are not added to
this count, and replicated GPU assertions are not counted as independent cases.

Full-update timing includes batch validation/copy, captured compute/reduction,
global health/loss checks, gradient clipping, fused Adam and scheduler. It excludes
fixture construction, outer timing barriers, reporting, hashing and checkpoint
I/O. The examples are a bounded repeated-text fixture. **These are compute-path
benchmarks, not end-to-end data-loader throughput or model-quality results.**
Input throughput counts 512 tokens per sequence, while CE supervises 511.

At global B128, the analytic matrix-work estimate is **590.5–636.6 TFLOPs per
update**, including modeled activation-checkpoint recomputation. Larger batch
estimates scale with the number of examples. This is not measured hardware
FLOPs: norms, RoPE, other elementwise work, communication, optimizer/clipping,
launch overhead and hardware padding are excluded. Dao's optimization therefore
can improve elapsed time without changing this matrix-work estimate.

Per GPU, active gradients occupy 4,707,057,664 bytes and initialized Adam state
9,414,115,588 bytes; resident model parameters including dormant fusion occupy
4,740,612,096 bytes. These tensor inventories exclude extra communication storage,
graph pools, activations and allocator cache. The 32 MiB inactive fusion cost
does not explain the large differences between setup and steady reservation.

## Source and retention record

The native control records source HEAD `0c6ace4`; explicit ordinary Dao options
and the B1 integration check record `467bde7`. Dao capacity rows record
`1a98230`; the single-GPU and B64-repeat reports record `a69fe1b`. The source
inventories of both follow-ups match the first Dao B64 run exactly; the HEAD
change is documentation only. Per-stage source snapshots and
SHA256 pins, dependency records, configuration and W&B links establish what ran.
The first control predates the explicit RoPE flag; its frozen constructor and
arm mapping establish native RoPE. Subsequent rows record the option directly.

W&B completed rows:
[native B64](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ny0jx5v5),
[Dao correctness B1](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/y5noic5g),
[Dao B64](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ay5416z1),
[Dao B128](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/74bfhe6g),
[Dao B192](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/868y5jy9),
[Dao B32](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/hu9io4py),
[Dao single B128](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/yqfpvlue),
[Dao B64 repeat](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/qmqgesfz).

The prospective [protocol](protocol.md) is retained with each run. Local evidence
is under `.runtime/olmo-ordinary-two-gpu/`; stage retention receipts and original
pretrained O1 checkpoint references are preserved. Disposable short capacity
updates do not require extra full checkpoints. **All eight stage receipts are
verified**, under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T195200Z/`.

The final local audit finds **eight passed reports, no failed/OOM/incomplete
reports, seven throughput rows and 1,303 matching report-pinned source snapshot
pairs**. There are 55 physical distributed optimizer executions (110 rank-level
Adam calls) and eight single-GPU executions, or 118 optimizer calls overall.
These include duplicated correctness branches, not steps of one learning run.
Source, retained-archive and final-report/retention-manifest checks find no mismatch.
Remote verification is recorded in stage receipts; the local audit does not
make a new cloud request. The storage receipt provides the retained-object details.

The generated audit and CPU-only plots live in
`.runtime/olmo-ordinary-two-gpu/audit/`. They exclude nonfinal reports, check true
ordinary execution from mode/count/resource records, and keep native/Dao and
optimizer/GPU-count series separate. The earlier RT/combined inventory remains
unchanged. No quality-training run was started by this benchmark milestone.
