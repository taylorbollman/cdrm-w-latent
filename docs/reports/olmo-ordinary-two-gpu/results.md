# Ordinary OLMo throughput on two H100s

2026-09-25. **Draft:** the B32 follow-up, matched single-GPU reference and B64
repeat are pending. Completed measurements below support a provisional default
of **DDP B64 per GPU, global batch 128**, using Dao RoPE. Finalize the marked
rows and retention inventory before publishing.

Ordinary OLMo reaches **85,152 input tokens/s** at B64 per GPU. Increasing the
physical batch to B128 or B192 adds only **0.8% or 1.3% throughput**, while
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
| Dao | 64 / 128 | **85,151.90** | **0.76964** | **23.487** | 30.606 | 53.039 | 32.725 | **44.906** |
| Dao | 128 / 256 | 85,804.03 | 1.52757 | 23.309 | 38.658 | 76.734 | 41.703 | 35.926 |
| Dao | 192 / 384 | 86,282.50 | 2.27865 | 23.180 | 46.708 | 76.666 | 51.803 | 25.824 |

Memory columns use the larger reservation/allocated peak or smaller sampled
free value across ranks. Dao improves the matched B64 native control by **8.9%**
in this pair. Its current reservation is about 0.69 GiB lower; the main measured
benefit is speed. The B64/B128/B192 curve is nearly flat. Small differences need
the pending repeat before being interpreted as durable throughput gains.

Setup reservation is not live activation size. The B128/B192 runs transiently
reserve about 76.7 GiB during warmup, then release unused cache before capture.
Conversely, graph replay reuses its pool, so the roughly 22 GiB shown as
allocated during replay does not represent its entire footprint. Consider
setup allocated and reserved peaks, steady reservation and sampled device free
memory together. Free memory is sampled at phase boundaries, not continuously;
these values do not establish a maximum supported batch.

### Follow-ups pending final reports

| Follow-up | Purpose | Status | Aggregate tokens/s |
| --- | --- | --- | ---: |
| Dao DDP B32/GPU, global B64 | Check whether a still smaller physical batch remains near the plateau | Pending | — |
| Dao single GPU B128 | Match the global batch and examples of DDP B64/GPU for scaling | Pending | — |
| Fresh-process Dao DDP B64/GPU repeat | Check the recommended point's repeatability | Pending | — |

<!-- CLOSEOUT: fill these rows, update the default if B32 changes the tradeoff,
record matched scaling and repeat variation, then refresh final receipts/counts. -->

The recommendation is provisional until these finish. **Skip B256 and ZeRO-1
for this sweep:** the observed plateau and existing DDP headroom answer the
question of a useful batch without maximizing capacity. Larger global optimizer
batches can still be chosen for a learning experiment, but that is a separate
decision; batch-size changes are not learning-equivalent merely because their
tokens/sec are similar.

## Correctness and what the timings include

The actual Dao B1/GPU graph check passes initial exact raw eager/graph loss and
gradient parity, exact replicas, and **two changed-input complete Adam update
comparisons**. Each completed capacity row also passes initial and changed-weight
terminal raw parity plus initial/final replica checks. No tolerance was relaxed,
and all completed runs in this ordinary-only sweep pass so far. These are own
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
`1a98230`, with the same runtime implementation. Per-stage source snapshots and
SHA256 pins, dependency records, configuration and W&B links establish what ran.
The first control predates the explicit RoPE flag; its frozen constructor and
arm mapping establish native RoPE. Subsequent rows record the option directly.

W&B completed rows:
[native B64](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ny0jx5v5),
[Dao correctness B1](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/y5noic5g),
[Dao B64](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ay5416z1),
[Dao B128](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/74bfhe6g),
[Dao B192](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/868y5jy9).

The prospective [protocol](protocol.md) is retained with each run. Local evidence
is under `.runtime/olmo-ordinary-two-gpu/`; stage retention receipts and original
pretrained O1 checkpoint references are preserved. Disposable short capacity
updates do not require extra full checkpoints. Evidence is progressively verified
in `gs://fast-chunks`; final receipt totals remain pending the queue closeout.

Regenerate `.runtime/olmo-ordinary-two-gpu/audit/summarize.py` after all final
reports and receipts, then run its CPU-container plot tool. It excludes running
reports, checks true ordinary execution from mode/count/resource records, and
keeps native/Dao and optimizer/GPU-count series separate. The earlier RT/combined
inventory remains unchanged.
