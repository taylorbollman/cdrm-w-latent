# Resource ledger: completed two-H100 pilots

2026-09-30. CPU-only aggregation of existing terminal reports; no new profiling
or training. **These measurements describe the correctness/adaptation pilots,
not an optimized campaign throughput benchmark.** Rates pool both H100 80GB
GPUs and count each real input token once, even when K4 processes it four times.

B is the ordinary backbone; F adds FBT fusion; NF adds NextLat to F;
NFR adds native RT at layers 0 and 15 to NF. K is the number of full model
passes: B uses K1; the other rows use K4. NextLat includes both the latent
prediction loss and its KL term; the two NF branches differ only in external
KL weight after the same saved update 32 model and Adam state.

## Measured segment costs

All rows use length 1024, BF16 mixed precision with FP32 master weights/Adam
state, fused AdamW, prepared CUDA graphs, activation checkpointing, static
replicated DDP, and Flash SDPA for ordinary attention. NFR additionally uses
the native Triton RT tiles and recomputation. This is not FA4. Common backbone:
16 layers, width 2048, 16 heads, MLP intermediate 8192, RoPE and tied 50304-token
embedding/readout. Full executor time excludes launcher-side setup and final
W&B teardown; see timing definitions below.

| Completed segment | New real inputs | Compute inputs/s | + materialization inputs/s | Full executor inputs/s | Executor min | Peak allocated / reserved GiB per GPU |
|---|---:|---:|---:|---:|---:|---:|
| B128, updates 33–128 | 50,331,648 | 73,266 | 70,935 | 30,625 | 27.39 | 35.133 / 42.430 |
| F128, updates 1–128 | 67,108,864 | 12,782 | 10,746 | 7,014 | 159.47 | 30.079 / 49.562 |
| NF KL1, updates 33–64 | 16,777,216 | 8,783 | 7,780 | 5,291 | 52.85 | 42.833 / 58.707 |
| NF KL0.1, updates 33–64 | 16,777,216 | 8,787 | 7,636 | 5,231 | 53.45 | 42.833 / 58.707 |
| NFR32, updates 1–32 | 16,777,216 | 3,746 | 3,566 | 2,457 | 113.81 | 33.352 / 59.076 |
| NFR KL1, updates 33–64 | 16,777,216 | 3,747 | 3,571 | 2,375 | 117.73 | 42.834 / 59.076 |

“Inputs” means valid **input tokens**, not documents or pass-token work. B128
is a 96-update continuation from 32; each NF row is 32 updates from 32; F128
starts at 0 and stops at the declared review boundary 128; NFR32 starts at 0.
The rates use only the new tokens in those intervals, not the inherited
cumulative counters. The NFR KL1 continuation from 32 is now terminal; its
reduced-KL partner remains **pending**, with no projected rates or paired result in this ledger.

Every optimizer update contains 512 real packed rows globally, hence 524,288
real input tokens. B uses physical batch 32/rank with 8 accumulated microbatches
per rank: 64 rows per simultaneous microbatch, without dummy rows. F/NF/NFR
use physical batch 12/rank with 22 accumulated microbatches per rank: 24 physical
rows per simultaneous microbatch, 528 physical row slots per update, 512 real
rows plus 16 empty padded slots. Their deterministic partition allocates 260
real rows to rank 0 and 252 to rank 1; the remaining 4/12 slots are dummy rows.
Padding and repeated passes contribute work but never increase the throughput
numerator. Logical global batch 512 therefore does not mean physical batch 512.

## Parameter accounting

Counts are scalar parameters, deduplicating shared tensors and counting the
tied embedding/readout once. Optimizer ownership was reconstructed from each saved parameter layout
and ownership list and equals the trainable count.

| Arm | Resident parameters | Trainable / optimizer-owned | Extra active modules |
|---|---:|---:|---|
| B | 1,185,153,024 | 1,176,764,416 | None; dormant fusion remains resident |
| F | 1,185,153,024 | 1,185,153,024 | Fusion: 8,388,608 |
| NF and NFR | 1,267,879,936 | 1,267,879,936 | Fusion: 8,388,608; predictor: 82,726,912 |

The backbone itself has 1,176,764,416 parameters. The two selected RT layers
reuse backbone parameters; switching NF to NFR adds computation and state,
not trainable tensors. K4 repeats the shared model, not four independent
parameter copies. B's dormant 8,388,608 fusion parameters are excluded from
its optimizer but are part of resident memory.

## Timing and memory scope

For each update, compute seconds are the **maximum over ranks of the sum**
`backward + optimizer_and_cursor` in
`observations[update].timing_by_rank`; `backward` includes forward/loss and
backward graph execution. Materialized seconds add `materialization_host`
before taking that maximum. Pooled rates are total new input tokens divided
by summed seconds, not an average of per-update rates and not a sum of rank
throughputs. These selected regions exclude graph preparation, evaluation,
checkpoint callbacks, and most reporting work. Concurrent background storage
can still contend with them.

The broader timed update-region rates from
`update_wall_seconds_by_rank` are B128: 68,293; F128: 10,207; NF KL1: 7,671;
NF KL0.1: 7,531; NFR32: 3,247; NFR KL1 continuation: 3,241 input tokens/s.
Their timers include per-update validation, observation/gather, intermediate report persistence and scheduled
development/probe callbacks. They stop before the final update-wall evidence
persistence, outer tracker logging and checkpoint callbacks; graph preparation
also precedes the timer. Thus these are broader update regions, not the whole
training loop.

For F128 specifically, the same 67,108,864-input numerator gives **12,782/s over
5,250.347135 compute seconds**, **10,746/s over 6,245.082280 compute-plus-
materialization seconds**, and **10,207/s over 6,574.690342 timed update-region
seconds**. The extra 329.608061 seconds in the last scope includes the host
work and scheduled dev/settling probes described above; it is not an additional
materialization estimate. The previously reported 10,207/s and this ledger's
10,746/s therefore measure different valid regions. All are distinct from
7,014/s over 9,567.969001 full-executor seconds. The source is the terminal
training report, not the curve-summary report (which has no throughput field).
The exact update-timer boundaries are visible in the frozen run's
[source snapshot](../../../.runtime/olmo-fbt-stability/native-f12-to128-01/source-snapshot/scripts/olmo_fbt_stability_engine.py).

Full executor rates instead divide new inputs by `elapsed_seconds`. This
includes stage setup, model/optimizer construction or restore, graph capture,
training, development evaluations, synchronous checkpoint observation/write/
postcheck work, and foreground worker waits including terminal retention drain.
It begins after distributed runtime initialization and ends before final
report/W&B teardown. Background transfer durations overlap execution and must
not be added again. Source/input hashing and interruption protection are
intentional costs in these short studies.

Checkpoint submissions / foreground worker wait seconds were B128: 3 / 533.8 s; F128: 17 / 1482.5 s; NF KL1: 4 / 398.1 s;
NF KL0.1: 4 / 390.2 s; NFR32: 9 / 453.1 s;
NFR KL1 continuation: 8 / 397.3 s.
F additionally ran 18 settling probes and 10 named development evaluations;
B ran 7 named evaluations, each NF continuation 3, NFR32 ran 2,
and the NFR KL1 continuation ran 3 (including its restored 32 boundary).
Different interval lengths, initialization versus resume, physical batches,
probe schedules and checkpoint cadence prevent treating whole-wall ratios as
controlled architecture speedups. Even the selected compute regions are
observational pilot timings, not a batch/allocator/kernel optimization sweep.

Memory values are maximum allocator **peak** counters across both ranks in
`memory_after_capture` and every update's `memory_by_rank`. Allocated means
live tensor allocations; reserved includes allocator pools/cache. They are
per-GPU extrema, never summed across GPUs, and can include preparation or
preceding evaluation work. They are not NVML totals, backward-only peaks, or
proof of unsampled free headroom. In particular, the resumed NF runs and fresh
NFR run have different setup/allocation histories; their peak difference does
not establish an RT memory saving.

**FLOPs per update are unspecified.** This ledger has no validated estimate
covering these captured K4 paths, activation/RT recomputation,
NextLat loss work, padding and two-rank accumulation. A generic 6×parameters×tokens
estimate would hide those scope differences and is not used here.

## Exact report authorities

All report SHA256 values were verified from current immutable local files.
The formulas above and saved fields suffice to reproduce every table entry.
Convenience aggregations are `.runtime/olmo-fbt-stability/resource-ledger-01.json`
and `.runtime/olmo-nfr-kl-continuation/control-resource-ledger-01.json`.

- **B128**: [.runtime/olmo-fbt-stability/native-b32-to128-01/report.json](../../../.runtime/olmo-fbt-stability/native-b32-to128-01/report.json); SHA256 `9b8f44ce52163eca6679f78f4603fb26e023d6354273f2369749fd5b4798f7f3`.
- **F128**: [.runtime/olmo-fbt-stability/native-f12-to128-01/report.json](../../../.runtime/olmo-fbt-stability/native-f12-to128-01/report.json); SHA256 `8a07a7fc2a5ecafc4523a1f5adb6a9b2073ebd1f46a586c9034e3815977cbdfe`.
- **NF KL1**: [.runtime/olmo-kl-continuation/native-nf-control-32to64-01/report.json](../../../.runtime/olmo-kl-continuation/native-nf-control-32to64-01/report.json); SHA256 `d0b9c32c0f3465d8a9950915cf8a5582c47ee16c6df16347e8c2ed50879708b2`.
- **NF KL0.1**: [.runtime/olmo-kl-continuation/native-nf-reduced-32to64-01/report.json](../../../.runtime/olmo-kl-continuation/native-nf-reduced-32to64-01/report.json); SHA256 `bd7fbc420ad5472d3f032b006a4df3bb0c87a4bdbceab14c61bed3761bc67e72`.
- **NFR32**: [.runtime/olmo-adaptation-pilot/native-nfr12-first32-01/report.json](../../../.runtime/olmo-adaptation-pilot/native-nfr12-first32-01/report.json); SHA256 `01bceb2a1191adb513bea974d8dcb0a5b52e8b5c3b384dbd0e69d21cfea665f7`.
- **NFR KL1 continuation**: [.runtime/olmo-nfr-kl-continuation/native-nfr-control-32to64-01/report.json](../../../.runtime/olmo-nfr-kl-continuation/native-nfr-control-32to64-01/report.json); SHA256 `4e47728173364a6a3a6ed14887517cea9df8bcd2247d1b5d8ca743eb2f3b27fe`.

B/F retention authorities are in [storage-receipt.md](storage-receipt.md).
The NF continuation's independent results/qualification are in
[its report](../olmo-kl-continuation/results.md); the NFR continuation
has separate [storage authorities](../olmo-nfr-kl-continuation/storage-receipt.md).
This ledger makes no quality, numerical-equivalence, or campaign-capacity claim.
