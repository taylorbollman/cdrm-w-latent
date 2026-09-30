# H100 capacity evidence for H200 allocation planning

2026-09-30. Read-only synthesis of completed reports; no new GPU measurement or
H200 performance prediction. B denotes ordinary OLMo, N NextLat training, F FBT,
and R native RT at layers 0/15. The user's prospective H200 option has 141 GB
advertised memory; all measured memory below is in GiB and is per GPU.

The strongest current evidence favors testing a larger physical batch for NFR
on H200. Ordinary OLMo has already reached a useful throughput plateau with
substantial H100 headroom in historical measurements. That is evidence against
an urgent **capacity** need, not proof that it is independent of memory
bandwidth. None of these reports establishes a compute-versus-bandwidth roofline.

## Direct current-architecture batch evidence

The [T1024 K4 calibration](../olmo-campaign-two-gpu/results.md#resource-calibration-and-next-scope)
used two H100s, RT0/15 on all four passes, full CE and auxiliary masks, eager
SwiGLU, separate local/synchronized CUDA graphs, and two accumulation slots per
rank. Rates count each original input token once.

| NFR physical batch/rank | Global real rows/update | Aggregate input tokens/s | Peak reserved GiB/rank | Sampled free GiB/rank |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 3,428.23 | 50.35 | 22.97 |
| 12 | 48 | 4,311.01 | 59.06 | 14.23 |
| 16 | 64 | 4,969.84 | 69.97 | 3.30 |

B16 was 15.3% faster than B12 but had insufficient comfortable headroom, which
motivated B12. B32 was skipped, not observed to fail. This is a concrete reason
to revisit physical batch on a larger-memory GPU. It does not establish how
much benefit survives at the campaign's fixed logical batch of 512 rows or how
far throughput continues increasing beyond B16. More accumulation does not
enlarge RT's per-invocation matrix batch.

Those short timings include CPU preflight/refill, graph forward/backward,
NCCL, clipping, Adam and scheduling; exclude fixture creation, reporting and
extra post-update health scans. They are not end-to-end campaign throughput.

## Current packed-data pilot operating points

These are observational segment measurements with T1024, BF16 mixed, FP32
masters/Adam, activation checkpointing, replicated DDP and Flash SDPA. They
are not matched physical-batch sweeps. All use 512 real rows/update. B has
physical B32/rank and 8 slots/rank; F/NF/NFR have B12/rank and 22 slots/rank.

| Arm / completed segment | GPUs | K | Input tokens/s, compute plus materialization | Peak allocated / reserved GiB/rank | Capacity interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| B, updates 33–128 | 2 | 1 | 70,935 | 35.133 / 42.430 | Considerable measured headroom; current batch sweep absent |
| F, updates 1–128 | 2 | 4 | 10,746 | 30.079 / 49.562 | B12 inherited; not shown to be capacity-limited |
| NF, KL0.1, updates 33–64 | 2 | 4 | 7,636 | 42.833 / 58.707 | More state/work, but no current batch sweep |
| NFR, KL0.1, updates 65–128 | 2 | 4 | 3,560 | 42.83 / 59.08 | B12 chosen using tight B16 calibration above |

Sources: [common B/F/NF ledger](../olmo-fbt-stability/resource-ledger.md),
[latest NFR report](../olmo-nfr-stability-128/results.md).
The memory peaks can include preparation and evaluation. Graph reservation,
live allocation and device-used memory differ. These measurements do not prove
that NF uses less or more actual graph storage than NFR. Executor rates are
lower because evaluations, checkpointing and startup are excluded here.

There is no corresponding modern T1024 K4 physical-batch sweep for N, R, NR,
or FR in these completed reports. The eight-arm resource ledger gives parameter
and analytic matrix-work counts, not measured memory or utilization.

## Historical evidence and its limits

The [one-H100 ordinary T2048 measurements](../olmo-ordinary-long-context/results.md)
show B16 to B32 improving throughput only about 0.2–0.3%, while B32 leaves
40.9 GiB sampled free. Flash SDPA measured about 41,480 input tokens/s; FA4 about
43,135. Similarly, the [ordinary T512 two-GPU sweep](../olmo-ordinary-two-gpu/results.md)
increased physical B64/rank to B192/rank for only about 1.3% more throughput,
with considerable remaining memory. Both support a low-priority capacity
upgrade for ordinary OLMo under those execution paths. Neither excludes a
benefit from H200 bandwidth, changed checkpointing or future objectives.

The older [F4 one-H100 T512/K2 matrix](../olmo1b-f4/results.md#complete-update-resources)
covered all eight arms. It predates later performance improvements and current
K4/all-pass RT, so its absolute speeds and capacities must not be projected
onto the current runner:

| Arm | Input tokens/s at B64 → B96 | Peak reserved GiB at B96 | Directional observation |
| --- | ---: | ---: | --- |
| B | 31,113 → 31,259 | 45.410 | Essentially flat |
| N | 24,026 → 23,938 | 49.145 | Essentially flat |
| F | 15,798 → 15,764 | 61.910 | Essentially flat |
| R | 19,437 → 21,618 | 62.453 | Larger physical batch helps |
| NF | 12,177 → 12,049 | 73.887 | Flat despite high setup reservation |
| NR | 16,444 → 17,833 | 64.967 | Larger physical batch helps |
| FR | 12,136 → 12,893 | 78.391 | Some gain, tight setup; old numerical qualification retained |
| NFR | 9,892 → 10,305 | 78.223 | Some gain, tight setup |

This may explain the user's recollection that an FBT/RT combination was
constrained by memory. High reservation alone did not guarantee a benefit from
larger batch: NF is a useful counterexample in this historical configuration.

After later optimizations, [RT-only T512](../olmo-rt-large-batch/results.md)
improved 6.18% from B128 to B192; B256 failed graph capture after eager updates
succeeded. The older [K2 combined T1024](../olmo-combined-t1024/results.md)
improved 12.92% from B32 to B64, leaving 9.20 GiB sampled free at B64.
These independently support physical-batch sensitivity in RT-containing paths,
but do not quantify H200 gains for current K4.

## Allocation implications

1. Keep capacity and bandwidth questions separate. First compare H100/H200 at
   the same physical batch and logical examples, then increase physical batch
   on H200 while retaining the same logical update. This separates hardware
   effects from batch utilization effects as far as a directional test permits.
2. Prioritize NFR for the larger-memory sweep, followed by FR/R/NR if those arms
   are about to run. For F/NF/N, first determine whether existing H100 headroom
   already permits a throughput plateau; there is no need to force B12 across
   all arms solely because NFR used it.
3. Use the current two-H100 allocation milestone to compare one distributed
   two-GPU job against two independent one-GPU jobs at matched physical batch
   and effective 512-row updates. Do not change physical batch in that primary
   comparison; otherwise allocation and batch-utilization effects are mixed.
4. Before committing an eight-H200 node to training, briefly qualify the exact
   one-/two-/eight-rank paths, then measure relevant physical batches, memory
   peaks during capture and updates, useful tokens/sec/GPU, per-run update
   latency, and concurrent checkpoint/storage contention. Rank count does not
   pool VRAM with the current replicated optimizer.
5. More memory may also permit reducing checkpointing or additional graph
   storage. Those are separate changes requiring their own bounded checks;
   do not include an assumed benefit in a purchase/allocation forecast.

No H200 speed multiplier, maximum supported batch or all-eight-arm memory
classification is established. Existing measurements are sufficient to make
H200 particularly worth testing for NFR, and insufficient to choose the most
cost-efficient node allocation without actual-node measurements.
