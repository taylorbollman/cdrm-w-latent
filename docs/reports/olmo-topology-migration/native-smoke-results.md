# Native N/R/NR/FR integration smoke

2026-09-30. **All four cells completed both optimizer updates, passed the
functionality checks, synchronized their online W&B runs, and released their
CUDA graphs, DDP reducers and process groups cleanly.** This closes the four
current native combination gaps identified in
[integration-scope.md](integration-scope.md). It does not establish useful
learning, precision equivalence, or long-run stability.

## Exact scope

Each cell runs in a fresh project Docker container with two `torchrun` ranks on
the two H100s. Each rank has physical batch **12**, sequence length **1,024**,
and **two accumulation slots**. An optimizer update therefore consumes **48
real packed rows / 49,152 input tokens**, with no dummy rows. There are exactly
two changed-input updates per arm, using the same ordered corpus prefix:
chunks 0–47, then 48–95. This is a small functionality fixture, **not the
512-row learning-update budget** used by the allocation benchmark.

The backbone is the original pinned OLMo-1B `step60000-tokens252B` checkpoint,
strictly loaded without resetting its weights. Active fusion/predictor modules
and all active Adam state are fresh. Common ambient initialization seed is
`20261001`; fusion/predictor/jitter seeds are respectively `20260922`,
`20260921`, and `20260928`. The module constructors already use isolated seeded
generators. The added CPU constructor tests confirm identical initialization
from different ambient RNG states and unchanged supplied backbone weights.
Initial common parameter digests also match across the four actual GPU cells.

| Arm | Actual computation | Active objective weights: CE / latent / KL |
| --- | --- | --- |
| N | One ordinary pass plus NextLat | 1 / 1 / 0.1 |
| R | One pass, native RT at layers 0 and 15 | 1 / 0 / 0 |
| NR | One pass, native RT0/15 plus NextLat | 1 / 1 / 0.1 |
| FR | Four feedback passes, native RT0/15 on every pass, active fusion | 1 / 0 / 0 |

FR uses the current campaign CE pass weights `(1/2, 1/6, 1/6, 1/6)`. NextLat
is disabled in FR. N/NR use the accepted reduced KL weight **0.1**, explicitly
installed in both the model and predictor configuration, with latent weight 1.

All cells use BF16 mixed execution with FP32 master parameters, gradients and
Adam moments; fused AdamW; forced ordinary Flash SDPA; native Triton RT forward
and backward tiles with recomputation; ordinary activation checkpointing;
reused native RoPE; and two captured DDP forward/backward graphs. TF32,
autocast weight caching and `torch.compile` are off. Pointwise operations use
the current eager backend. The warmup/plateau recipe remains explicit: the two
updates use learning rates `2e-5` and `2.016875e-5`, with norm clipping at 1.

## Functionality checks

Every arm passed the following checks on both actual updates:

- Literal token-by-token eligibility counts equal the model counts, immutable
  packed plan, executed denominators and counters. Distributed rows are complete,
  unique and in the prescribed logical update; actual tokens change between updates.
- Every active parameter has a finite FP32 raw gradient; frozen modules have
  none. Full tensor digests of raw gradients agree exactly between the two ranks.
- Every active parameter owns Adam state with the correct step counter and
  FP32 finite moments; no frozen, missing, foreign or duplicate ownership exists.
  Full parameter and Adam tensor digests agree exactly between ranks after each update.
- Each active component changes parameters; dormant fusion remains unchanged.
  Original input/readout tying is retained. Capture preparation leaves parameters
  and fresh empty Adam unchanged and gradients zero.
- Both graph objects, parameter/gradient/input storage, and completed zero-gradient
  boundaries remain valid. Each rank executes exactly **two local and two
  synchronized graph replays** across the two actual updates.
- Scheduler exposure, data cursor and counters advance together. Original artifact
  file stats and all pinned runtime source bytes remain unchanged during each cell.

The independent report review additionally checked the completed statuses,
shared source pins, common initial parameter digests, preparation equality,
active Adam-state cardinalities and replay totals. This is a consistency audit
of execution evidence, not another numerical oracle or checkpoint readback.

### Counts and parameters

CE includes valid within-chunk next-token transitions across internal EOS;
latent pairs and KL triples do not cross document identities. Attention context
remains continuous across packed documents: there is no extra EOS insertion or
EOS-triggered RT/FBT reset. The final chunk position does not predict into the
following chunk.

| Arms | Update | CE targets | Latent pairs | KL triples |
| --- | ---: | ---: | ---: | ---: |
| N, NR | 1 | 49,104 | 48,997 | 48,843 |
| N, NR | 2 | 49,104 | 48,999 | 48,846 |
| R, FR | 1 and 2, each | 49,104 | 0 | 0 |

All cells end at 2 updates, 8 global physical microbatches, 96 packed rows,
98,304 input tokens and 98,208 CE targets. N/NR additionally record 97,996 latent
pairs and 97,689 KL triples. The runner's `documents` counter is 96 nonempty
packed rows here; it is **not** the number of original source documents.

| Arm | Resident parameters | Trainable parameters | Active gradient / Adam tensors | Changed tensors per update: backbone / fusion / predictor |
| --- | ---: | ---: | ---: | ---: |
| N | 1,267,879,936 | 1,259,491,328 | 69 | 65 / 0 / 4 |
| R | 1,185,153,024 | 1,176,764,416 | 65 | 65 / 0 / 0 |
| NR | 1,267,879,936 | 1,259,491,328 | 69 | 65 / 0 / 4 |
| FR | 1,185,153,024 | 1,185,153,024 | 67 | 65 / 2 / 0 |

The inactive resident fusion wrapper accounts for 8,388,608 frozen parameters
in N/R/NR. Native RT adds no parameter tensors to the backbone.

## Startup observations and resource scope

| Arm | Objective, update 1 → 2 | Raw gradient norm, update 1 → 2 | Graph preparation, max rank seconds | Total cell seconds | Peak allocated / reserved GiB, max rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| N | 4.64369 → 4.44386 | 3.35537 → 2.73860 | 12.59 | 126.43 | 24.25 / 34.81 |
| R | 6.03960 → 3.57910 | 117.29398 → 42.44651 | 103.96 | 215.03 | 23.47 / 36.00 |
| NR | 7.53008 → 5.20216 | 114.39385 → 42.08240 | 111.79 | 230.52 | 24.25 / 37.55 |
| FR | 7.57037 → 6.27251 | 107.68509 → 87.65494 | 413.44 | 544.40 | 32.15 / 54.55 |

**Heavy startup clipping remains visible in the RT-containing arms.** Their
finite updates and matching replicas do not establish that this clipping is
scientifically desirable or that later feedback passes improve CE. These are
two different training batches, not a fixed development panel. N/NR objectives
include auxiliary losses, so their absolute values are not CE-only comparisons
against R/FR. No learning-speed or quality ranking follows from this table.

Timings include setup and substantial CPU tensor hashing/finite-state audits;
recorded update windows, which include these audits, range from approximately
38–46 seconds. These are not tokens/sec measurements. Peak memory describes
these two-update physical shapes, not a capacity sweep or a prediction for
H200, eight ranks, or concurrent full-node jobs. No large disposable native
checkpoint was written for these four smokes.

## Evidence and reproduction

The new helper is
[`scripts/olmo_topology_native_smoke.py`](../../../scripts/olmo_topology_native_smoke.py).
Its [17 focused CPU tests](../../../tests/test_olmo_topology_native_smoke.py)
passed before GPU launch. Existing execution/core files were unchanged.

| Arm | Immutable completed report | Online W&B |
| --- | --- | --- |
| N | [native-n-01/report.json](../../../.runtime/olmo-topology-migration/native-n-01/report.json) | [ck22yp4u](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ck22yp4u) |
| R | [native-r-01/report.json](../../../.runtime/olmo-topology-migration/native-r-01/report.json) | [1ps75ckx](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/1ps75ckx) |
| NR | [native-nr-01/report.json](../../../.runtime/olmo-topology-migration/native-nr-01/report.json) | [pcdgipq6](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/pcdgipq6) |
| FR | [native-fr-01/report.json](../../../.runtime/olmo-topology-migration/native-fr-01/report.json) | [bk4jo7c4](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/bk4jo7c4) |

Report SHA256 pins:

```text
N   8a4d73b8d807fde3e8265d356410b7f2b45c592fe9d43be27c67af1416060a18
R   76a2e6f6cbfae9955edce0bc6ae16e98289a6254e8728da93bd854731ddde09a
NR  0e550b21323b6ad7affe2cc86194caaee9f4704dc6d37a495ef66aabe1ea335d
FR  6a84f1783b541d6430f20674865b98a10cc4fb11f002d88f5d92e4e70d1c6f8a
```

Shared authorities:

```text
Original repo revision: 81b71efbce6f4dada57c94860301af4298bcd351
Original model SHA256: ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c
Ordered index SHA256:  e890bdb6721cdf454251a45b9376d66622d62e6c164feeae3d624cd4194ea76d
Smoke helper SHA256:   499a000c8a7e57d2fcbb07ae51d1d268853be4c83e02c38ef80ddc6c8ac5c89d
Source-map SHA256:     f32c0d86314dfc467bad64a41c3ed6a942649b0aec07f69a36d4ffc9da00966a
```

The source-map digest uses sorted compact JSON of the report's complete
`sources` mapping. Each run includes its source snapshot. The host matrix
`.runtime/olmo-topology-migration/smoke-matrix.py` launches fresh jobs in
N → R → NR → FR order, each with a 1,800-second container timeout and no automatic
restart; adjacent `native-*-01.log` files retain the launch/termination evidence.
Cloud retention receipts are recorded separately in the milestone closeout.

With these native combinations covered, continue the separately scoped
production-state topology migration and independent-job failure-isolation
acceptance. Native restart/retention, abrupt failures, eight-rank behavior and
H200 tuning are not established by this smoke matrix. Broad FP32/BF16 trajectory
qualifications and the open scientific question of useful feedback refinement
remain unchanged.
