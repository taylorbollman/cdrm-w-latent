# Ordinary pretrained B: captured host-loop and cloud restart

2026-09-29. **The actual pretrained ordinary-model acceptance passes.** Three
updates complete through the common host loop, with live-graph checkpointing
and verified cloud retention. Fresh processes restore checkpoint 1 from exact
GCS generations, rebuild DDP/graphs with Adam resident, and reproduce complete
updates 2 and 3 **bitwise**, including all 65 native gradient tensors, metrics,
model/Adam/scheduler/counters, committed cursor and rank-local RNG.

| Stage | Result | Reported stage time | Host launcher time |
| --- | --- | ---: | ---: |
| [Reference](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/m6vnrnqx) | Three updates; checkpoints 1/3 retained | 900.190 s | 910.110 s |
| Cloud restore | Exact generation, size and SHA256 verification | — | 75.562 s |
| [Fresh resume](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/vgls4mor) | Updates 2/3 exact; final checkpoint retained | 568.573 s | 578.625 s |

Every stage exited normally with status 0, within its 1,200-second limit. Both
W&B runs synced. The combined host orchestration took approximately 26.1 minutes.
Large checkpoint writes, immutable uploads/readbacks and diagnostic hashes
dominate these times. Root also ran CPU-only endpoint analysis concurrently;
this is not a throughput measurement.

The model uses the original pinned **OLMo-1B step 60000** checkpoint: 16 layers,
width 2,048, 16 heads, MLP width 8,192 and tied 50,304-row embedding/readout.
All **1,176,764,416 native parameters** train. The resident wrapper also contains
8,388,608 frozen, unused fusion parameters, giving 1,185,153,024 resident
parameters. There is no predictor, NextLat loss, FBT feedback or RT execution.
Native weights remain FP32 masters; execution uses deterministic BF16 mixed
precision with ordinary Flash SDPA, activation checkpointing, native RoPE,
fused AdamW and two captured DDP backward graphs on two H100 80GB GPUs.

Physical batches are **8 rows per GPU at T1024**. M1/M2/M3 accumulation gives:

| Logical update | Global input tokens | CE targets | Observed CE, nats/target |
| --- | ---: | ---: | ---: |
| 1 | 16,384 | 16,368 | 2.539888 |
| 2 | 32,768 | 32,736 | 2.719925 |
| 3 | 49,152 | 49,104 | 2.752614 |

The reference presents **98,304 unique stream inputs in 96 non-overlapping
chunks**, with 98,208 CE targets and zero enabled latent/KL targets. There is one
actual within-chunk document transition, in update 2; there are no dummy rows or
padding in this prefix. CE includes that document-boundary transition and omits
the final position's cross-chunk prediction. The different losses correspond to
different input chunks and are not an evaluation curve. Original token-based
warmup is preserved; this finite plan does not choose a production batch size.

The reference begins with empty Adam state, while resume loads populated moments
before DDP/capture. Both preserve their complete boundaries through preparation.
Each rank performs 20 preparation backwards and two capture backwards, with no
optimizer or cursor advance. Reference then executes six actual graph replays;
resume executes five. Checkpointing preserves the live graph's storage contract,
and cloud retention does not consume training RNG.

| Memory scope, maximum per GPU | Peak allocated | Peak reserved | Lowest sampled free |
| --- | ---: | ---: | ---: |
| Reference through three updates | 22.70 GiB | 31.12 GiB | 46.50 GiB |
| Cold resume through two updates | 29.34 GiB | 31.11 GiB | 46.52 GiB |

Cold Adam-resident preparation accounts for the higher resumed allocation peak.
These observations establish comfortable capacity for this specific B8/T1024
acceptance, not an optimal batch, sustained throughput, or another topology.

Checkpoint 1 contains a **14,154,933,413-byte** state file, SHA256
`5ed5a23f7d9f2753e76bbf95ed45797f8ddafb30f82ca7b276dc8e98ce2203e5`.
Its 84,792-byte manifest has SHA256
`748bb7a190c2d136119d4c6924b9ab6324b1bf3041c2903bae108ea55a5f7718`.
The restore downloaded state generation **1790680957093316** and manifest
generation **1790681021056156** beneath
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/base-loop-base-reference-01/update-000001/`.
Both objects were fully rehashed and the distributed manifest inspected before
GPU resume. The reference final and replay final checkpoints are separately
retained under their stage prefixes; original checkpoint 0 was not duplicated.

Independent read-only audit: **460 checks passed**, including 182 source/snapshot/
live-file pairs, full update/final-boundary comparisons, scalar finiteness and
counts, checkpoint manifests and exact-generation restore receipts. It compares
state-file sizes and previously verified hashes without redundantly rereading
each 14 GB state file. Audit report:
`.runtime/olmo-campaign-lifecycle/base-audit-01/report.json`, SHA256
`32a9501ad88b71cad169e51c8b4ad72ed7ce9fc5eecff095ef83cc4a48082778`.
The prelaunch focused CPU scope passed 24 tests; initial test-only setup failures
remain retained. See the [predeclared protocol](base-loop-protocol.md).

Final report SHA256 pins, beneath `.runtime/olmo-campaign-lifecycle/`:

- `base-reference-01/report.json`:
  `6e945cf271c1791a3dcb4e71f2f481e8389bb59477dad1f78097db2a23563d25`.
- `base-cloud-restore-01/report.json`:
  `733b9cbdd6a6c4475f0908860db990c7f189e7194843fc7e9bcb3580fa186fd6`.
- `base-resume-01/report.json`:
  `fadc76f4636ce06b5bd9bb2ab48eaaf761f26b87bef9e041d2bc1fdac069814c`.

This closes the ordinary pretrained scale/ownership integration with the common
loop. It does not qualify NFR BF16 numerical fidelity, adapted startup, H200 or
changed topology, pretrained evaluation insertion, long-run stability, production
data/quality training, or a general manifest-driven campaign launcher.
