# Manifest-driven ordinary-B execution and restart

2026-09-29. **The bounded ordinary-B adapter passes on two H100 80GB GPUs.**
The model, optimizer, data plan and schedule are constructed from the pinned
manifest and resolved evidence. Three updates complete, checkpoint 1 is restored
from exact cloud generations, and fresh processes reproduce complete updates
2 and 3 **bitwise on both ranks**. This includes input/mask/key digests, all
65 native gradient tensors, metrics, model/Adam/scheduler/counters, committed
cursor and rank-local RNG. The final complete boundaries also match exactly.

| Stage | Result | Host orchestration time |
| --- | --- | ---: |
| [Reference](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/y8sfg1bc) | Updates 1–3; checkpoints 1/3 retained | 901.284 s |
| Cloud restore | Exact generation, size and SHA256 verified | 90.924 s |
| [Fresh resume](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/dy45suwn) | Updates 2/3 exact; checkpoint 3 retained | 579.787 s |

All stages exited with status 0 within their bounds; both W&B runs synced.
The approximately 26.2-minute orchestration includes construction, preparation,
hashing, serialization, cloud upload and readback. It is not a throughput result.
Reported in-process reference/resume times are 887.202/565.580 seconds; measured
update regions, including gradient hashing, total only 21.729/14.752 seconds.
The remaining time is not fully attributed here.

The original pinned OLMo-1B step-60000 weights remain authoritative: 16 layers,
width 2,048, 16 heads, MLP width 8,192 and tied embedding/readout. All
**1,176,764,416 native parameters** train. An additional 8,388,608 dormant fusion
parameters remain frozen, for 1,185,153,024 resident parameters. There is no
predictor, NextLat objective, FBT feedback or RT execution. The execution path
uses deterministic BF16 mixed precision, FP32 masters/moments, ordinary Flash
SDPA, activation checkpointing, native RoPE, fused AdamW and captured DDP
backwards, as declared in the [frozen protocol](run-protocol.md).

Physical batches are **8 rows per GPU at T1024**, with one physical microbatch
per rank per logical update:

| Update | Global valid inputs | CE targets | CE, nats/target | LR used |
| --- | ---: | ---: | ---: | ---: |
| 1 | 16,384 | 16,368 | 2.539888 | 0.0000200000 |
| 2 | 16,384 | 16,368 | 2.685670 | 0.00002005625 |
| 3 | 16,384 | 16,368 | 2.708998 | 0.0000201125 |

The reference consumes **49,152 unique stream inputs and 49,104 CE targets** in
48 non-overlapping full chunks, ending at chunk cursor 48 and update cursor 3.
There is one genuine within-chunk document transition, no padding or dummy rows,
and no enabled latent/KL targets. The data manifest also records potential
latent/KL counts; those are not active losses in B. CE includes the document
transition and excludes cross-chunk prediction. This canonical prefix contains
only two books documents and is an unrepresentative readiness sample. Different
update losses use different chunks and are not a held-out learning curve.

The adapter repeats the exact CPU resolution before CUDA, then checks actual
data membership, rank allocation, architecture ownership and token schedule
against the resolved plan. It preserves the original token warmup while using
the explicitly declared 16,384-token diagnostic update budget. The reference
starts with empty Adam; resume loads populated moments before capture. Both
preserve their complete boundaries through 20 warmup and two capture backwards
per rank. Actual synchronized replays number three for reference and two for
resume, with no accumulation replay in this fixed-budget example.

An additional direct control compares reference update 1 against the earlier
[hardcoded ordinary-B runner](../olmo-campaign-lifecycle/base-loop-results.md).
On both ranks, the full inputs, all 65 raw gradients, every step metric,
post-Adam model and optimizer state, counters and cursor are **exactly equal**.
The only scheduler differences are its future token prefix and plan hash:
the older run accumulates more data in updates 2/3. Process RNG and the nominal
effective-update budget differ explicitly, so complete configuration/RNG
identity was not required for this control. Their current LR and scheduler
history match. This directly checks the adapter's constructor and first-update
behavior without another GPU run.

Peak allocated/reserved memory was 22.70/34.93 GiB per GPU in reference and
29.34/31.11 GiB in fresh resume; minimum sampled free memory was 42.69/46.52 GiB.
These observations establish capacity for this particular acceptance, not an
optimal physical batch or sustained training rate.

Checkpoint 1's state file is **14,154,945,317 bytes**, SHA256
`3dfd330775c3f79372dabf6b7a70ab6dfdc4177272ede87081bd9a78b0632879`.
Its 97,446-byte manifest has SHA256
`15fd141f260c8bb096ba90f0c068438a3a1c363b913083b133f5157514bc4502`.
Restore downloaded state generation **1790682559220328** and manifest generation
**1790682635372673** beneath
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/manifest-b-loop/reference/update-000001/`.
Both full object byte streams were verified before GPU resume. Reference and
resume final checkpoints are separately retained under their corresponding
stage prefixes. Original checkpoint 0 was not duplicated; no local pruning ran.

The independent stdlib audit passed **566 checks**, including 200 pairs of
source snapshots and current files, declarations, data membership, scheduling,
ownership, replicas, complete replay, first-update adapter parity and retained
checkpoint/restore pins. It compares state sizes and already verified hashes
without repeating the 14 GB state-file reads. Its initial descriptor-hash
reconstruction omitted the packed helper's canonical trailing newline; correcting
that auditor-only formatting assumption made the membership checks pass without
changing execution evidence. The focused prelaunch CPU suite passed 19 tests.

Evidence is beneath `.runtime/olmo-campaign-manifest/`. Final SHA256 pins:

- `b-draft-01/manifest.json`:
  `a4476b4b7dcfff231bc07d8ccbcd39b56efc2a57624686529c1e330fa8718d78`.
- `b-draft-01/resolved/resolved.json`:
  `f6fba00a640e3aa153e0e30887208f6f2146eda6757be69bb5f4c4ca0a20d5b9`.
- `b-reference-01/report.json`:
  `db2e7ebf82ccb8197b2930e8d176c30a4f871817c74b6147e2585474835bc85f`.
- `b-cloud-restore-01/report.json`:
  `738aa7d7ec634e1bb67cbb73833a8da37c684178491a74501abfaf183ba3e921`.
- `b-resume-01/report.json`:
  `80e85ab33f81207c598e2cc571e549cb04d760d7f9e8299a142ba91377a8c689`.
- `b-execution-audit-01/report.json`:
  `ba9f2e27f80af2ce5a5355e01a80cb90dfe8864493963ed008f4e9884f35408f`.

This closes the explicit ordinary-B execution bridge. The resolver continues
to grant neither launch authorization nor numerical clearance by itself. The
general eight-arm launcher, adapted-startup lineage, evaluation insertion,
production data mixture, sustained throughput, changed hardware/topology and
long-run quality remain outside this result. It does not extend the separate
NFR BF16 numerical qualification.
