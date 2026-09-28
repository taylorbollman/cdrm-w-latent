# Portable campaign readiness results

2026-09-28. **The portable implementation milestone passed.** The changed model,
loss, optimizer and data contracts now have bounded checks and interruption-safe
records. This is the portable part of readiness milestone 1; a production
multi-GPU campaign trainer is not yet qualified.

Runtime commit: `714f31c` (after `77f9761` model/loss/data contracts).
[Protocol](protocol.md), [usage and remaining integration](usage.md),
[data contract](data-contract.md), [progress](progress.md).
[W&B diagnostic](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/kb0lbu1k).

## What changed

- Explicit eight-arm recipes, K4 when FBT is enabled, and selected RT layers on
  every pass including the initial pass. Legacy bootstrap behavior stays the
  default outside the new opt-in policy.
- Campaign CE weights `[1/2,1/6,1/6,1/6]` and independently averaged NextLat
  SmoothL1/KL terms; both auxiliary coefficients remain 1.0. Legacy loss policy
  stays available and is regression-tested.
- Explicit AdamW groups excluding tied embeddings, normalization vectors,
  biases and scalars from weight decay; token-based warmup/plateau and strict
  same-plan checkpoint contracts.
- Externally keyed feedback jitter, stable across row partition and padding,
  with no model-internal random draw. Static layout buffer ownership is explicit;
  graph trainers reject nonzero jitter pending full execution qualification.
- Ordered document windows, target accounting, logical updates, rank partitions
  and resumable data cursors. A bounded raw JSONL/gzip adapter verifies source
  bytes/tokenizer identity and records deterministic splits/preprocessing.

## Evidence

The broad CPU-container suite passed **706 tests in 38.88 seconds**. A subsequent
Unicode JSONL line-separator fix passed **28 ingestion tests**, including two new
regressions; these scopes overlap and should not be summed. Earlier narrower
runs are supplementary, not additional unique tests. The only broad-suite
warning concerns a future Google/grpc dependency minimum, not model behavior.

The fresh-process recovery test saves a tiny CPU model after one update during
warmup, starts a new Python process, restores it, and performs the next update.
Model/optimizer tensor hashes, schedule, metrics, counters, data cursor and next
RNG draw match exactly. This is actual process restart on the same CPU fixture;
it does not claim distributed, graph or H100-to-H200 restart equivalence.

The actual-checkpoint probe used native OLMo-1B step60000 (about252B pretraining
tokens), pinned revision `81b71efbce6f4dada57c94860301af4298bcd351` and weight
SHA256 `ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
NFR has **1,267,879,936 resident/trainable parameters** in this configuration.
Execution was B1/T16 on one H10080GB, BF16 mixed/FP32 masters, ordinary Flash SDPA,
native Triton RT0/15, activation checkpointing/recomputation, and full CE/latent/KL
supervision. No CUDA graphs, compile or Q/K normalization change.

| Check | Result |
| --- | --- |
| Canonical K4 backward | Finite loss and all71 active parameter gradients |
| RT execution | Layers0 and15 each called4 times in every probe/update |
| Independent literal loss assembly | Objective difference0; gradient relative L2 and maximum absolute difference0 |
| Feedback jitter RNG ownership | CUDA RNG unchanged on every backward |
| Two fused-Adam updates | Finite gradients and parameters; token clock16 then32 |
| Runtime source integrity | All60 archived source hashes unchanged |

The GPU probe finished in **41.9 seconds** including loading/verification and
initial setup before W&B final synchronization. All four stages passed. Peak
reserved GPU memory was21.10GiB for this tiny fixture; this is **not a production
capacity or throughput estimate**.

The initial canonical objective was14.0920 (CE7.1250, latent0.8950, KL6.0720).
The disposable updates used LR2e-5, near the configured warmup floor. Their
pre-clipping gradient norms were1008.84 and167.18; both were substantially clipped
to the configured maximum1.0. This is a finite/functionality result, **not evidence
that the proposed LR or initial retrofit is well calibrated**. Representative
data/large effective batches must still measure per-loss gradients, clipping and
adaptation before fixing a training recipe. The short repeated-text fixture and
two updates cannot establish learning quality or longer-run stability.

## Remaining gates

The next slice should connect the logical data updates to padded Flash execution,
per-rank jitter inputs, graph-safe changing masks and accumulation. It needs a
bounded two-GPU update/reference comparison and fresh-process distributed restart,
followed by representative K4/T1024 memory and throughput on the chosen hardware.
Empty local rank slots and zero local KL targets require correct global counts
and synchronized participation; their data representation alone is not enough.

Production disk-backed Dolma preparation, generation/evaluation, cooldown/SFT
schedules and LR calibration remain later work. The current scheduler freezes
its complete logical-update plan; extension requires an explicit audited fork.
H200 and final world-size qualification remain separate. Existing BF16/native-RT
qualifications are unchanged; exact gradient agreement here compares two ways
of assembling the **same BF16 objective**, not BF16 against FP32.

No quality training, new corpus download, hardware allocation or long-running
job was launched. Code was committed/pushed throughout. Disposable smoke updates
were not saved as billion-parameter checkpoints; every stage can be reproduced
quickly from the immutable original checkpoint. Evidence retention is recorded
in [storage-receipt.md](storage-receipt.md).
