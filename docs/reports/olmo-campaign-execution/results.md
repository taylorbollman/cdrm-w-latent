# Shared component execution and recovery

**Completed, 2026-09-29.** The shared component runner passes bounded execution,
observation and cloud-recovery acceptance. The native BF16 combined model
reproduces the uninterrupted reference exactly after restoring update 1 from
GCS and reconstructing DDP/CUDA graphs. This closes an operational readiness
milestone; the BF16-versus-FP32 numerical question remains qualified. PR 46.

The common launcher now executes B/N/F/R/NF/NR/FR/NFR with explicit startup and
finite packed-data plans. B means the ordinary backbone, N adds NextLat, F adds
K4 feedback passes, and R selects native RT layers0/15. Original-weight/fresh-Adam
startup supports all eight arms. The initial adapted allowlist is fusion128
weights plus fresh all-active Adam for NF/NFR. Same-lineage recovery uses the
committed checkpoint directly, without needing a successful reference report.

Lean observations retain losses, global counts, gradient norm/clipping estimates,
LR and data clocks while avoiding per-update complete gradient/model/Adam hashes.
Acceptance observations use the identical training path with those hashes added.
Complete checkpoint integrity checks remain in both modes. The native model,
losses, kernels, Q/K behavior and historical acceptance implementations are unchanged.

## Completed evidence

The integrated CPU suite passes **242 distinct tests**. All-eight checks cover
parameter/optimizer ownership, actual objective updates,
packed document boundaries, masks, accumulation, empty rank slots, observations,
strict startup/recovery metadata and optimizer/scheduler clocks. Separate operational
helpers verify generation-pinned cloud assets and audit recorded execution evidence.
See the [test ledger](test-ledger.md) for exact test scopes and retained attempts.

Tiny two-H100 acceptance passes:

- Three-update reference versus lean full run: identical metrics and final complete
  model/Adam/schedule/RNG/cursor state.
- Lean stop at update1, GCS generation-pinned restoration, fresh-process updates2/3:
  identical restored origin and every resumed input, raw gradient and step boundary.
- Resume of completed update3: validates and exits with no graph preparation or
  optimizer step; state remains exact.

The tiny fixture uses width32/two layers, T16, FP32 math/eager RT, B2 per rank and
five global rows/update. It checks operations and graph ownership; it does not
exercise native Triton head dimensions or establish BF16 accuracy.

## Native configuration and observed updates

Native OLMo-1B step 60000 backbone: 16 layers, width 2048, 16 heads, MLP 8192,
tied readout. NFR adds K4 (four total passes), RT at layers 0/15 in every pass, NextLat
latent and KL losses, with 1,267,879,936 active parameters:
1,176,764,416 backbone, 8,388,608 fusion and
82,726,912 predictor. Import uses the selected complete fusion-only update128
state, while the backbone/predictor remain at their pinned original authorities.
The new optimizer/data/schedule clocks start at zero. Prior adaptation exposure
is recorded separately; this is not exposure matched to untouched original weights.

Execution is two H100 80GB GPUs, BF16 mixed with FP32 master weights/Adam,
ordinary Flash SDPA, native Triton RT forward/backward, backward recomputation,
ordinary activation checkpointing, reused RoPE and prepared CUDA graphs.
T1024, B1/rank, three global rows/update give two microbatches per rank; rank 1's
last slot is empty. This tiny physical batch is for bounded recovery acceptance,
not a production batch recommendation or steady-state throughput test.

The native stop-at-1 run passed all strict import/transition checks and preserved
the full starting boundary through 20 warmup and two capture backwards per rank.
Adam started empty and acquired state for all 71 owned parameter tensors.
Each update consumes 3,072 inputs, 3,069 CE/latent targets and 3,066 KL triples.
The three-update reference completed with the following finite training metrics:

| Update | CE | Latent | KL | Raw gradient norm |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6.99714 | 0.80674 | 4.86984 | 697.082 |
| 2 | 6.77916 | 0.80761 | 4.61707 | 121.132 |
| 3 | 6.20720 | 0.74456 | 2.50378 | 62.437 |

All three activate the existing norm-1 clipping; update 1's estimated clipping
coefficient is 0.00143455. These are different training batches, not a held-out
learning curve or evidence that longer optimization is stable. Final counters
are 9,216 inputs, 9,207 CE/latent targets, 9,198 KL triples, nine packed rows and
12 physical microbatches across both ranks.

The reference and lean stopped run match exactly at update 1, including all
losses and the complete model/Adam/schedule/RNG/cursor boundary. Independent
auditing verifies both frozen source snapshots. Cloud restoration also recovers
that boundary exactly on both ranks, with Adam resident before DDP/graph setup.
Resumed updates 2/3 match the reference inputs, losses, all 71 raw parameter
gradients and complete step boundaries on both ranks. Final model, Adam,
scheduler, RNG, counters and cursor are exact. The process exited successfully,
W&B synced, and its final checkpoint was retained and verified in GCS.

In the lean stop run, forward/loss/backward took 6.969 s and optimizer/cursor
work 0.097 s. After-update memory was 23.711 GiB allocated and 36.373 GiB reserved
per GPU. Preparation took 375.05 s. Cloud retention took 100.11 s for the 5.071 GB
origin and 294.34 s for the 15.215 GB update-1 payload; segment elapsed time was
974.27 s. Exact-generation streamed recovery verified 15.215 GB in 81.72 s.
No host peak-memory or comparative I/O speed claim is made.

The reference's forward/loss/backward regions were 6.559–6.970 s per update.
Acceptance adds approximately 27 s per update for gradient and full-boundary
hashing, demonstrating the practical reason for the lean observer. Reference
peak memory was 24.384 GiB allocated and 37.051 GiB reserved per GPU. These are
instrumented observations, not steady-state throughput or capacity recommendations.

The resumed segment took 926.50 s: full tensor restoration took 20.66 s,
graph preparation 388.34 s, final checkpoint write 62.69 s and verified cloud
retention 268.50 s. Boundary hashing and other regions are recorded separately.
The preceding CPU cloud asset download is outside this segment's elapsed time.

W&B: [native lean stop](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/qjk60oku),
[native reference](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/55lv1nww),
[native cloud continuation](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/n8pchjsp).
These runs log training metrics for functionality checks, not held-out quality.

## Scope and next work

Existing BF16-vs-FP32 discrepancies remain qualified. Exact same-precision replay
establishes recoverable execution; it does not resolve precision equivalence,
longer-run quality or optimization sensitivity. No Q/K normalization was changed.
The short native prefix contains no internal source-document boundary; independent
CPU fixtures cover that policy. It is readiness data, not a selected pilot mixture.

Next implement [declared per-pass held-out evaluation](next-steps.md), including
live training-state preservation. Then review the actual mixture, budget, startup
exposure and target-hardware capacity/throughput before a quality pilot. See
[operator notes](operator-notes.md) for launch/recovery boundaries and limits,
and [storage receipts](storage-receipt.md) for checkpoint authorities.
