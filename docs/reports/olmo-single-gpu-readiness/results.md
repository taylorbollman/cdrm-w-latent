# One-H100 objective and recovery readiness

**Completed 2026-09-25: RT-only and combined K2 FBT + RT + NextLat pass the
bounded one-GPU preparation and graph-recovery gates.** The forward adapter
matches canonical losses, every participating raw gradient, and two complete
accumulated optimizer updates exactly. Saving, reconstructing and resuming the
graph-based training state also reproduces the reference continuation exactly.
The next useful milestone is actual two-GPU DDP correctness and recovery.

This establishes the single-GPU prerequisites in the
[one-to-two-GPU plan](../../native-rt-single-to-two-gpu-plan.md), not DDP/NCCL,
distributed CUDA graphs, sharding, throughput or task quality. The initial
Flash-dispatch failure remains failed in the [five-report summary](summary.json).
Four successful cases pass **70 required gates** and execute **20 physical
optimizer updates in total**.

## Configuration and completed cases

All cases use the original OLMo-1B step60000 checkpoint, about 252B source
pretraining tokens: 16 layers, width 2048, 16 heads, SwiGLU 8192 per branch and tied
50304-token embeddings/readout. Shape is physical **B2/T512**. RT is selected
at indices 0/15. Combined means one ordinary bootstrap pass followed by one
feedback pass containing those RT layers, with training-only NextLat; K2 counts
both passes. No learned-latent inference recurrence or Q/K normalization is added.

The selected execution path is `compiled-native`: deterministic PyTorch Flash
SDPA for ordinary attention, native ordinary FP32 RoPE, compiled rounded
SwiGLU, activation checkpointing on ordinary layers, fused AdamW and native
Triton tiled RT with recompute backward, reused casts/RoPE and K/V-only writes.
BF16 mixed keeps model weights, gradients and Adam moments in FP32. TF32 and
autocast weight caching are off; CE chunks 2048 and KL grouping 128 are unchanged.
These runs use neither the optional Dao RoPE nor FA4 overlay.

| Case | Result | Gates | Physical / logical endpoint updates | Frozen runtime | W&B |
| --- | --- | ---: | ---: | --- | --- |
| Objective preparation, original RT attempt | Failed before adapter comparison | 0 completed | 0 / none | `ed4333d` | [9vijknoo](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/9vijknoo) |
| Objective preparation, corrected RT | Passed | 15/15 | 4 / 2 | `f8be057` | [rdhvqpcb](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/rdhvqpcb) |
| Objective preparation, combined | Passed | 15/15 | 4 / 2 | `f8be057` | [tbmtchvi](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/tbmtchvi) |
| Graph recovery, RT | Passed | 20/20 | 6 / 4 | `ed4333d` | [w8b68887](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/w8b68887) |
| Graph recovery, combined | Passed | 20/20 | 6 / 4 | `ed4333d` | [dj4loi7u](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/dj4loi7u) |

Physical counts include the independently executed comparison branches.
They are not steps of one long training run. No quality checkpoint is produced.

## Objective and complete-update reference

`ObjectiveForwardAdapter.forward` calls the canonical `model.loss_sums` and
returns one attached scalar objective, with detached diagnostics. This supplies
the forward entry point a future DDP wrapper needs. For each objective separately,
the intended local scaling under default DDP gradient averaging is
`world_size * objective_weight * local_loss_sum / global_target_count`.
The GPU checks here use world size 1; CPU rank-averaging tests establish the
algebra without exercising real reducers or collectives. Existing FBT weighting,
`base + gamma * mean(extra passes)`, is preserved; gamma is 1 in these cases.

Two same-shaped microbatches have independent supervision masks. The second
has no latent or KL targets; its CE targets remain active. Thus a shared CE
denominator could not silently substitute for the auxiliary denominators.

| Case | Expected gradient tensors | CE counts, microbatch 1 / 2 / global | Latent counts | KL counts |
| --- | ---: | --- | --- | --- |
| RT | 65 | 682 / 510 / 1192 | 0 / 0 / 0 | 0 / 0 / 0 |
| Combined | 71 | 682 / 510 / 1192 | 680 / 0 / 680 | 764 / 0 / 764 |

For both cases, all local/per-pass loss records and raw gradients are bitwise
equal for canonical versus adapter on one batch and for same-order accumulated
backwards. An independent reference sums completed microbatch VJPs on CPU;
that comparison is also bitwise equal, with zero global L2, worst tensor L2 and
maximum-relative errors. Its prospectively fixed budgets remain 2e-6, 2e-6 and
1e-5 respectively; no tolerance was relaxed. Each gate independently checks
the expected active parameter-name set, not merely agreement between two paths.

Two canonical accumulated updates and two adapter updates start from identical
full weights and RNG. Each optimizer update uses two B2 microbatches, global
batch 4, clipping to 1 after both backwards, fused AdamW at LR 1e-5 with a
two-update warmup, betas (0.9, 0.95), epsilon 1e-8 and matrix decay 0.1.
The canonical optimizer is released before restoring initial weights in place
and constructing the adapter optimizer. Every corresponding update has exact
metrics, batch hashes and full model/Adam/scheduler/counter/RNG digests. All
expected gradients participate; probe weights change and parameters/moments
remain finite. This compares complete updates, rather than inferring optimizer
equivalence from raw-gradient agreement alone.

This eager adapter preparation deliberately uses **no CUDA graphs**. Graph
reconstruction is checked separately below.

## Save, reload and recapture

Each recovery case executes two preparation updates, saves an update-boundary
checkpoint, then executes two reference continuation updates. A reconstructed
model and optimizer reload the boundary state and execute the same two updates.
This is 6 physical updates across preparation/reference/restored branches 2/2/2,
with logical counters advancing from 2 at the saved boundary to 4 at either end.

The boundary loss/raw-gradient digests, restored model/optimizer/scheduler,
RNG, input cursor, subsequent update metrics and final state are exact. All
graph/eager loss and full-gradient comparisons pass bitwise, including comparisons
after graph release. Capture preserves RNG and expected gradient ownership.
Clipping, Adam and the scheduler remain outside the forward/loss/backward graph.

**Both continuation branches rebuild their CUDA graphs.** This validates
save/reconstruct/reload/recapture and an exactly reproduced continuation. It does
not compare a rebuilt branch with an uninterrupted live graph. Checkpoint/RNG/
cursor recovery is world size 1; distributed recovery remains to be tested.

After successful verification, disposable full training-state checkpoints were
deleted with matching SHA256/size receipts: 14,154,866,857 bytes for RT and
15,214,707,013 bytes for combined. The summary checks that those files remain
absent. The original pretrained checkpoint is retained by its existing
[O1 storage receipt](../olmo1b-o1/storage-receipt.json); temporary diagnostic
weights are not uploaded as new quality artifacts.

## The integration failure and bounded fix

The first eager canonical RT forward passed an explicit all-valid attention mask
to ordinary Flash SDPA. This environment rejects non-null Flash masks, so it
raised `No available kernel` before any adapter comparison or optimizer step.
The established prepared graph path already supplied implicit causal attention,
which is why recovery could proceed independently.

The correction adds opt-in `OLMoFBT.forward(..., full_valid_causal=True)`.
Normal input, RoPE-position and document validation runs first; the option
requires a boolean flag, all tokens valid, no cache and no packed documents.
Only then does each fresh stack call omit the redundant mask and use implicit
causal attention. Default eager behavior is unchanged. Every canonical and
adapter branch in the corrected harness uses and records the same option.
CPU explicit-math forward/all-gradient comparisons cover ordinary/RT,
standalone/FBT, tiled/non-tiled references and offset positions, together with
rejection and default-preservation tests. Recovery remains pinned to its
previous runtime; its prepared execution path was not changed by this opt-in.

## Evidence and next milestone

The [test ledger](test-results.txt) records overlapping CPU scopes explicitly:
256 tests at the initial integrated runtime; 247 in the isolated dispatch-fix
scope; 52 after the evidence allowlist adjustment; and 166 in the final focused
integrated scope. These counts must not be added together. Full protocols are
in [objective preparation](../olmo-distributed-prepare/protocol.md) and
[graph recovery](../olmo-graph-recovery/protocol.md).

The summary verifies all five finished reports against explicit runtime Git
revisions, source/protocol snapshots, dependency records and the same immutable
O1 checkpoint reference: 300 source pairs checked. The original failure remains
failed. Current-source differences for the older reports are displayed without
rewriting their provenance. The [retention procedure](usage.md) includes reports,
logs, tests, protocols and source snapshots and excludes credentials, datasets
and model/optimizer weights. The accompanying [storage receipt](storage-receipt.json)
records create-only GCS objects and downloaded SHA256 verification.

These same-shaped eager accumulation checks do not validate graph accumulation
or BF16 equivalence to one larger physical batch. Fully valid GPU fixtures do
not establish optimized padding, packed-document, online or prefix-cache support.
The final inventory shows one idle H100 and no remaining GPU process.

Proceed to actual two-GPU eager DDP first: establish rank-local/global count
normalization, independent gradient ownership and reduced/clipped updates;
exercise tied parameters, locally empty terms and asymmetric participation,
especially a predictor used in an earlier `no_sync` microbatch but unused in the
last synchronized one. Then verify same-world-size recovery and graph/collective
compatibility before measuring scaling. Compare equal global workloads separately
from larger aggregate batches. Consider ZeRO 1 for measured optimizer-state pressure,
then ZeRO 2 only if needed; two GPUs do not pool their local activation memory.

The prior [capacity results](../olmo-rt-large-batch/results.md) remain the one-H100
performance reference: RT B192 and combined B128 at T512. These B2 correctness
checks make no new capacity or throughput claim, and do not resolve historical
native/author mixed-precision or RT/FBT modeling qualifications.
