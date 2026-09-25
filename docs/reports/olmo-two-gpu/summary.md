# Two-H100 milestone summary

2026-09-25. **Milestone complete.** [results.md](results.md) contains the detailed
scope, measurements and retained qualifications. No quality run is queued.

The native RT stack now passes real two-GPU DDP, CUDA-graph and checkpoint
reconstruction checks in the selected configuration. Both ordinary gradients
and custom RT backward participate in actual captured NCCL reductions. The
combined RT + K2 FBT + NextLat path works with fused Adam and with opt-in ZeRO-1
optimizer-state sharding. These are functionality and efficiency results; no
model-quality training was run.

We retain OLMo-1B step 60,000, approximately 252B pretraining tokens: **16 layers,
width 2,048, with RT at indices 0 and 15**. Combined uses an ordinary bootstrap
pass followed by a feedback/RT pass, plus NextLat training losses. Native RoPE,
ordinary activation checkpointing and the selected Triton RT kernels remain.
BF16 mixed keeps parameters, reduced gradients and Adam state in FP32. Q/K
normalization is unchanged. This does not establish all-16-layer RT performance.

At the same global batch of 128 and sequence length 512:

| Mode | One H100, B128 | Two H100s, B64 each | Speedup | Total GPU time/token change |
| --- | ---: | ---: | ---: | ---: |
| RT | 28,000 tokens/s | 47,384 tokens/s | **1.69×** | +18.2% |
| RT + K2 FBT + NextLat | 12,362 tokens/s | 23,406 tokens/s | **1.89×** | +5.6% |

These measure five complete updates after preparation, including clipping,
optimizer and communication. Own eager/graph raw checks pass before and after
changed weights. Setup peaks, graph-pool reservation and sampled device memory
are reported separately; low replay allocator usage is not the whole footprint.

For architectural development, use **RT DDP B128/GPU** (55.8k tokens/s,
24.8 GiB sampled free) or **combined DDP B64/GPU** (23.4k/s, 30.5 GiB free).
For the fixed larger-batch configuration, **ZeRO-1 RT B192/GPU** achieves
58.2k/s with 14.5 GiB minimum sampled free, and **ZeRO-1 combined B128/GPU**
achieves 24.4k/s with 7.5 GiB free. ZeRO-1 costs roughly 1–1.5% throughput
versus matched DDP and recovers about 4 GiB of net headroom in these probes.
Combined DDP B128 fits at24.7k/s but leaves only3.4 GiB free; it is not the
recommended default for additions. DDP RT B192 is also viable at59.0k/s with
10.5 GiB free. These are physical per-GPU batches; global batches are doubled.

Recovery checks reproduce the next gradients, loss, full optimizer/model state,
data cursor and local RNG draws exactly: actual RT eager recovery passes 24
coordinated checks; combined graph reconstruction passes 42. Actual combined
ZeRO-1 also matches full Adam on fixed gradients and resumes exactly. Recovery
reconstructs model/optimizer/DDP inside the same process group; it does not yet
test a fresh `torchrun` process restart or live-graph checkpointing.

Two scoped optimizer fixes matter. The ZeRO-1 loader now restores the native
local Adam shard with correctly placed fused step counters and no duplicated
outer state. A buffer-view conversion removes a slow sender-side byte-copy
path during consolidation while preserving checkpoint bytes and optimizer
math. The corrected full combined checkpoint save takes **83.4 seconds**,
including consolidation, disk writing and hashing; GCS upload is separate.

This milestone retains an additional numerical qualification: the independently evolving combined BF16
comparison passes raw-gradient budgets and has exact rank replicas, but its
second update misses strict tolerances in **20 parameter and 7 moment tensors**.
The largest parameter difference is **1.38e-6**. Both complete updates pass when
compared from identical canonical starting state. This supports fixed-state
distributed correctness while leaving longer-trajectory equivalence unresolved.
The original failure and four corrected setup/restoration attempts remain in
the evidence; tolerances were not relaxed. The older native-versus-author
BF16 compatibility qualification also remains open.

RT executes 1.177B parameters. Combined executes 1.268B during training,
including 8.389M fusion and 82.727M training-only predictor parameters; its
deployable architecture is 1.185B. The full report separates active architecture,
resident dormant tensors, optimizer state and analytic matrix-FLOP estimates.

Sources, reports and approximately 13.2–14.2 GiB full checkpoints are progressively
verified in `gs://fast-chunks`; large local checkpoint copies are cleaned only
after retention verification. W&B runs are in
[pretrained-fbt-rt-nextlat](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat),
group `olmo-two-gpu`. [Usage instructions](usage.md) describe the exact runtime,
checkpoint and launch contracts.

All 31 stage attempts are retained:26 passed and five failed. The source/archive
audit verifies4,492 pinned source pairs and all31 stage receipts, including eight
checkpoint stages. See [storage receipt](storage-receipt.md) and
[test ledger](test-ledger.md).

**Next review decision:** select the next workload and execution point. Before a
long campaign, rehearse a fresh-process restart with its real data cursor. There
is no need to add ZeRO-2 or chase physical B512 first. Bucket-view adoption,
dynamic graph layouts, new RT-layer selections and quality runs remain outside
this completed scope.
