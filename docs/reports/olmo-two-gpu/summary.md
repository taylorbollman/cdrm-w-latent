# Two-H100 milestone summary

2026-09-25. **Draft pending the larger-batch capacity closeout.** Completed
functionality and matched-batch results are summarized here; [results.md](results.md)
contains the detailed scope, failures and pending rows.

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
changed weights. Setup memory can greatly exceed steady memory, so the pending
larger-batch measurements will determine comfortable operating points. Completed
RT DDP reaches **55,768 tokens/s at B128/GPU** and **59,035 at B192/GPU**;
sampled free memory is 24.8 and 10.5 GiB/GPU, respectively. The latter buys
another 5.9% throughput with less room for additions. High setup reserved peaks
include releasable warmup cache; low allocated memory during graph replay also
omits the full reserved graph footprint. The report keeps these measures separate.
Combined B128/GPU and matched ZeRO-1 capacity rows remain pending.

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

**Remaining closeout:** finish larger-batch DDP/ZeRO-1 timings and memory checks,
retain them, update the final inventory and recommend the operating batch and
optimizer. ZeRO-2, bucket-view adoption, dynamic graph layouts and quality runs
remain outside the completed scope.
