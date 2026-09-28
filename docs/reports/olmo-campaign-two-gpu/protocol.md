# Campaign two-H100 integration protocol

Started 2026-09-28 from main `dde3240`. The user supplied two H100 80GB GPUs
and enlarged the boot disk to 1TB. This is a functionality and interruption
recovery milestone, not a learning-quality experiment.

## Scope and order

1. Verify the container, GPU topology and real NCCL collectives.
2. Implement campaign accumulation using separate local and final-synchronized
   DDP CUDA graphs. Normalize each objective by its own global target count;
   zero once, accumulate all slots, synchronize once, and clip/update once.
3. Compare independent eager reference, eager DDP and captured DDP on tiny
   models. Cover unequal and entirely empty ranks, empty final slots, changing
   masks/counts/jitter, and one, two and three microbatches. Use all eight
   campaign arms where practical, including K4 and RT on every pass.
4. Check the actual pinned OLMo-1B ordinary and combined NFR paths at short
   context. NFR uses K4, native RT0/15 on every pass and both NextLat losses.
5. Save at a cleared optimizer boundary, retain the checkpoint in GCS, end the
   processes and reconstruct in new torchrun processes. Compare the next raw
   gradient, loss, Adam update, RNG state and data cursor against continuation
   through the original graph. Do not disguise a reconstruction mismatch by
   rebuilding both sides of the only comparison.
6. Only after correctness and recovery pass, calibrate representative K4/T1024
   memory and throughput. Packing policy and final production mixture are
   separate decisions; current fixtures use isolated right-padded documents.

The actual pretrained probe retains FP32 masters/gradients/Adam, BF16 mixed,
TF32 off, ordinary forced Flash SDPA, native Triton RT recomputation and
ordinary activation checkpointing. No FA4 or new Q/K normalization change.
CPU/Gloo checks do not count as CUDA/NCCL acceptance. Accumulation does not
simulate a larger physical batch for RT utilization.

## Frozen numerical and operational gates

Reuse the previous campaign graph budgets: raw-gradient elements atol3e-5,
rtol3e-4; raw-loss sums atol1e-5, rtol3e-6; objective atol1e-6, rtol3e-6.
Parameter elements atol3e-6, rtol3e-5 and update-delta relative L2 <=1e-3.
Adam moments atol3e-5, rtol3e-4 and aggregate relative L2 <=1e-3. Optimizer step
counters, schedule/data clocks and intended RNG preservation are exact checks.
Report actual errors, including failures; do not loosen limits after results.
Fresh-process exactness is measured separately, with any graph-reconstruction
roundoff identified explicitly. Prior BF16/native-versus-author qualifications
remain open; same-path replay parity does not resolve them.

Preflight invalid layouts on all ranks before mutation/collectives. Keep
parameter participation static, gradients attached for empty slots, and
globally enabled objectives nonempty. Warmup/capture consumes no optimizer or
data-clock steps. Verify replica agreement and graph-owned storage stability.
No optimizer step is included inside the forward/backward CUDA graph.

## Interruption safety

Each GPU stage has an external timeout, a fresh output directory, atomic stage
records, W&B logging and a source inventory. Root alone launches GPU probes.
Save/push source and progress at least every20–30min. Keep model states on SSD
and verify GCS retention before treating them as recovery checkpoints. Publish
state before its checkpoint manifest. Preserve failed/interrupted attempts.
Any unavoidable single unsaveable span over20–30min needs advance notice.

Initial environment: PyTorch2.13.0a0+8145d630e8.nv26.06, CUDA13.3,
NCCL2.30.5, two H10080GB with NV18 peer topology. Boot disk has about400GiB
free; local SSD is empty after migration. Existing retained document shards
can be restored and hash-verified without retokenization.
