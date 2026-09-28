# Campaign graph integration protocol

2026-09-28, before GPU execution. One H100 preparation only. User will provide
two GPUs when actual distributed CUDA/NCCL update and restart checks are needed.

The new path keeps one fixed physical shape and refills token, validity,
feedback-eligibility, loss-mask, normalization and jitter buffers in place.
Right-padded rows permit ordinary causal Flash without an explicit padding
matrix; native RT still receives actual key validity. Invalid query outputs
are zeroed. No document packing, left padding or prefix cache is accepted by
this fast path. Legacy fixed-layout execution remains unchanged.

The loss adapter uses constant-capacity pair/triple projections and masks
individual position losses. Counts vary outside capture; device coefficients
normalize each term by its own global logical-update target count. This spends
some arithmetic on padding to keep execution shape fixed. A bucketed production
loader and actual throughput measurements remain later work.

CPU checks compare right-padding outputs and all gradients to the existing
masked reference; dynamic losses to canonical selections and detachments;
all-eight-arm accumulated gradients to canonical eager execution; and rank-local
jitter transport with two CPU/Gloo processes. Tests include empty local slots,
zero local KL, changing masks/noise/counts, preserved storage, rejection before
mutation, and checkpoint publication with persistent graph gradients.
These are not GPU/NCCL distributed acceptance tests.

GPU probe uses actual pinned pretrained OLMo-1B, NFR K4, native RT0/15 on every
pass, full CE/latent/KL with changing masks and jitter. B2/T16 (or32), BF16 mixed
with FP32 parameters/gradients/Adam, TF32 off, ordinary Flash SDPA, native Triton
RT and recomputation/checkpointing. No FA4, torch.compile or quality training.

Each logical update contains three microbatches: regular/short rows, a short
row plus dummy, and an entirely dummy microbatch. The second update changes
lengths, tokens, loss selections, jitter and denominators while keeping the
same captured graph and buffer addresses. Test:

- Eager versus replay loss sums and raw gradients for both layouts, using
  ten warmup backwards plus initial gradient preparation before capture.
- Capture consumes no optimizer updates, changes no CPU/CUDA RNG state,
  and leaves gradients zero. Empty microbatches contribute zero raw loss/gradient.
- Changed inputs have a measurable effect; no stale captured masks/noise/counts.
- Two fused-Adam updates versus eager continuation from the identical initial
  parameters/optimizer, with schedule/counters and optimizer moments compared.
- All archived runtime source hashes remain unchanged throughout the probe.

Raw-gradient element budgets retain the previous smoke's `atol=3e-5, rtol=3e-4`;
report actual absolute/relative errors, not only pass flags. This is a replay
comparison within BF16 execution, not renewed BF16-versus-FP32 qualification.
The executable harness fixes complete-update comparison budgets before launch.

Each stage atomically records results and syncs W&B. Use a 15-minute launcher
timeout and a fresh output directory per attempt. CPU snapshots support this
short diagnostic; no new full checkpoint is written to the nearly full persistent
disk. Retain all attempts, source snapshots and logs in verified GCS storage.
If compilation or capture needs a longer unsaveable span, stop and tell the user.

Distributed acceptance remains: real two-GPU eager/captured accumulated updates,
NCCL participation with empty local slots, fresh-process checkpoint restart,
then representative T1024 capacity/throughput. The new local graph runner
explicitly rejects world_size>1; supplying scaled coefficients alone is not DDP.
