# Campaign trainer integration: one-H100 acceptance

2026-09-28. **Complete; ready for two-GPU integration and restart checks.**
The new local training path supports changing right-padding, loss masks,
feedback jitter and normalization counts while reusing one CUDA graph across
accumulated microbatches. The full pretrained NFR fixture matches eager BF16
execution exactly, including two complete Adam updates. This establishes local
execution correctness for the tested fixtures; distributed acceptance is next.

Runtime: `be11ac1`; [usage and next milestone](usage.md),
[frozen protocol](protocol.md), [test ledger](test-ledger.md),
[interruption record](progress.md), [retention receipt](storage-receipt.md).

## What changed

- Ordinary causal Flash can process validated right-padded rows without a dense
  padding matrix. Native RT keeps actual key validity; invalid outputs are zero.
- Fixed-capacity NextLat losses preserve target/teacher detachments and full
  vocabulary chunking while allowing selection masks to change in place.
- Each objective uses its own global logical-update target count. Graph replay
  accumulates gradients; clipping, Adam, schedule and counters run once per
  logical update outside capture. Warmup consumes no updates or data clock.
- Eager DDP now accepts independently validated rank-local jitter per
  microbatch. Two-rank CPU/Gloo reference tests cover unequal and empty slots.
- Checkpoint publication temporarily exposes cleared gradients to the atomic
  saver, then restores graph-owned storage, including after a save failure.

These are opt-in campaign APIs. The graph runner explicitly rejects multiple
ranks. It requires a positive global count for every enabled objective; local
zero counts and entirely empty local microbatches are supported. Different
global parameter participation requires a separate execution plan.

## Actual pretrained GPU checks

One H100 80GB, actual OLMo-1B step60000 (approximately252B source tokens),
revision `81b71efbce6f4dada57c94860301af4298bcd351`. Model weight SHA256:
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
NFR means NextLat + FBT + RT: K4, native RT at layers0/15 on every pass,
1,267,879,936 trainable parameters including fusion and predictor.

BF16 mixed with FP32 parameters/gradients/Adam, TF32 off, forced ordinary Flash
SDPA, native Triton RT, activation recomputation/checkpointing, native reused
RoPE. No FA4 or torch.compile. PyTorch2.13.0a0+8145d630e8.nv26.06, CUDA13.3.

The fixture deliberately uses B2/T16 and three microbatches per logical update.
Lengths change from `(16,7), (2,0), (0,0)` to `(13,5), (6,0), (0,0)`;
tokens, noise and individual loss masks also change. Global CE/latent/KL counts
change from22/22/19 to16/14/13. There is one captured graph,11 warmup backwards
(including initial preparation), one capture backward and13 replays.

| Check | Final result |
| --- | --- |
| Eager versus replay raw gradients, first fixture | Relative L2 0; all elements exact |
| Same graph with changed inputs/masks/noise/counts | Relative L2 0; all elements exact |
| CE/latent/KL sums and normalized objectives | Exact on both fixtures and both Adam updates |
| Entirely empty microbatch | Zero losses and zero gradient contribution |
| Warmup/capture | Gradients cleared; no optimizer updates or RNG changes |
| Buffer/gradient storage across refills and updates | Stable |
| Parameters after two Adam updates | Exact; update-delta relative L2 0 |
| Adam moments and step counters | Exact; aggregate moment relative L2 0 |
| Schedule and data counters | Exact;49 valid input tokens across two updates |
| Runtime provenance | All63 archived source hashes unchanged |

All11 stages passed in each attempt:

| Attempt | Runtime | Elapsed | W&B | Qualification |
| --- | --- | ---: | --- | --- |
| gpu-01 | `0a074ea` | 55.78s | [g3sqgfc0](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/g3sqgfc0) | Passing numerics; diagnostic eager-after-capture emitted an autograd stream warning |
| gpu-02 | `be11ac1` | 54.97s | [7fumon57](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/7fumon57) | Final repeat passes without that warning |

The warning came from retaining the completed Python autograd graph in the
diagnostic scalar output. Detaching that scalar after backward releases the
completed graph while preserving the output tensor storage used by replay.
A targeted lifecycle test and the full second GPU probe qualify this fix.

Final probe peak allocation24.064GiB, peak reservation29.5625GiB, sampled
remaining memory48.752GiB. These include diagnostic state and short sequences;
they are **not** production T1024 capacity or throughput measurements.

## Validation limits

The broad CPU suite passed867 tests in51.58s. After the lifecycle fix,28 focused
tests passed in5.14s; these overlap the broad suite and should not be added as
unique coverage. Tests include all eight campaign arms, independent selected-loss
and explicit-padding references, and two-process CPU/Gloo accumulation/Adam.

The GPU comparison is eager versus captured execution of the same BF16 path.
It does not reopen or resolve independent FP32/native-versus-author precision
qualifications. Both disposable Adam trajectories used max-norm1 clipping;
pre-clip norms were1470.987 and196.283. Raw gradient comparisons precede clipping,
so matching updates do not hide a raw-gradient discrepancy, but these two tiny
updates do not validate learning rate, training stability or model quality.

No production corpus, quality run, T1024 capacity sweep or distributed restart
was performed. Dense graph-compatible losses compute some unused positions;
measure actual workload throughput before making efficiency claims.

## Next hardware boundary

Move to two GPUs now. Qualify real NCCL eager accumulation first, then implement
and validate captured local/final-synchronized microbatches with changing global
counts and empty local slots. Save a completed distributed update, verify GCS
retention, terminate the processes, reconstruct DDP/graphs in fresh processes,
and compare the next update and data cursor to uninterrupted continuation.
Only then calibrate K4/T1024 physical batch and accumulation on the target GPUs.
H200 and a changed rank count need their own bounded acceptance.

Both attempts and source snapshots are retained; no new full model checkpoint
was necessary for these sub-minute probes. Source was committed/pushed during
work. Final container check reports no GPU processes,0MiB used,0% utilization.
