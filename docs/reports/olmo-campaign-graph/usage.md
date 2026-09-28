# Refillable campaign execution and two-GPU handoff

This is an opt-in path for the campaign introduced in PR35. Existing
`StaticFBTTraining`, `PreparedDDPObjective` and `DDPGraphTraining` keep their
fixed-layout behavior; their nonzero-jitter rejection is intentional. Do not
assume a successful local graph qualifies the old DDP graph runner for this
new objective.

## Local interfaces

`OLMoFBT.forward(..., right_padded_causal=True)` and native tiled forwarding
validate a contiguous valid prefix per independent row. Empty rows are allowed.
RT receives true validity; ordinary causal attention does not allocate a
quadratic padding mask. Invalid output positions are zeroed. Cache/prefix
execution, packed rows, left padding and internal validity gaps are rejected.

`PreparedFBTLayout(..., right_padded_causal=True)` fixes physical shape and
canonical arange positions/RoPE. `validate_replacement_batch(batch)` preflights
changed masks without mutation; `load_batch(batch)` refills validity and feedback
eligibility while preserving storage. `validate_batch` still checks equality
with the currently loaded layout. The dynamic lifetime never claims every token
is valid just because its first batch happened to be full.

`DynamicNextLatLayout` in `campaign_losses.py` owns fixed pair/triple indices and
mutable FP32 CE/latent/KL selection masks. `load_batch` updates counts outside
capture. `compute_dynamic_nextlat_loss_sums` returns raw tensor sums only, keeping
teacher/target/readout stop-gradients and chunked checkpointed full-vocabulary
losses. Dense predictor/projection execution keeps empty slots attached to the
same parameters. It computes some unused positions; bucket padding by length
and measure actual workload cost before claiming efficiency. T>=3 and zero
predictor dropout are explicit current requirements.

`CampaignObjective(model, batch, mode=..., global_counts=..., world_size=...,
feedback_noise=..., config=...)` composes those layouts with stable token,
batch-metadata, jitter and coefficient buffers. Its validated refill preserves
graph storage. `forward` returns the normalized scalar objective and detached
raw term sums. Scaling is `weight * world_size / global_count`, per term.
All positive objective weights require positive **global** counts; local counts
may be zero. Global participation changes need a separate plan, preventing
unintended Adam decay from invented zero gradients on a globally unused predictor.

`CampaignGraphTraining(adapter)` is deliberately local (`world_size=1`). It
captures one forward/objective/backward and reuses that graph for every physical
microbatch. Clear gradients only at a logical-update boundary; each replay adds
its contribution, with no extra division by accumulation length. `backward`
replaces previous unstepped diagnostic gradients with one complete accumulated
update. `optimizer_step` checks the token schedule, clips once, applies Adam once
and advances counters/schedule once. External noise tensors must remain unchanged
until the synchronous API call returns. Parameter/gradient/input storage and
execution flags are checked outside capture.

Call `discard_backward()` after retaining an unstepped diagnostic reference and
before capture. Warmup/capture consume no optimizer updates or data clock. Use
`with runner.checkpoint_boundary(): save_training_checkpoint(...)` only at a
completed/discarded boundary: the context temporarily exposes `grad=None` to the
atomic saver, restores the same graph-owned gradient storage even on a save
failure, and rejects execution inside the context. Do not load/mutate state in
that context. On restart load the canonical model/optimizer first and reconstruct
the adapter/graph. Actual distributed restart is the next acceptance gate.

## Eager distributed reference

`EagerDDPTrainer.backward/optimizer_step(..., microbatch_inputs=...)` accepts one
rank-local mapping per physical microbatch, currently only `feedback_noise`.
Put mode/right-padding options in shared `backbone_kwargs`; do not put tensors
there. Noise shape, dtype, device, unit bounds and training mode are checked for
every microbatch in coordinated preflight, before any DDP forward. Tensor values
never enter object collectives. Existing independent global denominators,
`no_sync` accumulation and unused-parameter handling remain unchanged.

## Next two-GPU acceptance

1. Freeze environment/source and launch real two-rank NCCL. Reuse CPU-validated
   data partitions and deterministic jitter keys, with one full effective update
   shared by eager and graph references. Start with small physical shapes.
2. Qualify eager rank-local jitter and unequal/empty local slots on actual GPUs.
   Verify raw global gradients, one complete Adam update, counters and per-term
   global sums against a single-process bounded reference.
3. Implement/qualify actual DDP graph accumulation around `CampaignObjective`.
   Ensure reduction only at the intended accumulation boundary; distinguish
   captured DDP collectives from local graphs plus explicit communication. Validate
   synchronized parameter participation, bucket storage and all-rank failures.
   The local graph runner must not be mislabeled distributed-ready.
   The intended implementation is a separate `CampaignDDPGraphTraining` with
   real DDP constructed on its dedicated stream: `static_graph=True`,
   `find_unused_parameters=False`, `broadcast_buffers=False`, initially no
   bucket views. Warm up synchronized and accumulated execution until addresses
   stabilize. Capture two bodies: forward+backward inside `ddp.no_sync()`, and
   the final synchronized forward+backward. Neither body zeros gradients.
   Zero once outside the graphs, replay local work M-1 times, then synchronized
   work once. Wrapping replay in `no_sync()` cannot remove already captured
   collectives. Begin with separate memory pools and qualify accumulation
   lengths1/2/3, wholly empty ranks and an empty final synchronized microbatch.
4. Save a completed distributed update to ephemeral SSD, verify durable GCS
   retention, stop processes, relaunch and reconstruct DDP/graphs. Compare next
   update and data cursor with uninterrupted continuation. No changed-world-size
   resume is implied. Keep the prior verified checkpoint until upload completes.
5. Only then measure K4/T1024 physical batch/accumulation choices and useful
   throughput on the actual hardware. H200 and the final rank count need their
   own bounded acceptance; no one-GPU capacity sweep is a prerequisite.

Production Dolma preparation, shuffling/bucketing, inference/evaluation, scheduler
extension/cooldown and LR/quality runs remain later campaign milestones. Existing
native-RT BF16 qualifications are unchanged. Preserve 20–30 minute progress
boundaries; warn before any longer non-resumable operation.
