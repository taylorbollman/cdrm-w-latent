# Two-H100 DDP / CUDA graph implementation audit

2026-09-25. Read-only implementation audit; no GPU launch, installation or runtime edits.
Installed CPU-container inspection: PyTorch 2.13.0a0+8145d630e8.nv26.06,
DeepSpeed 0.19.0, native `ZeroRedundancyOptimizer` available. Parent verified
CUDA 13.3 / NCCL 2.30.5 / two H100 80GB with NV18 peer links. The NVIDIA skill
catalog was checked; no directly useful native PyTorch DDP skill was identified.

## Recommended first graph implementation

Use actual DDP over a new fixed-layout forward/objective adapter. Keep the existing
`ObjectiveForwardAdapter` for eager dynamic masks. Keep canonical parameter names,
optimizer ownership and checkpoints on the original `FBTNextLatLM`.

1. Build a prepared layout before DDP wrapping. The adapter registers the canonical
   model exactly once and keeps the prepared layout as non-module execution state.
   Its `forward()` calls the prepared tensor-only loss-sum path and returns one
   attached scalar objective plus detached metrics. No `.cpu()`, validation,
   data-dependent host branch or collective count calculation occurs in this body.
2. All ranks compute CE, latent and KL counts before capture. For default DDP
   averaging use `sum_t(weight_t * world_size / global_count_t * local_sum_t)`.
   The existing FBT pass weighting is already in each sum; do not multiply counts
   by K. Initial graph scope is one physical batch per update, fixed masks and
   denominators, all positively weighted terms present. Reject layout or objective
   changes instead of silently reusing an incompatible graph.
3. Construct DDP on the capture/warmup side stream with `static_graph=True`,
   `find_unused_parameters=False`, `broadcast_buffers=False`. Preserve ordinary
   checkpointing with `use_reentrant=False`, tied parameters and shared FBT passes.
   For the first correctness capture use `gradient_as_bucket_view=False`; it is
   simpler and leaves substantial room at tiny B2. Later explicitly test the view
   option at useful batches because an extra gradient-sized bucket is material.
4. Execute at least 11 DDP-enabled forward/backward iterations on that side stream,
   no optimizer updates required. Every one must call `ddp_adapter()` and backward
   its returned objective. Clear existing participating gradients in place between
   iterations. Confirm reducer bucket rebuilding has completed, then freeze the
   exact per-parameter gradient addresses and used/unused set. Capture another
   `zero_existing_gradients(); ddp_adapter()["objective"].backward()` on that stream.
5. Replays only replace static token storage, replay forward/loss/backward/NCCL,
   then perform coordinated health checks, clipping and fused Adam outside capture.
   Do not call `optimizer.zero_grad(set_to_none=True)` after addresses are frozen.
   Global gradient clipping uses the replicated reduced gradient norm on each rank;
   do not sum that norm over ranks a second time.
6. Start graph processes with both `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` and legacy
   `NCCL_ASYNC_ERROR_HANDLING=0` before `init_process_group`. Installed header gives
   the TORCH-prefixed variable precedence; an inherited value otherwise defeats the
   legacy setting in the official example. Use an external bounded launcher timeout
   and per-rank phase files because automatic asynchronous abort is disabled for this
   graph compatibility probe. Do not force this setting on eager-only runs.

The existing `StaticFBTTraining._tensor_backward()` is not a DDP integration point:
it invokes prepared model work directly, so wrapping a different module and then
calling `plan.backward()` bypasses DDP forward bookkeeping. Its initial gradient
address contract also precedes DDP bucket rebuilding. Reuse its tensor loss/layout
components and validation ideas; implement a distinct distributed execution plan.

## Why this is plausible, and what remains empirical

Installed `DistributedDataParallel._pre_forward` prepares the reducer and rebuilds
buckets. `_post_forward` prepares backward and traverses returned tensors only for
non-static unused-parameter detection. Static graph caches stable participation,
including permanently inactive fusion/predictor parameters, but cannot be used for
changing masks or dynamically changing auxiliary-loss participation.

Native RT's custom backward creates detached local leaf copies for inner
`autograd.grad` calls and returns the completed VJP once to original parameter
edges. Its inner VJPs therefore do not directly trigger original DDP parameter
hooks. Shared FBT pass edges converge on the same canonical leaves, and ordinary
activation checkpoints are nonreentrant. This makes the design promising; it does
not prove full DDP hook/capture behavior. Real two-rank RT and combined tests remain
essential.

Official PyTorch documentation permits whole-backward capture with sufficiently
recent NCCL, side-stream DDP construction and at least 11 eager DDP iterations.
NCCL requires all participating ranks to capture and replay matching collectives;
one GPU per process avoids the documented multi-GPU single-thread launch deadlock.
Do not use a rank-zero-only graph or launch the second rank later as a separate test.
Do not pass newer docs-only flags to the installed build: its DDP signature still
uses `broadcast_buffers`, whereas current main docs also mention its replacement.

Sources: [PyTorch CUDA graph/DDP contract](https://docs.pytorch.org/docs/main/notes/cuda.html#usage-with-distributeddataparallel),
[DDP options and static graph](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html),
[NCCL graph requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html),
[ProcessGroupNCCL environment controls](https://docs.pytorch.org/docs/main/torch_nccl_environment_variables.html).
Installed source inspected in `/usr/local/lib/python3.12/dist-packages/torch/nn/parallel/distributed.py`
and `torch/include/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp`.

## Bounded test protocol

- First finish real eager DDP coverage for all eight modes, independent per-term
  global denominators, unequal rank counts, locally empty/global active terms,
  globally unused predictor, and predictor used only in an earlier `no_sync`
  microbatch. Static graph is not the workaround for those dynamic-mask tests.
- Tiny graph checks for RT and combined: fixed full-valid layout and stable masks,
  11+ warmups, two different token batches and at least two changed-weight steps.
  Verify exact same-candidate eager/graphed loss and reduced raw gradients if the
  deterministic path supports it; retain any failure instead of loosening budgets.
- Compare rank gradients/parameters/Adam state, exact active and inactive ownership,
  gradient pointer stability and complete optimizer updates. Compare DDP to an
  independent reference summing the same-shaped rank/microbatch VJPs, not only a
  concatenated large batch whose BF16 kernel rounding differs.
- Compare graph continuations after same-world-size checkpoint reload/recapture,
  preserving every rank's RNG and data cursor. Graph handles and process groups
  themselves are recreated; never serialize them as the resume state.
- Then useful physical batches: matched local B64/128 (and RT B192 only if memory
  allows) at T512. Report aggregate input tokens/sec using slowest-rank synchronized
  duration, per-rank tokens/sec, setup/current reserved memory, sampled free memory,
  communication overhead, parameter/state bytes and total global batch explicitly.
- A secondary view-bucket run must verify all parameter `.grad` addresses stabilize
  after rebuilding and survive graph replay, clipping, optimizer and checkpoint
  recapture. No `.detach_()` on bucket views. Log bucket sizes and DDP settings.
- Capture failure must retain per-rank phase/memory/error records and stop the whole
  launcher under timeout. A healthy rank must not blindly enter a barrier after a
  peer has OOMed or exited.

## Fallback if whole-DDP capture fails

The bounded fallback is **graphed local compute plus explicit gradient reduction**,
not a successful DDP capture. It can reuse the single-GPU prepared graph more
closely, but must replace local objective means with correct global denominators.
Use either `local_sum/global_count` followed by SUM, or
`world_size*local_sum/global_count` followed by SUM/world_size; never mix them.
Build a deterministic globally active parameter/bucket order outside capture,
materialize zero local contributions where a parameter is only active remotely,
and preserve `grad=None` for parameters globally inactive so Adam does not decay
otherwise unused weights. Reduce outside replay before clipping/Adam. Avoid adding
one NCCL call per parameter; use validated fixed buckets. No DDP wrapper should
remain with dormant hooks while explicit reduction pretends to replace them.

This fallback loses DDP's backward/communication overlap and needs independent
reference/update/recovery tests. It is useful if it preserves most RT graph speed,
not an automatic final choice. Partial-network `make_graphed_callables` is supported
by PyTorch but would require a more intrusive refactor of the custom RT backward;
it is a later alternative if whole capture and the explicit boundary are unsuitable.

## ZeRO-1 feasibility after DDP graphs

Native `torch.distributed.optim.ZeroRedundancyOptimizer` is already installed and
provides the smallest optimizer-state-sharding experiment. Start with canonical
parameter groups and `optimizer_class=torch.optim.AdamW, fused=True`,
`overlap_with_ddp=False`, `parameters_as_bucket_view=False`; optimizer and parameter
broadcast remain outside the compute graph. Verify canonical `None` gradients,
weight decay, scheduler param groups and unchanged parameter storage. Do not enable
its DDP-overlap hooks or parameter-storage rebinding in the initial experiment.

Its `step()` executes a local optimizer step then synchronizes updated parameter
shards. Thus this is ZeRO-1 state sharding, not activation, gradient or parameter
sharding; two-way Adam moment savings are only roughly 4.4/4.7 GiB for RT/combined
before communication/workspace overhead. It cannot make a B512 local batch fit.

All ranks must call `consolidate_state_dict(to=0)` before rank-zero `state_dict()`.
Consolidation uses CPU retention but collective object-transfer staging can create
additional GPU memory peaks. Perform saving after releasing graph pools if needed,
then recapture. Existing checkpoint ownership/optimizer-class checks need an explicit
ZeRO-aware path rather than masking the wrapper as a standard AdamW.

Installed `ZeroRedundancyOptimizer.load_state_dict` mutates the provided mapping
for nonlocal state and explicitly moves scalar tensors (Adam step) to CPU. Fresh
fused Adam allocates its step tensors on the parameter device. This discrepancy
must be checked in recovery; loading through the underlying optimizer's proper
`load_state_dict` path or an explicit validated step-device normalization may be
needed. Do not claim fused ZeRO resume works based only on eager step success.

DeepSpeed 0.19.0 is available too, but a second runtime/precision/checkpoint wrapper
is unnecessary for the first ZeRO-1 question. ZeRO-2 can be considered only if
measured moment savings are insufficient and communication/memory justify it.

Source: [PyTorch distributed optimizer API](https://docs.pytorch.org/docs/main/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer),
installed `torch/distributed/optim/zero_redundancy_optimizer.py` and
`torch/optim/adam.py`. These feasibility conclusions are source-audit inferences,
not GPU validation.
