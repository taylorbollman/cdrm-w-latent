# Shared campaign execution acceptance

This milestone implements the frozen campaign objective for B/N/F/R/NF/NR/FR/NFR
through one execution path. It is operational readiness, not a production launch,
quality comparison, new optimizer recipe, or general BF16 clearance. Historical
model, data, graph, checkpoint and numerical acceptance code remains unchanged.

## Actual startup and execution

The separately versioned declaration and pinned resolution bind the finite packed
data prefix, per-arm physical allocation, exact original or adapted startup,
training recipe, runtime and source files. See [contract.md](contract.md).
Native execution currently requires two ranks, FP32 master weights and Adam
states, BF16 mixed forward/backward, ordinary Flash SDPA, native Triton RT,
recomputed RT backward, ordinary activation checkpointing and prepared CUDA
graphs. No backend fallback or automatic schedule extension is allowed.

Original weights with fresh all-active Adam support all eight arms. The adapted
allowlist is NF/NFR with the complete saved fusion128 state and fresh all-active
Adam. Import first uses the historical isolated NF configuration and unchanged
strict importer. A separately checked state-preserving transition activates the
declared packed policy and selected RT mode. No old optimizer, scheduler, RNG or
data cursor is inherited. Historical adaptation exposure remains visible.

One model/objective/optimizer path serves lean and acceptance observations. Lean
records existing losses, norms, clipping estimates, exact counters and cursors;
acceptance additionally hashes inputs, raw gradients and update boundaries.
Neither observer performs a forward pass, optimizer operation or collective.
Full checkpoints still validate model, Adam, scheduler, RNG, counters and cursor.
Observation verbosity and segment stop boundaries are excluded from immutable
training identity. This does not remove the accepted runner's finite/norm checks.

## Recovery and interruption

Save the origin and requested completed boundaries to persistent project storage.
Publish a recovery pointer only after generation-pinned GCS upload and independent
download/hash verification. Save cadence is ten minutes at a completed update
boundary, plus explicit milestones and normal/controlled-stop endpoints; slow
serialization/transfer or arbitrary VM loss can exceed that interval. Keep all
milestone checkpoints locally during this bounded acceptance. Never save unknown
partially applied optimizer updates. External launchers bound CUDA/NCCL failure.

Resume requires a committed manifest with an independently supplied SHA256 and
the full checkpoint bytes. It does not require a successful reference report.
Metadata preflight precedes the unchanged generic tensor/Adam/RNG loader; data
cursor and enabled-loss clocks must match the immutable prefix. Reconstruct DDP
and graphs only after restoration. A fully completed checkpoint validates and
exits without preparing another update. The constructor's original/adaptation
authorities remain required for this initial recovery implementation.

## Bounded acceptance stages

1. CPU checks cover all eight ownership/objective/optimizer selections, real
   packed document boundaries, enabled loss counts, multiple microbatches with
   one empty rank slot, historical import policy transition, strict startup and
   recovery metadata, and observer non-interference in actual CPU/Gloo updates.
2. A separately labeled tiny fixture uses random width-32/two-layer weights,
   native vocabulary, T16, FP32 math/eager RT, two ranks and prepared CUDA graphs.
   Five global rows per update, physical B2/rank, give two microbatches and an
   empty final slot on rank1. Three updates establish complete clocks. This is
   GPU operational coverage, not Triton or pretrained-model numerical coverage.
3. Run uninterrupted tiny references and boundary stops/recovery. Compare
   preserved origin, raw gradients where observed, model/Adam/schedule/RNG,
   counters and cursor exactly. Exercise a cloud-restored stopped checkpoint
   and terminal-checkpoint resume; a successful prior report is not a launch
   dependency. Compare lean and acceptance execution under one identity.
4. Native adapted NFR uses the selected fusion128 startup, K4, RT layers0/15,
   NextLat latent and KL losses, T1024, physical B1/rank and three global rows
   (3,072 valid input tokens) per update. Three updates include two microbatches
   and an empty rank slot. Compare uninterrupted execution with recovery of a
   cloud-restored update1 checkpoint, including Adam present before DDP/graph
   construction. Capture must preserve the restored state exactly.

Resolve declarations after candidate source freeze. Archive every attempt,
including failures, commands and evidence. Any implementation change invalidates
that source lineage and requires new declarations and acceptance stages. Retain
final reports and checkpoint receipts in GCS after processes and W&B close.
Timing separates execution, observation, graph preparation, serialization and
transfer; these heavily instrumented short tests are not throughput benchmarks.
Native B acceptance already exists in the prior milestone. No claim is made
that every original/native combination received another GPU test here.

## Qualifications carried forward

All BF16-vs-FP32 qualifications in the fusion-startup reports remain. Exact
same-precision graph/restart behavior does not establish precision equivalence,
long-run stability, quality, larger-batch capacity or production data readiness.
No Q/K normalization or model math is changed. Evaluation is explicitly deferred.
