# Background checkpoint retention and bounded adaptation preparation

The user authorizes continuing training after a complete immutable SSD save
while upload and verification proceed independently. Preserve all accepted model,
optimizer, graph, data and loss semantics. Historical sources remain frozen;
new engine/entrypoint and host lifecycle carry the explicit transport policy.

One worker owns checkpoint validation, upload, full generation-pinned readback,
publication and verified local pruning. Only immutable paths and JSON metadata
cross that boundary. The cloud SDK runs in a fresh CPU-only process, with rank
variables removed, so it cannot perturb training RNG or participate in NCCL.
Only the main training thread logs W&B or calls distributed collectives.

There is at most one outstanding job. Drain before another save/storage access;
otherwise continue updates and poll at completed boundaries. Synchronous local
save includes the existing complete state snapshot and preservation checks.
Normal stop/terminal drains the final job. A worker failure stops at the next
healthy boundary without claiming a new durable checkpoint. Unknown GPU/update
failure tears down the worker without inventing recovery collectives. Pending
SSD files remain intact. Host filesystem work has no hard real-time guarantee.

Keep the prior verified cloud checkpoint until the next publication succeeds.
VM loss during upload rolls back to that prior checkpoint. Initial fresh runs
have no new durable origin until initial upload completes. Background completion
may precede the main report's next poll; persistent publication receipts/journal
are authoritative. Local and cloud completed-update counters are distinct.

CPU acceptance covers single-job ownership, deep-copied metadata, delayed and
failed workers, timeout/cancel cleanup, terminal drain, source integrity and
old math path preservation. Tiny two-H100 tests compare blocking versus async
retention with the same updates, then cloud-restore a completed boundary and
check exact continuation. Compare actual inputs/raw gradients/Adam/model/RNG/
cursors and evaluation, not nondeterministic elapsed time. No old evidence is
relabeled or overwritten; new runtime sources freeze before GPU acceptance.

A bounded native T1024 NFR check may use the proposed 524,288-input logical batch
with physical B12/rank to measure real accumulation, overlap and host cost.
Different finite schedule/startup/exposure defines a diagnostic, not a learning
cohort. A larger learning run is not required to accept this transport change.

Prepare concrete B and paired NF/NFR pilot declarations, fixed development
membership, finite ceiling/first stop and cost estimates. Explicitly preserve
heavy clipping and worse later-pass CE as open adaptation concerns. Monitor
per-pass CE, gradient norms/clip coefficients, auxiliary losses and backbone
pass-one damage. Use normal pilot observations first; localize losses only for
a concrete persistent issue. Do not reopen broad precision/QK/architecture work.

Checkpoints remain on local SSD and gs://fast-chunks; small source/evidence on
persistent disk and GCS. Save/push/retain progress every20–30minutes. A learning
campaign and cross-topology/H200 clearance remain outside this milestone.
