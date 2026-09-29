# SSD checkpoint readiness protocol

This version adds storage operations to the accepted PR47 trainer. Model math,
precision, data, optimizer, schedule, evaluation and graph implementation remain
unchanged. Historical executors, model core and source authorities stay frozen.
The new execution identity includes this version's sources and declared storage
policy. An older checkpoint is not resumable across the new identity.

Evidence stays under the persistent project checkout. New checkpoint directories
must be beneath `/mnt/localssd/cdrm-checkpoints/<namespace>/<segment>` and the
local SSD mount must exist; no fallback to a boot-disk directory is permitted.
Reject traversal, symlinks, existing run roots and mismatched ownership markers.
Register each new checkpoint destination before save. The existing distributed
saver writes state before its commit manifest. Each completed boundary is retained
synchronously with create-only GCS objects, exact generations and verified bytes.
Small immutable receipts and a durable journal must be published before local
pruning. Keep at least the newest two verified boundaries per segment. Do not
prune a resume source, historical directories, unknown/uncommitted checkpoints,
or anything not owned by this explicitly created run root. Failed transfer or
publication cannot authorize pruning. A prune failure must preserve the new
cloud authority and must not delete its checkpoint. Unknown CUDA/NCCL failures
still require process teardown; resume starts a fresh process.

CPU restore streams independently pinned manifest/state generations to a fresh
SSD destination, publishes the manifest last, and writes small evidence to the
persistent checkout. It never loads tensors, selects a cloud 'latest', or changes
source identity. The runner separately validates topology, runtime, scientific
configuration, source pins and the full model/Adam/schedule/RNG/cursor state.

Acceptance uses the existing real-data tiny NFR fixture on two H100s: three
updates, evaluation after update two, initial and every-update checkpoints.
Compare uninterrupted SSD execution with the previous persistent-storage run;
require identical update inputs, losses, gradients and full boundaries, allowing
only the explicitly added storage policy/source/schema identities. Then stop at
two, restore its published boundary to a new SSD directory and resume to three;
require exact same-lineage continuation. Exercise terminal evaluation-only resume
if needed to cover no new checkpoint publication. CPU tests cover path/mount and
ownership guards, corruption, publication/prune interruption, retention failure,
unknown files, immutable receipt authority and streaming recovery. Native-size
asset restore can validate 15GB streaming behavior without replaying unchanged
native model math or claiming native cross-version resume.

No historical checkpoint deletion, larger training campaign, new startup recipe,
capacity claim, BF16 clearance or data-mixture acceptance follows from this test.
The checkpoint cadence remains a completed-boundary target, not a hard deadline
inside graph preparation, evaluation or a cloud transfer.
