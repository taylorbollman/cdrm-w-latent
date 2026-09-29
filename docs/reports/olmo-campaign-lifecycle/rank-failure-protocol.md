# Abrupt rank-exit recovery acceptance

This new opt-in test addresses a remaining operational gap: earlier checks
covered ordinary host failures while both ranks were alive. VM interruptions
can instead remove a rank without executing cleanup. It changes no model,
optimizer, data, graph or checkpoint math, and does not modify frozen helpers.

Run three fresh two-GPU stages using `olmo_campaign_rank_failure.py` and the
same tiny T16 NFR, real corpus index and three-update plan as the guarded loop:

1. A complete three-update reference, with immutable retained checkpoints.
2. A fresh run with `--inject-rank-one-exit`. After update1's checkpoint has
   been published and verified in GCS, rank1 writes and flushes an explicit
   marker, then exits immediately with code73 before entering update2's
   backward. Rank0 may already be entering that update's collectives.
3. After the failed process group has been removed, fresh processes resume
   the failed stage's last complete update1 checkpoint. Updates2/3 must match
   the uninterrupted reference exactly under its new shared source inventory.

The failed launch must return nonzero under torchrun or its bounded external
timeout. It must not publish an update2 checkpoint or replace the last committed
pointer. Record whether torchrun or the external timeout actually ended it;
do not claim graceful model shutdown on the forcibly exited rank. The fresh
resume is the recovery action. No attempt to reuse a broken process group or
save uncertain in-flight state is permitted.

Use the required project container, online W&B, verified GCS retention, both
NCCL async-error flags 0 and a 180-second external bound with 30-second kill grace.
Root owns both GPUs while these stages run. This is a tiny fault-injection
check, not interruption of an active research trajectory, a literal whole-VM
power-loss test, mid-Adam atomic rollback, H200 qualification or BF16 clearance.

CPU tests verify exactly one intended exit, no second backward on the doomed
rank, unchanged peer/reference paths, authenticated retained boundary before
the exit marker, and preservation of the frozen source inventory. GPU evidence
must additionally establish actual process teardown and exact fresh recovery.
