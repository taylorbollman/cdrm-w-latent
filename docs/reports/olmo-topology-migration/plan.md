# Remaining two-GPU readiness work

Authorized 2026-09-30 after allocation PR57. This work qualifies checkpoint
migration, fresh-process continuation and independent-job failure isolation.
It does not extend the scientific NFR experiment beyond update128 or measure
H200/eight-GPU performance.

1. Add explicit migration contract/import wrappers around the frozen replicated
   checkpoint format. Preserve model, tied ownership, populated Adam, complete
   scheduler/token clocks, original logical cursor and document masks. Map RNG
   streams explicitly and keep jitter keyed to row occurrence and logical step.
   Preserve past microbatch counters; count future microbatches under the new
   allocation. The historical same-world-size loader remains unchanged.
2. Use tiny FP32 CUDA/NCCL fixtures to test unchanged topology,2→1,1→2 and fresh
   restart after saving the migrated boundary. Include uneven work, dummy rows,
   separate CE/latent/KL masks and a live-graph reference. Compare raw gradients
   and actual parameter updates against the matched control.
3. Authenticate saved native NFR127 and compare bounded2-rank and1-rank updates
   to128. Save migrated127 first, then resume it in a fresh process and check
   the same-topology result. Keep T1024/effective512/B12 perGPU/K4/RT0,15/
   latent1/KL0.1/BF16-mixed/FP32master+Adam and the existing finite LR plan.
   Cross-topology numerical differences will be measured and assessed rather
   than assumed bitwise-identical or divided by the full pretrained norm.
4. Verify independent-job isolation with separate one-rank containers. Exercise
   cooperative stop and abrupt victim exit while the peer completes real graph
   updates and retains/restores its own checkpoint. Cleanup is restricted to
   the test's owned containers/process groups.
5. Close remaining native N/R/NR/FR integration gaps with short packed-data
   updates through the same graph/objective core. These are functionality
   checks, not learning, capacity sweeps or architecture comparisons.

All GPU commands run inside the project Docker launcher. One parent owns GPU
scheduling; parallel development/review agents do not launch GPU work. Large
new state/gradient files use volatile SSD, with retained checkpoints and final
artifacts in gs://fast-chunks. New output paths and source snapshots preserve
failed attempts. Code/progress is pushed about every20–30minutes. The accepted
asynchronous save/publication implementation is reused where applicable; a
completed SSD save can be uploaded while graph setup/training continues, and
terminal success requires draining retention. No unverified local state is
presented as cloud recovery authority.

The next hardware-dependent work is target-node runtime/restart acceptance,
H200 physical-batch calibration and eight-singles/four-pairs/eight-rank scaling.
