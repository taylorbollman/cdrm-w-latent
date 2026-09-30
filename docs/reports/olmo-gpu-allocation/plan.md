# GPU allocation readiness: one job per GPU versus distributed jobs

Authorized 2026-09-30. This milestone is engineering readiness, not a learning
comparison or an extension of the completed NFR128 experiment.

## Sequence of milestones

1. **Current two-H100 milestone:** use a versioned entrypoint around the accepted
   campaign/DDP graph core with explicit rank count. Check tiny one/two-rank
   normalization and updates, then compare two concurrent single-rank jobs with
   one two-rank job for ordinary OLMo (B) and all-three (NFR). Keep real examples
   per update, sequence length, losses, precision, physical batch per GPU, and
   inherited model/Adam state fixed within each comparison.
2. **Portable restart:** explicit checkpoint topology migration, preserving model,
   Adam, logical data position and LR/token clocks; bounded fresh-process restart
   and failure isolation checks. Changing reduction order/rank RNG layout is not
   a promise of bitwise trajectory continuity. Do not weaken the historical
   same-world-size loader or silently relabel existing checkpoints.
3. **Target machine acceptance:** software/kernel and restart smoke on H200 or
   H100; physical batch/memory calibration; actual eight singles/four pairs/one
   eight-rank comparison, including shared CPU, SSD and upload contention. The
   best allocation may differ among the eight experimental conditions.

## First milestone protocol

- Native OLMo-1B, 16 layers, width2048, original tied embedding/RoPE architecture.
- B checkpoint: saved adaptation update32. NFR checkpoint: saved reduced-KL
  update128. Each condition compares its own identical starting state across
  layouts; differences between B and NFR are not causal quality comparisons.
- T1024, 512 real packed training rows =524,288 input tokens per logical update.
  Count tokens once, even with K4. Preserve true document masks for NextLat.
- Physical batch B32/GPU for B; B12/GPU for NFR. NFR uses K4 FBT, native RT0/15,
  latent coefficient1, KL0.1. BF16 mixed, FP32 master/Adam, fused AdamW, accepted
  activation checkpointing and captured local/synchronized backwards.
- Graph preparation retains existing warmup. Two complete optimizer warmup
  updates precede four measured updates. Extend measurement only if instability
  in timing requires it. Every measured update must be finite.
- Clone saved weights and populated Adam into new benchmark processes. Declare
  a fixed saved LR and fresh benchmark counters/data-prefix cursor. These are
  explicitly disposable performance clones, not resumed learning trajectories;
  they do not overwrite checkpoint128 or advance its schedule/cursor.
- Two independent single-rank jobs use disjoint GPU visibility, output paths,
  process groups and rendezvous ports. A ready/release barrier aligns measured
  windows after both have warmed. Report actual window overlap.
- Timing separates setup/import/capture from full update and selected compute
  regions. Report per-job latency, node aggregate useful tokens/sec, tokens/sec
  per GPU, allocated/reserved peaks and sampled free memory. Do not extrapolate
  measured two-GPU rates into claimed eight-GPU measurements.
- Tiny fixture checks global sample membership, positive denominators, dummy-row
  masking and same-precision update behavior across rank counts. Existing BF16
  numerical qualifications remain; no new broad precision sweep.

## Interruption and retention

Code and incremental reports live under the persistent project directory;
original learning checkpoints already have verified GCS copies. Benchmarks write
progress per update; retain the final small evidence and source snapshots in
gs://fast-chunks. Disposable benchmark weights need not become new large cloud
checkpoints. A VM interruption can require repeating at most the affected short
benchmark cell; do not rerun completed cells automatically. Save/push work at
roughly20–30minute intervals. Production asynchronous checkpoint transport is
unchanged and is not included in a steady-compute throughput claim.

## H200 assessment

Distinguish memory capacity limiting physical batch from bandwidth limiting
individual kernels. Extra capacity does not by itself imply proportionally
higher throughput. Use existing physical-batch slopes and memory headroom as
suggestive evidence; see memory-assessment.md. H200 hardware measurements remain
necessary before choosing batch sizes, checkpoint policy or node allocation.
