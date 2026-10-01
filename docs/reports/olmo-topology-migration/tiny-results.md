# Tiny topology migration and independent-job isolation

Completed 2026-09-30. The four required tiny migration/restart audits pass, and
both independent-job isolation scenarios pass. This closes the tiny execution
checks; native BF16 behavior is a separate result.

## Migration and fresh-process continuation

The CUDA/NCCL jobs use a tiny NFR model, FP32 master parameters and FP32
computation, real Adam state, a finite schedule, explicit packed-document masks,
row-keyed jitter, and captured local/synchronized backward graphs. The
independent auditor runs on CPU against authenticated reports and saved tensor
artifacts. It checks the global logical update rather than assuming that a new
rank allocation preserves it.

| Required comparison | Optimizer update | Raw-gradient relative L2 | Actual Adam-displacement relative L2 |
|---|---:|---:|---:|
| Two-rank control versus migrated one-rank execution | 2 | 4.835e-8 | 8.198e-7 |
| Migrated one-rank execution versus fresh one-rank restart | 2 | 0 | 0 |
| Live one-rank update 3 versus fresh restart from saved update 2 | 3 | 0 | 0 |
| One-rank execution versus reverse migration to two ranks | 2 | 4.835e-8 | 8.198e-7 |

Values are fractions, not percentages. Adam displacement is the actual change
in saved parameters from the common starting state. It is not normalized by
the much larger norm of all model weights, and tied aliases are counted once.
The declared aggregate FP32 gradient and displacement budgets are both 1e-4;
per-tensor elementwise checks also apply.

Exact checks cover imported model, populated Adam state, scheduler, historical
counters, canonical cursor, real examples, masks, feedback jitter, objective
denominators and learning rates. Graph preparation preserves the imported
boundary and destination RNG assignments. Future physical microbatch increments
are checked against each allocation; logical exposure remains common. Both
same-topology controls are bitwise exact through their final model, Adam,
scheduler, counters, cursor, rank RNG and raw gradients.

Evidence: [independent audit summary](../../../.runtime/olmo-topology-migration/independent-audits/summary.json)
and [readable audit summary](../../../.runtime/olmo-topology-migration/independent-audits/summary.md).

An additional comparison of update 3 across two independently checkpointed
histories is retained as a **strict identity noncomparison**, not an accepted
audit. Each history wrote a different update-2 manifest. That literal manifest
identity is the sole failed check: their root origin and imported, prepared and
final model/Adam/scheduler/counters/cursor/RNG are exact. No training discrepancy
was detected. The auditor was not weakened to accept this case; the direct
live-update-2-to-fresh-update-3 control already supplies the required evidence.

An initial reverse-migration audit also correctly rejected a report hash while
the report was receiving its final rewrite. Repeating only the CPU audit against
stable final bytes passed. The original log is retained; no GPU rerun or source
change was needed.

## Independent-job isolation

Both scenarios launch separate one-rank jobs with disjoint GPU visibility,
containers, rendezvous, outputs, W&B runs and retention identities. Each worker
performs two updates and writes a checkpoint. The victim then stops; the peer
keeps its existing model and captured graphs and performs four additional
updates. Its model must change between checkpoints 2 and 6, so progress before
the fault cannot satisfy the post-fault gate.

| Scenario | Victim worker / launcher exit | Peer outcome |
|---|---|---|
| Cooperative boundary stop | 0 / 0 | Updates 3–6 finite; same live graphs; changed model; checkpoint 6 restored exactly |
| Abrupt exit without graph/process-group cleanup | 73 / 1 | Updates 3–6 finite; same live graphs; changed model; checkpoint 6 restored exactly |

Checkpoint readback verifies finite model and Adam tensors, complete optimizer
ownership and matching Adam clocks. The surviving worker loads the final
checkpoint into a fresh CPU model and Adam instance and compares it exactly.
Owned-container cleanup completed with zero remaining containers in both
scenarios.

- [Cooperative scenario evidence](../../../.runtime/olmo-topology-migration/isolation-controlled-01/supervisor.json);
  [peer W&B run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/6s22bgs5).
- [Abrupt scenario evidence](../../../.runtime/olmo-topology-migration/isolation-abrupt-01/supervisor.json);
  [peer W&B run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/6iw9rh1w).

This deliberately tests a peer paused at a completed optimizer boundary while
the victim exits, followed by successful reuse of its live graphs. It does not
test failure during a peer kernel, loss of one rank within a shared process
group, or a whole-machine failure. The isolation checkpoint restoration is a
fresh CPU object restore in the same surviving worker; the migration tests above
provide the separate fresh-process CUDA continuation evidence.

## Durable evidence

Completed tiny and isolation reports, pinned source snapshots, logs, audit
results and isolation checkpoint files are retained separately from pending
native work. All tiny SSD checkpoint states, manifests and raw-gradient files
are included in the retention inventory.

The create-only destination is
`gs://fast-chunks/cdrm-w-latent/olmo-topology-migration/20260930/tiny-isolation/`.
See [tiny-retention.json](tiny-retention.json) for object generations, sizes,
transport MD5, SHA256 metadata and independent download-SHA256 verification.
The evidence archive excludes W&B internal directories, environment files,
credentials and generated Docker launcher shims. Runtime source snapshots remain
included. These retained results do not establish native BF16 equivalence,
native campaign throughput, H200 performance or eight-rank execution.
