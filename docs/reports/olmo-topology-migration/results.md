# Production-state topology migration and remaining native integration

**In progress.** Tiny migration/restart and independent-job isolation are
accepted. The four remaining native integration cells have passed. Native
NFR127→128 migration, cloud restoration and exact restart are still running;
this report will be finalized after their independent audits.

## What changes

A separate migration importer wraps the unchanged replicated checkpoint format
and strict same-topology loader. It authenticates the source before applying an
explicit destination allocation/RNG map. Model, populated Adam, finite token LR
schedule and logical data position are preserved. Historical microbatch counts
are retained; future counts reflect the destination's actual physical slots.
An imported boundary is saved before continuation, making strict fresh-process
restart independently testable. A failed destination process is discarded and
the original source remains available; this is not an in-memory transactional
rollback after an arbitrary device failure.

The historical model, objective, CUDA graph, optimizer and checkpoint core
files are unchanged. New scripts implement the acceptance runner, migration
identity/import, independent auditor, independent-job supervisor, native
component smoke and an explicit campaign-identity CPU retention worker.

## Completed checks

- **109 focused CPU tests pass.**
- Four required tiny CUDA/NCCL audits pass:2→1 and1→2 FP32 raw-gradient relative
  L2 difference4.835e-8 and actual Adam-displacement relative difference8.198e-7.
  Both fresh-process same-topology checks are bitwise exact. See
  [tiny results](tiny-results.md) for the separately recorded strict-identity
  noncomparison between independently written intermediate manifests.
- Cooperative stop and deliberate abrupt exit of one independent GPU job leave
  the peer's live graphs usable for four more parameter-changing updates. This
  tests a peer paused at a boundary during the failure, not mid-kernel failure
  or recovery from a failed rank within the same distributed job.
- Native N/R/NR/FR each complete two changed-input, accumulated updates at
  T1024/B12 per rank on two GPUs (48 real rows/update). Active gradient and Adam
  ownership, finite FP32 state, exact rank replicas, component updates, preserved
  preparation state, graph pointers and clean teardown pass. See
  [native integration results](native-smoke-results.md).

The native component checks use original OLMo weights and fresh modules/Adam.
They are functionality fixtures. Raw gradient norms remain large at R/NR/FR
startup and cause heavy clipping; no learning-quality or long-run stability
claim follows from these two updates. Their timings include full CPU state
inspection and are not throughput measurements.

## Storage correction found during acceptance

The first native attempt saved its imported127 boundary locally, but its cloud
worker rejected the identity schema: SSD staging accepts campaign identity,
while the reused historical pilot worker expects pilot identity. That attempt
was stopped during preparation, with its logs and local checkpoint preserved;
no cloud publication or completed update is claimed. Neither scientific127 nor
128 changed.

The fix uses the existing asynchronous manager's explicit `retain_hook` with a
new CPU worker that validates campaign identity. Historical validators and
manager/storage lifecycle are unchanged. A real SSD→CPU child→GCS→published
recovery-record probe passed, including independent exact-generation downloads,
source/parent-RNG preservation and absence of CUDA/distributed initialization
in the child. The native retry uses this source-pinned worker.

## Native migration and restart

The two-rank control has completed, exited cleanly and published both boundaries.
An independent comparison against the original scientific update 128 report
passes all 14 scalar/full-boundary digest checks: model, Adam, scheduler,
counters, both rank RNG streams, cursor, loss sums, gradient norm and learning
rate agree exactly. This compares authenticated recorded digests; it is not an
additional independent tensor readback of the historical endpoint. The control
objective is 3.296812589 and its pre-clipping gradient norm is 0.944218695.

One-rank migration and cloud restart remain pending. The fixed scope is original
native NFR127→128, T1024,512 real rows per
update, physical B12, K4, RT0/15, latent1/KL0.1, BF16 mixed with FP32 master
weights/Adam and the original finite LR plan. Compare two ranks with one rank;
then restore migrated127 from exact GCS generations and replay on one rank in a
fresh process. Compare raw gradients and actual Adam displacement, with exact
initial model/optimizer/schedule/cursor and identical data/masks/keyed jitter.
Changed-rank BF16 measurements are not assumed bitwise-identical or granted
blanket numerical clearance. The strict same-topology restart must be exact.
No update129 is authorized by this fixture.

Native reverse 1→2 migration is not covered by this matrix; the tiny FP32
fixture covers that direction. See the [operator guide](usage.md) for the
bounded executor, reusable importer, and recovery commands.

## Retention and destination-machine scope

Completed small evidence and tiny checkpoints are verified in GCS:
[tiny/isolation receipt](tiny-retention.json),
[native smoke/storage receipt](native-smoke-retention.json).
Native migration checkpoints use asynchronous verified publication; their final
receipts and raw-gradient artifact retention will be added at closeout.

Eight-rank execution, eight independent jobs, H200 physical batches and the new
provider/runtime require destination-node acceptance. See
[target-node plan](target-node-plan.md). The current tests do not infer H200 or
eight-GPU performance, resolve fresh RT optimization, or extend the NFR science
run beyond its existing128-update endpoint.
