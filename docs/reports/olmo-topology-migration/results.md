# Production-state topology migration and remaining native integration

**Completed, 2026-10-01.** Explicit checkpoint migration, fresh-process restart,
independent-job isolation and the four remaining native integration checks are
accepted in the bounded scope below. Native two-to-one-rank differences are
negligible, and the cloud-restored same-topology restart is bitwise exact.
Both H100s are idle; no GPU job or scientific training extension is queued.
Here N means NextLat, F means FBT feedback passes, and R means native temporal RT.

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
- **428 preservation checks pass:** all 422 historical source entries and
  original B32/NFR127/NFR128 manifests plus full checkpoint state bytes remain
  unchanged. Large state verification uses streamed SHA256 on the original
  files; no model execution or checkpoint rewriting occurs.
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

One-rank migration has completed. All 23 independent structural checks pass,
including exact common initial tensor artifacts, input/mask/jitter identities,
preparation preservation and logical counters. Raw-gradient relative L2
difference is **6.297229e-8**; actual Adam-displacement relative L2 difference is
**3.817187e-7**. Corresponding cosines are 0.999999999999997 and
0.9999999999999275. All 1,267,879,936 unique active parameter elements are finite.
These measured differences are negligible for this specific replay and support
operational rank migration. They do not compare BF16 against FP32, qualify a
different physical batch, or establish long-run trajectory equivalence. The
auditor deliberately records changed-rank BF16 as measured-only rather than
granting an automatic precision pass.
The largest per-tensor relative differences are 1.276375e-7 for gradients and
2.213100e-6 for Adam displacement, both at the tied embedding tensor; maximum
absolute differences there are 7.450581e-9 and 1.490116e-8, respectively. Thus
the small aggregate errors do not hide a large relative tensor outlier.

| Component | Gradient relative L2 difference | Actual Adam displacement relative L2 difference |
| --- | ---: | ---: |
| Backbone | 6.540097e-8 | 3.899200e-7 |
| Fusion | 2.616252e-9 | 6.809261e-8 |
| NextLat predictor | 2.250211e-9 | 9.143766e-8 |

Historical physical microbatches remain 5,588 at import, then become 5,632
(two ranks, +44) or 5,631 (one rank, +43). Both consume exactly 512 real rows /
524,288 input tokens with CE/latent/KL denominators 523,776 / 522,852 / 521,417
and LR 0.0002 in all groups. The reported loss and raw-gradient norm are exactly
equal. The migrated host launcher's exit status was lost during a chat
interruption; completed producer report, graph teardown, W&B sync and both
verified checkpoint publications were authenticated before adopting the result.
The separate interruption receipt preserves that qualification.

The exact-generation GCS restoration and strict fresh-process restart passed.
The independent auditor reports **zero difference and bitwise equality** for
all raw-gradient and actual Adam-displacement tensors. Initial and final
boundary records, including model/Adam/scheduler/counters, rank RNG and cursor,
are exact; graph preparation preserved the complete imported state. The
restart exited 0 after verified publication and clean teardown, with W&B synced.
Checkpoint serialization/file hashes need not match: the exactness claim is
about tensor state and execution, not incidental serialized provenance.

The fixed scope is original native NFR127→128, T1024, 512 real rows per update,
physical B12, K4, RT at layers 0/15, latent weight 1 / KL weight 0.1, BF16 mixed
with FP32 master weights/Adam and the original finite LR plan. Exact initial
model/optimizer/schedule/cursor, data/masks/keyed jitter and physical batch were
preserved. Changed-rank BF16 measurements do not grant blanket numerical
clearance. No update 129 occurred or is authorized by this fixture.

| Cell | Online W&B | Final report SHA256 |
| --- | --- | --- |
| Two-rank control | [f3xumz84](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/f3xumz84) | `eaf1a0eca1603cf62c01a8fff7d8cfb9fc384fa1e4a58adc6e0688da522fe2f6` |
| One-rank migration | [du7ptlwr](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/du7ptlwr) | `a68296027726921d135d66b7cd2911d16fd6c0b018afe01ac66678700abf3cff` |
| Cloud-restored one-rank restart | [ricgnpga](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ricgnpga) | `cc10cec584142e9bedf8d68f810d91d2ca2799e8c652fef247f6660f4dbf1a05` |

Independent audit artifacts under `.runtime/olmo-topology-migration/independent-audits/`:

- `native-control-vs-historical128-json.json`: all 14 historical replay checks pass.
- `native-control2-vs-migrated1-u128-recovery01.json`: measured BF16 comparison,
  SHA256 `0fbed46c72d936de5c68a78ed5d5bfb53adb7e7f877d4995fbb4a9865d65ff46`.
- `native-migrated1-vs-cloud-restart1-u128-recovery01.json`: accepted bitwise restart,
  SHA256 `b8616ab3310372e17b30e491b06b299d25ae6769f7df83f769cab1463f082b3f`.

Native reverse 1→2 migration is not covered by this matrix; the tiny FP32
fixture covers that direction. See the [operator guide](usage.md) for the
bounded executor, reusable importer, and recovery commands.

## Retention and destination-machine scope

Completed small evidence and tiny checkpoints are verified in GCS:
[tiny/isolation receipt](tiny-retention.json),
[native smoke/storage receipt](native-smoke-retention.json).
Native migration checkpoints use asynchronous verified publication; their final
receipts, raw-gradient files, cloud restoration, interruption/failure evidence,
independent audits and source snapshots are catalogued by the separately
published [native retention receipt](native-retention.json). That receipt is
created only after final evidence publication and exact-generation verification;
the six native checkpoint states are referenced rather than uploaded again.

Eight-rank execution, eight independent jobs, H200 physical batches and the new
provider/runtime require destination-node acceptance. See
[target-node plan](target-node-plan.md). The current tests do not infer H200 or
eight-GPU performance, resolve fresh RT optimization, or extend the NFR science
run beyond its existing128-update endpoint.

No further two-H100-only acceptance is recommended before the move. Expanding
native rank count and qualifying independent-job interference now belong on the
actual destination node, together with its runtime, credentials and physical
batch/memory measurements. The small reverse-rank fixture already exercises the
added-rank RNG mechanism; it does not replace that destination acceptance.
