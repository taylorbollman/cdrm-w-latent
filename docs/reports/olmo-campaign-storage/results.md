# SSD checkpoint readiness results

2026-09-29. **Storage execution and recovery acceptance passed.** The new executor
stages checkpoints on Local SSD, retains them in GCS with verified generations
and bytes, and then keeps the newest two locally. Small receipts, journals and
source evidence stay on the persistent disk. Historical checkpoints were not
modified or deleted. Model math and numerical settings are unchanged.

Implementation frozen at `54ae688`, with 172 execution source pins. Read the
[protocol](protocol.md), [operator notes](operator-notes.md),
[test ledger](test-ledger.md) and [storage receipt](storage-receipt.md).
Final PR/merge state is recorded in [progress](progress.md).

## Actual acceptance

| Check | Result |
| --- | --- |
| Tiny two-H100 NFR, three updates with scheduled evaluation | Passed; 20.05 s including initialization, checkpoints and retention |
| Same run versus PR47 persistent-checkpoint reference | All update inputs, losses, gradients and final model/Adam/schedule/RNG/cursor state exact; 2,121 audit checks |
| Stop after update 2, restore pinned GCS objects, resume to 3 | Exact next update and final state; 1,649 audit checks |
| Restore update 2, stop at 2 and repeat scheduled evaluation | Exact state/evaluation; no graph capture, optimizer update or new checkpoint; 1,425 audit checks |
| Native-size asset recovery | 15,215,060,305 bytes streamed to SSD and verified in 70.63 s; no tensor load or GPU execution |
| CPU coverage | 215 distinct tests passed: 102 new and 113 existing regression tests |

The tiny test uses the same T16/B2-per-rank NFR acceptance fixture as PR47:
K4, RT layers0/1, latent plus KL, FP32, prepared CUDA graphs and real packed
Dolma data. Three updates expose240 valid inputs,225CE/latent targets and210KL
triples. This is a lifecycle check, not a quality or throughput benchmark.
Nondeterministic observation durations are excluded from numerical parity.

The reference saved boundaries0/1/2/3. It deleted local0 only after2 was fully
retained and published, and local1 only after3; directories2/3 remain. The stop
segment similarly retained1/2 after deleting its newly owned0. Unknown files,
incomplete checkpoints, resume sources and historical roots are protected.
Keep-two is **per segment**, not a global cleanup of past segments. Cloud copies
of all eight newly saved checkpoints remain available.

The full-size restore used PR47's native NFR update3 checkpoint, independently
pinning its15.2GB state and manifest generations. This demonstrates streaming
recovery and the SSD/persistent-evidence split at actual checkpoint size. It
**does not** establish native GPU parity for the new executor or authorize
resuming a PR47 identity under the new version. Native model math was already
accepted in PR47 and was not rerun here; the storage transition's live-model
acceptance is the small two-GPU case.

## Durability and remaining limitations

A checkpoint becomes a deletion candidate only after local byte/ownership checks,
verified create-only GCS publication, an immutable persistent receipt, a durable
journal and latest-pointer publication. Pruning records its intent before
removing only the two authenticated checkpoint files. Transfer/publication
failure cannot trigger cleanup. A cleanup failure preserves the newer cloud
authority and stops the segment; a fresh process resumes that verified boundary.
CPU tests exercise corruption, path/mount failures, unknown files and interruption.
Actual cloud and GPU checks are bounded successful-path recovery rehearsals,
not an injected VM shutdown during each individual filesystem operation.

No source of historical checkpoints was swept, so the boot disk still has about
91GiB free. Future new states can use the1.4TiB-class SSD working area; account
for a temporary third checkpoint plus protected restore sources. Synchronous
hashing, upload and readback still cost time. The71-second restore is one asset
transfer, not a sustained-storage or training-throughput result.

All small stage evidence and new checkpoints were retained in `gs://fast-chunks`.
W&B runs are synced:
[reference](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ohnga8ik),
[stop](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/c0d0v6xp),
[resume](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/b8hepjmv),
[terminal evaluation](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/58ziwlr2).
Both H100s are idle; no larger training run is queued.

## Next

Prepare the reproducible bounded pilot pool in [data-plan.md](data-plan.md),
then measure the intended packed T1024 batch/accumulation and evaluation memory
allocation. The present equal-source readiness fixture and two-C4-document dev
prefix remain diagnostic fixtures. The proposed new data recipe includes all
Common Crawl strata and explicit held-out membership, with its sampling limits
stated. Startup/exposure fairness and the larger quality budget remain explicit
choices; see [next steps](next-steps.md). BF16 findings and qualifications are
unchanged by this storage work.
