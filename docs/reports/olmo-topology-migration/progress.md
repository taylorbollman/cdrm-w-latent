# Topology readiness progress

2026-09-30: branch feat/olmo-topology-migration from f93e673. User authorized
remaining two-GPU-suited steps. Read plan.md and the detailed implementation map
in ../olmo-gpu-allocation/topology-migration-plan.md.

Two H10080GB verified idle inside the required project container. Boot has
about87GiB free; SSD about985GiB free. Native checkpoints/gradient artifacts
will use SSD; small evidence/code live in the persistent project. No native
learning extension is authorized or scheduled.

Parent owns scripts/olmo_topology_execute.py and GPU scheduling. Parallel work:
allocation_runner owns new contract/checkpoint wrappers and tests;
allocation_acceptance owns independent numeric/restart auditor and tests;
memory_evidence owns independent-job isolation helper and gap assessment.
Historical runtime and checkpoint source inventories stay unchanged.

## 2026-09-30 22:16 UTC — pre-GPU implementation review

- Native NFR127 authenticated in full (manifest and15.215GB state); all422
  historical source entries remain unchanged. Both H100s were idle at preflight.
- Explicit migration/import and independent numerical audit are implemented.
  Destination checkpoints receive a fresh allocation identity compatible with
  the existing asynchronous SSD-to-GCS publication path.
- Review added TF32-off controls, source/precision checks, historical counter
  validation and cancellation-safe retention cleanup before any GPU execution.
- Tiny execution helper CPU tests pass; independent-job isolation review found
  no ownership/cleanup blocker and is adding explicit post-failure displacement
  evidence. Native N/R/NR/FR smoke helper is being prepared independently.
- No new GPU run or scientific continuation has started. Next: freeze sources,
  tiny CUDA/NCCL migrations, native127-to128 replay/restart, isolation and gaps.

## 2026-09-30 22:21 UTC — tiny GPU acceptance and native import

- Four required tiny CUDA/NCCL audits passed:2→1 and1→2 FP32 raw-gradient
  relative L2 error4.835e-8; actual Adam-displacement relative error8.198e-7.
  Fresh-process imported-boundary and live-graph-boundary restarts were bitwise
  exact. A redundant cross-history comparison correctly rejected distinct
  intermediate checkpoint manifest identities despite exact tensor/RNG state;
  this is retained explicitly rather than weakening the strict auditor.
- Cooperative and deliberate abrupt-exit independent-job isolation both passed.
  Peer graphs survived, four additional real updates changed parameters, and
  peer checkpoint bytes and fresh CPU model/Adam restore were verified.
- Native two-rank127→128 control is running from authenticated original127,
  preserving the actual finite schedule and populated optimizer. No update129
  is authorized or supported by this fixture. Later native migration and strict
  restart remain pending. Retention of the completed tiny evidence is underway.

## 2026-09-30 22:27 UTC — native publication integration issue

Native control attempt01 authenticated/imported/saved127 but its asynchronous
CPU worker rejected the identity schema: SSD staging uses campaign identity,
whereas the reused pilot worker hardcodes pilot identity. This was an omitted
end-to-end storage compatibility case, not a numerical failure. Parent stopped
only its authenticated owned container during graph preparation; original127
and128 remain untouched, imported127 remains local, and no publication is
claimed. Logs plus operator-stop receipt are preserved. The fix uses the
existing retention manager's explicit worker hook with a new campaign-identity
CPU worker; historical validators, files and checkpoint formats stay frozen.
Before retrying native, exercise actual tiny asynchronous cloud publication.

## 2026-09-30 22:45 UTC — publication fix accepted

The explicit campaign-identity retention hook passes18 focused CPU tests and
an independently reviewed real SSD→CPU worker→GCS→publication probe (4.823s).
Both state and manifest were independently downloaded by exact generation and
SHA verified; parent RNG/source remained unchanged and the child initialized
neither CUDA nor distributed state. Existing SSD publisher, async manager and
historical validators are unchanged. Main executor now injects this hook and
pins all its dependencies. The full focused suite passes109 tests.

N, R and NR native smokes have completed successfully; FR is still running.
Bare pretrained RT startup has heavy clipping (raw gradient norms about117 and
114 on the first R/NR updates), retained as telemetry rather than cleared as
stable optimization. After FR, retry bounded native migration/restart using
new immutable output paths. Original scientific128 remains the endpoint.

## Native integration completed; production-state matrix running

All N/R/NR/FR native cells completed two accumulated, changed-input updates and
clean shutdown. Independent inspection confirms identical starting tensors
across common modules, exact ranks/counts/gradient/Adam checks and stable graph
storage. Heavy fresh-start RT clipping remains a training-dynamics concern.
The source-fixed native matrix is now sequentially running control2, migrated1,
then a strict1-rank restart from an exact-generation cloud restoration of the
migrated127 checkpoint. All three stop at128. Parent exec session66753 owns
this queue; no other agent launches GPU work. Updated runtime status is in
`.runtime/olmo-topology-migration/session-status.json`.

## Native control update completed; retention and comparisons pending

Retry02 completed its two-rank replay of update128 with finite objective
3.296812589 and raw gradient norm0.944218695. Both replicas passed the runner's
checks. Imported127 is fully verified in GCS; its background retention took
347.9seconds while graph preparation continued. Final128 publication and clean
teardown are still pending. The sequential queue will then perform one-rank
migration, cloud restore and exact fresh-process restart. Independent auditors
will compare finalized reports only. Current checkpoint/gradient files are on
SSD; completed smoke/tiny evidence is already retained and referenced in docs.

## 2026-09-30 23:42 UTC — resumed after chat interruption

Original host queue/session66753 disappeared, but its active migrated1 container
completed successfully. Both control2 and migrated1 now report completed graph
teardown, synced W&B, and verified cloud127/128 publications. Migrated host
launcher exit status was lost; it is not reconstructed or claimed. No container
or queue process remained at inspection. Its final report is authenticated at
SHAa68296027726921d135d66b7cd2911d16fd6c0b018afe01ac66678700abf3cff.
The missing cloud-restore/restart tail now runs detached as PID2104081 via
`.runtime/olmo-topology-migration/native-resume-tail.py`, with persistent status
and per-child exit receipts. No completed GPU update is repeated.
`interruption-adoption-01.json` records the recovery decision. Independent CPU
cross-topology audit is running; strict GPU restart follows exact-generation
restoration. All execution sources remain unchanged.

The recovered CPU cross-topology audit completed/exit0: all23 structural checks
pass, initial tensor artifacts exact, raw-gradient relative L2 6.297229e-8 and
actual Adam-displacement relative L2 3.817187e-7. Measured-only BF16 semantics
remain explicit. Cloud restoration passed exact generation/SHA validation;
native-restart-1r-02 is active. Final same-topology exactness remains pending.

## 2026-10-01 00:04 UTC — final replay finished; publication/audit pending

Cloud-restored restart completed update128, with the full recorded final
boundary and metrics equal to migrated1. Its terminal checkpoint publication,
clean exit and independent raw-tensor restart audit remain pending. No new GPU
cell follows. Read-only final preservation passes428 checks:422 historical
source entries plus original B32/NFR127/NFR128 manifests and full state hashes.
All final evidence retention is prepared as an interruption-resistant CPU task;
it cannot publish final acceptance before strict restart passes.

## 2026-10-01 00:10 UTC — all acceptance complete; native retention underway

Tail completed/exit0 at00:06:22UTC. Independent strict cloud-restart audit exited0
at00:09:31UTC:23 structural checks, all71 gradient/displacement tensors bitwise
exact over1,267,879,936 elements, exact full boundary/RNG/cursor records. Both
H100s verified idle inside the required container; no GPU work queued. Native
control/migrated/restart checkpoint127+128 all published and W&Bsynced.
Final results.md is frozen at SHA51a4e259294de44dd140dcc964bfe58c6efb3eb21e68e81c866316110becd8d2
for evidence publication. Native final retention is authorized at the unique
`native-final-01` GCS prefix, using a detached CPU-only helper. Check its receipt
before claiming final raw-gradient/evidence durability. Historical source/core
files remain unchanged; no additional numeric test or learning extension remains.

## 2026-10-01 00:18 UTC — final evidence verified

Detached CPU retention exited0 after294.52seconds. All6 new objects passed
full exact-generation SHA/MD5/size readback:3 raw gradients totaling15,214,634,961
bytes, a4,528,412-byte archive with442 members, inventory and receipt. Existing
12 checkpoint objects were metadata-reverified and referenced, never reuploaded.
Prefix: `gs://fast-chunks/cdrm-w-latent/olmo-topology-migration/20260930/native-final-01/`.
Published receipt generation1790813898412657, SHA
f33f1dd6e59918ea03462460751476cdac4adf029e4b6a8b2fd6cee9e19d25f9.
Persistent native-retention.json SHA
93d222876c3068930dcdedc0fe2c67f8c0eea99932b4dfd65752ca48d376d486.
Independent tar inspection verified all442 member bytes/digests and exclusions;
results.md remains at its declared final SHA. All checkpoints, audits, failed
attempt and interruption evidence are retained. Both GPUs remain idle and no
GPU job or scientific continuation is queued. PR58 closeout follows.
