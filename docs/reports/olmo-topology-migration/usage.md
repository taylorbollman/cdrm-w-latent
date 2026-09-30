# Topology migration and restart operator guide

See [results](results.md) for completed acceptance and outstanding checks. These
commands reproduce bounded readiness fixtures; they do not authorize extending
the scientific NFR run beyond update128. Preserve completed and failed output
directories and always choose a fresh job name.

## Interfaces and scope

| Interface | Responsibility and limit |
| --- | --- |
| `scripts/olmo_topology_contract.py` | Declares a destination rank count, physical batch, lineage, source pins and RNG mapping. Preserves the source configuration, finite schedule, logical cursor and historical counters. Metadata admits up to64 ranks; this is not64-rank execution acceptance. |
| `scripts/olmo_topology_checkpoint.py` | Authenticates the original manifest/full state and imports model, Adam, scheduler and mapped RNG. It neither advances training nor saves a checkpoint. The caller restores and validates the ordered reader. |
| `scripts/olmo_topology_execute.py` | Bounded acceptance runner. Tiny: FP32, physical B2, at most3 fixture updates. Native: NFR only, physical B12, source boundary127/128 and stop127/128. Native T1024/effective512/K4/RT0,15/latent1/KL0.1 and the existing schedule come from authenticated source configuration. |
| `scripts/olmo_topology_storage.py` | Campaign-identity CPU upload hook injected into the unchanged async manager. Uses verified create-only GCS publication; it does not modify either historical identity validator. |
| `scripts/olmo_topology_audit.py` | Independently compares pinned reports/tensor artifacts: `changed_fp32`, `changed_bf16`, or `same_topology`. BF16 topology differences are measured, not automatically cleared by a tolerance. |

The reusable importer requires a freshly constructed compatible model/optimizer/
scheduler, an initialized process group, cleared gradients, and an explicit
authenticated migration contract. Its returned destination configuration and
fingerprint must accompany the unchanged distributed saver. Subsequent
**same-topology migrated-child recovery uses the unchanged strict loader**;
changing ranks again requires another explicit migration. Preserved ranks keep
their declared source RNG streams; added ranks receive deterministic streams
seeded by lineage/rank. Feedback jitter remains keyed to logical rows/updates.

Native acceptance scope here is2→1 plus one-rank strict restart. Tiny fixtures
also cover1→2 and live-graph-boundary restart. Native1→2, eight ranks, different
hardware/runtime and arbitrary longer continuations require separate acceptance.
The current executor's `resume` path expects a migrated child, not a seed-only
tiny checkpoint.

## GPU launch and reproduction

Run the host supervisor from `/home/taylorbollman/cdrm-w-latent`:

```bash
python3 .runtime/olmo-topology-migration/launch.py JOB_NAME \
  --world 2 --phase migrate --scale native --stop 128 \
  --source /mnt/localssd/cdrm-checkpoints/nfr-stability-128/native-nfr-reduced-64to128-01/update-000127 \
  --pin c9504decfe0e13fc6537979502a165ba73a0db984ed1ed5be997b659b01dd445
```

Replace `JOB_NAME` with a fresh identifier. Use `--world 1` for the matched
migration, with the same source/pin. For a strict restart, use `--phase resume`
and the independently authenticated migrated127 directory/manifest pin, keeping
the child's world size and physical batch. `--stop 127` saves the imported
boundary without an update; `--stop 128` permits only the remaining update.

The retained launcher enters the required Docker environment, checks
`nvidia-smi` and an idle GPU allocation, then starts fresh `torchrun` workers
with independent rendezvous, zero automatic restarts and an external timeout.
It sets both NCCL async-error flags to0 for captured collectives and uses the
installed Flash package. Native jobs have a3,600-second timeout; tiny jobs900.
It is a sequential acceptance launcher, not a concurrent-job allocator. Do not
run CUDA/training directly in the host shell.

Each job writes:

- Persistent report, W&B link and source snapshot:
  `.runtime/olmo-topology-migration/JOB_NAME/`; host log is adjacent as
  `JOB_NAME.log`.
- Large local checkpoint and raw-gradient artifacts:
  `/mnt/localssd/cdrm-checkpoints/topology-migration/JOB_NAME/`.
- Native checkpoint objects:
  `gs://fast-chunks/cdrm-w-latent/olmo-topology-migration/20260930/JOB_NAME/`.

The frozen `.runtime/olmo-topology-migration/native-matrix.py` records the actual
order: `native-control-2r-02`, `native-migrated-1r-02`, CPU cloud restoration of
migrated127, then `native-restart-1r-02`. It uses fixed names and is **not** an
idempotent resume command. `.runtime/olmo-topology-migration/tiny-matrix.py`
records the tiny comparison/restart sequence; tiny source creation used
`launch.py NAME --world 2 --phase seed --scale tiny --stop 1`.
Historical tiny checkpoints require their recorded source snapshot: the later
storage-hook wiring changed source pins even though tensor computation stayed
unchanged. Do not suppress that strict source-identity check.

## Cloud recovery and publication authority

A committed SSD checkpoint can exist while upload is still pending. Cloud
recovery authority is the full **published receipt**, including each object's
exact generation and size/SHA/MD5 verification. Native jobs retain it in
`checkpoint-publications/update-000127.json` and in the `receipt` field of
`async-retention/update-000127/result.json`; the latter must say `published`.
`latest-checkpoint.json` points to the latest verified publication. Terminal
success drains publication before declaring the job complete.

The retained CPU wrapper restores a pinned publication receipt to SSD while
copying only small recovery evidence back to the persistent project:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
bash scripts/docker_shell.sh \
  timeout --signal=TERM --kill-after=15s 900s bash -lc \
  'env -u GOOGLE_APPLICATION_CREDENTIALS python .runtime/olmo-topology-migration/restore-cloud.py \
     --publication .runtime/olmo-topology-migration/native-migrated-1r-02/cloud-source-publication-127.json \
     --pin PUBLICATION_JSON_SHA256 --name FRESH_RESTORE_NAME'
```

`PUBLICATION_JSON_SHA256` authenticates the **serialized publication receipt**;
it is distinct from the checkpoint manifest SHA required by the GPU runner.
The matrix extracts that receipt and records its pin before invoking this
command. Use authenticated retained values, not a guessed pin or an unverified
local file relabeled as authority.

The wrapper checks the actual SSD mount/path, uses the unchanged campaign
restore API, downloads exact object generations, and publishes the restored
manifest only after both files pass verification. The recovered checkpoint is
`/mnt/localssd/cdrm-checkpoints/topology-migration/FRESH_RESTORE_NAME/checkpoint`.
Its persistent counterpart under `.runtime/olmo-topology-migration/` contains
the intent, authorities, source snapshot and restore report or failure record.
`checkpoint_assets_verified_launch_pending` means byte recovery passed; model/
Adam/scheduler/RNG and execution-contract validation still occurs at strict load.

## After VM interruption

1. Read this milestone's results/progress, retained launch logs and the selected
   job's report. Treat `session-status.json` as a location hint, not checkpoint
   authority. Confirm actual process/GPU ownership in the required container
   before starting another job; never assume an interrupted agent stopped it.
2. Recover the selected **published** receipt and its independently retained
   pin. Prefer its exact GCS generations when volatile SSD contents were lost.
   A pending submission, local save alone, or incomplete run report does not
   establish cloud durability. Preserve failure and partial-download evidence.
3. Restore the matching pinned source revision/snapshots and required ordered
   corpus/index assets. Do not bypass source, model, data, precision or schedule
   checks. Runtime/hardware changes need the target-node acceptance in
   [target-node-plan.md](target-node-plan.md).
4. Use fresh restore/job directories. Recover the checkpoint with the CPU
   wrapper, then launch fresh processes: strict `resume` for the same migrated
   allocation, explicit `migrate` for a changed allocation. Do not reuse a
   failed process group or captured graph, reset Adam/warmup, replace historical
   microbatch counts, or extend the128-update horizon with this fixture.
5. Require completed launcher/report/teardown and verified publication, then
   run the independent pinned-report audit. The retained
   `native-audit-watch.py` records exact CLI flags and selected update128:
   `changed_bf16` for control2 versus migrated1; `same_topology` for migrated1
   versus cloud-restored restart1. Preserve gradient artifacts separately:
   checkpoint publication uploads only `state.pt` and `manifest.json`.

Small evidence and tiny checkpoint receipts are linked from [results](results.md).
Final native publication, gradient-retention and audit receipts belong there as
well; a failed attempt remains evidence and must not be overwritten.
