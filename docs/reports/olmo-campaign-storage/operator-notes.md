# Storage execution and recovery

Use `scripts.olmo_campaign_ssd_execute`, with the same pinned PR47 declaration
and resolution contracts, in the required GPU container on two H100s. Add a
fresh SSD checkpoint root:

```text
--output-dir /workspace/cdrm-w-latent/.runtime/<campaign>/<segment>
--checkpoint-root /mnt/localssd/cdrm-checkpoints/<campaign>/<segment>
```

Both the evidence directory and checkpoint segment must be new. The mount and
all path components are checked; a directory on the boot filesystem is not an
acceptable substitute for the SSD mount. Physical/model execution settings are
unchanged. Native manifests must declare `keep_local_completed` in2..32; the
tiny acceptance fixture uses2. Changing retention count changes execution
identity. A resumed segment gets a new SSD root and protects the supplied resume
source. Keep-two applies **per segment**, not globally across past segments;
this implementation does not reclaim completed historical segment roots.

The small persistent directory contains `ssd-journal.json`, immutable
`checkpoint-publications/update-NNNNNN.json`, and `latest-checkpoint.json`.
Each receipt pins state and manifest object generations, sizes, MD5 and SHA256.
An interrupted run need not have a successful final report: a fully published
receipt is sufficient to start verified asset recovery. Do not infer authority
from an SSD directory name, an incomplete manifest or an unverified cloud list.

Run streaming asset recovery in the CPU-only container:

```text
python -m scripts.olmo_campaign_ssd_restore \
  --publication <persistent-receipt.json> \
  --publication-sha256 <independent-sha256> \
  --output-dir /workspace/cdrm-w-latent/.runtime/<campaign>/<restore-evidence> \
  --checkpoint-dir /mnt/localssd/cdrm-checkpoints/<campaign>/<restore-name>
```

The restored state is never deserialized by this command. Resume the **matching
executor/version** with `--resume <restored-checkpoint-directory>` and the
receipt's `--resume-manifest-sha256`. The runner validates source, runtime,
configuration, topology, counters and complete model/Adam/schedule/RNG state.
PR47 assets can be recovered by this helper, but cannot resume under the new
SSD executor identity merely because their bytes are valid. The cross-version
acceptance comparison checks unchanged math; it does not authorize migration.

The external launcher still needs a bounded timeout and both NCCL async flags0.
Use `CDRM_DOCKER_GPUS=none` for asset restore and retention; unset stale
`GOOGLE_APPLICATION_CREDENTIALS` when using the instance's existing credentials.
Do not print `.env` or credentials. Preserve source snapshots and receipts in
GCS as well as the persistent project.

If retention fails, training stops before another update and no older checkpoint
is pruned. If pruning fails after publication, the newer cloud receipt remains
authoritative; inspect the durable journal. A partially pruned old directory is
not a valid resume source. This implementation deliberately does not sweep
unknown/incomplete files or automatically restart a crashed storage manager.
A new process/segment resumes the last verified boundary. Old owned SSD roots
may require a separate explicit cleanup operation later.

Capacity must include the temporary third completed checkpoint before old data
can be pruned, plus any protected restore source. Native full checkpoints are
about15.2GB each; initial optimizer-empty checkpoints are smaller. Keeping two
is not a promise of only30.4GB peak disk usage. Retention is synchronous and
fully verified; it does not promise faster transfer. Record checkpoint hashing,
write, cloud readback and cleanup separately from useful training tokens/s.
