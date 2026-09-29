# Restore a published execution checkpoint

The new CPU-only `olmo_campaign_execution_restore.py` consumes a pinned
`latest-checkpoint.json` produced by the shared execution engine after verified
retention. It needs no completed reference report or successful final run report.
Its only cloud operations are exact-generation reads of the two checkpoint
objects. It does not import torch, deserialize tensors, construct a model,
restore data, execute a resume command or use a GPU.

The receipt is independently SHA256 pinned. It must bind one complete committed
distributed manifest and exactly two create-only, producer-verified objects:
`manifest.json` and `state.pt` under the same `gs://fast-chunks/cdrm-w-latent/`
checkpoint directory. Each object has its own explicit generation, size, SHA256
and MD5. Their generations normally differ; neither a latest-object lookup nor
an inferred shared generation is permitted. Receipt-local paths are ignored.
This helper accepts state payloads up to **20 GiB**, manifests up to **16 MiB**
and publication receipts up to **32 MiB**. It leaves the old tiny recovery helper's
256 MiB limit and source contract unchanged.

Use a fresh destination under the persistent project checkout. Existing output
directories, symlink parents, traversal, foreign buckets, ambiguous object pairs,
unknown identity versions and incomplete publication receipts fail closed.
Available disk is checked before transfer. The selected checkpoint's execution
identity digest, topology, basic counters and rank-cursor agreement are checked
without importing the model-dependent execution contract. The eventual runner
must still verify its own exact expected identity and finite plan.

## Download and publication ordering

1. Preserve the exact input receipt and the restore helper's own source snapshot.
2. Download the pinned manifest to `checkpoint/manifest.json.partial`. Check its
   complete byte size/SHA256/MD5 and require exact metadata equality with the
   publication receipt. It remains uncommitted locally.
3. Stream the pinned state to `checkpoint/state.pt.partial`, using an append-only
   file sink with a strict byte limit and incremental SHA256/MD5. The sink retains
   no object-sized memory buffer. The SDK's default retry policy remains active;
   generation preconditions stay in place, and rewind or transcoding is rejected.
4. Verify digests, flush/fsync bytes, and recheck source/input authority. Publish
   `state.pt` with a no-overwrite hard link, then publish the exact downloaded
   `manifest.json` last. Directory entries are fsynced.
5. Publish `report.json` with status
   `checkpoint_assets_verified_launch_pending`, the exact generations/pins and
   restored checkpoint path.

Interrupted or corrupt files remain visibly partial. No committed checkpoint
manifest is installed on a failed manifest/state download. A failure after valid
state promotion but before manifest promotion leaves an uncommitted state file.
The helper records failures where possible and never deletes or overwrites an
earlier recovery attempt. Retry into a new directory. If a later evidence-report
write fails after both checkpoint files committed, `failure.json` distinguishes
the already published checkpoint manifest; the verified byte authority remains
separate from completion of the operational report.

## Operator interface

```bash
cd /home/taylorbollman/cdrm-w-latent
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_campaign_execution_restore.py \
    --publication /workspace/cdrm-w-latent/PINNED-STAGE/latest-checkpoint.json \
    --publication-sha256 INDEPENDENT_PUBLICATION_SHA256 \
    --output-dir /workspace/cdrm-w-latent/.runtime/olmo-campaign-execution/NEW-RESTORE'
```

Replace the illustrative paths and pin with the selected retained boundary.
The output contains `authorities/latest-checkpoint.json`, `source-snapshot/`,
`checkpoint/{state.pt,manifest.json}` and `report.json`. The source inventory is
independent of execution sources: adding this operational helper does not change
the training lineage. Its source snapshots authenticate the downloader, not the
training checkout.

Pass the restored `checkpoint/` directory and returned manifest SHA to the
matching execution runner. Corpus/index restoration, declared original-backbone
dependencies, runtime/topology, sources, optimizer/scheduler ownership, tensor
contents and rank-local RNG remain that runner's responsibility. A completed-plan
checkpoint can be restored; the runner must recognize it and exit without another
update. Download success is not a model evaluation, BF16 clearance, hardware
portability result or authorization for a new campaign.

Focused CPU tests use real JSON/file bytes and generation-addressed fake storage:
no cloud requests. They cover stopped/completed boundaries without run reports,
distinct object generations, malformed authorities, disk/destination limits,
stream corruption/interruption/overflow, manifest-last publication, unchanged
metadata and receipt hashes, and the real SDK retry-to-file path without network.
The parent agent owns any actual operational cloud restore after review/freeze.
