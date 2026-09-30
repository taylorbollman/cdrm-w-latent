# Ordinary-control evidence retention

The completed ordinary OLMo continuation, `native-b32-to128-01`, and its
metadata closure were retained on 2026-09-30 after the report reached
`completed_plan`, checkpoint 128 finished cloud verification, and W&B synced.
No historical execution files changed, no local files were deleted, and this
closure did not reload or rehash the 14.16 GB model/optimizer files.

The authoritative terminal report has SHA256
`9b8f44ce52163eca6679f78f4603fb26e023d6354273f2369749fd5b4798f7f3`.
All 200 historical source hashes matched both the live files and retained source
snapshots. Local checkpoint 128's manifest, state-file size and saved boundary
matched its publication receipt and final report. The run's existing publication
worker had already verified remote state and manifest size, MD5, SHA metadata
and downloaded SHA256 at updates 64, 96 and 128.

The common evidence prefix is:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/`

| Directory / object | Generation | Bytes | SHA256 |
|---|---|---:|---|
|`fbt-stability-baseline/evidence.tar.gz`|1790754067621228|1,825,301|`379a987d8589500cd84caf4d48733f331ab6858007973b82e9d3ffa3ee0e12da`|
|`fbt-stability-baseline/retention-manifest.json`|1790754067891130|56,122|`7af04a96e88c68fb1ca0f859fdb1428576cd7645bf4983d34e6eb2d1c11139bb`|
|`fbt-stability-baseline/storage-receipt.json`|1790754068167664|1,293|`7b53bb0caa32e571cf1f620cf3089317bd5072b9cf0e503434b001bd09849caf`|
|`fbt-stability-baseline-inventory/evidence.tar.gz`|1790754123804005|194,872|`35f55e5001dc4f9bf83a5bfa2b07f9842b0f7e94c1415aa054c2f207a8ceeed2`|
|`fbt-stability-baseline-inventory/retention-manifest.json`|1790754124100200|2,929|`64f0f65f074d1bb1ade4ca69d4c1c5958fb4b14eeca17c76ccbcc8a984b98560`|
|`fbt-stability-baseline-inventory/storage-receipt.json`|1790754124366704|1,320|`09c7d064d142c44e032461e50720046f32acbb8c13aba7f9d946c7b63c74a586`|

Both archives and their manifests/receipts passed size, server MD5, SHA metadata
and downloaded SHA256 verification. The first archive has 224 members, including
the terminal report and 200 source snapshots. The second has 11 members:
the closure analysis/report, exact publication receipts under neutral filenames,
the final local manifest, host launcher/terminal metadata and first retention
receipt. This separate inventory is intentional: the generic evidence archiver
excludes paths with a `checkpoint-*` component, so the three publication receipts
are copied byte-for-byte to `neutral-publications/boundary-000064.json`,
`boundary-000096.json` and `boundary-000128.json` before retention.

The already published full checkpoints remain under the original declaration's
destination:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T204500Z/pilot-async/b/B/native-b32-to128-01/`

There are six state/manifest objects across updates 64, 96 and 128, totaling
42,466,635,984 bytes. The inventory report records every generation, size, hash
and verification result. The final restart pair is:

| Object | Generation | Bytes | SHA256 |
|---|---|---:|---|
|`update-000128/state.pt`|1790753836155274|14,155,077,413|`bba44812dc9acd9674a86e5164f68448a5e68664b4655da2505dd22a14b797b4`|
|`update-000128/manifest.json`|1790753899233106|467,917|`3199abb2d40c1b2f0fec365e1b0bec77b0f0214e84a778c6db41e535faac46b8`|

The local final checkpoint is
`/mnt/localssd/cdrm-checkpoints/fbt-stability/native-b32-to128-01/update-000128`.
Its publication receipt SHA256 is
`b9b5aacb6160f5c562fabc8a79958d04c3ffb3d43dc117c744ff81492561f651`.
W&B's last history row has stale local/cloud checkpoint counters; the terminal
report and receipts are authoritative, as explained in [baseline.md](baseline.md).

Exact retention commands, run from the repository root in the CPU container:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'env -u GOOGLE_APPLICATION_CREDENTIALS python -m scripts.olmo_two_gpu_retain \
    --input-dir .runtime/olmo-fbt-stability/native-b32-to128-01 \
    --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/fbt-stability-baseline \
    --receipt .runtime/olmo-fbt-stability-retention/baseline-01.json'

CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'env -u GOOGLE_APPLICATION_CREDENTIALS python -m scripts.olmo_two_gpu_retain \
    --input-dir .runtime/olmo-fbt-stability/baseline-closure-01 \
    --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/fbt-stability-baseline-inventory \
    --receipt .runtime/olmo-fbt-stability-retention/baseline-inventory-01.json'
```

No `--checkpoint-dir` was supplied: the execution worker had already published
and verified the full checkpoints. The retained closure's baseline prose copy
precedes minor spacing corrections in the repository; its numerical content and
authority are the same. Further experiment conclusions belong in `results.md`.
