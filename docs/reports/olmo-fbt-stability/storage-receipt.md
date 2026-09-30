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

## Completed acceptance and origin curves

The three completed tiny GPU executions, their two independent audits, the
CPU cloud-restore receipt, and the immutable update-0 curve summary are also
retained. Every execution report is `completed_plan`, W&B is synced, final
update 3 is published, and no checkpoint worker remains pending. The restore
report intentionally records its earlier `checkpoint_assets_verified_launch_pending`
state; the later resumed execution and restart audit establish that its downloaded
update-2 checkpoint was subsequently used successfully.

The insertion audit passed 8,531 recorded checks; the cloud-restart audit passed
10,150. These counts describe report checks, not independent statistical tests.
The archives preserve their exact reports. The separate inventory also retains
the five auditor source files against the hashes in both audit reports.

All destinations below are fixed directories under the common evidence prefix
above, with `fbt-stability-` prepended to the stage name. Each directory contains
`evidence.tar.gz`, `retention-manifest.json`, and `storage-receipt.json`. Every
object passed size, server MD5, SHA metadata and downloaded SHA256 verification.

| Stage | Members | Archive generation | Archive SHA256 |
|---|---:|---|---|
|`tiny-old-control-01`|221|1790754584184360|`d9fc884fff235232806dcde603d774d439bbb51bd839b466ca5cce3f8d025119`|
|`tiny-stability-01`|233|1790754586166430|`cb7c762b8d0a434e7db8a2bd5257dc6eb0df92f671f270c628a32337a7c2f3eb`|
|`tiny-resume-01`|219|1790754587964371|`d2eaf202511ffb0665417bdece44030e71e15a7b6a76cc557adf8c23dcd48fab`|
|`tiny-insertion-audit-01`|1|1790754589301712|`504ccf06dbb902cc0a4b25356beda872304b5d74f7a4c63edf1f770de60b8c39`|
|`tiny-restart-audit-01`|1|1790754590622640|`2e638d33cd3fe554c9801ec8347263ee151faf5b5a30d505b9994ce3908eb3d3`|
|`tiny-restore-01`|13|1790754591933204|`15f10ada041f0253a41730be3cc2f75bf207ede2fffbd9377b9c6d682723c2d1`|
|`summary-update000000-01`|8|1790754593381403|`8b3c55b19b7555f7e15bb287436ccf83eececaf52117e5e00b13e70726b1b3c6`|
|`tiny-retention-inventory-01`|23|1790754594801748|`a954aea4e4e5575ff2068fc68779d23e8f580419e0ae0e4d6a42377c6f57cc6a`|

| Stage | Receipt generation | Receipt SHA256 |
|---|---|---|
|`tiny-old-control-01`|1790754584694726|`f46fc1a3ca91947db08f1a3f426c9f8566143bd1df97686fd64ceb38018b9b51`|
|`tiny-stability-01`|1790754586683850|`7809fff8e8aa1929a3bdd9720e549b1fea1eaa4a94eeea6fafb75f707e9b1414`|
|`tiny-resume-01`|1790754588457825|`250ec97dba2454ea77cb77a0e1da5821d5fa33263c281a88a6b7645dffdf45de`|
|`tiny-insertion-audit-01`|1790754589781745|`1c06940dc91a411e28899d3e2ca62a987a155f40a310c3e54cce6a7c548b5cb9`|
|`tiny-restart-audit-01`|1790754591082833|`0a2d3a40b298d1251362923533eadadc3ffecfb763dbfe15b5cba045772f75fb`|
|`tiny-restore-01`|1790754592433624|`77dfa065719896329743660709d028fe3e54a9e3405b52ab3558c57023812d8c`|
|`summary-update000000-01`|1790754593902495|`98b6635fceadcebb1af8aea1ec853212e9152c62d43dbb27c7b95906fdf7776a`|
|`tiny-retention-inventory-01`|1790754595293568|`9cedd27daabb079134e9a4b1a9883912eaab67e58c18efb6fbac800af8180209`|

The neutral inventory preserves all nine tiny publication receipts (old control
updates 0–3, instrumented execution updates 0–3, and resumed execution update 3),
covering 18 previously verified cloud state/manifest objects. In particular,
update 2 used for cloud restart and all three final update-3 publications are
included. No model states were rehashed or uploaded again. Source inventories
of 200, 208 and 208 files remain tied to the three execution reports.

The origin summary archive contains the immutable probe snapshot, summary report,
CSV, two PDF/PNG plot pairs and summary source snapshot. Its input probe SHA256
is `6594b98d436efa822c3a953460b835348859b15f9b3ad0ac438435a9f1a28e95`.
It contains only update 0, not subsequent training measurements. The active
`native-f12-to128-01` directory was neither archived nor read during this closure.

Local receipts are `.runtime/olmo-fbt-stability-retention/<stage>.json`;
`<stage>-result.json` additionally records the remotely verified receipt object.
The neutral inventory includes the exact eight retention argument sets in
`retention-commands.json`. They were executed sequentially through the unchanged
`scripts.olmo_two_gpu_retain.retain` helper, using this CPU-only command:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'env -u GOOGLE_APPLICATION_CREDENTIALS python .runtime/olmo-fbt-stability-retention/retain_acceptance.py'
```

For any listed stage, the equivalent individual command is:

```bash
python -m scripts.olmo_two_gpu_retain \
  --input-dir .runtime/olmo-fbt-stability/<stage> \
  --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/fbt-stability-<stage> \
  --receipt .runtime/olmo-fbt-stability-retention/<stage>.json
```

Run that individual command inside the same CPU container with the credential
environment adjustment above. No new GPU checks or training tests were run for
retention. Full checkpoint objects remain at the destinations recorded in the
neutral publication inventory.
