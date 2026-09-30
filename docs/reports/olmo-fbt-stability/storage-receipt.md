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

## W&B terminal-summary correction

On 2026-09-30, the finished ordinary control run `37uu86ip` received a summary-only
correction using its pinned terminal report and update-128 publication. The
Public API changed `checkpoint/last_local_update` from 96 to 128,
`checkpoint/last_verified_cloud_update` from 64 to 128, and
`checkpoint/worker_pending` from true to false. A provenance annotation records
both authority SHA256s, completed update, helper source hash and correction scope.
This supersedes the earlier qualification about the current W&B summary being
stale. Existing history rows retain their original values.

Installed W&B SDK 0.27.2 was inspected before use: its public summary update
writes `summaryMetrics` only. The helper did not initialize/resume a run or call
history logging. A fresh API readback confirmed all three corrected values,
finished run state, unchanged `_step=128`, and identical unrelated summary
metrics. Both pinned local authority files remained unchanged. The unrelated
summary SHA256 before and after was
`15d138933ff63c1172669ddff7d8f52f7a2226e638310b06284e4697802e20c0`.

The first immediate readback was stale, so the first attempt conservatively
recorded a verification failure even though the mutation had succeeded.
W&B also normalizes nested annotations into dotted keys. That attempt and its
source snapshot remain intact. A later, separately recorded **read-only**
verification established success without repeating the write. The reusable
helper now handles dotted annotations and bounded readback retries; nine focused
CPU tests passed. All 208 F-only and 215 conditional-NFR source pins remain unchanged.

The successful readback report SHA256 is
`13fab6979458a1f7d8bdad8b561a6adef787f5c6b7de3f29474721a19f4ad2b3`.

Two additional fixed directories under the common evidence prefix retain both
attempts. Each contains the usual archive, manifest and receipt:

| Directory | Members | Archive generation | Archive SHA256 |
|---|---:|---|---|
|`fbt-stability-baseline-summary-correction-01`|6|1790756340374664|`93317bae79871ca60f7daebf784b76b6ce9bf8ad237f9ea3fb70b9305f8b99ac`|
|`fbt-stability-baseline-summary-readback-01`|5|1790756341690414|`451c2b84be4f1add50679eba87c782dc6436f30155b2062270a22e1a94f53caa`|

| Directory | Receipt generation | Receipt SHA256 |
|---|---|---|
|`fbt-stability-baseline-summary-correction-01`|1790756340894970|`c6b2e6e0c3c0f656272f7476919a3c83f618eeb3a9539d43c773aaffab12f8d3`|
|`fbt-stability-baseline-summary-readback-01`|1790756342185849|`f081e3f2b6d88d8f2da2f68430ba8b8d65f22993b1a9db3483d28647819fb496`|

Both archives passed server size/MD5, SHA metadata and downloaded SHA256 checks.
The first includes the original terminal-report and publication snapshots; both
include before/after summary values, a compact receipt and the applicable helper
source snapshot. The later receipt identifies the earlier attempt by SHA256.

The initial mutation used `scripts.olmo_terminal_summary_reconcile` with the
report/publication paths and pins stated above and output
`.runtime/olmo-fbt-stability/baseline-summary-correction-01`. The successful
read-only verification used the same arguments plus:

```text
--verify-attempt .runtime/olmo-fbt-stability/baseline-summary-correction-01
--output .runtime/olmo-fbt-stability/baseline-summary-readback-01
```

Both operations used the CPU container. The two directories were archived with
`scripts.olmo_two_gpu_retain` using the same individual-command pattern documented
above, substituting their names for `<stage>`, with no `--checkpoint-dir`.
No model/checkpoint changes, new GPU tests or historical runtime edits occurred.

## Curve summaries through updates 32 and 64

The completed, W&B-synced summaries through updates 32 and 64 were retained
independently while native F training continued. These are immutable analysis
directories with pinned probe snapshots, source snapshot, CSV and PDF/PNG
figures. The mutable live training report was not read or archived. The origin
archive above remains unchanged.

| Directory under common evidence prefix | Included probe updates | Members | Archive generation | Archive SHA256 |
|---|---|---:|---|---|
|`fbt-stability-summary-update000032-01`|0, 32|9|1790758721777901|`fd5fe1d5a58ccec5d799885a92d351eff763c00f5b19361da800d14784aa0260`|
|`fbt-stability-summary-update000064-01`|0, 32, 64|10|1790758723465646|`65098c13876c794e135d524b310bc612046c7b2a5c72e2e4316d16f73f7f8910`|

| Directory | Receipt generation | Receipt SHA256 |
|---|---|---|
|`fbt-stability-summary-update000032-01`|1790758722288735|`0a64ca3cca0359650d034148411d2fb4eed03530bffb531be6a1469274fc1b77`|
|`fbt-stability-summary-update000064-01`|1790758723962304|`6f02a5232cf1d69d578cbe7c813caf97e29a64dcaf0bc877d6ef229c983cead6`|

Local receipts/results use the same stage names under
`.runtime/olmo-fbt-stability-retention/`. The unchanged `olmo_two_gpu_retain`
helper was called sequentially with each summary as `--input-dir`, its directory
in the table as `--prefix`, and the corresponding local `--receipt`. The CPU
container used `CDRM_DOCKER_GPUS=none`, `CDRM_FLASH_ATTENTION_SOURCE=installed`,
and `env -u GOOGLE_APPLICATION_CREDENTIALS`; no `--checkpoint-dir` was supplied.
All remote archive/manifest/receipt objects passed size, server MD5, SHA metadata
and downloaded SHA256 checks. No model execution or state rehash was involved.

## Immutable F64 context bundle

The separate seven-member context archive preserves only the completed F64
evaluation, verified publication receipt, exact optional-summary arguments,
preparation helper and analysis source pins/snapshots. It contains no mutable
whole-run report. The checkpoint state and manifest were already cloud-verified
by the execution worker and were not uploaded or rehashed again.

Directory under the common evidence prefix: `fbt-stability-f64-context-01`.

| Object | Generation | Bytes | SHA256 |
|---|---|---:|---|
|`evidence.tar.gz`|1790759155710271|97,177|`fa1e43eee15333ca7ed6f41ff5bd89200a3fce3e976d2be28919a0c7ad2604a2`|
|`retention-manifest.json`|1790759155979018|2,074|`3e66921c276978ece92696cdc8fd49cad7e9688057a593d1b62e1611da9d283e`|
|`storage-receipt.json`|1790759156268942|1,306|`d8282c93fbcf0cd06721fe0d8964234ecf7b070548376a48f15699132a603bff`|

The evaluation SHA256 is
`3ebee2842db92e8c532e22fb267ff624f50b6b7da7035d1a832a36038894cef2`;
the publication SHA256 is
`d25506b4607d1a6fce0448b5b2ef22db2646ee34f2d2b42a8ff894b67090cfd4`.
Exact arguments and interpretation are in
[f64-context.md](../olmo-nfr-kl-continuation/f64-context.md). Both were checked
against the same development membership as original NFR parent32. This is
F-only context at update 64, not a paired optimizer fork or F128 endpoint.

The CPU retainer used
`--input-dir .runtime/olmo-nfr-kl-continuation/f64-context-01`,
`--prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/fbt-stability-f64-context-01`,
and `--receipt .runtime/olmo-fbt-stability-retention/f64-context-01.json`,
with `env -u GOOGLE_APPLICATION_CREDENTIALS` and no `--checkpoint-dir`.
No GPU work or frozen runtime-source edits occurred. A metadata check caught
and corrected an optional-summary guard in the unpinned reporting helper only:
FBT mode is enabled while NextLat weights and predictor parameters are zero.
The fix is separately committed as `2ff12d8`, with 14 focused CPU tests passing.

## Curve summaries through updates 96 and 100

The completed, W&B-synced summaries through updates 96 and 100 were retained
as separate immutable analysis archives while native F training continued.
Each report, source snapshot, input-probe snapshot and plotted artifact hash was
checked before upload; the report hash was checked again afterward. No live
native-F report was read or archived.

| Directory under common evidence prefix | Included probe updates | Members | Archive generation | Archive SHA256 |
|---|---|---:|---|---|
|`fbt-stability-summary-update000096-01`|0, 32, 64, 96|11|1790761003490234|`45aa49e29e5add97bcbcc4e70ac770a9652103912ad3467c7f274fd6a549bc9b`|
|`fbt-stability-summary-update000100-01`|0, 32, 64, 96, 100|12|1790761275372185|`012d4e289b0bb7e7801c34886249bf3bf6372785e7a760de1e51b9d6227e2bf8`|

| Directory | Receipt generation | Receipt SHA256 |
|---|---|---|
|`fbt-stability-summary-update000096-01`|1790761004038280|`dbb7c94f4b3331764ead3c52278bd8b0fadbd3c3281cd85aaab444dc9323c4ba`|
|`fbt-stability-summary-update000100-01`|1790761275925820|`7663a73e3c10f76b757439f93083e16d706e300fd10890bb4c7ccf4394111adc`|

Terminal summary report pins:

- `summary-update000096-01/report.json`: `6e29ec3119aa18ed2154a1266613867c2013666077fd3a68d3962476ca85f27a`.
- `summary-update000100-01/report.json`: `cab6429e25fd48668a125932fc8229b83b0fa66bc64a66ae00f45746b099076c`.

Local receipt/result files use those same names under
`.runtime/olmo-fbt-stability-retention/`. Every remote archive, manifest and
receipt passed size, server MD5, SHA metadata and downloaded SHA256 verification.

The unchanged `scripts.olmo_two_gpu_retain.retain` helper was invoked by this
bounded CPU-only watcher, which waited for each summary to be completed and synced:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'env -u GOOGLE_APPLICATION_CREDENTIALS python .runtime/olmo-fbt-stability-retention/retain_late_curves.py'
```

For each stage, its exact arguments are the same documented individual-command
pattern: `--input-dir .runtime/olmo-fbt-stability/<stage>`,
`--prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/fbt-stability-<stage>`,
and `--receipt .runtime/olmo-fbt-stability-retention/<stage>.json`.
No `--checkpoint-dir` was supplied. No GPU execution, model-state rehash,
frozen-source change or new experiment launch occurred.

## Terminal F128 evidence, audit and summary reconciliation

Native F reached its initial review boundary at update 128; the original
192-update declaration remains a ceiling, not completed exposure. The host
launcher exited zero, the terminal report is `stopped_at_boundary`, W&B is
synced, and checkpoint 128 is fully cloud-verified with no pending worker.
The original report SHA256 is
`8a07a7fc2a5ecafc4523a1f5adb6a9b2073ebd1f46a586c9034e3815977cbdfe`.
The report, training history, checkpoint state and frozen runtime sources were
not changed by this closure.

All 208 live/source-snapshot pins matched. The final saved boundary matched
checkpoint128; its local manifest and state-file size matched the existing
publication authority. No state tensor was loaded or large state file rehashed.
The final publication receipt SHA256 is
`11264d9d07edc532ec4a6ff9ce34bccfc30f67471297d191c1ed3f2c144a5e39`.

The terminal run selected 131,433,171 evidence bytes (125.34 MiB), below the
unchanged 128 MiB cap, so no core/async split was needed. Its primary archive
contains 312 members. The separate inventory preserves all **17** publication
receipts under neutral names so the generic `checkpoint-*` exclusion cannot
drop them. These include scheduled and time-based recovery saves at updates:

`0, 11, 16, 28, 32, 44, 48, 60, 64, 76, 80, 92, 96, 100, 112, 124, 128`.

The terminal audit archive preserves the independent native-F audit (160,485
passed checks), B-prefix audit (24,617 passed checks), pinned source snapshots
and exact report/input snapshots. Its retained inventory hashes were checked
before upload. Check counts are audit-record counts, not independent samples.
The final curve archive includes immutable probe snapshots at 0/32/64/96/100/128.

Every directory below is under the common evidence prefix stated above.

| Directory | Members | Archive generation | Archive SHA256 |
|---|---:|---|---|
|`fbt-stability-summary-update000128-01`|13|1790763214373935|`2d7d2220fd4c8a29c34905877d509069876fe609a0fc0dd00d526c8c6d53aaf1`|
|`fbt-stability-native-f128-01`|312|1790763830551185|`a132746048853a6bd32716894e5e9d7b15cdf718bdd3971bbca8d009f1ce5216`|
|`fbt-stability-terminal-inventory-01`|23|1790763832465424|`c40e2178766a75835d372a4eec79040c0d65a965bd1727c52c8b8f32555e35cf`|
|`fbt-stability-native-f128-audit-01`|16|1790763920013455|`6bfdb7a8940c359ecd51b5590148d5d21ae68c91ed1e78e3e00de1bdef8114c0`|
|`fbt-stability-native-f128-summary-correction-01`|6|1790763914257435|`0a75d685a291ecb8df66af0ef7f73a7ffffa42654d282e4ac6a781048cf07048`|

| Directory | Receipt generation | Receipt SHA256 |
|---|---|---|
|`fbt-stability-summary-update000128-01`|1790763214941004|`dae4cd7e295464f584aa739127310fa40752aff625552e8606fde4302b1ecbd2`|
|`fbt-stability-native-f128-01`|1790763831124090|`d97d7a1d27921899cc2a3fc12fe261a2edf8ca2d73eafa95c29b82ac58f4a399`|
|`fbt-stability-terminal-inventory-01`|1790763833005680|`732e5d70fbc3befeb665a2fdad6810f6117e8b7eaf05bad45f702d3a1073a03c`|
|`fbt-stability-native-f128-audit-01`|1790763920588332|`b32e39b2943019f55ed19ac5f6a54351aaaf774f54dff766baca65d110b8ba7b`|
|`fbt-stability-native-f128-summary-correction-01`|1790763914808686|`e484a67950459b5dd59da6f1fa7d6590a02ef9c1c4f6d726296fcedde9366dcb`|

All archives, manifests and receipts passed size, server MD5, SHA metadata and
downloaded SHA256 verification. Local receipt/result files use these stage names
under `.runtime/olmo-fbt-stability-retention/`.

The full final checkpoint remains at:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/fbt-stability/20260930-overnight01/F/native-f12-to128-01/update-000128/`

| Object | Generation | Bytes | SHA256 |
|---|---|---:|---|
|`state.pt`|1790763589570207|14,222,290,081|`6a9842a21835dab572ee6c4d540fa814d999f733e2b98daa8b4977955305290c`|
|`manifest.json`|1790763654943708|730,991|`b0c8cd0a94db8d33efba77e8533846e782ccf8833db891f8f124457ffd6350b7`|

W&B run `taylorbollman/pretrained-fbt-rt-nextlat/32sqvp7e` initially retained
stale checkpoint summary values `124 / 112 / true`. The existing standalone
reconciler changed only the summary checkpoint fields to **128 / 128 / false**
and added the authority/source provenance annotation. Fresh readback succeeded
on its second bounded attempt. Run state remained finished, `_step` remained
128, and every unrelated summary value was identical. The before/after hash of
unrelated summary values is
`ec4df5f1aeb641b560bc9142f199c18e36b55b2135f890892cd404c7bc937906`.
No history was rewritten; the terminal report/publication pins remained unchanged.

The correction command, run in the CPU container, was:

```bash
python -m scripts.olmo_terminal_summary_reconcile \
  --report .runtime/olmo-fbt-stability/native-f12-to128-01/report.json \
  --report-sha256 8a07a7fc2a5ecafc4523a1f5adb6a9b2073ebd1f46a586c9034e3815977cbdfe \
  --publication .runtime/olmo-fbt-stability/native-f12-to128-01/checkpoint-publications/update-000128.json \
  --publication-sha256 11264d9d07edc532ec4a6ff9ce34bccfc30f67471297d191c1ed3f2c144a5e39 \
  --output .runtime/olmo-fbt-stability/native-f128-summary-correction-01
```

The unchanged retainer used the standard CPU container and
`env -u GOOGLE_APPLICATION_CREDENTIALS`. Each stage used its matching
`fbt-stability-<stage>` GCS directory and `<stage>.json` local receipt. Input
directories were `.runtime/olmo-fbt-stability/<stage>` except
`native-f128-01`, whose input was `native-f12-to128-01`, and
`terminal-inventory-01`, whose input was `terminal-closure-01`.
No call supplied `--checkpoint-dir`; checkpoint objects were already published
and verified. No new GPU execution or experiment launch occurred here.

## Completed post-F diagnostics and plots

Retained the four completed component diagnostics (NF32, NF64 KL1, NF64
KL0.1, NFR32), the F128 online comparison, their three completed CPU plot
summaries, and a separate immutable launch-authority bundle. All five helpers
finished and W&B synced before archiving. Original report bytes and source
snapshots were preserved. The retainer independently checked each report's
source inventory (211/221/221/211/210 files respectively).

The online helper returned `complete` after successful execution and W&B sync;
the pinned host launcher expected `completed` and therefore exited 1 in its
final report validation. The original launch record and traceback are retained,
alongside the root's explicit `terminal-adoption.json`; neither the report nor
launcher was rewritten. The other four host launchers exited 0. This is a host
closeout spelling mismatch, not a repeated or failed model diagnostic.

The separate authority bundle contains both preparation/bound plans, original
commands and launcher, all five launch directories, neutral-name copies of the
five checkpoint publication receipts and manifests, and a record checking all
255 bound-plan pins. No model state was loaded or rehashed. CPU plot archives
include their input snapshots, rendered artifacts and SHA-matched producer
source copies; the original plot report pins are unchanged.

All stages use the common `20260930T070700Z` evidence prefix stated above:

| Stage | Members | Archive generation / SHA256 | Receipt generation / SHA256 |
|---|---:|---|---|
| `fbt-stability-post-nf32-01` | 220 | 1790764664037239 / `bd6126d855a72239b7c96b0189b44ee476db929d14f409bfd1ff1dce1a9113fa` | 1790764664551458 / `7a9a147a725d36c7ed1ffc03a5ca5b8f94876051ff86454e2633e5402f1acb6c` |
| `fbt-stability-post-nf64-control-01` | 230 | 1790764665879988 / `41a92ebf9b87e7399ffa7c815e367f18080ea36aff9c6a8cbe05888e9f6dc479` | 1790764666436605 / `c781b0727c061b0ae5810aa1bc7661bd0076aa8cf8c021abbf9632158eebd2b1` |
| `fbt-stability-post-nf64-reduced-01` | 230 | 1790764667743588 / `cc5371b92add8167ade61a0b0bea2032670c88f212f4f8876c03a94a4fd240d2` | 1790764668240214 / `27c7bbb4cc2c64966a1798bf3e0b77aa93a1fd036cc1cbd7464464cf2d5f02ec` |
| `fbt-stability-post-nfr32-01` | 220 | 1790764669582064 / `6fc5201e6181eaf6af7ec4a466393f35483ee97ddfa5f0c9e5e8ebb3658e0ced` | 1790764670105224 / `64c11cbbb80e80946503f371fc58c835a8066042f9fd237c0787cce977d125b0` |
| `fbt-stability-post-online-f128-01` | 214 | 1790764671622351 / `1e151cb9f5d2d79e72ae00eb28898f38aa1e0769464eb3439b571d9ed2b9c87a` | 1790764672148384 / `9bdaba1f6b8dc395e50c458cb69b244b82e9e2a558bd5af0ab7625b5883e3dfd` |
| `fbt-stability-post-authorities-01` | 38 | 1790764673242007 / `38d685743d5eee01ba570ca8bf18ca2778dc5c9f2c5b64de1e67a7a5237dd53f` | 1790764673770707 / `0aafa359d5a6e671f537bfd950c062efb943ee92df2d238f0c7e20db5bddf27b` |
| `fbt-stability-post-components-update32-summary-01` | 8 | 1790764676508312 / `5a14e0f84e1e85d9bd01a67c5aac1caf824feadf5de5a194f41236fc5fb03605` | 1790764677025993 / `e851b55f9e5672b530a8a1f5b1c56f371d4a41d0b4e7a8502509315c5d17f75b` |
| `fbt-stability-post-components-update64-summary-01` | 8 | 1790764678147522 / `2f234120e4e85fbd45612c689916eef275dfd1b78952116683f5575cc01bdad6` | 1790764678632380 / `837ed6be4af039198b039b83aa93f9e088aacf292938ac70008dd755f476fb6a` |
| `fbt-stability-post-online-summary-01` | 5 | 1790764679676473 / `fe69618a2a3230d39356bdfeaaf81bcd3b8ca499f6c2d0a9bdff3ffdad32e019` | 1790764680219257 / `cffdd02b40820818066bdaccd5848dcad9e7f1c5e96365aa0ece1f06736d6899` |

Terminal diagnostic report pins:

| Case | Report SHA256 |
|---|---|
| `nf32` | `b0d4630d31b325d4b833b7324eaaadd9fe7c7fcfd546c8d65632c47fe0ed37a8` |
| `nf64-control` | `cbaee8ddbe833616f7c47ab189d90196358235e571e9974a24cc5f9f97b98392` |
| `nf64-reduced` | `922d4c9dfdad3305a75e91d69a68e1f16da3abcd079a980913d8565b68986dec` |
| `nfr32` | `b6083334682ae09f81969d77af28e3f2720bedf09c242c448eed6979c1e6c3ee` |
| `online-f128` | `04b7133b93341f354903524b00dd6ff0ddaa3ef8590c12648983af0fff007616` |

All archive/manifest/receipt objects passed size, server MD5, SHA metadata and
downloaded SHA256 verification. Each directory stayed below the unchanged
128MiB evidence cap. Receipts and full returned object metadata are in
`.runtime/olmo-fbt-stability-retention/<stage>.json` and
`<stage>-result.json`. The standard CPU-only launcher used
`env -u GOOGLE_APPLICATION_CREDENTIALS python -m scripts.olmo_two_gpu_retain`,
with each exact input/stage mapping recorded in
`.runtime/olmo-fbt-stability-retention/post-diagnostic-retention-plan.json`.
The batch caller is `retain-post-diagnostics.py` in the same directory.
No checkpoint objects were uploaded again and no GPU jobs were launched.

The concurrently activated NFR pair's static scope, queue source and activation
authority were retained separately; see
[its storage receipt](../olmo-nfr-kl-continuation/storage-receipt.md).
No active training report, mutable queue state, events or log was archived as
terminal evidence.
