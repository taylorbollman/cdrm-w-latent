# Overnight storage snapshot and independent readback

The completed September 29 readiness work is retained in `gs://fast-chunks`.
After the experiment, restart and checkpoint-readback producers closed, a
CPU-only audit independently downloaded **13 selected evidence receipts' 26
objects at their exact generations**. All byte hashes, sizes, archive inventories,
authoritative reports and **1,212 report/source-snapshot pairs** matched. This
audit downloaded **11,998,596 bytes** and checked **1,274 archive members**; it
did not download or load model checkpoint tensors or run a GPU.

The separate local inventory records this fixed snapshot:

| Inventory | Count / bytes |
| --- | ---: |
| Producer-verified stage receipts | 79 |
| Evidence archives and retention manifests | 158 objects; 63,105,952 bytes |
| Listed evidence archive members | 5,561 |
| Distinct checkpoint payload URI-generation pairs | 65 |
| Checkpoint payload bytes | 208,579,046,712 bytes (194.25 GiB) |

The checkpoint count includes 14 full ordinary/NFR model-and-optimizer payloads
and 51 smaller compact fusion or tiny lifecycle states. It counts retained
objects, including equivalent states published in distinct lineages, rather than
unique model weights. Checkpoint JSON manifests are not included in the payload
count. Retained failed diagnostics remain in the evidence inventory: successful
storage verification does not change their experimental status.

These counts precede retention of this audit, its inventory, the final independent
readback audit and the later closeout package. Those later objects are deliberately
outside this snapshot, avoiding a self-referential claim of complete final totals.

## Cloud locations and saved endpoints

Checkpoint payloads and their committed manifests use:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/`

Small evidence archives, source snapshots and retention manifests use:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T075900Z/`

Useful checkpoint locations relative to the first prefix are:

| Saved state | Relative path | Generation |
| --- | --- | --- |
| FP32 fusion-only warmup, update 128 | `train-02/update-000128.pt` | `1790671431225622` |
| Full NFR BF16 common continuation origin, update 4 | `nfr-updates-01/bf16_flash_triton-update-000004.pt` | `1790677664012201` |
| Full NFR FP32 continuation, update 20 | `nfr-continue-fp32-01/update-000020.pt` | `1790680435443040` |
| Full NFR BF16 continuation, update 20 | `nfr-continue-bf16-01/update-000020.pt` | `1790680265436959` |
| Manifest-driven ordinary B restart origin, update 1 | `manifest-b-loop/reference/update-000001/state.pt` | `1790682559220328` |
| Manifest-driven ordinary B fresh resume, update 3 | `manifest-b-loop/resume/update-000003/state.pt` | `1790683656001589` |

The inventory report contains the complete exact URI, generation, SHA256, size,
verification flags and originating report paths for all 65 payload objects.
Use those records and the associated checkpoint manifests when restoring;
filenames alone are not checkpoint authority.

## What was independently checked

The selected readback covers fusion warmup, the four-update NFR pair, both
16-update continuations and their comparison, the packed sparse/prepared/captured
bridge, abrupt-rank recovery, live evaluation insertion, the resource ledger,
ordinary B recovery, both manifest-driven B reference/resume runs, and the
checkpoint streaming-readback comparison. It checks the stored bytes against
the local receipts and source snapshots, including safe regular archive paths
and every listed member hash. All selected final reports retain their original
status; this storage audit supplies no new numerical or model-quality clearance.

The 79-receipt inventory aggregates already completed producer verification.
It is not a fresh download of all checkpoint objects. Producers verified full
checkpoint uploads by exact-generation readback. Separate actual cloud restores
and fresh-process replays are documented in the
[ordinary B result](../olmo-campaign-lifecycle/base-loop-results.md) and
[manifest-driven B result](../olmo-campaign-manifest/run-results.md).
The latter is the B-only three-update acceptance on the same topology/runtime:
checkpoint 1 reproduces updates 2 and 3. It does not validate an eight-arm launcher.
The [readback comparison](../olmo-campaign-lifecycle/readback-results.md) also
re-read one full checkpoint using both byte-buffer and streaming paths. Neither
the audit nor the inventory deleted local files or altered any experiment source.

An initial audit invocation failed before cloud access because its selection
file had not yet been written. Its log is preserved as
`storage-audit-01/execution-selection-missing.log`; the corrected, pinned
selection and completed audit are separate final artifacts. The container logs
also retain the installed Google SDK's future compatibility warnings.

## Recovery scope

Persistent project files remain under `/home/taylorbollman/cdrm-w-latent`.
The prepared token corpus is separately retained at
`gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/readiness-20260928/tokenized`:
86 files, 21,016,338 bytes. An earlier complete cloud restore and the latest
local 86-file verification are recorded in the
[corpus recovery instructions](../olmo-campaign-lifecycle/corpus-recovery.md).
The present selected audit did not repeat that corpus download.

After local-SSD loss, restore the corpus payloads as well as the packed index;
the index contains metadata, not the corpus. Use the matching source checkout,
container/runtime, topology, pinned configuration and original pretrained
backbone when required by a compact fusion checkpoint. The
[recovery-bundle helper](../olmo-campaign-lifecycle/recovery-bundle-results.md)
was tested for the guarded tiny-model/T16 asset bundle and emits a conditional
command. Full ordinary B recovery used the separately documented exact-generation
restore path; that tiny-bundle helper is not a general 14 GB checkpoint restorer.
Neither path bootstraps an arbitrary VM or authorizes a new campaign. The full-state loader
still validates model, optimizer, scheduler, cursor and RNG identity.

Checkpoint cadence is checked at completed update boundaries. Slow updates or
cloud retention can exceed the intended ten-minute interval; arbitrary instant
recovery or H200/runtime portability is not established. After SSD loss, work
beyond the last successfully retained checkpoint must be replayed.

## Evidence pins

All paths below are relative to the project root. These runtime directories are
closed for retention; the readable receipt itself is part of the later closeout.

| Artifact | SHA256 |
| --- | --- |
| `.runtime/olmo-fusion-startup/storage-audit-01/report.json` | `3ada35d7ba7090fc2f121964bdeb8c396095ebb1f282e478e61fa033d9faebb1` |
| `.runtime/olmo-fusion-startup/storage-audit-01/selection.json` | `ce5f4567d8106225ce2541af2e99819d21d3c7007c8199d048ff2a02ff8b2c48` |
| `.runtime/olmo-fusion-startup/storage-audit-01/audit.py` | `5f947b9741e1b3f76d1f7664134b3124759d1b92c23a3a20a6ead3ff5ec23dd4` |
| `.runtime/olmo-fusion-startup/storage-inventory-01/report.json` | `b5a7d0767f7f37f6a69c849a596d03c781541d0f66a240fd8303ec0738fc076f` |
| `.runtime/olmo-fusion-startup/storage-inventory-01/build.py` | `427d60981c8efbe904b0135cd5b387c32074dd1c288b80118246c4415fd8e92d` |
