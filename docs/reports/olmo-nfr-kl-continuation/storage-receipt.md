# NFR KL continuation storage receipt

The original NFR32-to64 KL1/KL0.1 pair was activated after the F128 and
post-F diagnostic review. This receipt preserves **initial static authority**,
not a completed training run. The active training directories, live queue
state/events and training logs are deliberately excluded.

Evidence prefix:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/`

| Stage | Members | Archive generation / SHA256 | Receipt generation / SHA256 |
|---|---:|---|---|
| `nfr-kl-initial-static-authority-01` | 226 | 1790764674883619 / `cb6fc6f9698582304d27148450774f3e4dccbba1bef82ca2c2eff89b8626babd` | 1790764675387523 / `6ec028728e6ebbdefe08c4fbe485cab21ac2085016760623e9a7369b14e5e861` |
| `nfr-kl-status-v2-validation-01` | 2 | 1790764740859220 / `fb61d1aceeef8a09235497c605b885f4a5be346a24a9d9770b7bd600972f3ba0` | 1790764741382537 / `969b4aa2f9afd0eb9cb1d2eeacc38ee7b6471a2653fc238ab191a4302b473773` |

The 226-member static authority archive comes from
`.runtime/olmo-nfr-kl-continuation/initial-static-authority-01`. It includes
`activation-01.json` (SHA256
`c9b209280ac92c7b1a86655e90a52bfffe6a47f155c3e9cfb86586a7eb75d2e3`), scope,
dry resolution, preparation records, pinned host launcher, sequential queue,
original status helper and the complete 215-file runtime source snapshot.
Every activation authority and runtime source pin was verified before copying.
The preparation records retain their historical `prepared_not_launched` labels;
activation is separately and explicitly recorded afterward.

The optional read-only status helper is now:

```bash
python3 .runtime/olmo-nfr-kl-continuation/live_status_v2.py \
  --queue-name queue-after-f128-01
```

Version 2 skips the evaluator's temporary `panels: {}` placeholder, retains the
last completed development evaluation, and reports `dev_pending_at_update`
while a newer one is running. Four bounded metadata fixtures and the actual
live update32 placeholder passed. The original prepared utility remains
unchanged at SHA256
`12fa8792b56606330f848af079f9e500bceb70ce8d5d9572483b781bac30557c`.
Version 2 SHA256 is
`1de47a94a253e0f19c715f2e7246a963e3e90bc4ae594773dc7dbdfb83b32b18`.
The separate two-member archive preserves that helper and its validation
receipt. No training, queue or optimizer code was changed.

Both stages used the existing `scripts.olmo_two_gpu_retain` in a CPU-only
container with `env -u GOOGLE_APPLICATION_CREDENTIALS`, without
`--checkpoint-dir`. Archive/manifest/receipt size, server MD5, SHA metadata and
downloaded SHA256 checks passed. Local receipt/result files are under
`.runtime/olmo-fbt-stability-retention/` using the stage names above.
The static-input file inventory records exact source/destination paths and
hashes. No local files were deleted, no model state was loaded or rehashed,
and no GPU work was launched by retention.

## Completed control 32→64 closure

The original-KL control finished at its review boundary 64 with host exit 0,
`stopped_at_boundary`, completed segment, W&B synced and final cloud 64
publication verified. The broader 128-update plan is not complete. At that control closeout, the reduced
branch was running independently under the pre-existing queue; it was not
paused, modified or included in the control terminal archive.

Terminal report: `.runtime/olmo-nfr-kl-continuation/native-nfr-control-32to64-01/report.json`.
SHA256: `4e47728173364a6a3a6ed14887517cea9df8bcd2247d1b5d8ca743eb2f3b27fe`.
Final publication: `checkpoint-publications/update-000064.json` in that directory.
SHA256: `6b484f991be6121b403f07529a0627f314da7143cf6c997f3a204c3d4cc1bad7`.

All 215 live runtime source files match the control source snapshot. Final 64
boundary hashes match the saved checkpoint boundary; the local manifest and
state-file size agree with the already verified cloud publication. Closure
performed no checkpoint tensor loading or state-content rehash.

The eight publications are updates 33, 38, 43, 48, 53, 58, 63, 64. Exact receipts are
retained under neutral filenames in the separate inventory, since the standard
small-evidence retainer excludes `checkpoint-*` directories. That inventory
also contains the two remaining local manifests (63, 64), final manifest,
terminal host receipt/launcher and metadata closure record. Source reports
and training history were not rewritten.

| Stage | Members | Archive generation / SHA256 | Receipt generation / SHA256 |
|---|---:|---|---|
| `nfr-kl-control32to64-01` | 256 | 1790771636506962 / `f3c4c9f2be0158fbeff30a315fe4b04e11520694e9fa77c3256523f56481c8a9` | 1790771637050364 / `85fdd7464a4b0cd6f15ae0769611ac25406a9b93051f3a425a78059fd8eb6a14` |
| `nfr-kl-control-terminal-inventory-01` | 15 | 1790771638121382 / `871f6d4f6585feebbf45ed47239796c8a21cba2012ab6274a6cbbf67411222e3` | 1790771638642225 / `78122f7732640fec5323f2fb00c3c8186b37f9031c83d0103b553408a297af78` |
| `nfr-kl-control-summary-correction-01` | 6 | 1790771640476842 / `9654f3939de8776fa9091222bfa1b9caab6bfcbcee52c215fa4dca7972879106` | 1790771641012084 / `bc70f83540c4d8aac6b3730bf9893180dd746ced260fa7cb5712146047701e48` |

All three directories use the common GCS evidence prefix above. The main
control archive selected 41,875,765 bytes before compression (39.94 MiB), below
the unchanged 128 MiB cap; the inventory selected 6,920,334 bytes and correction
selected 26,966,184 bytes. Archive/manifest/receipt size, server MD5, SHA metadata
and downloaded SHA256 verification all passed. Exact local receipt/result
objects are `.runtime/olmo-nfr-kl-retention/<stage>.json` and
`<stage>-result.json`. No full checkpoint was uploaded again.

Final checkpoint directory:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/nfr-kl-continuation/20260930-pair01/control/native-nfr-control-32to64-01/update-000064/`

| Object | Generation | Bytes | SHA256 |
|---|---|---:|---|
| `state.pt` | 1790771374621236 | 15,214,973,825 | `fa7ef205a9b93da8002fc82c5de78dddf820fd7190845edb8a6f6757c7e031f6` |
| `manifest.json` | 1790771452892688 | 623,321 | `f46420d563943d4e28f29b67c3f182c0c1e61f96406827061cb4e7beca4a14a3` |

W&B `taylorbollman/pretrained-fbt-rt-nextlat/ujz924fj` had stale summary values
**local 63 / cloud 58 / worker pending true**. The standalone reconciler corrected
only these checkpoint summary fields to **64 / 64 / false**, plus provenance.
Fresh readback passed on attempt 2; `_step` remained 64 and the run remained
finished. The unrelated-summary digest before/after was
`7373f8ed7c87a5aac139ce8452eb87c4938f2fe426ab815cea2e18e8bc8f5b60`.
The immutable report/publication pins are unchanged. No history was altered.

The helper explicitly accepts the existing `olmo-kl-continuation-report-v1`
schema, retaining every terminal/drained/matching-publication/W&B guard. This
metadata-only extension is outside the 215 frozen training source pins and
passed one focused CPU regression; original reports were not relabeled.
The correction's source snapshot records the precise helper bytes.

Execution used the CPU-only project container and
`env -u GOOGLE_APPLICATION_CREDENTIALS PYTHONPATH=/workspace/cdrm-w-latent python .runtime/olmo-nfr-kl-continuation/retain-control-terminal-01.py`.
This wrapper called the established reconciler with the report/publication
pins above, outputting `control-summary-correction-01`, followed by the unchanged
retainer without `--checkpoint-dir`. Closure is reproducible from
`control-terminal-closure-01/build.py` and its retained report. No additional
GPU job or paired comparison was run.

Completed control resource measurements were appended to the
[resource ledger](../olmo-fbt-stability/resource-ledger.md); reduced-branch
throughput and paired results were still pending at the control closeout;
see the reduced closeout below.

## Completed reduced 32→64 closure

The KL 0.1 branch also finished at `stopped_at_boundary` 64 with host exit 0,
completed segment, W&B synced, and final cloud 64 verified. The queue is now
`completed_pair`; no further training was launched. This closes the authorized
32→64 pair, not the broader 128-update plan. Independent paired analysis is
recorded separately in [validation.md](validation.md) and the paired report.

Terminal report: `.runtime/olmo-nfr-kl-continuation/native-nfr-reduced-32to64-01/report.json`.
SHA256: `9de466ea50ea837db1aaf6f0e88f23675c6a13af1f24a196fab992ea54a96e8c`.
Final publication SHA256 (`checkpoint-publications/update-000064.json`):
`094934f8391c6323f9a63e35758af296d80df69da2db9c8538492321f3d7ad7c`.

The same terminal guards passed: all 215 live/source-snapshot pins unchanged,
final 64 boundary equal to the saved checkpoint boundary, local manifest and
state-file size matching the published receipt. The eight neutral-name
publication receipts cover updates 33, 38, 43, 48, 53, 58, 63, 64; the inventory also
contains remaining local manifests 63/64, final manifest and terminal host
launcher/receipt. No state tensor was loaded or rehashed during closure.

| Stage | Members | Archive generation / SHA256 | Receipt generation / SHA256 |
|---|---:|---|---|
| `nfr-kl-reduced32to64-01` | 256 | 1790778756838315 / `d531186712d6cebd45f56a7a0d545fced48d49c5703388bd257ae634ee17fa39` | 1790778757403199 / `6a033aa91b59451406e4e72cf83e98d54501be3f62c52d620ffc298ce66a7427` |
| `nfr-kl-reduced-terminal-inventory-01` | 15 | 1790778758538877 / `ce0ad85716f485c4b26cbfc1b74ec188f7972b84c22c63e3052b94d2ee050bc3` | 1790778759076963 / `d00a4b2221336e8e9287e8337e0ae627aea540e4c476b8b7a953f0ef6b970a3e` |
| `nfr-kl-reduced-summary-correction-01` | 6 | 1790778760939589 / `c149361865a4886c34720a82d19c48679e8b5f123838bf468bd8a75b7a3af123` | 1790778761471173 / `9fb83d59c869ac73ef0fbe3f273ea08846bf34c4e405f89fb88c8bfe960f0dc0` |

The common GCS evidence prefix above contains all three verified stages.
Selected pre-compression bytes were 41,875,789 (39.94 MiB), 6,920,377 and 26,966,235
respectively, each below the unchanged 128 MiB cap. Size, server MD5, SHA metadata
and downloaded SHA256 checks passed for archive, manifest and receipt objects.
Local receipts/full returned metadata remain under
`.runtime/olmo-nfr-kl-retention/<stage>.json` and `<stage>-result.json`.

Final checkpoint directory:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/nfr-kl-continuation/20260930-pair01/reduced/native-nfr-reduced-32to64-01/update-000064/`

| Object | Generation | Bytes | SHA256 |
|---|---|---:|---|
| `state.pt` | 1790778527701002 | 15,214,973,825 | `098050b609821c7fe12895a9f94c79770e3ee0be16c950f6d66448c0ccfc03be` |
| `manifest.json` | 1790778585850850 | 623,325 | `4f24a233c89e2eb20a0c355d1a3dda530fb906981402297278b2decffde611ea` |

W&B `taylorbollman/pretrained-fbt-rt-nextlat/1xu07xdf` had the same stale
checkpoint summary values **local 63 / cloud 58 / pending true**. Summary-only
reconciliation changed them to **64 / 64 / false** and added pinned provenance.
Fresh readback passed on attempt 2, `_step` stayed 64, and all unrelated summary
values remained identical (digest
`e24f5504fef2a85f5c2943de3c3c6dfdf81796c5c36f6602994a4d0eacd27769`).
Original training report bytes, source pins, history and checkpoint data are
unchanged. No paired quality comparison was made by this closure.

The CPU-only wrapper was
`.runtime/olmo-nfr-kl-continuation/retain-reduced-terminal-01.py`, invoked with
the same `env -u GOOGLE_APPLICATION_CREDENTIALS PYTHONPATH=/workspace/cdrm-w-latent`
container setup as control. It reused the accepted reconciler and retainer
without `--checkpoint-dir`; no checkpoint was uploaded again. The final resource
row is in the [resource ledger](../olmo-fbt-stability/resource-ledger.md).
