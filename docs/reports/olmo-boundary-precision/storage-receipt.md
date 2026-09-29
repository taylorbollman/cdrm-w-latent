# Fixed-boundary evidence storage receipt

2026-09-29. The completed diagnostic is retained and independently downloaded
again at its **exact cloud object generations**. Archive bytes, manifest bytes,
all 92 inventory members, the 89 source snapshots and the boundary tensor fixture
match their recorded pins. No model update or newly trained checkpoint was
created, and no local/SSD file was deleted.

The namespace is
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T061000Z/`.
The historical `olmo-two-gpu` directory name does not imply DDP or two-GPU
execution: this diagnostic used one device; verification used a GPU-disabled
CPU container.

## Completed diagnostic stage

The verified receipt is
`.runtime/olmo-boundary-precision/retention/boundary-01.json`.
The two listed objects below are relative to the namespace above.

| Object | Generation | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `boundary-01/evidence.tar.gz` | `1790663005779698` | 5,208,751 | `89b36d135def3ab84b478667ffcc73441bff98affb5fa4fe8bc48cd9807b9e00` |
| `boundary-01/retention-manifest.json` | `1790663006171237` | 23,544 | `3878a3f0518fb6899865a94c554617539269ea81ecf76b77013623bd8b0c274b` |

This is **one receipt, two listed objects, 5,232,295 downloaded bytes and 92
inventory members**. The archive also contains two control files, `RESTORE.md`
and `evidence-members.json`, for 94 regular files in total. Separate uploaded
`storage-receipt.json` control objects are outside the listed-object count.

The inventory includes the final report, launcher log, 89 frozen source files
and the **9,474,929-byte boundary fixture** containing all 44 tensors. Report SHA:
`60432b04554d1eeadc069b9b63f562049be38f23b65317ca4ff60775df559ff1`.
Fixture SHA:
`a86a1a667ac07c0f2ae6d236f023ff9a86ebbdfbd53691b1209a616e0be584aa`.
The original pretrained checkpoint remains pinned by the experiment's provenance;
this stage does not duplicate that large source weight file.

## Independent verification

The bounded reusable script and its atomic report are under
`.runtime/olmo-boundary-precision/storage-audit-01/`:

| Artifact | SHA-256 |
| --- | --- |
| `audit.py` | `5819df1b06e3beda87233773aadf1518e2afb4a402f57b6e11c235449696144a` |
| `report.json` | `d78cedb1bfbfab7838f4d3847cf9bd829696e4cc00661858e60085b452857060` |

The independent readback checked generation, server size/MD5/SHA metadata,
downloaded whole-file size/MD5/SHA and retained local archive/manifest bytes.
It inspected the archive without filesystem extraction, rejected unsafe names,
duplicates and special files, and checked every inventory member. Source bytes
match both the archived snapshots and current frozen sources; report/fixture
bytes also match the completed local stage. The audit stores its own source
snapshot and the verified receipt.

The separate local boundary audit verified tensor payloads, masks, strides,
shared cotangents and exact A/C endpoints; its pins are in the
[test ledger](test-ledger.md). Both audits are CPU-only and do not rerun a model.
Their reports and documentation await the final closeout bundle; this receipt
claims only the completed diagnostic stage above.

Storage integrity does not resolve the model's numerical qualification. The
diagnostic passed operationally; the reproduced full-model BF16 discrepancy
remains open, and local error norms are not additive causal fractions.
