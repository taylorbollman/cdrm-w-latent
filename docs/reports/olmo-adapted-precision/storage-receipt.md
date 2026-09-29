# Adapted-state evidence storage receipt

2026-09-29. The completed diagnostic is retained and independently downloaded
at both exact object generations. Whole-file size/MD5/SHA, server metadata,
all 106 inventory members, the final report and 104 source snapshots verify.
No new trained checkpoint was created or uploaded; no local/SSD files were deleted.

Namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T064900Z/`.
The historical directory name does not imply DDP: this probe used one GPU;
independent readback used a GPU-disabled CPU container.

| Object relative to namespace | Generation | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `adapted-01/evidence.tar.gz` | `1790664950650456` | 464,949 | `fa79b1183c72a520bbe84dde76c085ee9cefd49343f350c049d031956b95d8a6` |
| `adapted-01/retention-manifest.json` | `1790664950950381` | 26,881 | `ee3ccdcec430725fafb688a4cfea2217da56a984c25a1f2a4df851b6c9dd3846` |

Receipt: `.runtime/olmo-adapted-precision/retention/adapted-01.json`.
Totals: **one receipt, two listed objects, 491,830 downloaded bytes and 106
inventory members**. Two additional archive control files make 108 tar files.
Separate uploaded `storage-receipt.json` controls are outside this object count.
The archive contains the final report, launcher log and 104 frozen sources.

The reused independent CPU readback script/report are under
`.runtime/olmo-adapted-precision/storage-audit-01/`:

| Artifact | SHA-256 |
| --- | --- |
| `audit.py` | `0b952c18398ab904a80e0a9ea30a6246bd043df74781f3c1cca573f42951155b` |
| `report.json` | `bf02d52b36a2c891e59bdd567a73054e5c9d012f73f98dac3019ed794f8afee6` |

It verifies downloaded archive/manifest bytes against the receipt and local
retained artifacts, safely inspects members without filesystem extraction, checks
every member hash and compares archived report/source bytes to the frozen stage.
Its own source snapshot and receipt copy are retained locally for closeout.
The final diagnostic report SHA is
`6af988581eac19a2d74dcb32558b63c80911f44569ee9dcd2db442dceb0dbfa4`.

The imported O5c checkpoint remains at its previously retained object/generation
documented in [baseline and controls](baseline-and-controls.md). This small
evidence archive does not duplicate its 4.8 GB weights. Independent review did
not repeat that large checkpoint hash/read or download.

Final documentation, audit and CPU-log closeout is pending separately. Storage
integrity does not establish BF16 clearance: the adapted-state gradient result
retains the intermediate forward-state qualification and adapted-backbone confound.
