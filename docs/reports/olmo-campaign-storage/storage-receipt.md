# Durable storage receipt

All completed stages were retained with create-only objects and verified server
size, MD5, SHA metadata, generation and downloaded SHA256. Small receipts are at
`.runtime/olmo-campaign-storage/receipts/` on the persistent project disk.

Small-stage prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T172500Z/campaign-storage-<stage>/`.
Each stage contains `evidence.tar.gz` and `retention-manifest.json`. Runtime stages
preserve nested source snapshots, logs, declarations, journals and individual
checkpoint publications. Large state files are not duplicated in these archives.

Checkpoint prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T172500Z/campaign-storage/NFR/<segment>/update-NNNNNN/`.
The historical subtree name is required by the unchanged accepted retainer; it
does not change the scientific startup identity of these tiny acceptance runs.

The inventory snapshot contains11small-stage receipts /22objects /4,697,498bytes,
and8new checkpoints /16objects /134,467,186bytes. It excludes its own and later
closeout receipts. This is a verified-receipt inventory, not an additional fresh
cloud download. Its report SHA256 is
`64c1015ed6a27a6ba4b12a4584c239db3e6fe91a6c2b0d3f4c0dd12ccada217c`
at `.runtime/olmo-campaign-storage/inventory-01/report.json`.

## Restore authorities

The tiny stop-at2 manifest SHA256 is
`1e52b8226f51503670d9180eb1c8bfb67ac16029ca80d85e97da9a066a8af283`.
Both continuation and evaluation-only recovery bind to that stop run's exact
published receipt and its independently pinned state/manifest generations.
Restored assets live under
`/mnt/localssd/cdrm-checkpoints/campaign-storage/tiny-restored-01/`.

The native asset restore re-read the already retained PR47 update3 objects:

- Manifest SHA256:
  `1df150e15d822929e0ff8683c5fffc6eddfaaa6ab5a174a6f1eac265631e4d1a`,
  generation1790699951813406.
- State SHA256:
  `b46d2f501521a34e517c93f0e6946d0e6a2f5a1edec3d0cfdb654324d9b0423b`,
  generation1790699881131039,15,214,852,353bytes.
- Total with manifest15,215,060,305bytes; verified streaming restore70.631s.
- Restore report SHA256:
  `72d8ef783f5e853b1b82c434a61c58418dfbe55710bfc4bbbb830bf5ef697978`.

Native assets are at
`/mnt/localssd/cdrm-checkpoints/campaign-storage/native-assets-01/`; persistent
receipts/source evidence are in `.runtime/olmo-campaign-storage/native-assets-01/`.
This is an asset recovery check only, not cross-version training migration.

## Independent audit report pins

| Audit | SHA256 |
| --- | --- |
| Transition | `ab9967b4ff3e6a698c7a59d9c768de7ed83f8579377fe06d82b985458dc26a6d` |
| Resume | `f770525461ea652bd776f5692e2cf89a4b06329629ae92792bb5e11c83e453c1` |
| Terminal | `e73ee5d0e0175221aa3bf71986f71fee2e34e21c40c5a22f5aa57de88e55b9d8` |

Only three newly owned local checkpoint directories were pruned: reference0/1
and stop0. Their cloud objects and persistent receipts remain. All historical
boot-disk checkpoints and both restore sources were retained. Final closeout
bundle receipt and PR state are recorded below/in progress.md after publication;
the bundle cannot contain its own receipt or later administrative note.

## Final closeout

PR: https://github.com/taylorbollman/cdrm-w-latent/pull/48. Closeout contains 1,023 members and was fully verified in GCS.

Closeout report SHA256: `43690247ab3fab5d1dc2a768c8a2ec251d59399e8d970ca0507f6d4db8c6e9e7`.

Receipt SHA256: `25c926a73a873f22abf8acbb06b94a3d1061e8335dc8a7e964160bb57a1593f9`.

- `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T172500Z/campaign-storage-closeout-01/evidence.tar.gz`: generation `1790703387157541`, 4,641,146 bytes, SHA256 `bf2010bd7f151e3f028d1e93dc241baa6a613116eea338ef6c147d18e9099a81`.

- `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T172500Z/campaign-storage-closeout-01/retention-manifest.json`: generation `1790703387453254`, 279,624 bytes, SHA256 `e9850efb423bdc4f0903b5ef273cf1dc0ed6a93a933b9a27cd654cf10925e418`.
