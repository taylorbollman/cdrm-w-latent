# Combined T2048 storage receipt

2026-09-28. All four new stages and the separate closeout bundle are retained
and verified. No failed or incomplete execution is being presented as passed.

Prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T165900Z/`.

Each stage contains an evidence archive, retention manifest and cloud storage
receipt. Evidence and manifest objects were checked for server size, MD5 and
SHA256 metadata, then downloaded for SHA256 verification. Local receipts are
`.runtime/olmo-combined-long-context/retention/<stage>.json`.

| Stage | Evidence members | Archive bytes | Archive SHA256 |
| --- | ---: | ---: | --- |
| `sdpa-b2-check-01` | 165 | 670,389 | `41e5462b642d74be1916eb2544d734e24c8c031a4551348414ea04a3603b7d1a` |
| `sdpa-b16-01` | 165 | 670,418 | `0c342b15064ffa758ad0482147ac4a2f3d7e84d618b8a65a728e3811e6da2e8f` |
| `sdpa-b32-01` | 165 | 670,418 | `acd3d96cdb716f1cb5a2c75173d4827ac43813703c4d362b2e86d2c9a24ddc59` |
| `sdpa-b32-02` | 165 | 670,203 | `47d38e6e59049e1a63145374aeb57a323848d73b4c4653819dea71a3fbbbdd0d` |

The final audit verifies 652 new report-pinned source snapshot pairs. All 37
pretrained runtime sources and dependency versions match the historical T512
reference. Four stages represent 32 physical optimizer updates and 20 passing
operational checks. B2 is a diagnostic; the performance pool contains B16 and
two B32 runs. There are no FA4 or Dao dependency source snapshots in these
SDPA/native-RoPE runs; package identities are recorded.

Historical reference remains at
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T172000Z/single-combined-b128-01/`.
Its saved receipt and frozen sources/report were checked without rerunning it.
Original O1 weights remain retained by reference; no new full checkpoint is
needed for these disposable benchmark updates.

The separate closeout archive contains audit tools, generated JSON/CSV, plots,
CPU/process/final-GPU-state logs, four stage receipts and the historical report
with source snapshots and saved receipt. It is not another execution stage.
Credentials, W&B caches and model weights are excluded.

Closeout verified: 190 retained members.

| Object | Bytes | SHA256 |
| --- | ---: | --- |
| `evidence.tar.gz` | 851,643 | `df7da3eb0f813091be41e05560fb0a93212f9765e1d2a2437bbd173f133e3f09` |
| `retention-manifest.json` | 49,600 | `5f942318b3cb11bcd56e56102046f0db8e5fbd145e2e22491525fb8f864c1e60` |
