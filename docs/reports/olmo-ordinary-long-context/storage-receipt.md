# Ordinary T2048 evidence storage receipt

2026-09-28. All seven new stages have verified GCS receipts: six passing
capacity runs and one retained strict-loss numerical failure whose operational
checks pass. No failed stage is relabeled by later successful timings.

Prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T163200Z/`.

Each stage contains `evidence.tar.gz`, `retention-manifest.json` and a cloud
`storage-receipt.json`. Evidence and manifest objects were checked for server
size, MD5 and SHA256 metadata, then downloaded for SHA256 verification. Local
receipts are `.runtime/olmo-ordinary-long-context/retention/<stage>.json`.

| Stage | Evidence members | Archive bytes | Archive SHA256 |
| --- | ---: | ---: | --- |
| `fa4-b16-01` | 218 | 1026838 | `aec4a144d98a3b0c5068e91c31c1f0d109eaff225a9b94169f20613cf0c7be63` |
| `fa4-b32-01` | 218 | 1026862 | `708152cce9562c0c91663e7af685b6a79909df615405d256c94d06c274bf0566` |
| `fa4-b32-02` | 218 | 1026839 | `9da39a8b6dcf3b5b064110f75c0849c596e397ac2d35c6dcf5d1ce34a8ee6957` |
| `fa4-check-b2-01` | 218 | 1030804 | `19ae1499003d17a6a7402100280a38fb94b220cc672337996aa2b90d02a91922` |
| `sdpa-b16-01` | 168 | 673995 | `0c70e453b76f54de26119a147e69915b76c5d590b252c3681a1dcb66c84cdfb5` |
| `sdpa-b32-01` | 168 | 673999 | `cc933d2c5653a1b221a2842860cb6ee94fb7a67a889da3f9758e558e246e55e1` |
| `sdpa-b32-02` | 168 | 673953 | `5d268bf529446ed9caed550cdc950c57c911969d780217027c32fbbb7237c51a` |

The final audit verifies1,148new report-pinned source snapshot pairs and214
installed dependency source pairs, with no source/dependency/retention mismatch.
All37 `cdrm/pretrained/` files match the saved T512 runtime. Seven new reports
represent56physical optimizer updates; historical T512 is not a new stage.

Historical reference remains at
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T195200Z/ordinary-dao-single-b128-01/`.
Its existing local receipt, frozen report/sources and hashes were checked without
rerunning it. Original pretrained O1 weights remain retained by reference;
no additional full checkpoint is needed for the disposable capacity updates.

The separate `closeout` archive contains audit tools/README, generated JSON/CSV,
plots, CPU/preflight/launcher logs, queue scripts and seven stage receipts. It
also contains a copy of the historical reference report/source/dependency
snapshots and its receipt, explicitly labeled historical. It is not an eighth
execution stage. No credentials, W&B cache or model weights enter this bundle.

Closeout verified: 217 retained members.

| Object | Bytes | SHA256 |
| --- | ---: | --- |
| `evidence.tar.gz` | 1141459 | `05dc39f792bd38004c90d5304a1da82615532e1d064647d0390b2f5ff48ffbe6` |
| `retention-manifest.json` | 55933 | `48ca00d5611d00fb252a403b949e198428789e56f223e92f61e300318664c497` |
