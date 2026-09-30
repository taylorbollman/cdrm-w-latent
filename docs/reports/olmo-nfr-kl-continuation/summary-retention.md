# Paired summary, figures and retained evidence

2026-09-30. The CPU reducer ran only after both NFR branches completed update
64, verified their terminal cloud checkpoints, synchronized W&B and passed the
independent 16,483-check pair audit. It used the unchanged predeclared reducer,
the original common NFR32 parent and optional descriptive F64 context.

All four generated PNGs were visually inspected before publication. The PDF
and PNG copies in this directory match the summary's recorded artifact hashes.

| Figure | PDF | PNG |
| --- | --- | --- |
| Development CE, latent and KL | [PDF](figures/development-raw-losses.pdf) | [PNG](figures/development-raw-losses.png) |
| Gradients, clipping, LR and training losses | [PDF](figures/training-dynamics.pdf) | [PNG](figures/training-dynamics.png) |
| Timing and per-GPU memory | [PDF](figures/timing-memory.pdf) | [PNG](figures/timing-memory.png) |
| Absolute CE and pass-4 penalty | [PDF](figures/ce-refinement.pdf) | [PNG](figures/ce-refinement.png) |

[W&B summary: ni8f0ch6](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ni8f0ch6)
is synchronized and contains the four figures and per-pass raw-loss table.
Publication did not modify either training run's history. No new optimizer
update, GPU inference, numerical test or checkpoint-state reread was performed.

## Exact authorities

| Report | SHA256 |
| --- | --- |
| [Summary](../../../.runtime/olmo-nfr-kl-continuation/summary-01/report.json) | `b521e7546a3e4afef6fc81addb8dbf78b2e86585cb8f28e1d9c1bbd53907378a` |
| [Tracking](../../../.runtime/olmo-nfr-kl-continuation/summary-wandb-01/report.json) | `18f7a344af8597fbce6d1a278be55a88b9e318b1fd05316207051fa5761d6bc4` |
| Original NFR32 parent | `01bceb2a1191adb513bea974d8dcb0a5b52e8b5c3b384dbd0e69d21cfea665f7` |
| KL1 terminal NFR64 | `4e47728173364a6a3a6ed14887517cea9df8bcd2247d1b5d8ca743eb2f3b27fe` |
| KL0.1 terminal NFR64 | `9de466ea50ea837db1aaf6f0e88f23675c6a13af1f24a196fab992ea54a96e8c` |
| Independent paired audit | `2ee9b737c8269ef071fd63792f9d4ed2d87598b900347b571d7811e1aadcb211` |
| F64 evaluation | `3ebee2842db92e8c532e22fb267ff624f50b6b7da7035d1a832a36038894cef2` |
| F64 publication receipt | `d25506b4607d1a6fce0448b5b2ef22db2646ee34f2d2b42a8ff894b67090cfd4` |

The summary retains all six input snapshots, its analysis sources, three CSVs
and four figure pairs. F64 is same-panel context, not a matched KL intervention.
Weighted objective totals are absent from the presentation because their
definitions differ between branches.

## Cloud retention

Both disjoint stages are under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/`.
The normal CPU retainer verified size, server MD5, SHA256 metadata and downloaded
SHA256 for the evidence archive, retention manifest and final receipt. No
checkpoint was uploaded again and no original branch or audit directory was
re-archived. The summary's selected evidence was 83,431,913 bytes, below the
128 MiB cap; no split was needed.

| Stage/object | Generation | SHA256 |
| --- | --- | --- |
| `nfr-kl-summary-01/evidence.tar.gz` | `1790778927493447` | `e57c50745a044d1f1e8289f24cf593abb56d05ac308a3d2d7c4883e9d4a0752e` |
| `nfr-kl-summary-01/retention-manifest.json` | `1790778927821541` | `199910d7cb3f2ed2a4f174414160690bff5cc66b6c7c34df48f6cc1014097620` |
| `nfr-kl-summary-01/storage-receipt.json` | `1790778928112910` | `87bf1ac3f52f59658f2c83483c44024f1557195cfe0c60ef3fb674fddf0a33c3` |
| `nfr-kl-summary-wandb-01/evidence.tar.gz` | `1790778924393194` | `9c301826d67792ecf0b684733d0ad7f31784e0e71820bf7f115f765d474b5a58` |
| `nfr-kl-summary-wandb-01/retention-manifest.json` | `1790778924691069` | `d945a6d53f963456a64d1e9430600210ede9d2b3ef73d28122d802871cfde14f` |
| `nfr-kl-summary-wandb-01/storage-receipt.json` | `1790778924966596` | `5fe74cc6692e700d0dfca540bd8fa43025f10a2293d521a1eec5a0f06f4e3770` |

Local receipts are under `.runtime/olmo-nfr-kl-retention/` with those stage
names. The first retention attempts stopped before cloud upload because the
environment named a nonexistent credentials file. Repeating the unchanged
deterministic bundles with `env -u GOOGLE_APPLICATION_CREDENTIALS`, as used by
the accepted launcher, completed successfully. No environment file or model
source was changed.
