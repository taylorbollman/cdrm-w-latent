# Portable campaign readiness evidence retention

Status: **verified**, 2026-09-28. The existing bounded retention helper was reused;
its `olmo-two-gpu` storage namespace does not imply this was a two-GPU run.

Prefix:

```text
gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T203800Z/campaign-portable-readiness
```

| Object | Bytes | SHA256 |
| --- | ---: | --- |
| `evidence.tar.gz` | 240068 | `fd94f5e1dbe3e822068ba060cfcd708099799e0849435b774fa4527a54a09e35` |
| `retention-manifest.json` | 17854 | `3e79327c3e1acbcc8dfdd802a98394b969503c925648931b5ee949daa73c93e5` |
| `storage-receipt.json` | 1306 | `107dcce3f1bbd8f81848a0fabcba3a1208b768164236e64fc97d6ab317b956b9` |

All objects passed server-size/MD5/metadata checks and a downloaded-byte SHA256
check. Archive: 71 members, including report, launcher/CPU logs, protocol/results/
usage/data/progress notes, and all 60 declared runtime source snapshots. Runtime
source commit is `714f31c`; tests and final closeout notes are also retained in Git.
W&B run `kb0lbu1k` is synced. No local evidence was deleted.

Local receipt: `.runtime/olmo-campaign-readiness/retention/gpu-smoke-01.json`.
The archive is immutable; later documentation edits do not alter its contents.
No full optimizer/model checkpoint was uploaded: these were two disposable
diagnostic updates, reproducible in under a minute from the pinned original.
The fresh-process resume test uses a small temporary CPU fixture checkpoint.

After the probe, container `nvidia-smi` reported 0MiB used, 0% utilization and no
compute processes. No training or benchmark job remains queued.
