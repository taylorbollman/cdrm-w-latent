# Allocation benchmark storage receipt

Verified 2026-09-30 21:47 UTC.

All benchmark processes and the suite stopped before final retention. Original B32 and NFR128 learning checkpoints remain unchanged and retain their existing cloud authorities. No disposable native benchmark weights were saved as new learning checkpoints.

Cloud prefix:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T202000Z/`

| Retained stage | Members | Local receipt |
| --- | ---: | --- |
| `allocation-b-pair-teardown-01` | 76 | `.runtime/olmo-gpu-allocation-retention/b-pair-teardown-01.json` |
| `allocation-b-pair-02` | 76 | `.runtime/olmo-gpu-allocation-retention/b-pair-02.json` |
| `allocation-b-singles-01` | 154 | `.runtime/olmo-gpu-allocation-retention/b-singles-01.json` |
| `allocation-nfr-pair-01` | 76 | `.runtime/olmo-gpu-allocation-retention/nfr-pair-01.json` |
| `allocation-nfr-singles-01` | 154 | `.runtime/olmo-gpu-allocation-retention/nfr-singles-01.json` |
| `allocation-complete-01` | 591 | `.runtime/olmo-gpu-allocation-retention/complete-01.json` |

The complete 591-member archive includes all successful and failed cell reports/logs/source snapshots, suite and launcher records, origins, CPU test log, tiny acceptance reports, final summary/PDF/PNG, the 931-check audit and its producer, idle-GPU evidence, and a separate snapshot of the new scripts/tests and prior receipts. W&B local working directories and large model files are excluded. Final documentation and copied figures are also committed to Git.

Final complete archive:

- `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T202000Z/allocation-complete-01/evidence.tar.gz`
  - Generation `1790804720235578`; 2,678,227 bytes; SHA256 `b2c881de87020c15131391796d511bb59ad6147228921400f878778d0b31c789`.
- `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T202000Z/allocation-complete-01/retention-manifest.json`
  - Generation `1790804720527438`; 156,335 bytes; SHA256 `e9d6eafd5ddd494c667d8b87b15126cdea6a43db254111eeb39135940137fb01`.

Each evidence archive and manifest was published create-only and verified by exact generation, server size/MD5/SHA256 metadata and downloaded SHA256. Retention does not convert a failed run into successful evidence: the initial Bpair teardown failure stays excluded from the primary timing comparison.

The four tiny FP32 model/optimizer fixture files are retained separately because the small-evidence archive excludes `.pt`:

- `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T202000Z/allocation-tiny-state/tiny-graph1-01/B-state.pt` (generation `1790802139568831`).
- `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T202000Z/allocation-tiny-state/tiny-graph1-01/NFR-state.pt` (generation `1790802139824319`).
- `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T202000Z/allocation-tiny-state/tiny-graph2-01/B-state.pt` (generation `1790802140070856`).
- `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T202000Z/allocation-tiny-state/tiny-graph2-01/NFR-state.pt` (generation `1790802140329556`).

Their verification records are in `.runtime/olmo-gpu-allocation-retention/tiny-state.json` and copied into the complete archive. These are tiny acceptance fixtures, not migrated native production checkpoints.

No local files were deleted. This milestone requires no new native checkpoint restore before reviewing its measurements. The next topology-migration milestone must independently authenticate its selected saved127 native checkpoint and preserve its original lineage.
