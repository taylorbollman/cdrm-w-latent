# NFR settling and continuation evidence retention

## Endpoint preparation

The completed endpoint scope, plan, host launcher, commands, readiness receipt
and both CPU preflight directories were copied into the isolated immutable
directory `.runtime/olmo-nfr-stability-128/endpoint-static-retention-input-01`.
Each copied file was hashed before and after copying; the four independently
provided scope/plan/report pins matched. No active GPU result or launch output
directory was selected.

The standard CPU-only retainer published 468 files, 6,609,959 bytes before
compression, with both 230-file source snapshots verified against their
preflight reports. Its container used `CDRM_DOCKER_GPUS=none` and
`env -u GOOGLE_APPLICATION_CREDENTIALS`. No checkpoint state was uploaded or
rehashed, no local file was deleted, and no GPU command was launched by this
retention work.

GCS prefix:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/nfr-endpoint-static-01/`

| Object | Bytes | Generation | SHA256 |
| --- | ---: | --- | --- |
| `evidence.tar.gz` | 1,740,834 | `1790782786598423` | `493e80d546773f6eab0dda652bf71d27a413616a1631a9b17736c4d33d8af3d4` |
| `retention-manifest.json` | 127,060 | `1790782786865719` | `73455ce471802094dc0467c0bd05efadefb7bc0dcfd398c658bbe69478e353ee` |
| `storage-receipt.json` | 1,294 | `1790782787137929` | `fa69a66144399ecd97553d0b006eb9ba37fa9f37aacfd54e836fac717b4998a1` |

Server size, MD5, SHA metadata and downloaded SHA256 checks passed for all
three objects. Local receipt and result:
`.runtime/olmo-nfr-stability-retention/endpoint-static-01.json` and
`endpoint-static-01-result.json`.

This archive preserves readiness and CPU preflight, not completed GPU curves or
continuation. Their later receipts must be recorded separately.

## Conditional continuation preparation

The closed `continuation-prepared-01` directory, its CPU-test/readiness receipt,
`launch_continuation.py` and all 222 pinned runtime sources were copied into
`.runtime/olmo-nfr-stability-128/continuation-static-retention-input-01`.
Scope, resolution and launcher pins matched the independent preparation
receipt before copying. A new static report and per-file copy inventory record
that this remains **prepared, not activated**. Mutable status helpers, future
activation receipts, live endpoint outputs and training directories were
excluded.

The same CPU-only retainer verified and published 229 files, 3,671,835 bytes
before compression, with the complete 222-file source inventory validated.
GCS prefix:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/nfr128-static-01/`

| Object | Bytes | Generation | SHA256 |
| --- | ---: | --- | --- |
| `evidence.tar.gz` | 875,681 | `1790782973850511` | `1d215a43af74bb2198aa4f3cf5b52ea32b55a58d8ced23f06a312b8911b25b28` |
| `retention-manifest.json` | 57,657 | `1790782974122928` | `e3a93b7e1c1b90d8895769bdf69dc0b29163cb86cb0ed8ab508904904e4dc926` |
| `storage-receipt.json` | 1,274 | `1790782974403930` | `0c0f7ac0c7c25f02a454814eba9d63f716cc7cf51ae3aa8582431d660592ffa2` |

All three objects passed server size, MD5, SHA metadata and downloaded SHA256
checks. Local receipt/result files are
`.runtime/olmo-nfr-stability-retention/continuation-static-01.json` and
`continuation-static-01-result.json`. No model state was uploaded or rehashed,
no local files were deleted, and no GPU command was run by retention.
