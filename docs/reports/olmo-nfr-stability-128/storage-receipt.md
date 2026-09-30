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
