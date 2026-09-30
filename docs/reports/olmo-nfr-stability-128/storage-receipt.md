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

## Completed endpoint probes, summary and activation

After both launchers exited 0 and both probes and their summary were completed
and synced, their closed directories were copied into
`.runtime/olmo-nfr-stability-128/endpoint-terminal-retention-input-01`.
The selection contains `result-control-01`, `result-reduced-01`, `summary-01`,
both launch completion directories, immutable `activation-01.json`, an
independent CPU closure receipt and a copy inventory. It excludes the active
64→128 training directory, its logs and every model checkpoint.

The standard CPU-only retainer published 496 files, 14,401,033 bytes before
compression, and verified both 230-file source snapshots. GCS prefix:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/nfr-endpoint-terminal-01/`

| Object | Bytes | Generation | SHA256 |
| --- | ---: | --- | --- |
| `evidence.tar.gz` | 2,958,605 | `1790783493220547` | `e1ff713c5c076698b1230e26745c2f05888b0e94131881b62d4c5ff04e2ed92e` |
| `retention-manifest.json` | 132,146 | `1790783493523907` | `44f37b4a07fde5d039b80c439bba0bfa53d88ca63d85f8e57560ddc92913415b` |
| `storage-receipt.json` | 1,300 | `1790783493787355` | `f7709b993156e45277038e6e7e508ac2ff68e6b7806c6d2acd86fc3c6ed5a404` |

All objects passed server size, MD5, SHA metadata and downloaded SHA256 checks.
Local receipt/result files are
`.runtime/olmo-nfr-stability-retention/endpoint-terminal-01.json` and
`endpoint-terminal-01-result.json`. No model state was uploaded again or
rehashed, no local files were deleted, and no GPU command was run by retention.

## Prepared final128 observer

The final observer's frozen `prepare.py`, preparation receipt, new helper,
tests and protocol were retained as a small additive source overlay, together
with a copy inventory. All 244 preparation pins matched before copying. The
historical training and endpoint source snapshots are already present in the
preceding static and terminal archives; this six-file archive avoids repeating
them. It is preparation only: no terminal128 checkpoint authority or actual
final-probe result is claimed.

Input directory:
`.runtime/olmo-nfr-stability-128/final-probe-static-retention-input-01`.
GCS prefix:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/nfr128-final-probe-static-01/`

| Object | Bytes | Generation | SHA256 |
| --- | ---: | --- | --- |
| `evidence.tar.gz` | 21,277 | `1790783852075089` | `18b5aa8eac10d97a8d7dbd0c14654870ff047beff0499d0b9191a43c010c29b6` |
| `retention-manifest.json` | 1,898 | `1790783852323984` | `cb0aa03dd76d711c2775c49ab8f958b9c4eee60b8b0999c5c6c1140dbc542ce4` |
| `storage-receipt.json` | 1,306 | `1790783852582314` | `7163a208e2d3ae668399450547267396cb4138edf9e8414dfd7b1b50f4c98a9f` |

The six files total 53,827 bytes before compression. Server size, MD5, SHA
metadata and downloaded SHA256 checks passed for all objects. Local
receipt/result files are under `.runtime/olmo-nfr-stability-retention/` as
`final-probe-static-01.json` and `final-probe-static-01-result.json`. The same
CPU-only container policy was used; no active directory or checkpoint state
was uploaded, no local file was deleted and no GPU command was run.

## Final-audit preparation and verified origin64

The static final-audit launcher, all twelve new/reused auditor source files,
immutable origin observation, completed development64 artifact and copy/check
receipt were retained in
`.runtime/olmo-nfr-stability-128/audit-origin-retention-input-01`.
Independent CPU checks confirmed exact inherited origin64 state and clocks,
an unchanged metadata-transition boundary and an exact raw development64
result. The active whole report was inspected but excluded from this archive;
its recorded observation hash is not terminal authority.

GCS prefix:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/nfr128-audit-origin-01/`

| Object | Bytes | Generation | SHA256 |
| --- | ---: | --- | --- |
| `evidence.tar.gz` | 76,229 | `1790784039480373` | `bb44bff92c43cb8985a598c3fd668623b7b9cf1c93d5334130473bf9c121649e` |
| `retention-manifest.json` | 4,438 | `1790784039770304` | `fec0f8fcb4a3918d8cfa3ed1fa53a2834205847be6bc3476d2f3499df529a63b` |
| `storage-receipt.json` | 1,289 | `1790784040052321` | `3370d7f09b2645a458f204edb9de2b1b4082fac18b8bb27f1220be8947df5207` |

Sixteen files total 506,371 bytes before compression. All objects passed server
size, MD5, SHA metadata and downloaded SHA256 checks. Local receipt/result
files are `.runtime/olmo-nfr-stability-retention/audit-origin-01.json` and
`audit-origin-01-result.json`. Final24-test evidence is explicitly owner-observed
stdout/session/exit status; no nonexistent raw log is claimed. This archive is
preparation and origin evidence, not a completed terminal128 audit. The CPU-only
retention neither touched training state nor uploaded checkpoint tensors.
