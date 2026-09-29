# Retained optimizer diagnostic and public reference

The fixed-state diagnostic created no new model checkpoint and changed no live
training state. Its existing update-20 checkpoint remains under the prior
continuation's retained authority. Small evidence was uploaded with create-only
writes and verified by generation, size, MD5, SHA256 metadata and downloaded
SHA256. No local files were deleted.

The diagnostic and public-reference stages are under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T155500Z/`:

| Stage | Local receipt |
| --- | --- |
| `optimizer-history-probe-01` | `.runtime/olmo-optimizer-history/receipts/probe-01.json` |
| `optimizer-history-public-loss-reference-01` | `.runtime/olmo-optimizer-history/receipts/public-loss-reference-01.json` |

Each stage contains `evidence.tar.gz` and `retention-manifest.json`; the receipts
hold exact object generations and byte/hash pins. The probe report SHA256 is
`035a417908b4c673c3b63983e8303909ec7b21866e4e3a26c45250319d5f284e`.
Its archive includes the closed report, launcher log and all 148 frozen source
snapshots. W&B run `yezbu1wv` is synced.

The public-reference report SHA256 is
`2015010d216a243dbc1560a10d01a8b81d9ebc9045ca941494e00f1be13c3804`.
Its archive preserves a rerunnable unauthenticated query, exact official W&B
response, 22 history rows, selected configuration and pinned historical trainer
source. The raw 39,110-byte response SHA256 is
`9f404d5f613a8b9835664b930befe3bb50c91de416d6a31d68da7cf25927fb05`.
This supports the published CE reference; it is not a new local model run or a
verification that original optimizer payloads are currently downloadable.

Final CPU logs and their snapshots are also retained in the companion
campaign-evaluation `cpu-02` stage. See the evaluation storage record for the
shared PR closeout bundle and checkpoint inventory.
