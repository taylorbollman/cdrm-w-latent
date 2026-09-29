# Operator notes

Work on the `feat/olmo-pilot-data` branch until its closeout records a merge.
Large files are in `/mnt/localssd/cdrm-data/olmo-dolma-v1_5-pilot-20260929`;
recovery uses the separate `...-20260929-recovery` directory. Persistent small
evidence is under `.runtime/olmo-pilot-data`. No old SSD or boot files are pruned.

Run CPU work in the project container with `CDRM_DOCKER_GPUS=none` and
`CDRM_FLASH_ATTENTION_SOURCE=installed`. Never silently use CPU for a GPU test.
For GCS tools, unset an inherited `GOOGLE_APPLICATION_CREDENTIALS` override and
use the instance identity, following existing project retention commands.

The declared source plan is
`.runtime/olmo-pilot-data/declaration-01/source-plan.json`, SHA256
`5954f4480268e50c9041ffc06651a6539efe93ec54f6eb36a2977a4d5e2a9707`.
Its helper/test source hashes are part of the immutable extraction contract.
Do not edit those files and attempt to continue old extraction in place.

`olmo_pilot_data_prepare.py acquire` processes one named source. Require a
retained receipt for every previous source, use the same plan/pin/raw root and
persistent acquisition evidence. A failed attempt is not a committed source;
there is no automatic retry or substitute URL. Completed sources have a raw
manifest, exact retained raw bytes, original line mapping and cloud generations.

`olmo_pilot_data_prepare.py prepare` requires every raw source and its receipt.
Each call writes at most four committed shards, retains those before continuing,
and reports either `retained_partial` or `complete`. Repeat only the same fixed
configuration. Raw receipt paths are operational locations, not corpus identity.
Each configuration retains the exact raw-fragment SourcePin plus upstream map.

`olmo_pilot_data_restore.py` restores one document-retention stage from a
separately pinned receipt. Use fresh data and evidence destinations. Raw stages
contain `raw.jsonl`, `source-lines.jsonl` and their manifest. Configuration stages
contain exact original `config.json` bytes and source authority. A partial
tokenized corpus can be reconstructed from that config and consecutively
committed shard directories; no complete root manifest should be fabricated.
The final root-summary stage retains exact `tokenized-manifest.json` bytes.

After recovery, run semantic verification and the matching preparation/reader
checks. A successful byte download is not a sufficient data validation result.
No training checkpoint or RNG state is involved in this milestone.

The ordered builder publishes only after all declared quotas and reserve minima
fit. If it fails for insufficient data, retain that failure and revise the
declaration explicitly; do not shrink panels or cycle examples in place.
SQLite indexes use their real extension and a separate authenticated schema.
Existing small-evidence archives accept SQLite but cap uncompressed members at
128MiB. Larger indexes require an explicit streaming index-retention helper,
not changes to old helpers or an opaque file-extension workaround.
