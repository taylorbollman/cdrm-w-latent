# Two-H100 evidence storage receipt

2026-09-25. All31 stage attempts and eight checkpoint stages have verified
retention receipts. This inventory records existing verification; the local audit
rehashes archives/source snapshots and checks report manifests, without fetching
every large checkpoint again. No credentials are included.

Prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T172000Z/`.

Each stage has `evidence.tar.gz`, `retention-manifest.json` and
`storage-receipt.json`; checkpoint stages also have `checkpoint/state.pt` and
`checkpoint/manifest.json`. Evidence archives get server size/MD5/SHA metadata
checks plus downloaded SHA256 verification. Large checkpoint objects get server
size/MD5/SHA metadata verification against the locally hashed manifest.

| Stage | Final result | Checkpoint |
| --- | --- | --- |
| `actual-eager-01` | failed | — |
| `combined-anchored-01` | passed | — |
| `combined-graph-01` | passed | — |
| `combined-graph-recovery-01` | passed | Yes |
| `combined-zero1-01` | passed | Yes |
| `combined-zero1-buffer-01` | passed | Yes |
| `combined-zero1-graph-01` | passed | — |
| `ddp-combined-b128-01` | passed | — |
| `ddp-combined-b64-01` | passed | — |
| `ddp-rt-b128-01` | passed | — |
| `ddp-rt-b192-01` | passed | — |
| `ddp-rt-b64-01` | passed | — |
| `nccl-01` | passed | — |
| `rt-graph-01` | failed | — |
| `rt-graph-02` | passed | — |
| `rt-recovery-01` | passed | Yes |
| `single-combined-b128-01` | passed | — |
| `single-rt-b128-01` | passed | — |
| `tiny-capacity-01` | passed | — |
| `tiny-eager-01` | passed | — |
| `tiny-graph-01` | failed | — |
| `tiny-graph-02` | failed | — |
| `tiny-graph-03` | passed | — |
| `tiny-graph-recovery-01` | failed | — |
| `tiny-graph-recovery-02` | passed | Yes |
| `tiny-recovery-01` | passed | Yes |
| `tiny-zero1-01` | passed | Yes |
| `tiny-zero1-buffer-01` | passed | Yes |
| `tiny-zero1-graph-01` | passed | — |
| `zero1-combined-b128-01` | passed | — |
| `zero1-rt-b192-01` | passed | — |

## Checkpoint objects

All paths below are relative to the prefix. These are diagnostic recovery
checkpoints, not checkpoints from a quality-training campaign.

| Object | Bytes | SHA256 |
| --- | ---: | --- |
| `combined-graph-recovery-01/checkpoint/state.pt` | 15214755393 | `b8d1fa1d2cb271c04a8145dc7eb5a0e9c5b43940c613471f7339cd3282425691` |
| `combined-zero1-01/checkpoint/state.pt` | 15214742681 | `2c504fe563dec068a668d1daba287c5861d771e819bb84f0ecd0c6bb6f418be0` |
| `combined-zero1-buffer-01/checkpoint/state.pt` | 15214742681 | `8df45ce25ff58c6c15e99017a3231b8315380d8511637883918d3b4247a5a1b5` |
| `rt-recovery-01/checkpoint/state.pt` | 14154909669 | `cfde20254c18b00da4b45c03e295441919ba0a45986735adff1b7b6d4d5457d1` |
| `tiny-graph-recovery-02/checkpoint/state.pt` | 729364 | `cedfce6a69d16a37913c60f3db07c228126a8399322de8cf6746c88e6dcc77d6` |
| `tiny-recovery-01/checkpoint/state.pt` | 727764 | `2e9ef7dfce788deaf2bc706307e392040d16e49489c0ced78191aa1d92caad8b` |
| `tiny-zero1-01/checkpoint/state.pt` | 717600 | `beab233308f9c9ea788b37dcec4663b216de1307c0caf3b0fc048dc73e698a00` |
| `tiny-zero1-buffer-01/checkpoint/state.pt` | 717664 | `8c3f2fa4bc0eddaaf08638dc3ee7ca544b330c8cf3bdd33aaa2ee97c7bfa16be` |

Generated full-model local duplicates were removed only after verified cloud
retention and local SHA recheck; local manifest and cleanup receipts remain.
Tiny checkpoint files remain local. Original OLMo weights are retained by the
earlier O1 manifest/receipt and were not duplicated in these archives.

## Closeout audit

The `closeout` archive retains the final inventory, audit generator/README,
performance and memory CSVs, all stage receipts, checkpoint cleanup receipts,
hardware inventory and CPU/launcher logs. It is a separate evidence bundle,
not a32nd execution stage. Its verified receipt is stored locally as
`.runtime/olmo-two-gpu/retention/closeout.json`.

Audit result:31final reports,26pass/5fail,0OOM/unfinished;4,492 pinned source
pairs verified with no source/archive mismatch. The two setup stubs lacking full
source pins remain explicit. All31 execution-stage receipts and8 checkpoint
receipts verify. See [results](results.md) and [test ledger](test-ledger.md) for
interpretation; failed reports are never relabeled by passing retries.

Rebuild the audit from the project checkout using standard-library Python:

```bash
python3 .runtime/olmo-two-gpu/audit/inventory.py
```

After restoring the closeout archive elsewhere, pass
`--evidence-root /path/to/restored/stage-root` explicitly to its audit script.
Source and stage archives must be restored too before a complete re-audit.

Closeout verification completed: 127 retained files.

| Closeout object | Bytes | SHA256 |
| --- | ---: | --- |
| `evidence.tar.gz` | 443369 | `b800c0b59cb6b1060a3adae682dfbcf770c3047bbdcd9490392449b465592cf2` |
| `retention-manifest.json` | 30217 | `28c6baaef7c66c0ec9a1a54025e81b2694ec486f0e6916be94f541d4d7dfc897` |
