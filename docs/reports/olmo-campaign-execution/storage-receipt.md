# Execution acceptance storage

All GPU stages are closed and W&B synced. All checkpoint payloads and small
stage evidence were retained with create-only writes, generation pins, byte
counts, SHA256, server MD5 and downloaded SHA256 verification. No local files
were deleted. Disk space should be reviewed before a longer campaign; this
milestone does not implement local pruning.

Runtime records: `.runtime/olmo-campaign-execution/`.
The pre-closeout `inventory-01/report.json` crosschecks **22 small-stage receipts,
44 evidence objects (12,779,261 bytes), and 34 checkpoint objects including
17 state files (55,990,621,427 bytes total)**. Objects are deduplicated by URI and
generation, not by identical content. Inventory and later closeout receipts are
excluded from those snapshot totals.

Inventory report SHA256:
`a359d12fc20a9c78ac3047fd53ea881c776326a662d2a6ea59acad0c6dcfbcbb`.
This is a local metadata/receipt consistency check, **not another complete cloud
readback**. Its verification flags describe the preceding producer readbacks.
The actual tiny and native stop checkpoints were additionally downloaded from
the pinned GCS generations into new persistent directories and used by the
successful fresh-process continuations.

## Small evidence

Prefix for each closed stage:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T142100Z/execution-<stage>/`.
Each has `evidence.tar.gz` and `retention-manifest.json`. Exact generation, size,
SHA256, MD5 and readback flags are in `receipts/<stage>.json`. The retained
inventory copies those receipts and enumerates all authorities.

Included scopes: CPU authority/declaration preflights (including superseded
preflight-only declarations-01), development/final CPU logs, tiny reference,
lean/stop/restore/resume/terminal stages, native stop/restore/reference/resume,
and the independent closed-report audits. Frozen source snapshots and commands
are included. A preflight is not labeled a GPU attempt.

## Native checkpoint authorities

Root:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T142100Z/campaign-execution/NFR/`.
The historical `olmo-fusion-startup` storage prefix does not change this milestone's
training lineage. Each directory below contains `state.pt` and `manifest.json`.

| Directory relative to root | State generation | Manifest generation |
| --- | --- | --- |
| `native-stop-01/update-000001` | 1790693082036481 | 1790693159573777 |
| `native-reference-01/update-000003` | 1790694178754330 | 1790694236177797 |
| `native-resume-01/update-000003` | 1790695160535063 | 1790695210766273 |

Manifest SHA256 pins, in the same order:

- Stop/update 1: `576d834696977f7c51e27a7e62c3a19bedd935264265cddd58721781b34fd686`.
- Reference/update 3: `a6b6e2a26d4714e61146fc2812be672f4fd7f6b988fe3408e2a643bf87c045d2`.
- Continuation/update 3: `a09c9bb37ef51e706cd18fc1b9cbfb81caa325e647ff17c0c3a5a868233e3c32`.

The first two state payloads are 15,214,848,577 bytes each; continuation state is
15,214,848,769 bytes. Their serialized files have different hashes; the numerical
training states compare exactly. Use each checkpoint's own manifest and state
pins, not another run's file hash. Origin and all tiny authorities are also in
the machine-readable inventory.

`native-restored-01/checkpoint` is the verified download of update 1.
`checkpoints-native-resume-01/update-000003` is the completed continuation.
The finite three-update plan is complete: a same-lineage resume validates and
exits; it does not authorize or silently extend training. See operator-notes.md
and next-steps.md for recovery scope and the next milestone.
