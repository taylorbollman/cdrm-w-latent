# Ordinary two-H100 evidence storage receipt

2026-09-25. All eight execution stages pass and have verified retention receipts.
The audit checks 1,303 pinned source pairs with no source, archive or retention
mismatch. There are no unfinished stages or newly generated full checkpoints.
The disposable throughput updates do not replace the retained original O1 weights.

Prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T195200Z/`.

Each stage has `evidence.tar.gz`, `retention-manifest.json` and
`storage-receipt.json`. The helper verifies server size, MD5 and SHA256 metadata
and downloads these small evidence objects to verify SHA256. Local receipts
are under `.runtime/olmo-ordinary-two-gpu/retention/`.

| Stage | Receipt | Archive members | Archive bytes | Archive SHA256 |
| --- | --- | ---: | ---: | --- |
| `ordinary-dao-correctness-b1-01` | verified | 169 | 690613 | `99073ff7aa2251949cc6354baca2f44ce76448b73274a7ee23919d0c6135907c` |
| `ordinary-dao-ddp-b128-01` | verified | 169 | 693323 | `c2011176084aee2c71d5ecf5a1462062bf55d895a1a0248d4feca20ccbc96e05` |
| `ordinary-dao-ddp-b192-01` | verified | 169 | 693283 | `e2d9a1612b36318bf76a85baf862d347563ba8f5904acf6cd4bae7fe542d87a0` |
| `ordinary-dao-ddp-b32-01` | verified | 169 | 693220 | `6264bf895a947f330ec5ef802886e17a7dc12aeec1005b6f116ed118b8c93736` |
| `ordinary-dao-ddp-b64-01` | verified | 169 | 693209 | `77ef71cda3664a4b7ae9b4712d42306b1b624e2b179e7ccd3c0cb76bbfb2daa0` |
| `ordinary-dao-ddp-b64-02` | verified | 169 | 693221 | `bdb5e78fa64d223b679bb9d430b66c0c2ed13ecffe98e3b036ff730bcfbb30f3` |
| `ordinary-dao-single-b128-01` | verified | 167 | 671043 | `7e085aa78a21197be2ed66c787ac15de6a814f8313d248aff8a6d5010fbfae02` |
| `ordinary-native-ddp-b64-01` | verified | 167 | 686194 | `23cd838db812b6d26c13385a5a5751fedd392a2e7f1235890408c06e666d74ad` |

## Closeout bundle

The separate `closeout` archive contains the final audit scripts/README,
JSON/CSV inventory, PDF/PNG plots, source-test log, launcher logs, final idle-GPU
check and eight stage receipts. It is not a ninth execution stage. Retained
stage archives include configuration, source snapshots and pins, rank reports,
W&B links and the prospective ordinary protocol. The previous RT/combined
milestone remains a separate, unchanged evidence collection.

Audit regeneration uses standard-library Python:

```bash
python3 .runtime/olmo-ordinary-two-gpu/audit/summarize.py
```

The CPU-container plotting command and restoration notes are in the archived
`audit/README.md`. The audit rehashes local retained artifacts; cloud status
comes from the saved verification receipts, without downloading them again.

Closeout verification completed: 49 retained members.

| Closeout object | Bytes | SHA256 |
| --- | ---: | --- |
| `evidence.tar.gz` | 449653 | `5291a1b8f256d4f5a873ba55e500a53278e0923be8b0fa88fe97f92b84a09073` |
| `retention-manifest.json` | 11858 | `82405f9fd89e2f559e5249950c015c88b98f11cdd0c331f8ea8f5b5e1e9768a0` |
