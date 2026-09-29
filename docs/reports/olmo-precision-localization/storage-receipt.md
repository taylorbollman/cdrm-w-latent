# Numerical localization storage receipt

2026-09-29. Four completed diagnostic stages have verified cloud retention:
**4 receipts, 8 listed objects, 7,159,549 downloaded bytes and 318 inventory
members**, including 308 source snapshots. Two separate CPU-only audits
downloaded all eight exact object generations and verified their full bytes,
metadata, archive inventories and source pins. Nothing was deleted locally.

The stages are single-process numerical probes on CUDA device 0. The historical
`olmo-two-gpu` storage prefix does **not** mean these probes used DDP or both
GPUs. They performed no optimizer updates and created no newly trained
checkpoint. Storage verification and `passed_operational_diagnostic` statuses
do not clear the outstanding BF16 numerical qualifications; see
[results](results.md) and the [test ledger](test-ledger.md).

## Retained objects

The common namespace is:

```text
gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T043105Z/
```

Each object below is identified by that namespace plus its complete relative
object name. Pin the generation as well as the SHA when restoring.

| Relative object name | Generation | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `precision-bridge-01/evidence.tar.gz` | `1790656484569699` | 1,823,808 | `aee0af12fc549a3e7b51a97739fdd44adf90aac739bd8278064616cc130f32b2` |
| `precision-bridge-01/retention-manifest.json` | `1790656484859129` | 20,162 | `f5931f2a60159ca2c17337613bfe43802e787fee759a3dbedc8274b9179393f3` |
| `precision-auxiliary-01/evidence.tar.gz` | `1790656623447135` | 313,130 | `ae8b89ec45f4771280b9ede71029f2de8e078aae9987e45568359b008777d90d` |
| `precision-auxiliary-01/retention-manifest.json` | `1790656623722010` | 19,436 | `544cc443e1996aa54504c883fe0ed655e45ddd27c6355cb264fa7329b41f2718` |
| `precision-backend-cross-01/evidence.tar.gz` | `1790657156820142` | 320,987 | `b01b703aeb640faac4e59ecec09faaf9d3160343e3acd31225f00fcbd83e4bf5` |
| `precision-backend-cross-01/retention-manifest.json` | `1790657157079200` | 20,472 | `1a80c4e0421f8b198d2063c0ee62941c9540fb099fa81b9dc50688ba7529d55f` |
| `precision-attention-local-01/evidence.tar.gz` | `1790657915104994` | 4,620,322 | `3b084382670afcffa8d3722585f8d6e9075a7e6127c287517d5341951b9f6f91` |
| `precision-attention-local-01/retention-manifest.json` | `1790657915373047` | 21,232 | `52ad5354af40b44fd70829e04cf56be07a3445a9dca3631e67a4185526136473` |

| Stage | Inventory members | Source snapshots | Final report SHA-256 |
| --- | ---: | ---: | --- |
| `bridge-01` | 79 | 76 | `39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b` |
| `auxiliary-01` | 76 | 74 | `c4946da63c6275a4fcd926292b0296337f57d10553233c7184e94e38b617d5fc` |
| `backend-cross-01` | 80 | 78 | `97ced83fd037c907bd6a8ad34c377b96a0c1bc04dc424c21950cbf2194da9a65` |
| `attention-local-01` | 83 | 80 | `aa105ea3d1678f787840dc84d85997ed9e8ba60c33ae007f007aa3723717f51e` |

These member counts follow the retention manifest. Each archive additionally
contains `RESTORE.md` and `evidence-members.json`: 326 regular archive files in
total, of which 318 are individually inventoried evidence files. Each stage
also has a separately uploaded `storage-receipt.json` control object; those
four control objects are outside the eight listed-object total above.

Local verified receipts are in
`.runtime/olmo-precision-localization/retention/{bridge-01,auxiliary-01,backend-cross-01,attention-local-01}.json`.
Their adjacent `.artifacts/` directories contain the uploaded archives and
manifests. Each receipt records `checkpoint_uploaded=false` and
`local_files_deleted=false`.

## Independent readback

The reusable audits and their atomic reports are:

- `.runtime/olmo-precision-localization/storage-audit-01/audit.py`
- `.runtime/olmo-precision-localization/storage-audit-01/report.json`
- `.runtime/olmo-precision-localization/storage-audit-02/audit.py`
- `.runtime/olmo-precision-localization/storage-audit-02/report.json`

The first audit covers the initial three stages; the second covers the local
attention stage without changing the first audit or its evidence. Both ran in
the project container with GPU passthrough disabled, confirmed
no GPU device nodes, and used standard ADC with
`GOOGLE_APPLICATION_CREDENTIALS` unset. They checked each pinned cloud object's
server generation, size, MD5 and SHA metadata, then independently calculated
SHA-256, MD5 and size from the entire downloaded object. Local retained object
bytes passed the same pins.

Every archive member was inspected without filesystem extraction. The audits
rejected unsafe names, duplicate names and special files, required the exact
inventory plus the two archive control files, and verified every inventoried
member's SHA, MD5 and size. All 308 report-to-snapshot source hashes matched
the archived snapshot and the current frozen source. The four archived report
bytes also matched the local final reports.

The bridge's exported auxiliary fixture is retained inside its archive:
3,517,265 bytes, SHA-256
`aeab58a88c7eba15448a1b7630c9af747e492b3760b2364da5cd53380e063b27`.
The auxiliary report references this exact fixture, and the crossed-backend
report references the exact bridge report. The fixture contains small tensor
payloads and provenance; it is not a trained model checkpoint.

The local attention fixture is 7,023,735 bytes, SHA-256
`c08fa685da76f211fb50132db44af98e1d0c747d26c96480a5c082ee6b4e147c`.
The second audit decoded and verified all 48 tensor/mask payloads, the matched
reference report, the eight exact local Flash output hashes and the 24 finite
VJP cases. It preserves the qualification that eight fixed-input sites do not
clear the full-model discrepancy or globally exclude a Flash bug.

Each audit preserves its own source snapshot, copies of its input receipts and
downloaded objects in its separate directory. Their source, reports and input receipts are also retained in the verified
final closeout described below; duplicate downloaded stage archives were not
uploaded a second time. Closeout is separate from the four-diagnostic total.

## Final closeout

All four diagnostic stages and the independent audit evidence are retained.
`precision-closeout-01` preserves the final diagnostic code/tests, protocols,
results, ledger, next-step plan, usage, CPU logs and both independent audits.
Its receipt is `.runtime/olmo-precision-localization/retention/closeout-01.json`,
verified by full-object download hashes and server metadata. Final publication
and PR status live in Git's handoff/progress records, avoiding a circular
archive that tries to include its own final receipt hash.

| Closeout object | Generation | Bytes | SHA-256 |
| --- | --- | ---: | --- |

| `evidence.tar.gz` | `1790658298937758` | 54,528 | `3b543562eb52d2cc35c58e326742a343555568fc38eb25df5e004e49b49d8b4f` |
| `retention-manifest.json` | `1790658299199032` | 9,046 | `f25d7e41067c09fdfdf3805add4ca4ee26b8c735186f12cec7ead0641688ecb2` |

The closeout adds 34 inventory members and two listed objects, separate
from the four diagnostic stages and their independent eight-object audit.
Its source and member checks passed at retention. No local files were deleted,
no trained checkpoint was created, and no numerical qualification was cleared.
