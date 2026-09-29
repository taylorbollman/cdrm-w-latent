# Recurrence/precision storage receipt

2026-09-29. The CPU baseline audit and both GPU diagnostics are retained:
**3 verified receipts, 6 listed objects, 748,652 downloaded bytes and 168
inventory members**. Independent CPU-only readback verified every pinned
object generation, full-object hash/size, inventory member and source
snapshot. No local files were deleted and no trained checkpoint was created.

The two model probes ran on one GPU/process; the baseline and independent
audits were CPU-only. The historical `olmo-two-gpu` namespace does not imply
DDP or two-GPU execution. Operational/storage passes do not clear numerical
qualifications: the fusion candidate did not improve forward and backbone
gradient agreement together and is not adopted. See [results](results.md).

## Exact retained objects

Common namespace:

```text
gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T053000Z/
```

Each complete relative object name below is appended to that namespace.
Use its pinned generation and SHA when restoring.

| Relative object name | Generation | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `recurrence-baseline-audit-01/evidence.tar.gz` | `1790659816052044` | 15,985 | `9b70d8faa2a0e51689b6119cfb5401a30c485bf77ad7b46dba926208fcd718e2` |
| `recurrence-baseline-audit-01/retention-manifest.json` | `1790659816336806` | 1,237 | `c307b3aa5f88c7a9ae1bbee6df33f5b461eb4317920acfd4271dd36999b3a3a2` |
| `recurrence-matrix-01/evidence.tar.gz` | `1790660177647439` | 344,456 | `3c84e6f91d6d6b80995b1f6fb5f7b4c463a7b895543115bd8637f58010d21ddb` |
| `recurrence-matrix-01/retention-manifest.json` | `1790660177923547` | 20,719 | `94278f7287c3324cd959994186cf25bea659387444d57a3a64c58790244f010f` |
| `recurrence-fusion-01/evidence.tar.gz` | `1790660767040519` | 344,758 | `c5bbf1a5e6bd3bd77f4c5355c4e6e0edfee540e054f67e866b571713b50dde78` |
| `recurrence-fusion-01/retention-manifest.json` | `1790660767308954` | 21,497 | `7d9c40a4471f0fd7be357152811ff97f6b6f27e94319dfd58efaf8ab0b00d743` |

| Stage | Inventory members | Archived source snapshots | Final report SHA-256 |
| --- | ---: | ---: | --- |
| `baseline-audit-01` | 3 | 1 | `917c5b59376f87ca409da3de7044969c27322d3dcda7ddc5592733d9234beb51` |
| `matrix-01` | 81 | 79 | `bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412` |
| `fusion-01` | 84 | 82 | `4f98763c43d5e94fc6c5f783712151a2f648f74ea4147b1cf2f6424bba6446a6` |

There are **162 archived source snapshots**: 161 across the two GPU stages
and the baseline audit's own source. This count is separate from the 308 older
PR40 source pairs rechecked inside the baseline audit. Each archive additionally
contains `RESTORE.md` and `evidence-members.json`, giving 174 regular archive
files including those six control files. Separately uploaded
`storage-receipt.json` control objects are outside the six listed-object count.

Local receipts are
`.runtime/olmo-recurrence-precision/retention/{baseline-audit-01,matrix-01,fusion-01}.json`,
with adjacent `.artifacts/` directories containing the uploaded bytes.
Every receipt records `checkpoint_uploaded=false` and
`local_files_deleted=false`.

## Independent readback and remaining closeout

The reusable script and atomic report are
`.runtime/olmo-recurrence-precision/storage-audit-01/{audit.py,report.json}`.
Report SHA-256:
`89c758b7258c90637bb1091f5f420a5bcb4a50d360bcdbd67c7e120ae656784b`;
script SHA-256:
`ceeba0505f9ba0bd9c7d0b0dc68c75c8990814e45edfdd2f96aa2f3c431285f9`.
It ran in the project container with GPU passthrough disabled, confirmed no
GPU device nodes, and used standard ADC with
`GOOGLE_APPLICATION_CREDENTIALS` unset. Each exact cloud generation's size,
MD5 and SHA metadata matched the receipt. Independently calculated SHA-256,
MD5 and size of the entire downloaded bytes also matched, as did local
retained copies.

Without filesystem extraction, the audit rejected unsafe/duplicate archive
names and special files, required the exact manifest inventory plus its two
control files, and checked each member's size, SHA and MD5. All archived report
bytes matched their pinned local final reports. All 162 source snapshots
matched the corresponding report inventories and current frozen sources.
The baseline-to-matrix-to-fusion reference pins also agree.

The independent matrix and fusion audits additionally verified exact controls,
reference reproduction, source integrity and the candidate's bounded scope;
their evidence lives in `matrix-audit-01/` and `fusion-audit-01/` under the same
runtime root. See [test ledger](test-ledger.md) for their report/script pins.

Final milestone closeout and retention of the matrix/fusion/storage audit
evidence are pending. The three stage receipts above are complete; no pending
closeout artifact is included in their totals. This milestone does not create
new model weights, validate checkpoint continuation or adopt a precision fix.
