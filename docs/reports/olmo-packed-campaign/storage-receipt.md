# Packed campaign storage and recovery receipt

2026-09-29. Code and small evidence live on the persistent project disk. Corpus
and model/optimizer files staged on local SSD also have verified GCS copies.
No SSD checkpoint, corpus or earlier verified cloud object was deleted.

The namespace for this milestone is:

```text
gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T023500Z/
```

Local receipts are in `.runtime/olmo-packed-campaign/retention/`. Each stage
contains `evidence.tar.gz`, `retention-manifest.json` and an uploaded
`storage-receipt.json`. Only `packed-checkpoint-boundary-01` additionally
contains the model/optimizer `checkpoint/state.pt` and its committed manifest.
The evidence archives include the source snapshots declared by their reports.
Retention preserves failures and qualifications as well as operational passes.

## Verified evidence at this entry

| Local receipt | GCS stage suffix | Evidence members | Objects listed in receipt |
| --- | --- | ---: | ---: |
| `index-01.json` | `packed-index-01` | 14 | 2 |
| `index-restore-01.json` | `packed-index-restore-01` | 7 | 2 |
| `tiny-eager-01.json` | `packed-tiny-eager-01` | 74 | 2 |
| `tiny-graph-01.json` | `packed-tiny-graph-01` | 74 | 2 |
| `pretrained-graph-01.json` | `packed-pretrained-graph-01` | 74 | 2 |
| `precision-components-01.json` | `packed-precision-components-01` | 74 | 2 |
| `checkpoint-boundary-01.json` | `packed-checkpoint-boundary-01` | 78 | 4 |
| `pretrained-write-01.json` | `packed-pretrained-write-01` | 80 | 2 |
| `checkpoint-restore-01.json` | `packed-checkpoint-restore-01` | 6 | 2 |

A local read-only audit checked all nine verified receipts, their **20 listed
objects**, each retained archive/manifest SHA256 and byte count, **481 archived
member hash/size pairs**, and **446 declared source/snapshot pairs**. The
separately uploaded receipt objects are not included in the 20-object count.
Archive and retention-manifest uploads had already been downloaded and SHA256
verified by the retention tool. The local audit did not re-download every
object; the index and checkpoint recovery exercises below independently did
download their pinned cloud objects.

The boundary evidence is the saved update-one snapshot. Completed write-phase
evidence is retained separately under `packed-pretrained-write-01`; its report
SHA256 is
`960e65563ac4b66a51197546cb14e8eccd6d457bfdb32838c8136333d5c8ce38`.

## Actual-data checkpoint

The checkpoint was saved after the first completed packed NFR update: 524,288
valid input tokens, 512 real rows, 523,776 CE targets, 523,768 latent pairs and
523,248 KL triples. It includes the actual packed cursor, model, optimizer,
scheduler/counters and rank RNG state. Its configuration is two ranks, T1024,
B12/rank, 22 slots per rank, K4 FBT, native RT at layers 0/15 on every pass and
both NextLat losses. This is a readiness checkpoint, not an approved quality
training starting point.

All paths in this table are relative to the common namespace followed by
`packed-checkpoint-boundary-01/`:

| Object | Exact GCS generation | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `checkpoint/state.pt` | `1790650643862025` | 15,214,757,825 | `7b5e948eea2f1102676b26b4d9b883df398fad06c4b0523f3b0c79d12584b830` |
| `checkpoint/manifest.json` | `1790650644130539` | 58,065 | `9238b186a33c80be85aa18aec11cc2ea15f097e137be34eece4e978ca2886a50` |
| `evidence.tar.gz` | `1790650644356165` | 371,343 | `970239b3a289bcac434b95bf9a3044239a9b7d3fe3c3610e94ebeac4005b2f1f` |
| `retention-manifest.json` | `1790650644606797` | 20,655 | `09bedccb87573b0d422bb22c45c080f21ffacb09a9a3742eef0d4eae5bc7c39e` |

The upload receipt verifies server byte count, MD5 and SHA256 metadata for the
checkpoint objects. Its checkpoint `download_sha256` fields are intentionally
false because the upload step itself did not re-download 15 GB. A subsequent
independent restoration downloaded **both exact generations**, hashed all
**15,214,815,890 bytes**, matched the table above and verified the committed
checkpoint manifest/state relationship.

Original checkpoint:
`/mnt/localssd/cdrm-checkpoints/packed-campaign/pretrained-write-01`.
Fresh restoration:
`/mnt/localssd/cdrm-checkpoints/packed-campaign/cloud-restored-01`.

The byte-verification report is
`.runtime/olmo-packed-campaign/checkpoint-restore-evidence-01/report.json`,
SHA256 `cca970fe759c30db934dda130a04a595a344873b459ec26a4acf8d878b2768db`.
Its frozen evidence includes the reusable restoration script and two verified
source snapshots, retained under `packed-checkpoint-restore-01`.

**Byte restoration is verified. Fresh-process model continuation is still
pending at this entry.** File recovery does not by itself establish cold
T1024 DDP/graph construction with resident Adam, next-gradient equality or
bitwise optimizer continuation. The active resume stage will supply that
separate evidence; do not infer it from this receipt.

## Packed index recovery

The immutable train index contains 12,283 documents, 6,947,277 tokens and 6,785
nonoverlapping T1024 chunks. Its manifest SHA256 is
`372a7e05f5164198bdeb1531fc45bd761f5d33222d424c6804b54f19fd75288d`.
Its SQLite database is 3,321,856 bytes, SHA256
`68f37f9c0ba304581c822ec0227ced4027b49df3a61ce88eb0cc8e7bc84df0ef`.

Under `packed-index-01/`, recovery downloaded:

| Object | Exact GCS generation | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `evidence.tar.gz` | `1790649584570774` | 1,831,548 | `54b0ee32fa520bdc27be7c50a81b86a1a41dc31acc30948caeb1c8471ac80927` |
| `retention-manifest.json` | `1790649584900376` | 3,776 | `99405bacaa0a1727efc8215b368f74ea3ca30fb6c5f4ec270974ff4ec92ec81f` |

The CPU-only restore verified full object SHA256/size and all 14 archived
member pins, rejecting links, traversal and unexpected members during
extraction. The restored index reopened against the original verified corpus;
manifest, first-update chunk keys/counts, initial cursor and serialized
boundary-cursor behavior matched exactly. The cursor exercise was a metadata
test and did not run an optimizer update.

Restored index:
`.runtime/olmo-packed-campaign/index-cloud-restore-01/evidence/index`.
Restoration report:
`.runtime/olmo-packed-campaign/index-restore-evidence-01/report.json`,
SHA256 `625218c61f111799c33d2745bc3fec529e797689eaeaa2fc8920c45e9976cf8c`.
The reusable script, two source snapshots and report are retained under
`packed-index-restore-01`.

The original tokenized coverage fixture remains separately retained at
`gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/readiness-20260928/tokenized/`.
The packed index references those verified bytes; no retokenization, production
mixture selection or corpus cycling occurred.

## Pending closeout

Fresh-process continuation is active. Its completed report, source evidence
and any failure records must be retained before claiming restart acceptance.
Final closeout retention and the final receipt/member/source audit remain
pending. This entry does not clear the numerical qualifications described in
[precision assessment](precision-assessment.md).
