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
`storage-receipt.json`. `packed-checkpoint-boundary-01` and
`packed-checkpoint-boundary-02` additionally contain the model/optimizer
`checkpoint/state.pt` and its committed manifest.
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
| `pretrained-resume-01.json` | `packed-pretrained-resume-01` | 80 | 2 |
| `flash-t16-d0-01.json` | `packed-flash-t16-d0-01` | 8 | 2 |
| `flash-t16-d1-01.json` | `packed-flash-t16-d1-01` | 8 | 2 |
| `flash-t1024-d0-01.json` | `packed-flash-t1024-d0-01` | 8 | 2 |
| `flash-t1024-d1-01.json` | `packed-flash-t1024-d1-01` | 8 | 2 |
| `checkpoint-boundary-02.json` | `packed-checkpoint-boundary-02` | 79 | 4 |
| `pretrained-write-02.json` | `packed-pretrained-write-02` | 81 | 2 |
| `checkpoint-restore-02.json` | `packed-checkpoint-restore-02` | 6 | 2 |
| `pretrained-resume-02.json` | `packed-pretrained-resume-02` | 81 | 2 |
| `closeout-01.json` | `packed-closeout-01` | 38 | 2 |

A local read-only audit checked the 18 experiment-stage receipts, their **40 listed
objects**, each retained archive/manifest SHA256 and byte count, **840 archived
member hash/size pairs**, and **779 declared source/snapshot pairs**. All 840
members also match their original local stage bytes; the 779 source pairs
match both local and archived snapshots. The separately uploaded receipt
objects are not included in the 40-object count.
The separately verified final closeout brings the complete milestone to **19
verified receipts, 42 listed objects, 878 archived member pins and 780 declared
source/snapshot pairs**. Its exact object pins appear below.
Archive and retention-manifest uploads had already been downloaded and SHA256
verified by the retention tool. The local audit did not re-download every
object; the index and checkpoint recovery exercises below independently did
download their pinned cloud objects.

The boundary evidence is the saved update-one snapshot. Completed write-phase
evidence is retained separately under `packed-pretrained-write-01`; its report
SHA256 is
`960e65563ac4b66a51197546cb14e8eccd6d457bfdb32838c8136333d5c8ce38`.

## First actual-data checkpoint and retained restart failure

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

**Byte restoration passed, but the first fresh-process continuation failed
bitwise equality.** Its failure report and source snapshots remain retained
under `packed-pretrained-resume-01`. Restored state before the update, graph
preparation, next input/noise, forward losses and predictor gradients matched;
backbone gradients differed, followed by a different gradient norm and updated
state. Do not label this first pair a successful exact restart.

The subsequent isolated Flash tests are also retained. At T16, deterministic
mode off/on both repeated exactly. At T1024, mode off produced query-gradient
differences (maximum aggregate relative L2 `3.917861750344298e-6`), while mode
on repeated exactly across three eager backwards and three graph replays. These
are fixed-input attention diagnostics, not whole-model continuation tests or
clearance of the separate BF16 gradient qualifications.

## Deterministic rerun checkpoint

The second write phase passed all 13 gates and saved a new update-one
checkpoint. Its harness records deterministic algorithms enabled,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN deterministic mode enabled and cuDNN
benchmarking disabled. Its logical data, model and update size match the first
pair; it has its own source/configuration fingerprint and checkpoint.

Completed write report SHA256:
`a3c1b48913ecc95f087cc1a79ddc3c70e6ac0642e0f0e5ee0f3883b42b230754`.
Completed write evidence is retained separately under
`packed-pretrained-write-02`.

These exact objects are under `packed-checkpoint-boundary-02/`:

| Object | Exact GCS generation | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `checkpoint/state.pt` | `1790653137378773` | 15,214,758,081 | `c27186bb4847d4325821ef96916e85e69e3c9246a73eb4886d343941467e1199` |
| `checkpoint/manifest.json` | `1790653137632395` | 58,373 | `c1cdb79fc58b767160bf6466777b434665f4271f731e03698549b28d85e74ed1` |
| `evidence.tar.gz` | `1790653137843529` | 373,961 | `dafcb23626b56e8a2cf0bdb829f95dc85bd774c4af5c14166c27dc6942244acc` |
| `retention-manifest.json` | `1790653138099686` | 20,906 | `a91cdf386a3805239c7177714b0f005645dc6144808d98fdf1dc5cee385d1bef` |

The separately retained completed write archive is 605,709 bytes, SHA256
`1476bdf65562f20ca1b06c2bc51a4bd858e208050d6200381d2a7117b67e5283`,
generation `1790653015576665`. Its retention manifest is 20,581 bytes, SHA256
`4d8e312da327f00cf388413ffce018c6111ead06d540339c2681baaab29560b7`,
generation `1790653015847241`. Local independent inspection verified both
archives/manifests, all their member pins and all declared source snapshots.

Original checkpoint:
`/mnt/localssd/cdrm-checkpoints/packed-campaign/pretrained-write-02`.
Fresh restoration:
`/mnt/localssd/cdrm-checkpoints/packed-campaign/cloud-restored-02`.
The state and manifest upload receipt verifies server byte count, MD5 and
SHA256 metadata. A separate generation-pinned restoration then downloaded and
verified all **15,214,816,454 checkpoint bytes**, including the committed
manifest/state relationship. Its report is
`.runtime/olmo-packed-campaign/checkpoint-restore-evidence-02/report.json`,
SHA256 `d49ce991cd97862143c9412e917b2de24e3d92fdadcd8979d2dab9fe4960e814`.
That byte-verification evidence is retained under `packed-checkpoint-restore-02`.

**Fresh-process continuation for pair 02 passed all 10 gates.** The loaded
model and Adam state were already resident before cold DDP warmup and graph
capture at T1024/B12. Both ranks' next input/noise hashes, raw gradients,
metrics, updated model/optimizer state, counters, cursor and RNG state/draws
matched the original live-graph continuation exactly. The completed resume
report is `.runtime/olmo-packed-campaign/pretrained-resume-02/report.json`,
SHA256 `18666d8cfed2ef79bb5a20f0f569c81c25dcf8b08b6a20842a5983f42e821603`,
retained under `packed-pretrained-resume-02`.

This acceptance is limited to the recorded deterministic configuration, two
H100s, same world size and runtime, the pinned packed fixture/index and this
single next-update comparison. It does not qualify changed world size, H200s,
arbitrary production mixtures or long training trajectories. The first
checkpoint and its failed continuation remain intact; the successful second
pair does not overwrite that result or clear the BF16 numerical qualifications.

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

## Final evidence audit and retained closeout archive

The bounded stdlib audit at
`.runtime/olmo-packed-campaign/closeout-01/audit.py` completed successfully in
an explicitly CPU-only container. Its atomic report is
`.runtime/olmo-packed-campaign/closeout-01/report.json`, SHA256
`486025c051404a0a501409ed1497fd52224e4b689a2998bbc49cd900e32090fe`.
It verifies the 18 stage reports and receipts, the artifact/source counts
above, both checkpoint restoration pin sets, index recovery, and pair 02's
exact continuation directly from both ranks' recorded values. It also requires
pair 01's failed comparison to remain failed, validates the four isolated
Flash diagnostics, and preserves the independent BF16 qualification failures
and component comparison results. It did not re-read either 15 GB state file;
the full cloud downloads are independently pinned by their completed
restoration reports.

The final closeout is retained under `packed-closeout-01/`:

| Object | Exact GCS generation | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `evidence.tar.gz` | `1790654319700429` | 123,441 | `103161b4f93876b906b367c376081139277af78306ff49174437be3dc6b6c667` |
| `retention-manifest.json` | `1790654319947551` | 9,222 | `904e5f6af17ffa6fb631973f3a63eafe0fa1224bd7814204773b1fbc7c4a7136` |

The upload receipt verifies server size/MD5/SHA metadata and full download
SHA256 for both objects. Independent local inspection matched both objects
against that receipt, all **38 archived member pins**, and the audit's source
snapshot. The archive contains exact copies of all **18 experiment-stage
receipts**, all **four CPU test logs**, the final GPU-status log, **11
documentation files**, and the completed closeout report/script. Neither
checkpoint was re-read or deleted for this final archive check.

This completes storage verification for **19 receipts and 42 listed objects**;
separately uploaded receipt objects remain excluded from the object total.
The closeout directory/archive is frozen. Its documentation records the
pre-upload snapshot; this final receipt addendum is recorded in the repository
without changing the archived bytes. No numerical clearance follows from the
evidence audit; see [precision assessment](precision-assessment.md).
