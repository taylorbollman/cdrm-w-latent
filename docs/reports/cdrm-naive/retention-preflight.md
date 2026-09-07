# CDRM retention preflight

The first section records the 12-block `source-pilot-v3` preflight before the user
changed the plan to six-block screening. The later `source-six-screen-v4` audit is
recorded separately below; changes between those freezes are intentional.

Read-only audit on 2026-09-07 while the pilot was running. Scope: the local
`source-pilot-v3` archive and checksum record, the frozen pilot decision, all local
MAD data references, and the existing GCS data upload/verification logs. No GPU,
checkpoint mutation, cloud mutation, `.env` read or unrelated-project access was
performed. This is a preflight; final checkpoint and report retention remains the
root controller's later step.

**The local source archive is complete for the frozen execution source set and
passes every recorded hash check.**

| Artifact | Identity |
|---|---|
| Archive | `.runtime/cdrm-naive/20260907T123830Z/source/source-pilot-v3.tar.gz` |
| Archive size | 8,521,920 bytes |
| Archive SHA256 | `3da35594e2d8bade58fb26a8152882d93d89e55d6eb0d8266a0d8c59746b7b59` |
| Per-file record | `source/source-pilot-v3.json`; SHA256 `0e7d12d11d6988a05b8916869107e12fd6c1afae5d8a8d36dcceb8536173051e` |
| Archive creation | 2026-09-07 13:10:44 UTC |
| Pilot-decision SHA256 | `68b544a9e40eebc04831d4e316a1e56b873ed0b32369b852b0cc656ac9755ca3` |
| Project / recurrent fork HEAD | `a605d3c167034e2a1da8f3acc4677f3d07b9c06f` / `4b07f023a61f8b3ecee52651b2bcbd0d8f4292ad` |

The tar was inspected without extraction. All **154 regular-file members** match
their per-file SHA256 values; there are no missing, extra or duplicate members.
All **39 execution-source hashes** in `pilot-decision.json` match both the archived
bytes and current frozen working-tree files. The other archived files also match
their current counterparts at audit time. Working-tree changes are included; the
archive, rather than Git HEAD alone, identifies the executed implementation. The
two pilot-decision copies match byte-for-byte, and all **10 referenced NUM/OPS
quality-gate reports** match their decision-record hashes.

Coverage includes:

- The OLMo core, `olmo/cdrm.py`, model/configuration changes and checkpoint
  conversion; 34 files under `recurrent-transformer/olmo`.
- All four CDRM common/train/validate/prepare scripts, their reused Stage A/B and
  R3 helper scripts, and execution queue scripts; 27 project scripts in total.
- `cdrm/mad_data.py`, the other six `cdrm` package files, all 14 CDRM configuration
  files, and the frozen baseline/configuration provenance.
- `test_cdrm_reference.py`, checkpoint-conversion and recurrent-foundation tests,
  MAD oracle tests and other retained tests: 29 recurrent-fork test files plus six
  project test files. Inclusion does not imply every archived test was executed.
- All 12 pinned MAD source/provenance/config/license files and three Zoology
  snapshot files, plus `recurrent-transformer/LICENSE`.
- The authorized brief copy, CDRM plan/data/usage documents, repository instructions,
  Dockerfile, pinned requirements, Docker launch/build scripts and recurrent-fork
  package metadata. Third-party installed packages and the Docker image itself are
  not embedded in this source archive.

No archive member is a link, absolute path or parent-traversal path. A filename
scan identified no environment files, credential stores, private-key filenames or
container-home directories. A supplementary byte-pattern check found no private-key
headers or service-account credential objects. This bounded check supports the
stated archive-file exclusion; it is not a general certification against every
possible secret representation. No credential/environment file was opened.

**All local data files are covered by the frozen manifest chain.** The top manifest
matches the pilot decision's SHA256:
`b4e2a7f4daca652737552cfb15be0b09b18fb74b9e5efc4f5aa47f3c67a22e7b`.
Following its references verifies exactly **21 files**: one top manifest, six split
manifests, six array archives, six metadata files and two epoch-permutation arrays.
Every file SHA256 matches; there are no unreferenced files under `data/`. Split
array identities agree between parent and child manifests, and every pinned MAD
source hash in the data provenance still matches. The existing preparation record
separately establishes array readback and oracle correctness; this audit hashes the
retained files and does not regenerate the corpus.

The upload destination is
`gs://fast-chunks/cdrm-w-latent/cdrm-naive/20260907T123830Z/data/`.
`execution/archive-data-upload.log` records a copy for each of the 21 local files,
with no omitted path or destination outside that prefix. Its SHA256 is
`4bfe2343488b51e62e4d2cf3ba7f893c06a03a6466671c67cb2418ae8ce8029d`.
`execution/archive-data-verify.log` records local and remote listings of 21 entries
and no difference/copy lines; its SHA256 is
`a355f8de7e72f62c44791186d3e8fd17d0d35edfd0118f1f2c31741bbffbecd6`.
These log files alone do not retain their exact verification command, checksum
flags or exit code, and this preflight did not independently download remote
objects. Consequently, they support upload coverage and an apparently empty
verification comparison, not a new independent remote SHA256-readback claim.

**Remaining retention scope is explicit.** The source tar does not contain data
arrays, learned checkpoints, NUM/OPS result artifacts, `pilot-decision.json`, or
Markdown/JSON reports under `docs/reports/cdrm-naive`; those belong in the separate
lineage retention pass. In particular, this preflight and the protocol-alignment
audit must be retained with the final reports. The reviewed logs establish data
upload activity only; they do not establish upload of `source-pilot-v3`, its JSON
record or future checkpoints. Final cloud retention should preserve these artifacts,
their per-file checksums and command/result evidence, and verify remote manifests
or equivalent readback before claiming complete retained lineage. No local
source/data integrity gap was found.

## Six-block screening source and data supplement

After the old SEQ-12 run completed and the source guard was released, the new
`source-six-screen-v4` freeze was independently checked without extraction or GPU
execution. Its creation timestamp is 2026-09-07 13:23:33 UTC.

| Artifact | Verified identity |
|---|---|
| `source/source-six-screen-v4.tar.gz` | 8,543,924 bytes; SHA256 `fe9ddad1ed225062fb177a4a691ba0d44d3e3a30faf5998a385b9bb638657720` |
| `source/source-six-screen-v4.json` | SHA256 `517cf8799b0f10d94517d4c6fba306046da3f10a8ce3c0425a159c3125ab78d3` |

All **159 regular-file members** match the JSON per-file hashes and their current
working-tree counterparts at this second audit. There are no duplicate, extra or
missing members, links, unsafe paths, credential/environment filenames, private-key
headers or service-account credential signatures. The v4 archive includes the
generalized data/preparation implementation and tests, the new six-block base
configuration, revised runner/validation helpers, early-stop logic test script,
six-block steering document and preflight queue. The changes from v3 are retained
as their own freeze; v3 remains the source reference for the earlier 12-block run.

Each new screening corpus contains eight referenced files: one manifest, two split
manifests, two array files, two metadata files and one epoch-order array. All **16
local file references** across `data-screening-v1/recall-v128-t128` and
`data-screening-v1/copy-v16-t256-k96` pass their SHA256 checks. Neither root contains
a final split. Both epoch-order files share SHA256
`d6639bb93cfa79d85b8da67f25bc9f34d9848919526ded04eab6ba190f5e750d`, confirming
identical retained permutations for the two fixed-size corpora.

The newer [screening data guide](../../cdrm-mad-screening.md) was written after the
v4 freeze and is not inside that tar; include it with the final documentation
retention. No cloud upload/readback claim for the screening corpora or v4 archive
is made here. The earlier data-upload logs cover the original two-task corpus,
not these newly generated roots. Future final splits and their supplemental
manifests also remain outside this preflight.

The already-declared optional third corpus, `data-screening-v1/copy-v128-t256-k16`,
was subsequently prepared with the unchanged v4 data/preparation scripts. All eight
of its referenced files hash-verify, its 12,800 train and 1,280 dev examples pass
the native-label/independent-oracle checks, and no final split exists. Its top
manifest SHA256 is
`385c24dea422d63ec5d97676f3e4368b969714a04315c1d4391d4fc4aae6c85e`.
`execution/data-screening-copy-v128-prepare.json` retains the exact CPU command,
exit code 0, completion timestamp and log/manifest hashes. No remote-retention claim
is made for this additional corpus.
