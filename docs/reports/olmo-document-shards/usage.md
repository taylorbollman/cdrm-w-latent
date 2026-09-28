# Complete-document preparation and recovery

This milestone prepares reusable tokenized documents. It does **not** construct
training chunks, choose a model boundary policy, or qualify packed RT/FBT/NextLat
execution. No model or training defaults change.

The readiness fixture covers seven explicitly selected Dolma v1.5 source prefixes,
targeting roughly one million tokens per source. Prefix selection and equal
source token targets are coverage checks, **not** a representative sample or the
final training mixture. See `source-pins.json`, `olmo-source-audit.md`, and the
eventual results/storage receipt for the precise source and retained artifact pins.

## Commands and API

Run preparation in the CPU container, with GPU passthrough explicitly disabled:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_prepare_documents.py --help'
```

`scripts/olmo_document_extract.py` obtains bounded complete-document prefixes.
`scripts/olmo_prepare_documents.py` validates those extracts and calls the reusable
API below. Its required arguments are `--raw-dir`, `--output-dir`, `--cloud-root`
and `--report`; `--max-new-shards 2` stops after two atomic shard commits. Omit that
argument on a later invocation to finish. Keep the source plan, cloud root and
preprocessing settings identical when resuming. Local paths may change.

```python
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained.document_shards import (
    prepare_document_shards, verify_document_shards, iter_documents,
)

summary = prepare_document_shards(
    sources,  # Ordered LocalJSONLSource records with verified SourcePin SHA256s.
    output_dir,
    tokenizer_path=tokenizer_json,
    split_policy=SplitPolicy(20260928,
                            (("train", 98), ("dev", 1), ("confirmation", 1))),
    max_documents_per_shard=1024,
    target_tokens_per_shard=250_000,
    max_new_shards=2,
)
verified = verify_document_shards(output_dir)
for document in iter_documents(output_dir):
    pass  # Complete unique TokenizedDocument; no context-window truncation.
```

The wrapper freezes the displayed split and shard settings. The first two actual
shards contain 638,155 tokens across ten documents: the 250,000-token target is
soft because a whole document is never split to meet that target. The row limit
counts source rows, including duplicate rows. The default 64 MiB JSONL-record
limit rejects an oversized record explicitly; it never truncates the text.

## Tokenization, identity and provenance

- The tokenizer JSON must match the native OLMo SHA256 and size in
  `cdrm/pretrained/olmo_artifacts.py:FILE_SPECS`. Its serialized normalization is
  retained. Implicit special-token insertion, padding and truncation are disabled.
- Tokens are stored as unsigned 16-bit little-endian values. Append EOS 50279
  only when the document does not already end in EOS. Literal internal EOS tokens
  are preserved and counted; offsets and source records define real document
  boundaries, rather than inferring them from every EOS occurrence.
- Split identity is SHA256 of the token bytes after excluding exactly one final
  EOS. Token-equivalent text, including normalization-equivalent Unicode and
  supplied-versus-appended terminal EOS, therefore receives the same split.
  The existing `SplitPolicy` hash allocation supplies deterministic 98:1:1
  weights, not exact document or token counts in each split.
- Exact token duplicates retain their first occurrence globally within this
  prepared artifact. Every later row still has provenance and a `duplicate_of`
  reference. A repeated source/document ID with conflicting normalized tokens
  is rejected. This is neither near-duplicate removal nor evaluation-set
  decontamination; those remain dataset-level decisions.
- Metadata retains source index/line, original document ID, unique document key,
  raw UTF-8 text hash, normalized content-token hash, stored-token hash, split,
  EOS information and token offsets/counts. Source extraction manifests and
  `source-lines.jsonl` map retained rows back to their original upstream lines.
- The extraction stage skips empty/whitespace-only strings with an explicit
  audited source-line record. The reusable tokenizer API itself accepts empty
  text as an EOS-only document subject to exact deduplication. This distinction
  is recorded in the respective manifests.

Every complete local source file is streamed through SHA256 verification before
any tokenization begins. Verified descriptors remain open; file identity and
modification metadata are checked before each commit. Inputs must stay immutable.
The bounded extracted bytes are fully verified. Remote ETags, URLs and revision
pins do **not** imply a verified SHA256 of the entire upstream gzip object, most
of which was deliberately not downloaded.

## On-disk commit and resume contract

```text
output/
  config.json
  .prepare.lock                 # Operational only; do not retain.
  shard-000000/
    tokens.bin
    documents.jsonl             # Unique-document and duplicate-row records.
    manifest.json
  shard-000001/...
  manifest.json                 # Present only after every source is consumed.
```

Each shard manifest pins its configuration, files, counts, preceding manifest and
start/end source cursor. Files and directories are fsynced; the manifest is written
last, then the pending directory is atomically renamed to its final shard name.
The exclusive writer lock prevents concurrent preparation of the same output.

A process interruption can leave `.pending-*`, which is never a committed shard.
Resume validates existing immutable shards, removes abandoned pending directories,
rebuilds the dedup/identity index and restarts at the last committed source cursor.
Changed source pins, tokenizer bytes or preprocessing settings are rejected.
Source paths themselves are relocation-independent. Plain JSONL and gzip are
supported; resuming a gzip source may require decompressing its already-consumed
prefix again. A bounded stop exactly at source EOF may need one final invocation
to publish the completion manifest without creating another shard.

`verify_document_shards()` checks file hashes, manifest chains, offsets, token
hashes, duplicate references, identities, counts, splits and cursor consistency.
`iter_documents()` performs verification by default and reads unique documents
lazily. Its `verify=False` skips the full metadata-validation pass, but still
checks shard file hashes; use it only after a separate completed verification.

## Storage and scaling limits

Use Local SSD for active raw extracts, prepared shards and scratch; retain raw
extract manifests/files, `config.json`, and each committed shard promptly in
`gs://fast-chunks`. A final root manifest marks complete preparation. Preserve
the verified previous recovery state until its replacement is verified. Do not
rely on Local SSD surviving an unexpected VM stop. `scripts/olmo_document_retain.py`
provides immutable cloud publication and integrity verification; consult its CLI
and the storage receipt before restoring or deleting local artifacts.

The actual interruption exercise retains raw inputs and the first two tokenized
shards, restores them to a fresh directory, and resumes in a new process. Compare
the final immutable bytes against an independently uninterrupted preparation.
This is data-preparation recovery, not a distributed training-checkpoint test.

Memory scales mainly with one input document; the corpus dedup and ID indexes
are disk-backed SQLite with a bounded page cache. Set `TMPDIR` to an existing SSD
scratch directory for a large preparation job. The index is rebuilt from all
committed metadata on each restart, so resume verification cost grows with the
artifact. All input source descriptors are currently held open simultaneously.
These choices suit the bounded readiness slice; a production corpus needs an
explicit source-file grouping/descriptor strategy and measured verification
costs before scaling to thousands of files. Such grouping must preserve global
dedup/split accounting rather than silently resetting it per group.

## Training boundary remains pending

These artifacts retain full document identity so a later adapter can build
OLMo-style concatenated 1,024-token chunks or another explicitly chosen layout
without repeating tokenization. There is currently no EOS-directed chunk offset,
shuffle, truncation, cross-chunk target or packed-attention implementation here.

The existing `CampaignData`/training adapters still enforce their earlier
single-document contract, including rejection of internal EOS and empty content.
Do not feed this artifact into them by removing checks. Packed CE targets,
NextLat masks, RT state continuity and FBT feedback boundaries require their
own implementation and bounded correctness tests, followed by the planned
two-GPU update/restart acceptance work.
