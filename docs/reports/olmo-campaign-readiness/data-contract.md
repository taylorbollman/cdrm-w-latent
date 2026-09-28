# Portable data and logical-update contract

The new `cdrm.pretrained.campaign_data` module accepts **ordered, already-tokenized
local documents**. It is a bounded in-memory reference adapter, not a Dolma
download/preparation job or a production-scale corpus loader. No corpus mixture,
licenses, held-out allocation, or pretraining-overlap claim is selected here.

Each document names an immutable source artifact (URI, revision, SHA256), stable
source document ID, split, raw-text SHA256, and token IDs. A tokenizer pin records
its repository, revision, artifact hash, and disabled implicit special-token
insertion. The adapter hashes normalized uint32 token bytes. Supplied raw-file
and raw-text hashes are **caller declarations**; verifying those absent bytes is
the responsibility of the future preparation stage. Exact duplicate text or
normalized tokens, including duplicates crossing splits, fail visibly rather
than silently changing source order or split membership.

`CampaignData` preserves the supplied document order and exposes separate split
streams. Windows have a maximum width of 1024 and overlap by one context token.
This retains short documents and final windows, preserves every same-document
CE/latent pair exactly once, and omits exactly one KL triple per window seam.
Documents must contain content. An existing single terminal EOS is preserved;
otherwise one is appended only at the actual document end. Internal/multiple EOS
values fail instead of being rewritten. Windows never pack multiple documents.

Every `DocumentWindow` has a deterministic `key`, `length`, `document_index`,
`start`, `split`, and `tokens`. The key is independent of rank, physical batch,
and accumulation layout and can key per-window randomness. It binds document,
source and tokenizer provenance plus the exact window boundaries.

The canonical JSON manifest includes ordered document identities, hashes,
normalization, source/tokenizer pins, and the window/counting policy. Its SHA256
binds a `DataCursor` (`manifest_sha256`, `split`, `next_window`). Restoring a cursor
against reordered data, changed tokens/provenance, or a changed window policy
fails. The cursor records data position only: optimizer/model/RNG state and
recipe/hardware-layout compatibility remain trainer responsibilities.

`next_update(cursor, target_valid_tokens)` consumes whole rows until reaching the
valid **presented** input-token target. Overshoot is strictly less than one maximum
window (1024 tokens). This count includes EOS and repeated window-overlap context;
it excludes padding and does not multiply by FBT passes. A final shorter update is
returned with `reaches_target=False`; the following request raises `StopIteration`.
There is no implicit cycling or dropped tail. Save `next_cursor` only with the
corresponding completed optimizer update.

`TokenCounts` separates:

- `new_unique_tokens`: tokens introduced for the first time in this ordered,
  noncycling split stream, including EOS; this is not deduplicated vocabulary.
- `presented_tokens`: all valid row tokens, including repeated overlap context.
- `overlap_tokens`: repeated context, even when its first occurrence was in the
  previous logical update.
- `ce_targets`, `latent_pairs`, `kl_triples`, and omitted boundary KL triples.

`partition_update(update, world_size, physical_batch_size)` returns
`[microstep][rank][rows]`. Flattening those axes recovers exactly the original
logical update, independent of physical layout. The final microstep can have
empty rank slots. `batch` can pad them to an explicit physical size using dummy
rows with no valid tokens or targets, document IDs -1, and legal PAD token IDs.
**This does not establish that the distributed graph trainer safely executes
empty slots:** synchronized zero-loss backward and graph execution remain an
integration gate. No real document is duplicated as filler. A two-token short
document can have CE/latent supervision with no local KL targets; global objective
normalization must handle that case without averaging local means.

Readiness evidence is download-free CPU testing of pair/triple conservation,
source/cursor integrity, EOS behavior, exact 1024-token tails, logical update
boundaries/resume, and physical partitions including empty slots. Production
disk-backed data, shuffled/repeated epochs, distributed collectives, graph
accumulation, and whole-training checkpoint recovery are outside this module.

## Bounded ingestion from actual local raw files

`cdrm.pretrained.campaign_ingest.prepare_local_preflight` bridges pinned local
JSONL or gzip JSONL to `CampaignData`. Pass ordered `LocalJSONLSource` entries
(source pin, local path, explicit compression, configurable text/id fields), a
`TokenizerPin`, and an explicit `SplitPolicy(seed, weights)`. There is no default
train/dev/test allocation. Document splits depend only on the seeded raw-text
SHA256, keeping identical text together independently of source order; duplicate
text/tokens still fail under the existing data contract. Source order and JSONL
record order define the resulting stream. Text is preserved without Unicode
normalization, and nonempty string text/IDs are required. Invalid/empty records
fail visibly instead of being skipped.

Every supplied source artifact's SHA256 is checked before any document is
tokenized, and the bounded verified byte snapshots are parsed directly. A local
tokenizer JSON path is also verified against its pin, its native EOS ID is
checked, and `encode(add_special_tokens=False)` leaves terminal-EOS handling to
`CampaignData`. Alternatively, a supplied tokenization callable is explicitly
recorded as **declared and unverified**, even though its raw source files were
verified. Byte verification establishes artifact identity, not the remote
publisher's authenticity or checkpoint-pretraining disjointness.

The result exposes `.data`, `.manifest`, and `.manifest_sha256`. Pin **both** the
ingest manifest hash and data manifest hash in the outer training/checkpoint
configuration: changing selected JSON fields or preprocessing settings can
leave resulting token bytes unchanged. The ingest manifest records settings,
split allocation, row/source provenance, verification mode, and resource bounds.
Default bounds are 1024 documents, 16 MiB total source bytes, 16 MiB total
uncompressed bytes, and one million normalized tokens. Exceeding a bound rejects
the preflight rather than silently selecting a prefix. Paths may relocate
without changing the recorded artifact identity.

This is a usable local preparation/verification exercise, **not** a Dolma
download, corpus mixture decision, production disk-backed loader, or an attempt
to materialize the full campaign in RAM. Large-corpus preparation, split audits,
and restartable orchestration remain separate work.
