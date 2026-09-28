# Reusable document preparation — progress and interruption record

2026-09-28. User authorizes the reusable data preparation described in the prior
turn, before moving to two GPUs. Branch `feat/olmo-document-shards`, based on
main `3b26007`. No GPU/model execution or quality training in this milestone.

Boot disk has about12GiB free. Local SSD has about737GiB free; all substantial
raw/token data and scratch will go under
`/mnt/localssd/cdrm-data/olmo-dolma-v1_5-readiness-20260928`. Small source,
reports and receipts remain in the project. Completed units will be uploaded
and verified in `gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/`.

Implementation split:

- `document_shards.py`: complete tokenized documents, compact disk arrays,
  offset/provenance records, explicit split/dedup policy, atomic shard commits,
  resume/byte verification. No context truncation or packing.
- Source audit: pin official Dolma v1_5 URL inventory and verify OLMo target
  shift and chunk semantics at the user's cited source revision.
- Retention helper: create-only per-shard publication, full streamed hash
  verification, cloud manifest last, safe retries.
- Root: bounded real-data extraction/orchestration, integration/restore check,
  documentation, durable progress and PR closeout.

Initial data is a source-coverage readiness slice, approximately1M tokens per
available v1_5 source, **not** a representative campaign sample or a settled
training mixture. Select source URLs deterministically and retain exact extracted
raw bytes, source metadata and extraction bounds. A partial remote gzip stream
does not establish the hash of the complete upstream file. Final source mixture
and sample-wide decontamination remain separate decisions.

Prepare complete documents with true document offsets. Preserve embedded special
tokens as data; do not infer actual document boundaries solely from EOS. Split
before future concatenation; no destructive truncation. Token-equivalent
documents must not cross splits. The initial module uses deterministic
normalized-token-hash splits and first-occurrence exact-token deduplication.
This does not guarantee disjointness from the checkpoint's original pretraining.

OLMo's cited trainer uses inputs[...,1:] as targets for logits[...,:-1,:]. Its
loader uses fixed consecutive chunks. This is not an EOS-aware packing offset.
Production packed-stream support remains unqualified for our model variants:
the current training adapters deliberately reject multi-document rows. Keep
new tokenized storage independent of that future execution policy.

Save source/progress every20–30min. Work commits a small shard/source at a time;
no long non-resumable operation is planned. Do not run two-GPU readiness checks
until actual hardware is available and environment is verified.
