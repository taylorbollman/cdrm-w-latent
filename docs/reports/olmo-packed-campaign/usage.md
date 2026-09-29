# Packed campaign data and recovery runner

This is an opt-in readiness path for the pinned OLMo-1B checkpoint, a finite
verified corpus and two GPUs. It does not launch a continuing quality campaign.
Commands below are templates: replace every uppercase placeholder with the
recorded path or pin. Actual run outcomes, W&B URLs and cloud receipts belong
in this milestone's results and storage records; no success is implied here.

## Document policy

`continuous-stream-v1` must agree between `CampaignRecipe`, `NextLatConfig`,
`FBTMode`, the packed index and checkpoint configuration. The default
`isolated-v1` remains unchanged and rejects multidocument rows.

| Component | Within a packed row |
| --- | --- |
| Tokens | Complete stored documents concatenated in canonical shard/record order within one existing split |
| CE | Every valid adjacent pair, including terminal EOS to the next document's first token |
| Latent prediction | Adjacent positions in the same actual document |
| KL loss | Three consecutive positions in the same actual document |
| Attention and native RT | Continuous causal history across document boundaries |
| FBT feedback and jitter | Valid adjacent positions, including document boundaries |
| Positions | RoPE positions start at zero for each row |

Chunks have stride equal to context length, without overlap, EOS-dependent
offsets or targets into the next chunk. The final partial chunk is right-padded.
All attention/RT/FBT state resets at each row. Literal embedded EOS tokens are
content; stored document offsets determine boundaries. Packing neither adds
EOS nor retokenizes documents.

The readiness corpus is a seven-source coverage fixture. Its canonical order
and source proportions are not a production sampling policy. No shuffle,
cycling or cross-split mixing is implemented by this reader.

## Build and inspect the immutable index

Use the completed tokenized corpus retained by the document-shard milestone.
Verify or restore its pinned bytes before building the index. The index stores
metadata in `documents.sqlite` and `manifest.json`; tokens remain in their
original shard files. `build_packed_index` publishes atomically to a new
directory and never overwrites an existing index.

Run this Python API in the CPU container, for example via
`CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed bash
scripts/docker_shell.sh bash -lc 'python PATH-TO-PREPARATION-SCRIPT.py'`:

```python
from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.packed_campaign_data import (
    PackedCampaignData, build_packed_index,
)

manifest = build_packed_index(
    corpus_dir, index_dir, split="train", length=1024,
    document_policy="continuous-stream-v1", pad_id=1,
)
index_sha256 = sha256_file(index_dir / "manifest.json")

with PackedCampaignData(corpus_dir, index_dir) as data:
    assert data.manifest_sha256 == index_sha256
    cursor = data.cursor()
    update = data.peek_update(cursor, target_valid_tokens=524288)
    packed = data.rank_batches(
        update, rank=0, world_size=2, physical_batch_size=12,
    )
    # Inspection and prefetch leave data.cursor() unchanged.
    # Only after the corresponding optimizer update succeeds:
    # cursor = data.commit(cursor, update)
```

The CLI's `--index-sha256` is the SHA256 of the **manifest file bytes**, not
the manifest's separate `identity_sha256`. The manifest pins corpus/tokenizer
identity, split, order, chunk policy, dimensions and SQLite bytes. Opening a
reader hashes pinned corpus/index files; later reads reject changes to file
identity or modification metadata. Keep these files immutable during a run.
Paths can change on restore while the verified bytes remain identical.

Use `descriptor()`, `read_chunk()` and `chunk_counts()` for bounded inspection.
`read_chunk()` retains original document/source provenance. The reader caches
at most eight token file descriptors and reads only requested token slices.
Metadata planning does not materialize the corpus's token tensors.

## Logical updates and physical batches

The shared logical target is **524,288 valid input tokens per optimizer
update**, formed before partitioning across ranks. Full length-1024 rows give
512 real rows. Physical batch 12 per GPU on two GPUs therefore uses 22
microbatches per rank: 528 physical row slots, including 16 dummy rows in the
last slot. Physical batch 8 uses 32 microbatches per rank and no dummy rows for
such a full update. These changes do not change the logical token exposure.

Each slot has fixed physical B/T. The first M-1 slots use the local CUDA graph;
the final slot uses the synchronized DDP graph. Global denominators are
computed separately for CE, latent pairs and KL triples. Clipping, Adam and
scheduler advance once per logical update. Keyed FBT noise follows immutable
chunk identity, logical update and feedback pass, rather than rank or physical
partition. The runner currently stages every microbatch/noise tensor for a
logical update on CPU; this is not a streaming-prefetch performance claim.

`peek_update()` is pure; prefetch never advances the committed cursor.
`commit()` rejects stale or changed updates and follows a successful optimizer
step. Fresh-process restore independently reconstructs the scheduled chunk
boundary and checks it against both optimizer-update and input-token clocks.
The finite final update may contain fewer tokens; exhaustion returns `None`
instead of silently cycling.

The legacy counter `documents` means nonempty packed-row presentations.
Telemetry separately records document segments/completions, valid inputs,
objective counts, cross-document CE, excluded boundary auxiliary targets,
omitted cross-chunk targets, tail padding and dummy slots. Do not interpret
`documents` as a count of unique source documents.

## Bounded write and fresh-process resume

Root alone launches GPU stages. Check the container and both GPUs before each
launch. The runner fixes T1024 and supports `--arm B|NFR` and physical
`--batch-size 8|12`. NFR means K4 FBT, native RT at layers 0/15 on every pass,
and both NextLat auxiliary losses. The default batch is 12; batch 8 is a
fallback requiring its own write/resume pair. This path uses BF16 mixed
precision with FP32 master parameters/Adam and forced ordinary Flash SDPA.

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
  set -euo pipefail
  test -f /.dockerenv
  test "$PWD" = /workspace/cdrm-w-latent
  nvidia-smi
  env TORCH_NCCL_ASYNC_ERROR_HANDLING=0 NCCL_ASYNC_ERROR_HANDLING=0 \
    timeout 1200s torchrun --standalone --nproc-per-node=2 \
    scripts/olmo_packed_campaign_run.py --phase write --arm NFR --batch-size 12 \
    --corpus CORPUS-DIRECTORY --index INDEX-DIRECTORY \
    --index-sha256 INDEX-MANIFEST-SHA256 \
    --output-dir .runtime/olmo-packed-campaign/NEW-WRITE-STAGE \
    --checkpoint-dir /mnt/localssd/cdrm-checkpoints/NEW-PACKED-CHECKPOINT
'
```

Each attempt needs a new evidence directory under the persistent checkout.
The runner records source snapshots, configuration, per-rank preparation
phases, counters and W&B metrics. Keep the outer launcher log too. Preparation
uses 11 synchronized plus 9 accumulated backward warmups and captures local
and final-sync graphs without advancing weights, Adam, RNG or training clocks.
The write phase performs one logical update, saves at its committed boundary,
then records the next update on the **original live graphs**. It does not save
the second update as the recovery starting point.

After writers stop, copy the launcher log into its evidence stage and retain
the checkpoint before depending on it across a VM interruption. CPU-container
retention example:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
    env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_two_gpu_retain.py \
      --input-dir .runtime/olmo-packed-campaign/NEW-WRITE-STAGE \
      --checkpoint-dir /mnt/localssd/cdrm-checkpoints/NEW-PACKED-CHECKPOINT \
      --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/YYYYMMDDTHHMMSSZ/packed-write-NN \
      --receipt .runtime/olmo-packed-campaign/retention/packed-write-NN.json
  '
```

The timestamp/stage prefix must be new for changed evidence. The command above
uses mounted application-default credentials, avoiding a stale explicit ADC
path; it does not expose credentials. Evidence retention accepts success and
failure reports and does not certify numerical correctness. Index files can
be retained separately as small evidence, including `.sqlite`; preserve the
existing independently retained corpus too.

Restore both `checkpoint/state.pt` and `checkpoint/manifest.json` from the
receipt's exact GCS object generations into a **fresh directory**. Hash every
downloaded byte and check sizes against the receipt. Then call
`inspect_distributed_checkpoint(restored_dir, verify_state=True)` and save a
restore receipt. Server MD5/size and SHA metadata from upload alone are not
the full-download recovery check. Also restore and verify corpus/index files
if SSD contents were lost. Keep the preceding verified copy until the new one
is verified.

Launch a new two-rank process group using the same GPU launcher/environment
as the write example, changing the arguments to:

```text
scripts/olmo_packed_campaign_run.py --phase resume --arm NFR --batch-size 12
  --corpus CORPUS-DIRECTORY --index INDEX-DIRECTORY
  --index-sha256 INDEX-MANIFEST-SHA256
  --output-dir .runtime/olmo-packed-campaign/NEW-RESUME-STAGE
  --checkpoint-dir RESTORED-CHECKPOINT-DIRECTORY
  --reference-report .runtime/olmo-packed-campaign/NEW-WRITE-STAGE/report.json
  --reference-sha256 COMPLETED-WRITE-REPORT-SHA256
  --expected-manifest-sha256 CHECKPOINT-MANIFEST-SHA256
```

Use the hash of the completed, finalized write report. Identical implementation
source hashes and execution configuration are required; a code/policy/batch
change means a new pair. The runner loads actual model, Adam, scheduler,
per-rank RNG and real cursor **before** DDP construction and graph preparation.
This is the cold T1024 setup-memory test with resident Adam state. Never load
into existing captured graph storage.

Resume compares the next input/noise fingerprints, raw gradients, loss and
update metrics, full model/Adam/scheduler/counter digests, cursor, RNG state and
actual RNG draws bitwise against uninterrupted continuation. Replicated raw
gradients and model/Adam state must also agree between ranks. Retain the
completed resume stage and cloud-restore evidence separately.

## Reading results and limits

The short loader/update rate counts valid global input tokens once, regardless
of K4 passes. Its timed segments include token reads, keyed jitter generation,
validation/refills, graph backward/NCCL, clipping, Adam, scheduling and cursor
commit. It excludes diagnostic hashing, gate publication, extra health scans,
W&B and checkpoint I/O. This is a bounded integration rate, not a long-run
end-to-end campaign estimate. Read memory phases separately from timed steps.

No operational pass clears the independently recorded BF16 sparse/dense
gradient discrepancy. Its component diagnostic remains separate. This runner
does not qualify changed world size, H200, ZeRO/FSDP, in-flight collective
recovery, asynchronous production loading, a production mixture, changed
packing policy or learning quality. The 20-minute external timeout bounds a
disposable phase; a timed-out attempt may need to repeat from the last safely
retained completed checkpoint. Save code and completed evidence every 20–30
minutes and preserve failed attempts rather than overwriting them.
