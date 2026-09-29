# Checkpoint and readiness cost assessment

The completed ordinary-B acceptance spends most of its wall time **outside its measured update regions**. Checkpoint retention contributes a measured substantial interval, but the evidence does **not** establish that checkpointing accounts for most of the remaining time. This corrects the stronger shorthand that “checkpoints dominate.” No GPU work, checkpoint payload download, upload or full local checkpoint rehash was performed for this assessment.

## What was actually timed

| Completed stage | Host command | Report interval | Sum of update regions | Outside those regions |
|---|---:|---:|---:|---:|
| Original reference, updates 1–3 | 910.11 s | 900.19 s | 22.82 s (2.53%) | 877.37 s |
| Fresh resume, updates 2–3 | 578.62 s | 568.57 s | 15.53 s (2.73%) | 553.05 s |
| Separate cloud restore of checkpoint 1 | 75.56 s | No separate internal timer | — | — |

The update timer surrounds captured backward, full active-gradient hashing, optimizer/scheduler/counter changes and cursor commitment. It excludes data materialization, full model/Adam boundary hashes, health checks, report serialization, logging and checkpoint work. It is not a pure training-compute timer or a throughput benchmark. Each process also performs 20 preparation and two capture backwards per rank; their durations were not recorded separately. The ~10 seconds between each host and report interval includes process/container initialization and finalization, and cannot be assigned exclusively to W&B.

The separate restore interval includes container/Python setup, generation-pinned download to a local file, full SHA256/size validation, and another full state verification through `inspect_distributed_checkpoint`. It is not isolated network bandwidth.

## Additional small metadata evidence

Read-only metadata requests authenticated the six already-retained objects against their exact generations, sizes, MD5 and SHA256 metadata. Explicit GCS `time_created` fields give these intervals; generation numbers were **not** interpreted as timestamps:

| Checkpoint | State-object creation to manifest-object creation |
|---|---:|
| Reference update 1 | 63.921 s |
| Reference update 3 | 63.370 s |
| Resumed update 3 | 75.332 s |

The code publishes each manifest only after the state upload, remote metadata validation, a complete state readback and its SHA256 validation. These intervals therefore locate a real, synchronous retention tail: 127.29 s across the reference (~14.1% of its report interval), and 75.33 s for resume (~13.2%). They also contain request/response and manifest-publication overhead; they are **not** isolated download/hash timings. Local serialization, upload and earlier checks fall outside this tail and remain untimed. There is no basis for assigning all 877/553 remaining seconds to retention.

## Concrete work performed by the current path

Each full state is about **14.155 GB / 13.18 GiB**: FP32 model weights, Adam moments and metadata. The reference saves two states and resume saves one. This unusually frequent saving is deliberate recovery acceptance, not a proposed training cadence.

1. All ranks hash their model/Adam boundary; rank zero serializes and fsyncs the state, hashes it, then commits the local manifest. The runner checks the boundary again.
2. The retainer makes another sequential file pass for SHA256 and MD5, then performs a create-only upload with SDK MD5 verification. Installed `google-cloud-storage` **3.12.0** uses resumable upload beyond 8 MiB and defaults here to 100 MiB sequential chunks (about 135 chunks for this state); this is not a multipart parallel transfer path.
3. `upload_verified(..., download_sha256=True)` calls `download_as_bytes`, then hashes the returned complete payload. The SDK builds a `BytesIO` buffer and returns its value. At least one whole ~13.18 GiB payload is therefore resident; actual peak RSS was not measured, and no claim of a second full copy is needed.
4. The manifest is uploaded/verified last, followed by local publication of the retained receipt. Other ranks wait while this rank-zero action completes. The RNG-preserving wrapper remains necessary for exact continuation.

Readiness hashing itself also has substantial potential cost: `tree_digests` transfers GPU tensors to CPU and hashes their bytes. Each update hashes ~4.7 GB of active gradients inside its timer, and the later complete model/Adam boundary (~14 GB per rank) outside it. Whole-state checks around saving repeat that work. These operations intentionally establish exact recovery; their individual durations are unknown.

## Recommended next check

A **single CPU-only, fixed-generation readback pair** is justified: compare the existing `download_as_bytes` path against `download_to_file` writing to a bounded SHA256/MD5/counting sink, in fresh child processes. Measure elapsed time and peak/incremental peak RSS, authenticate the same full size and both hashes, preserve generation-pinned retries, and perform it after active checkpoint transfers finish. This can establish whether the complete in-memory payload is avoidable without reducing verification. It does not test upload performance, end-to-end checkpoint pause or asynchronous checkpoint safety, and one ordered pair cannot establish a speed advantage.

The installed SDK streams ordinary downloads and retries interrupted requests from the number of bytes already written. A retry may return HTTP 206, for which its internal final checksum comparison is skipped; our final full-object SHA256/MD5/count checks are essential. A hashing-only sink must reject seeks and transcoding, require absent stored content encoding, bound its count, and fail closed on truncation or digest mismatch. Its `write` must accept complete chunks without partial writes. These semantics need focused CPU tests before any real transfer.

Before changing a production cadence or optimizing transfers, add explicit timers in a **new** operational runner for preparation, GPU update, diagnostic hash, serialization/fsync, local digest, upload, readback/hash and publication. Keep frozen acceptance code intact. Streaming verification is a smaller first change than asynchronous publication: asynchronous work would additionally require immutable snapshots, bounded queues/backpressure and preserved recovery/RNG behavior. No change to the existing retention policy is proposed here.

## Evidence and limits

- [Completed base-loop results](base-loop-results.md) contain the full stage/report/checkpoint authorities.
- `.runtime/olmo-campaign-lifecycle/checkpoint-cost-01/timing-assessment.json` records the arithmetic from those reports; `sdk-and-object-metadata.json` contains the exact metadata responses and SDK version/source hashes.
- The same directory contains the metadata-query script, its log, and selected installed SDK function snapshots, including streaming/retry behavior. Six metadata objects were queried; **zero payload downloads and zero uploads** were made by this assessment.
- Preparation, individual hash passes, local save, upload, full retention and peak host-memory costs still need direct instrumentation. Existing results do not determine the best transfer chunk size, concurrency, checkpoint cadence, H200 behavior or sustained training throughput.
