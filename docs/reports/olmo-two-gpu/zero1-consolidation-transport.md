# Bounded ZeRO-1 consolidation transport change

Native ZeRO-1 consolidation became a material checkpoint cost in the actual
combined-model recovery probe. This change replaces the sender's per-byte tensor
construction with a buffer view. It changes checkpoint transport preparation;
training partitioning, Adam updates and parameter broadcasts remain inherited.

## Installed source and observed cost

Source: PyTorch `2.13.0a0+8145d630e8.nv26.06`,
`/usr/local/lib/python3.12/dist-packages/torch/distributed/optim/zero_redundancy_optimizer.py`.
SHA256: `f390e40fa51e336549b7337bd876781f106baafdd87ad7636270fd978dd6500f`.
The relevant upstream functions begin at lines 73 (`_broadcast_object`) and 513
(`consolidate_state_dict`).

The native sender does `torch.save` into `BytesIO`, copies to a `bytearray`, then
constructs `torch.ByteTensor(data)`. Bounded CPU-only measurements on this VM:

| Serialized payload | Native ByteTensor conversion | Peak RSS increase, including bytearray | Buffer-view conversion |
| --- | ---: | ---: | ---: |
| 1 MiB | 0.0705 s | 10.2 MiB | 14.2 microseconds |
| 8 MiB | 0.5441 s | 80.0 MiB | 14.9 microseconds |
| 32 MiB | 2.3000 s | 320.1 MiB | 19.6 microseconds |

Each row used a fresh CPU-only container subprocess and one PyTorch thread.
Native timing covers `torch.ByteTensor(data)` after bytearray allocation; RSS
starts before that bytearray. Buffer timing covers `torch.frombuffer` on an
already allocated owned `BytesIO` payload; it added no measured peak RSS.
These measurements isolate conversion, not complete serialization, transport or
checkpoint time. They are directional single measurements, not a throughput
benchmark.

The actual sender's local Adam state was 5,072,093,328 bytes (4.724 GiB).
Extrapolating the 32 MiB conversion gives about 348 seconds and 47.2 GiB transient
host memory, plus serialized buffers. This was consistent with roughly 56 GB
sender RSS while the native run consolidated. The checkpoint phase had begun by
18:41:05 UTC and `state.pt` appeared at 18:47:24 UTC on 2026-09-25; that is only
a coarse observation, not an exact save duration. The new harness records each
rank's complete `save_zero1_checkpoint` wall time and logs the maximum once to
W&B as `checkpoint/save_seconds`. GCS upload is excluded.

## Exact scope

`FP32Zero1AdamW.consolidate_state_dict` narrowly adapts the pinned native loop.
The private `_broadcast_state_object` helper replaces sender construction with
`torch.frombuffer(buffer.getbuffer(), dtype=torch.uint8)`. On CPU the tensor
retains the exporter; on CUDA the blocking `.to(device)` finishes before host
ownership is released. There is no global monkeypatch or early `BytesIO.close`.

The following remain native:

- Synchronizing exposed parameter-group options into the local optimizer.
- Visiting shards in rank order, mapping process-group rank to global rank,
  skipping the target's own broadcast and discarding non-target receives.
- Synchronous length (`int64`) then serialized payload (`uint8`) broadcasts.
- `torch.save` bytes, CUDA payload staging, receiver `torch.load` device mapping,
  native recursive CPU copies, and consolidated state remapping.
- Existing global-to-local Adam loading and the post-consolidation CUDA sync
  before durable checkpoint serialization.

No checkpoint-format/schema change, process group, offload, rank-shard format,
optimizer overlap or runtime default is introduced. Source fingerprints identify
the implementation change. Large serialized payloads and receive-side CUDA
staging still exist; this is not a bounded-size communication redesign.

## Validation and pending scope

Focused CPU tests verify exact native serialized bytes, nested/empty-state round
trips, exporter lifetime after the helper returns, absence of the per-byte tensor
constructor, wire ordering and sender identity. Mocked subgroup tests exercise
noncontiguous global ranks and non-target discard order. Real two-rank Gloo tests
compare complete consolidated states against the native implementation for both
target ranks, before and after Adam initialization, while checking model/local
state preservation and parameter-group synchronization.

The existing real Gloo suite exercises checkpoint/restore and exact continuation
for ordinary and combined models with fused and unfused Adam. GPU/NCCL tiny and
actual-model recovery must pass before adopting this transport for training.
The ongoing native run remains an unchanged, separately retained reference.
