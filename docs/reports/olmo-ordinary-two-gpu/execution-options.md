# Bucket views and changing CUDA graph layouts

These are execution and memory options. Neither is intended to change the model's
architecture or objective, and neither is required for the fixed T512 baseline.

## Gradient bucket views

DDP groups gradients into communication buckets for cross-GPU reduction. The
current GPU-tested path keeps parameter gradients and those buckets as separate
storage. With `gradient_as_bucket_view=True`, each parameter's `.grad` points at
its section of a bucket. This can remove a gradient-sized allocation and copying
between the two representations. PyTorch documents the memory benefit and the
restriction on in-place gradient detachment in its
[DDP reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

For our ordinary model, the active FP32 gradients occupy
1,176,764,416 ×4 bytes =4.383 GiB per GPU. That is the approximate theoretical
storage opportunity, not a measured net saving. Graph pools, activations and
allocator behavior still determine usable headroom.

The local `DDPGraphTraining` already has an opt-in argument, but the completed
GPU milestone left it off. Adoption means verifying the actual gradients and
optimizer update with aliasing, ensuring bucket rebuilds finish before capture,
and checking that zeroing, graph replay and checkpoint reconstruction preserve
storage addresses. It is distinct from ZeRO-1: ZeRO-1 shards Adam moments across
GPUs; bucket views share two local representations of gradients. No GPU adoption
or combined-saving claim is made by exposing the argument.

## Changing graph layouts

Our current prepared graph accepts new token values on every step and updated
weights after each optimizer step. RT's internal states can also change. It fixes
the batch/sequence shapes, masks and document layout, loss denominators, selected
execution mode and participating parameters. This keeps the captured operations
and their addresses valid. CUDA graph replay uses the same tensor addresses and
layouts, as described in the
[CUDA semantics reference](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).

Examples that require extra handling in our implementation include a short final
batch, switching T512 to T1024, changing padding/supervised-token masks, or toggling
an auxiliary loss so a different parameter set participates. Some changes could
be supported with dynamic device-side values in a future implementation; current
validation deliberately rejects them.

The practical options are to rebuild the prepared DDP wrapper/graph when its
contract changes, maintain a small coordinated set of supported layouts, or
use eager execution for exceptional batches. Multiple resident graphs may use
additional memory and all ranks must choose compatible collective schedules.
This is separate work if the intended data pipeline needs it. Full fixed-length
ordinary batches need none of it; CUDA graphs remain enabled for this benchmark.
