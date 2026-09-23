# B32 isolated-block profile

One untimed CUDA-graph replay follows the five timing samples in each reverse
repeat: `capacity-author-l1-b32-r2-profile` and
`capacity-native-l1-b32-r2-profile`. Both use native geometry, identical fixed
fixtures and actual Adam updates. Traces are retained as compressed exact bytes
with hashes in each report. Profiling is outside the throughput timer.

| Observation | Native both/recompute | Author legacy/chunks4 |
| --- | ---: | ---: |
| Complete-update input tokens/s in preceding samples | 53,917 | 50,198 |
| Reported CUDA events | 72,020 | 68,231 |
| Sum of event self-device time | 251.617ms | 270.861ms |
| All names containing `direct_copy_kernel_cuda` | 25.858ms | 30.305ms |
| All names containing `CUDAFunctor_add<float>` | 22.891ms | 36.192ms |

The author path has fewer reported events here but greater total device time;
event count alone is not a performance verdict. Named FP32 additions and direct
copies take more time in the author trace. These string-based categories are
mutually distinct here but do not cover every addition or copy implementation. Both
profiles retain large small-row dense-matmul groups and thousands of pointwise,
normalization and cast operations. These event totals include non-kernel CUDA
events, and summed self time is not complete-update wall time.

The source provides a plausible explanation for the batch-size crossover:
author per-token writer weight gradients require repeated full weight-gradient
accumulation/conversion, whereas native performs that weight VJP in a batch.
Conversely, author avoids some native projection reconstruction and has compiled
batched helpers. The audited logical dense work is50ND²+36NDM for author versus
58ND²+36NDM for native K/V-only, plus separately different attention schedules.
More batch work can amortize fixed per-token overhead and favor the smaller
matrix ledger. This is an inference from implementation and measurements, not
an exclusive attribution of the generic copy/add kernels to a particular VJP.

The B32 traces do not establish which kernels explain the B128 advantage, nor
whether that advantage survives full vocabulary objectives and FBT/NextLat.
The stack/integration checks answer the latter question. No new broad fusion,
FA4 or kernel-rewrite project is included in this milestone.
