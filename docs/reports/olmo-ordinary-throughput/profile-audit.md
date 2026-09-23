# Ordinary throughput profile audit

Read-only analysis of the six-layer native ordinary model at B64/T512, H16,
half-target CE and 128-position CE chunks. The untimed CUDA graph replay covers
forward, loss and backward, excluding optimizer and input loading. This is a
profile of the existing implementation, not an optimization or a paper-model
reproduction.

## Evidence and timing scope

- Run: `.runtime/olmo-ordinary-throughput/l6-h16-b64-half-c128/`.
- Raw files: `report.json` and `operator-trace.json.gz`.
- Compressed trace SHA256:
  `c36899fb6a488d5451b28054fa27528a8a1c93312fd944325ab9f1f19f60c118`.
- Trace has 6,308 device events: 5,952 kernels, 228 memory sets and 128
  device-to-device copies. Their summed duration is 548.870796 ms. Summed
  device-event duration is not generally the same as elapsed wall time.
- Separate uninstrumented medians: complete update 571.067590 ms and captured
  forward/loss/backward 548.794455 ms. Their difference is 22.273135 ms, about
  3.90% of the full update. This comparison does not isolate Adam from copying,
  validation, synchronization or clipping individually.

## Broad kernel families

These mutually disjoint named families use the recorded device-event durations,
not CPU profiler attribution. Percentages use the 548.870796 ms sum above.
Unlisted pointwise, indexing, concatenation and other events account for the
remainder. Counts and durations include checkpoint replay and backward work.

| Family | Kernel-name selection | Time (ms) | Share |
| --- | --- | ---: | ---: |
| GEMMs and their reduction | `nvjet_` prefix or `cublasLt::splitK` | 167.945783 | 30.60% |
| Copy/conversion kernels | `copy_kernel_cuda` | 124.373185 | 22.66% |
| FP32 additions | `CUDAFunctor_add<float>` | 89.075541 | 16.23% |
| Fill kernels | `FillFunctor<` | 61.557957 | 11.22% |
| Flash attention | `pytorch_flash::` | 11.859272 | 2.16% |
| CE log-softmax and NLL | `LogSoftMax` or `nll_loss` | 11.299748 | 2.06% |
| Layer normalization | `layer_norm` | 10.302376 | 1.88% |
| SiLU | `silu_` | 14.538482 | 2.65% |

Flash has 12 forward invocations (six initial and six checkpoint replays) and
six invocations of each backward stage. Its small measured fraction means a
Flash-only change cannot explain or remove most of this run's elapsed work.
This statement is specific to the measured six-layer B64 configuration.

## Readout-sized traffic

The trace is a CUDA graph replay and lacks CPU operator input shapes. The
following attribution is therefore an **inference from launch dimensions,
kernel specializations, source semantics and the 128 CE chunks**, rather than
direct CPU-to-kernel shape correlation. The tied readout has
`50304 * 2048 = 103,022,592` elements, distinct from the block matrices.

| Inferred operation | Grid / block | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| FP32-to-BF16 readout conversion | `[100608,1,1]` / `[128,1,1]`, BF16 copy kernel | 256 | 51.850 |
| BF16-to-FP32 readout-gradient conversion | `[201216,1,1]` / `[128,1,1]`, direct copy kernel | 128 | 41.132 |
| Readout-sized FP32 addition | `[100608,1,1]` / `[128,1,1]`, vectorized FP32 add | 129 | 51.671 |

These operations total approximately **144.65 ms, 26.35%** of recorded device
duration, before CE GEMMs, softmax, zero/fill work or hidden-gradient slicing.
The 256 forward conversions match initial projection plus checkpoint replay for
each chunk. The extra addition beyond 128 chunks is consistent with tied
embedding/readout gradient ownership; it is not separately attributable from
this graph trace.

Readout-sized fill kernels add approximately 32.48 ms: 256 BF16 fills at16.181 ms
and 131 FP32 fills at16.300 ms. Deterministic uninitialized-memory filling is a
plausible contributor. **This trace does not establish that runtime setting or
which fills are avoidable**; some fills initialize required gradient storage.
Do not disable a determinism setting on the basis of this attribution alone.

## Paired chunk2048 measurement

The completed paired run changes only CE chunk size, retaining B64/T512, H16,
the random initialization/token seeds, half-target selection, vocabulary,
checkpointing, deterministic Flash, graph policy and precision. It uses the same
runtime source hashes and protocol. The mathematical objective is unchanged;
chunk regrouping can change floating-point accumulation and GEMM rounding.

- Run: `.runtime/olmo-ordinary-throughput/l6-h16-b64-half-c2048/`.
- Compressed `operator-trace.json.gz` SHA256:
  `d4b3d6f8df268be4aac917a00e4473edd15a49cdfea58aca4c4ff9ea93297baf`.
- CE chunks decrease from128 to8 for the same16,384 supervised targets.
- The isolated same-weight2048-position forward CE check returns the same
  scalar for both chunk sizes:10.84616756439209. The first full-model objective
  is11.321547508239746 versus11.321544647216797, a2.86e-6 absolute difference
  (2.53e-7 relative). This is not a gradient-equivalence or long-training test.

| Uninstrumented result | Chunk128 | Chunk2048 |
| --- | ---: | ---: |
| Complete update median | 571.068 ms | 353.054 ms |
| Forward/loss/backward median | 548.794 ms | 331.871 ms |
| Input tokens/s | 57,380.25 | 92,812.94 |

Chunk2048 is **61.75% faster in input tokens/s**, or38.18% shorter per complete
update. These are three-update directional medians, not a randomized benchmark
or proof of maximum throughput. Profiling was performed separately from timing.

| Same kernel family definitions as above | Chunk128 (ms) | Chunk2048 (ms) |
| --- | ---: | ---: |
| All recorded device events | 548.870796 | 332.395392 |
| GEMMs and their reduction | 167.945783 | 141.590008 |
| Copy/conversion kernels | 124.373185 | 36.756102 |
| FP32 additions | 89.075541 | 24.972597 |
| Fill kernels | 61.557957 | 20.982007 |
| Flash attention | 11.859272 | 11.863681 |
| CE log-softmax and NLL | 11.299748 | 13.601932 |
| Layer normalization | 10.302376 | 10.360791 |
| SiLU | 14.538482 | 14.579571 |

Copy/conversion, FP32 addition and fill durations decrease by192.296 ms in
aggregate, **88.83% of the216.475 ms total device-duration reduction**. Flash
time is essentially unchanged. The paired measurement therefore supports the
small CE chunk implementation as a major avoidable cost in this configuration.
It does not isolate every reduction to the readout matrix: chunks also create
and accumulate gradients for hidden-state slices and allocate other buffers.
Larger chunks change GEMM efficiency and softmax behavior as well.

The chunk2048 trace contains1,876 device events (1,744 kernels,124 memory sets,
eight device copies), versus6,308 for chunk128. Do not reuse the initial
readout-grid attribution mechanically: a2048-by50304 logit tensor has the same
element count as the50304-by2048 readout. Their copy/fill launch grids coincide,
so this graph trace cannot separate those arrays by grid size alone. The broad
kernel-family comparison above avoids that ambiguity. The source still
predicts fewer readout conversions, but exact attribution of all new copy/fill
events would require additional operator-shape evidence.

This pair leaves deterministic settings unchanged; it does not test the
uninitialized-memory-fill hypothesis. B512 results and memory behavior require
their own reported measurements rather than extrapolation from this pair.
