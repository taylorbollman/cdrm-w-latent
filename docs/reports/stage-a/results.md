# Stage A results — 2026-09-06

The GPU environment, mixed-block R3 implementation, exhaustive weight
conversion, numerical fixtures, and bounded training/benchmark infrastructure
are implemented. These artifacts establish a foundation for the next synthetic
pilot. They do not establish an R3 language-model improvement.

## Environment and implementation

The maintained fork is now the top-level `recurrent-transformer` submodule,
starting from `a21b42d2bc292edb86ed1b62cee4bcab809a9d21`. Implementation changes
remain local and uncommitted. The parent starting revision and exact container
image/package details are in [environment.md](environment.md).

The Docker bootstrap now reuses an existing image and mounted SSD. Its editable
import points to the relocated fork, and Python user-site packages cannot
silently override image pins. Both the home bootstrap and direct launcher have
been verified on the H100:

```bash
bash /home/taylorbollman/cdrm-w-latent/scripts/docker_shell.sh
```

Inside, `pwd` must show `/workspace/cdrm-w-latent`; `nvidia-smi` must succeed.
All GPU validation and benchmarks used this container. No SSD formatting was
performed. See [usage](../../stage-a-usage.md) for reproducible commands.

R3 replaces zero-based block 3 of 12. Block selection and backend selection are
independent, block configurations are independent, and helper modules retain
one canonical parameter owner. The naïve backend supports the persistent-write
rho ramp; the tiled backend supports rho=1. The converter copies all state,
including nondefault learned norms and biases, and maps fused/split QKV
gradients exhaustively. It rejects incompatible semantics before mutation.
Ordinary attention now enforces causality at direct block entrypoints as well
as normal model calls. CUDA graph capture remains disabled.

The supported initial contract is pre-norm ALiBi, full MHA, no dropout, no RoPE,
and no cache/document packing. Distributed recurrent training, frozen-prefix
tiled training, and tiled activation checkpointing remain outside the validated
contract. Detailed decisions are in [the decision log](../../semantic-decisions.md).

## Numerical evidence

The final CPU and GPU commands, results, and source hashes are retained under
[validation](validation/numerics.json): **62 passed, 16 GPU tests skipped** in
the explicit CPU container, and **55 passed** in the GPU foundation suite.
Logs are [CPU](validation/cpu.txt) and [GPU](validation/gpu-final.txt).
Tests cover rho-zero SEQ equivalence, an independent
functional recurrence oracle, causal/padding behavior, temporal credit to earlier
writes, write ordering, parameter ownership, conversion round trips, and naïve
activation checkpointing. Output and gradient checks use random cotangents;
sequence length one is an output/gradient case because shifted CE has no targets.

FP32 tiled comparisons use lengths 1/7/16, both helper modes, and 1/4 backward
chunks. BF16 compares both backends to the same FP32 reference and to each other,
checking every parameter separately, with both helper modes and chunk settings.
BF16 numerical bounds apply to the documented small bias-free fixture, not all
full-size configurations.

The first absolute BF16 gradient check failed a few final-head entries. The
[preserved diagnostic](bf16_diagnostic.json) measures the error against the same
FP32 cotangent and motivates scale-aware per-parameter bounds; the original
[failure](validation/bf16-initial-failure.txt) remains visible. FP32 tolerances
were unchanged. The unmodified upstream reproduction also had small FP32
random-cotangent failures, retained as failures in [environment.md](environment.md).
Current-fork passes are separate evidence.

## Operational evidence

The smoke runner checks SEQ and R3 fixed-batch learnability, target-count-weighted
gradient accumulation, empty-target handling, checkpoint round trips, and exact
interrupted/resumed trajectories. Exact resume includes optimizer state, rho,
schedule metadata, Python/NumPy/CPU/CUDA RNG state, and data position. A fresh
optimizer branch preserves the parent's next data batch while resetting only
branch training counters and optimizer state.

The 24-update fixed-batch check reduces CE from about 4.73 to 0.063 for both
models. This demonstrates that the training path learns this tiny reused batch;
it is not a held-out result. Strict FP32 accumulation/update equality is checked
on bias-free models. Bias-enabled gradients pass, but nearly zero redundant
key-normalization bias gradients cause Adam epsilon sensitivity; the report
records those weight differences rather than asserting optimizer equality.

Smoke checkpoint paths, hashes, and lineage are recorded in the checkpoint
[ledger](ops-checkpoint-ledger.json); all 18 recorded files were independently
checked against their hashes and sizes. All are explicitly **not research
pretrained checkpoints**. No
`LM-SEQ-u5000` parent, real corpus pipeline, tokenizer contract, or downstream
evaluation has been established by this work.

Verified operational reports are [naïve/eager](ops-naive-summary.json),
[tiled/eager](ops-tiled-eager-summary.json), and
[tiled/compiled](ops-tiled-compiled-summary.json). The final compiled suite
reports 32/32 successful frames, 50 unique graphs, no graph breaks, and no
fallback counters. Earlier [default-cache fallback](ops-tiled-compiled-initial-fallback-summary.json)
and [worker-context fallback](ops-tiled-compiled-worker-fallback-summary.json)
results remain distinct historical records.

## Performance and next-stage budget

The [benchmark ledger](benchmark-ledger.json) records all whole-model results
and links their raw JSON artifacts; planning projections follow below. The measured model has **216,843,264 parameters**, from the specified
12×1024, H16, MLP4096 dimensions and untied T5-shaped embedding/head vocabulary;
the protocol's informal “150M” label does not describe the actual total.
After adding the final fallback rejection, a separate
[tiny compiled BF16 entrypoint check](validation/benchmark-entrypoint.json)
also passed; it is excluded from the full-model throughput comparison.

Measurements use one H100 80GB, length 512, FP32 weights/AdamW state with BF16
autocast, synthetic tokens, two warmup updates, and three measured updates.
Timing includes shifted CE, backward, and AdamW, excluding prepared-input
transfer. Allocated/reserved peaks and forward times are retained in each JSON.
These are bounded performance probes, not long-run stability measurements.

The default helper compilation cache was too small for all tile sizes at length
512. Those initial fallback runs remain labeled separately. Subsequent tiled
benchmarks use a limit of 64 and report actual compiler counters; verified
results must have no fallback. Graph capture and whole-model compilation are off.
This PyTorch build stores compiler overrides in thread-local context, so the
autograd worker can retain defaults despite main-thread settings. The broad OPS
runner isolates independent fixture code caches while retaining cumulative
fallback counters; each accumulation/resume trajectory stays in one group.
Both entrypoints reject an unsupported/fallback counter instead of reporting a
compiled pass. A cache setting alone is not treated as execution evidence.

| Model / recurrent backend | Batch | Seconds/update | Input tokens/s | Allocated GiB | Reserved GiB |
|---|---:|---:|---:|---:|---:|
| SEQ | 1 | 0.0217 | 23,597 | 3.56 | 3.80 |
| SEQ | 4 | 0.0216 | 94,900 | 4.94 | 4.99 |
| SEQ | 16 | 0.0448 | 182,973 | 10.92 | 11.33 |
| SEQ | 64 | 0.1454 | 225,356 | 34.88 | 37.42 |
| R3 naive | 1 | 2.5534 | 201 | 3.96 | 4.34 |
| R3 naive | 4 | 2.5221 | 812 | 7.02 | 7.41 |
| R3 naive | 16 | 2.5866 | 3,167 | 19.18 | 23.27 |
| R3 tiled | 1 | 0.8146 | 629 | 3.56 | 3.88 |
| R3 tiled | 4 | 0.8190 | 2,501 | 4.86 | 4.94 |
| R3 tiled | 16 | 0.8439 | 9,707 | 10.59 | 11.13 |
| R3 tiled | 64 | 0.9361 | 35,004 | 33.50 | 36.61 |

Tiled rows use compiled helpers. The [benchmark ledger](benchmark-ledger.json)
links every retained run and records exclusions, skipped configurations,
forward timings, and actual token/update counts.

At batch 64, SEQ takes 0.1454 seconds/update (225,356 input tokens/s) and tiled
R3 takes 0.9361 seconds/update (35,004 tokens/s), approximately 6.4× slower.
Peak allocated memory is 34.88 GiB and 33.50 GiB respectively. Naïve R3 at batch
16 takes 2.5866 seconds/update, using 19.18 GiB allocated and 23.27 GiB reserved.
Naïve batch 64 was skipped because scaled reserved memory exceeded device
capacity; this is an unmeasured configuration, not an observed OOM.

For scale only, multiplying these microbatch timings to a provisional global
batch of 512 gives about 1.62 GPU-hours for 5,000 SEQ updates and 0.81 hours for
a 2,500-update SEQ continuation. A naïve-only R3 continuation projects to
57.5 hours. An illustrative 250-update naïve ramp followed by 2,250 tiled
updates projects to 10.4 hours, but requires a separately validated backend
switch preserving optimizer state. The combined program projects to 12.9 hours
with that switch or 59.9 hours using naïve R3 throughout, before evaluation,
checkpoints, data handling, and other overhead.

These are linear planning estimates from three-step probes, not measured
accumulation throughput or guaranteed upper bounds. Multiplication also repeats
the measured optimizer cost, whereas real accumulation updates the optimizer
once. A conditional 20 GPU-hour execution cap for the switched program would
leave planning margin, but has not been authorized or launched; remeasure the
actual data/accumulation path before adopting it. A much smaller synthetic pilot
should come first, with its own width-256 benchmark and explicit stop budget.

## Next milestone

Pin and test the selected MQAR/noisy-recall generators and the custom ordered
state-update oracle; audit label alignment, answer masks, splits, and shortcut
baselines. Benchmark the actual small diagnostic model, freeze pilot conditions,
and then run paired SEQ/R3 training. Preserve ordinary SEQ as the common parent
for later CDRM experiments. CDRM, latent objectives, and substantive LM training
are subsequent milestones, not completed features of Stage A.
