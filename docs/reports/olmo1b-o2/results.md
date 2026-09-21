# OLMo O2: native RoPE tiled RT

Completed 2026-09-21. Native tiled recurrence and checkpointed first-order
backward are implemented, with bounded actual-checkpoint validation. The
backward passes the FP32 semantic checks, including fractional recurrence,
attached caches and shared weights. One strict raw-input coordinate screen
required an explicit, documented calibration supported by an FP64 oracle;
the original failure is retained below. This is a correctness milestone and
an initial performance baseline, not a training-quality or speedup claim.

No pretrained optimizer update was made. FBT and NextLat remain subsequent
milestones. See the [usage guide](../../olmo1b-tiled-rt-usage.md),
[protocol and amendment](protocol.md), [machine-readable summary](validation-summary.json),
and [compaction handoff](../../fbt-rt-nextlat-handoff.md).

## Implementation and native model

`cdrm/pretrained/olmo_tiled.py` adds dyadic online attention tiling, native RoPE,
and a custom backward which reconstructs parallel intermediates from saved
block inputs/outputs. Reverse recurrent credit remains sequential. All four
block weights are explicit autograd inputs, and their gradients are returned
to the engine; there are no hidden writes to parameter `.grad`. Temporary self
KV and persistent historical KV remain separate. Fractional alpha and both
incoming and exported cache gradients are supported.

The model is unchanged original OLMo-1B at `step60000-tokens252B`, revision
`81b71efbce6f4dada57c94860301af4298bcd351`, with 1,176,764,416 parameters in
65 native tensors: 16 layers, D2048, 16 full-MHA heads, SwiGLU intermediate
8192, non-affine LayerNorm, native FP32 split-half RoPE, no Q/K normalization,
one tied 50,304-row embedding/readout. Full-model GPU checks select only layer
0 for RT; the other 15 layers are ordinary. Tiny tests also select the upper
layer and both layers. This is not an all-16-recurrent GPU clearance.

The new module is separate from O1's validated ordinary, sequential and
independent native-source oracle implementations. Those references and the
historical vendor implementation are unchanged.

## Correctness and precision

**230 CPU tests pass**, comprising the retained O1 suite and 82 new tests.
The new coverage includes irregular lengths/positions, masks, random raw output
and cache cotangents, terminal writes, parameter-only `autograd.grad`, frozen
weights, three shared calls, attached chunked training and cache provenance.
Structural tests confirm no sequential RT forward replay in backward and
linear retained forward activation state. The independent FP64 diagnostic has
ordinary-limit and finite-difference checks.

Actual checkpoint, H100 80GB, FP32 parameters, TF32 disabled:

| Full-model fixture | Comparison | Global parameter-gradient relative L2 | Worst parameter-tensor relative L2 |
| --- | --- | ---: | ---: |
| B1/T16, alpha 0 | Tiled versus ordinary | 2.055e-6 | 3.434e-6 |
| B1/T16, alpha 0 | Tiled versus O1 scan | 1.846e-6 | 2.784e-6 |
| B1/T16, alpha .37 | Tiled versus O1 scan | 1.543e-6 | 2.134e-6 |
| B1/T16, alpha 1 | Tiled versus O1 scan | 1.600e-6 | 2.197e-6 |
| B1/T128, alpha 1 | Tiled versus O1 scan | 2.319e-6 | 3.095e-6 |

All declared full-model logits/hidden/loss/input-gradient and joint per-tensor
parameter budgets pass. Stricter coordinate diagnostics flag a parameter in
the T16 alpha-0 scan comparison and a parameter in T128; both pass the unchanged
joint parameter budgets. Chunked decoding, causality and first-token temporary
self checks pass at alpha 0/.37/1.

BF16 autocast differences from the same tiled FP32 fixture are descriptive:
The T16 FP32 baseline uses math SDPA; T128 uses default SDPA. Thus the T16
default-backend rows also include the ordinary-layer dispatch change.

| Length | Ordinary-layer backend | Tiled attention | Global gradient difference | Worst tensor |
| --- | --- | --- | ---: | ---: |
| 16 | Math SDPA | Mixed | 1.516% | 3.499% |
| 16 | Math SDPA | FP32 | 1.648% | 3.689% |
| 16 | Default SDPA | Mixed | 1.489% | 3.723% |
| 16 | Default SDPA | FP32 | 1.660% | 3.659% |
| 128 | Default SDPA | Mixed | 2.714% | 3.197% |
| 128 | Default SDPA | FP32 | 1.675% | 2.219% |

Every checked BF16 output, loss, input gradient and parameter gradient is finite,
with complete ownership. FP32 attention improves the longer fixture, but does
not uniformly reduce full-model differences on the shorter fixture. Retain both
policies for future pilots; these measurements alone do not establish learning
stability or require a new architecture/norm. Other layers, dense projections
and MLPs still use mixed precision when only tiled attention is set to FP32.

### Raw-input coordinate qualification

The initial run stopped at a B2/T17 bottom-block stress test with Gaussian,
unnormalized cotangents on outputs and both persistent-cache tensors. All four
parameter tensors passed; the input-gradient norm difference was 5.548e-7,
but the strict coordinate rule `2e-6 + 3e-4*abs(reference)` failed near zeros.
That run remains [failed in its original report](initial-validation-01.json),
with a [reverse source patch](initial-validation-01-source.patch) that reproduces
its recorded source hashes from this milestone's source.

The final run preserves that strict screen: 65 of 69,632 coordinates flag in
the frozen-weight adjudication. An independent FP64 oracle, using the same
native FP32 sine/cosine constants but FP64 arithmetic, finds:

| Input-gradient comparison | Relative L2 | Maximum absolute error |
| --- | ---: | ---: |
| Tiled FP32 versus scan FP32 | 5.573e-7 | 6.485e-5 |
| Tiled FP32 versus FP64 | 6.143e-7 | 8.161e-5 |
| Scan FP32 versus FP64 | 5.439e-7 | 4.944e-5 |

The reference gradient's largest magnitude is 112.766. Both FP32 implementations
have sub-ppm norm error; neither is an exact oracle for tiny cancellation
coordinates. This supports ordinary rounding as the cause, rather than a
tiled-specific derivative defect. The explicit protocol amendment applies the
existing parameter-style joint tensor norm/maximum budget to this raw input
gradient, and requires it for both FP32 paths against FP64 as well as each
other. It passes. This was a post-failure calibration, not an unchanged original
input-coordinate threshold. Full-model input and all parameter budgets remain
unchanged. The final raw-block parameter-gradient relative L2 is 5.465e-7.

Two reviewed implementation improvements preceded the final run: exclude the
permanent current-position diagonal before mixed-precision reconstruction
matmuls, then add the temporary self contribution directly; and cast frozen
local-Jacobian weights once per backward. Both preserve the recurrence equations.

## Initial H100 performance

Three warmups and five timed forward/backward iterations per case; synchronized
wall time, no optimizer, compile or CUDA graphs. The full pretrained backbone
remains resident even for block-only measurements. Peak allocations include
the final finite-gradient check's scratch space; timing excludes that check.
Thus these are operational peaks, not pure-kernel memory or training-capacity
estimates. Parameter storage and gradients remain FP32.

| Scope | Execution/precision | B × T | Median wall time | Tokens/s | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| One block | Scan / FP32 | 1 × 128 | 0.201 s | 636 | 5.110 |
| One block | Tiled / FP32 | 1 × 128 | 0.319 s | 401 | 4.918 |
| One block | Scan / BF16 | 1 × 128 | 0.256 s | 499 | 4.918 |
| One block | Tiled / BF16 | 1 × 128 | 0.368 s | 348 | 5.000 |
| One block | Tiled / BF16 | 1 × 256 | 0.741 s | 345 | 5.033 |
| One block | Tiled / BF16 | 1 × 512 | 1.446 s | 354 | 5.108 |
| One block | Tiled / BF16 | 4 × 512 | 1.476 s | 1,388 | 5.527 |
| Full 16-layer model, RT layer 0 | Tiled / BF16 | 1 × 128 | 0.392 s | 326 | 9.506 |
| Full 16-layer model, RT layer 0 | Tiled / BF16 | 1 × 512 | 1.464 s | 350 | 9.509 |

The eager tiled implementation is about 1.6x slower than scan in FP32 and 1.4x
slower in BF16 for the checked B1/T128 block. Its explicit Python/local-autograd
work remains an optimization target. B4 nearly quadruples token throughput
at T512 without much changing per-step time. Do not extrapolate these small
checks to a maximum batch size or all-layer RT cost.

Forward retains linear activation state, but this first backward reconstructs
quadratic attention probabilities. It is not a FlashAttention backward kernel.
Higher-order derivatives, compiler/graph capture, distributed training, context
2048, optimizer memory, long-run precision and learning quality remain untested.

## Evidence and recovery

- Passing validation: `.runtime/olmo1b-step60000/tiled-validation-02/report.json`,
  W&B [nfys79l3](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/nfys79l3).
- Profiling: `.runtime/olmo1b-step60000/tiled-profile-01/report.json`,
  W&B [em18z6bm](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/em18z6bm).
- Initial failed attempt: `.runtime/olmo1b-step60000/tiled-validation-01/`,
  W&B [q4gu72t4](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/q4gu72t4).
- CPU record: [test-results.txt](test-results.txt).
- Full reports contain source hashes, native artifact manifest, exact tokens,
  runtime and per-tensor errors. The source checkpoint is reused from the
  verified O1 GCS object; O2 retention uploads evidence only. The
  [storage receipt](storage-receipt.json) records the evidence and checkpoint
  generations and hashes.
- Evidence prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-tiled-rt/20260921T205831Z/`.

Stop at O2 for review. O3 is the language-model objectives/platform milestone:
NextLat alignment/detachment/masks, optimizer/save/resume and practical batch
memory, with multi-GPU correctness when hardware is available. Neither O3
development nor a learning run is silently started by finishing O2.
