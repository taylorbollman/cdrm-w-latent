# Dao cross-entropy audit

2026-09-23. Source/import audit only: no Dao GPU timing or numerical result is
claimed here. The larger CE position chunk remains the first change to validate.
Dao's loss is a reasonable subsequent opt-in candidate, but it does not fuse the
vocabulary projection that dominated the small-chunk overhead.

## Source and environment

The local `vendors/flash-attention` tree reports package version **2.8.4**. It is
vendored into this repository, not a separate Git checkout; running `git
rev-parse` inside it would misleadingly report the enclosing project commit.
The five files below are byte-identical to upstream main at
**`2cb0a8e4710d6472fb1e0c5187af27cae6ed9ec2`**, resolved on the audit date
(upstream commit time `2026-09-23T16:58:00Z`). This is a file-content comparison,
not a claim that the entire vendor tree equals that upstream commit.

| File, relative to vendor root | SHA256 |
| --- | --- |
| `flash_attn/losses/cross_entropy.py` | `b63e48a1e519b92cc0d7ff36505afba3ed56ba81f22807358154ba7d6cc06c34` |
| `flash_attn/ops/triton/cross_entropy.py` | `cb9ea79ae3ada5237133bb0365c512817288b0811ac9fbafad4f1b015a20fe7c` |
| `training/configs/experiment/owt/base.yaml` | `1ee12f4027087c315c64cacc9a97e073f17d8b81e83ae8a66fd73e420a261f7f` |
| `training/src/tasks/seq.py` | `9494f0b360de43225d05aff7653990977b65169ebf4b20ae0411befc88bce4cd` |
| `tests/losses/test_cross_entropy.py` | `8b2d42efbe92fb7e44fdb7243962ef8caaec4f24d3a7a176b2bb621174d03603` |

A CPU-only container import with `CDRM_DOCKER_GPUS=none` succeeds for both loss
modules through the normal launcher. They resolve to the vendor paths, with
PyTorch `2.13.0a0+8145d630e8.nv26.06` and Triton `3.7.0`. No dependency installation
is needed for this import. GPU compilation, graph capture and execution are
separate checks that were not run in this audit. This CE implementation is Triton
code; selecting it would not change the ordinary attention backend or RT tiles.

## API and objective

`flash_attn.losses.cross_entropy.CrossEntropyLoss` takes already-computed
`[positions, vocabulary]` CUDA logits and `[positions]` labels. Its `sum`
reduction matches our chunk numerator; division by the shared valid-target
count should remain outside. Use `label_smoothing=0`, `logit_scale=1`,
`lse_square_scale=0`, `process_group=None`, `inplace_backward=False`, and no
precomputed LSE. These choices preserve the ordinary CE objective. The demo
uses the same module, with in-place backward enabled, after its model has
already returned logits. It does not fuse the language-model linear readout
with CE. [Loss module](https://github.com/Dao-AILab/flash-attention/blob/2cb0a8e4710d6472fb1e0c5187af27cae6ed9ec2/flash_attn/losses/cross_entropy.py),
[demo configuration](https://github.com/Dao-AILab/flash-attention/blob/2cb0a8e4710d6472fb1e0c5187af27cae6ed9ec2/training/configs/experiment/owt/base.yaml),
[demo training task](https://github.com/Dao-AILab/flash-attention/blob/2cb0a8e4710d6472fb1e0c5187af27cae6ed9ec2/training/src/tasks/seq.py).

The Triton kernels promote logits to FP32 for softmax arithmetic, return FP32
per-position losses, retain logits/LSE/labels, and return gradients in the
logits' dtype. Starting with our existing `F.linear(...).float()` output keeps
the FP32-logit interface. Passing BF16 logits directly could save the FP32
materialization, but deserves its own comparison. Non-unit last stride causes
a contiguous copy; misaligned int64 label storage can also cause a copy.
Out-of-range labels other than `ignore_index` are not rejected like PyTorch CE:
the loss can silently become zero. Keep existing target validation. This
custom backward is suitable for first-order training; no higher-order-gradient
claim is made. [Triton implementation](https://github.com/Dao-AILab/flash-attention/blob/2cb0a8e4710d6472fb1e0c5187af27cae6ed9ec2/flash_attn/ops/triton/cross_entropy.py).

Upstream tests compare forward and backward with PyTorch for FP32, FP16 and
BF16, ignore labels, smoothing and large vocabularies. They use FP32 gradient
`rtol=1e-5, atol=1e-6` and reduced-precision `rtol=1e-3, atol=1e-4`; these are useful
isolated-loss starting tolerances, not a full-model gradient budget.
[Upstream tests](https://github.com/Dao-AILab/flash-attention/blob/2cb0a8e4710d6472fb1e0c5187af27cae6ed9ec2/tests/losses/test_cross_entropy.py).

## Graph/checkpoint assessment and likely value

Code inspection suggests the single-GPU path can fit our static CUDA graphs:
after Triton warmup, allocations, launches and fixed-shape metadata branches are
capture-compatible in principle. It contains no tensor-value `.item()` in that
path. This is an inference, not tested capture support. Our non-reentrant CE
checkpoint should reconstruct its saved tensors normally with non-mutating
backward. Retain `inplace_backward=False` initially to avoid introducing saved
logit mutation into the checkpoint integration. Tensor-parallel collectives and
multiple GPUs remain outside this assessment.

The preceding six-layer B64/T512 profile measured CE log-softmax/NLL at
11.30 ms of 548.87 ms device-event duration with chunk128, and **13.60 ms of
332.40 ms (4.09%)** with chunk2048. Even completely removing that named family
would remove only that fraction of recorded device time; real acceleration
must be smaller unless additional allocations/copies are also eliminated.
This is an illustrative ceiling for those kernels, not a bound on every
possible fused-loss implementation or a measured Dao speedup. The large
chunk change instead removed repeated readout casts, full-weight gradient
adds, fill work and launch overhead. Dao CE still materializes the logits and
leaves those projection operations in place. See the retained
[profile audit](../olmo-ordinary-throughput/profile-audit.md).

Recommendation: complete shared CE-chunk validation and the 16-layer timing
first. A small later probe can compare the existing and Dao losses at
128/2048 positions and the actual 50,304-word vocabulary, checking logits and
tied readout/hidden gradients, non-reentrant checkpoint replay and graph replay
before measuring elapsed time/memory. Test FP32 logits first; consider BF16
input and in-place backward separately only if the first comparison warrants
them. Adopt an opt-in backend only if an end-to-end run shows a useful gain.
RoPE, SwiGLU and attention backend changes remain separate work.
