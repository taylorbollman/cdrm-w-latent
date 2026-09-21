# OpenELM-1.1B: attention backends and layer-wise scaling

> **Historical model-selection record.** The user subsequently selected original
> OLMo-1B at approximately 252B tokens. The
> [v3 plan](fbt-rt-nextlat-research-plan-v3.md) and
> [current handoff](fbt-rt-nextlat-handoff.md) govern active work.
> OpenELM recommendations and future-tense milestones below retain their
> original context; source audits and completed results remain useful evidence.

Follow-up: the [ordinary import milestone](reports/openelm-import/results.md)
now validates the native1.1B checkpoint, including the container's cuDNN fused
attention path. RT/FBT implementation remains subsequent work.

Source audit and planning decision, 2026-09-21. No model changes, dependency
installation, checkpoint download or GPU execution accompanied this review.
The [research plan](fbt-rt-nextlat-research-plan-v2.md) is authoritative for
milestones and experiment scope.

**Decision**

Use OpenELM-1.1B as the primary pretrained backbone. Its layer-wise scaling is
compatible with RT, FBT and NextLat. Native Flash-compatible attention can serve
ordinary layers, including those inside FBT passes. Selected RT layers still
need the generalized exact recurrent schedule and backward. A custom FA4/CuTE
kernel is optional optimization work, not a prerequisite for the experiment.

Feature matching remains a late diagnostic. Semantic Tube Prediction and
Nanochat experiments are deferred; 450M is an optional resource fallback.

**1. What OpenELM actually calls**

The [OpenELM paper, section2.1](https://arxiv.org/html/2404.14619v1#S2.SS1)
explicitly reports Flash Attention. Both audited implementations invoke
`torch.nn.functional.scaled_dot_product_attention`:

- [Native CoreNet](https://github.com/apple/corenet/blob/f9f83e616a34d02c422733a06a3fe5bde63ae575/corenet/modeling/models/language_modeling/general_gpt.py):
  full-sequence training uses no additive mask and `is_causal=True`.
- [Pinned HF implementation](https://huggingface.co/apple/OpenELM-1_1B/blob/ee559a10b14895dde9f8cfde3fdc77b3ff0dbc0f/modeling_openelm.py):
  passes an explicit causal mask to SDPA.

SDPA selects a supported kernel according to runtime, dtype, layout and mask;
an SDPA call alone does not prove which kernel ran. Inspect actual dispatch
during the bounded feasibility profile. FlashAttention is an implementation
of attention arithmetic, so a checkpoint does not require the identical kernel
version used in pretraining. Floating-point results need not be bitwise equal.
[PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

The pinned HF code contains a relevant integration hazard: forcing
`_attn_implementation="flash_attention_2"` can remove the causal mask while
the attention call still lacks `is_causal=True`. Do not enable that flag
blindly. Preserve native causality, padding, document boundaries and cached
position offsets explicitly. For cached unequal query/key lengths, validate
the intended absolute-position mask rather than assuming a causal flag alone
has the correct alignment.

**2. Why exact RT tiling is a separate operation**

Suppressing native norms, RoPE and projection details, full RT computes

\[
a_t=\operatorname{Attn}\left(q(x_t),
 [k(z_{<t}),k_{\rm temp}(x_t)],
 [v(z_{<t}),v_{\rm temp}(x_t)]\right),
\qquad z_t=\operatorname{BlockUpdate}(x_t,a_t).
\]

Only then can it publish persistent K/V from z_t. A stock FlashAttention call
receives already-computed Q/K/V. It cannot generate z_t, run the layer MLP,
publish its memory and schedule the newly available memory for future queries.

RT's exact tiling updates online-softmax statistics as completed K/V tiles
become available. It can potentially use fused attention suboperations, but
the causal schedule remains. The paper describes compiled PyTorch and leaves
custom kernels for future work. [RT implementation discussion](https://arxiv.org/html/2604.21215v1#S6).

The active local code agrees:

- [Ordinary backend selection](../recurrent-transformer/olmo/model.py):
  `OLMoBlock._scaled_dot_product_attention` uses SDPA or an optional legacy
  `from flash_attn import flash_attn_func` path.
- `block_attention_add` and `OLMoRecurrentBlockTiledFunction` in that file implement RT
  with compiled PyTorch and custom autograd, not `flash_attn.cute`.
- [Docker requirements](../docker/requirements-docker.txt) declare
  `flash-attn-4[cu13]==4.0.0b20`; a dependency declaration is not runtime use.

Saved layer inputs and outputs permit parallel reconstruction of attention
intermediates. Historical positions must use persistent K/V, while the diagonal
uses temporary input-derived K/V. Ordinary causal attention over persistent
K/V alone gives the wrong diagonal. The reverse dependency through recurrent
memory writes also remains; ordinary attention backward cannot replace it.
Current reconstruction materializes a B×H×T×T attention tensor, so its memory
behavior is not FlashAttention's. Measure this before choosing physical batch.

Finite FBT passes obtain all their inputs from an already-computed previous
pass, so per-layer RT tiling remains applicable. Exact online FBT generation
instead receives fresh previous-token feedback and remains sequential across
tokens. This distinction belongs in prefill/decode benchmarks.

**3. What FA4/CuTE could contribute**

The official CuTe implementation at
`edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d` advertises H100/Hopper and Blackwell
support. It provides FP16/BF16 training, GQA, custom masks, score modifications
and optional log-sum-exp outputs. OpenELM's head dimension64 and 4:1 GQA fit
its basic constraints. Backward requires FP16/BF16 Q/K/V; keep a separate FP32
reference. Individual feature combinations still require checking.
[CuTe README](https://github.com/Dao-AILab/flash-attention/blob/edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d/flash_attn/cute/README.md),
[interface](https://github.com/Dao-AILab/flash-attention/blob/edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d/flash_attn/cute/interface.py).

Potential uses, subject to profiling:

1. Ordinary OpenELM attention and ordinary layers in FBT passes.
2. Known rectangular Q/K/V tile contributions, with correct softmax merging.
3. Parallel reconstruction using historical persistent memory plus temporary
   self entries, integrated with the recurrent adjoint.
4. A custom fused RT kernel if the measured benefit warrants kernel development.

Per-token cached attention is another possible use, but one-query launches and
cache assembly may limit speed. A custom score/mask callback does not eliminate
the dependency on not-yet-produced K/V. Never detach history to fit an API.

The public package inspected was 4.0.0b31; the project still declares b20.
No upgrade is made or required by this note. Pin and validate whichever version
is actually selected during implementation.

**4. Layer-wise scaling and the native geometry**

The residual width is **2048 throughout all 28 layers**. Internal capacities
change by layer:

| Zero-based layers | Query heads | KV heads | Attention width before output projection |
| --- | ---: | ---: | ---: |
| 0–2 | 16 | 4 | 1024 |
| 3–9 | 20 | 5 | 1280 |
| 10–17 | 24 | 6 | 1536 |
| 18–23 | 28 | 7 | 1792 |
| 24–27 | 32 | 8 | 2048 |

Every head has dimension 64. The SwiGLU intermediate widths are

```text
1024, 1280, 1536, 1792, 2048, 2304, 2560,
2816, 3072, 3328, 3584, 3840, 4096, 4608,
4608, 5120, 5376, 5632, 5888, 6144, 6400,
6656, 6912, 7168, 7424, 7680, 7936, 8192
```

Preserve these dimensions from the native configuration/tensors, including
rounding. Q/K use learned RMSNorm over each 64-dimensional head before RoPE;
norm weights are shared across heads within a layer. Attention and MLP outputs
both project back to 2048.
[Pinned configuration](https://huggingface.co/apple/OpenELM-1_1B/blob/ee559a10b14895dde9f8cfde3fdc77b3ff0dbc0f/config.json).

Consequences for our design:

- **RT:** each layer maps its own 2048-dimensional state into its own K/V
  geometry. Caches are not shared across layers, so their different sizes do
  not conflict. Bottom-layer RT has smaller memory and MLP capacity than
  top-layer RT, which matters when interpreting later placement ablations.
- **FBT:** the top state and bottom residual input both have dimension 2048.
  Feedback therefore needs no additional width bridge caused by layer-wise
  scaling. Every pass reuses the same nonuniform stack.
- **NextLat:** prediction operates on the final 2048-dimensional state and uses
  the native tied readout. Internal head counts do not alter its interface.

This is compatible mathematically, but the existing RT implementation assumes
MHA, attention width equal to residual width and full-vector Q/K norms, and
rejects RoPE. Its adapter/backend must support separate residual, Q and KV
widths; native output projections/FFNs; GQA gradient reduction; per-head norm
gradients; and correctly positioned temporary/persistent RoPE keys.

The reference sources cache normalized unrotated keys and rotate the assembled
keys during attention. A rotated persistent cache is possible, but specify one
convention and never rotate stored keys twice. Preserve native KV cache size;
repeating activations is an acceptable small oracle, not duplicated parameters.

**5. Bounded implementation sequence**

1. Import the native 1.1B checkpoint and match ordinary outputs, loss, gradients
   and cached behavior, preserving tokenizer/vocabulary and tied ownership.
2. Generalize the RT reference and tiling at the actual bottom-layer geometry;
   cover alpha0/intermediate/1, causal isolation and temporary/persistent K/V.
3. Check a short native-geometry gradient/update case in the intended precision,
   plus shared FBT-pass and NextLat paths. Reuse the existing numerical methods;
   do not reopen the historical broad precision campaign.
4. Profile ordinary fused attention and RT separately on H100, including
   backward peak memory and actual kernel dispatch. Use these results to choose
   batches and whether any additional fusion work is warranted.
5. Continue to the agreed matched learning experiments. Keep backend/precision
   differences explicit; do not attribute a numerical-policy change to recurrence.

Architectural compatibility does not establish useful training speed, stable
late-checkpoint adaptation or quality gains. Those remain measured outcomes of
the staged plan, rather than reasons to change the pretrained layer geometry.
