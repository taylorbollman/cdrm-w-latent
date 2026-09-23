# Author-derived native RoPE port: source audit

Source audit, 2026-09-23, with the initial CPU port checks recorded below. The
user authorized proceeding to the author-derived comparison after Stage A.
This document records source findings and the adapter contract; it does not
establish GPU numerical, compilation/capture or performance qualification.

## Source identity

The authoritative upstream revision is
`a21b42d2bc292edb86ed1b62cee4bcab809a9d21` in the project-root
`recurrent-transformer/` repository. Its `olmo/model.py` SHA256 is
`8b552064f233b93f9ec348d93a55b972aac7b62a191460353de6fbdbd7bd3acb`.
The current local fork is clean at
`824767f9a6f8e29959a0c486d169ce10b7194d41`; its `olmo/model.py` SHA256 is
`dfc37f10c8da1480265e8b90c9245a5120c8e596d6550057d9769c588eb2c9f5`, and
`olmo/config.py` is
`c019ee9fb70e47d778f652ed7b410984bc65a75a6f829dc23a460a8d9262e904`.
The fork contains 472 changed lines in `model.py` relative to the pin and must
not be described as pristine upstream. Freeze the actual extracted/adapted
source and document each substantive deviation.

Relevant local-fork definitions in
[`olmo/model.py`](../../../recurrent-transformer/olmo/model.py):

| Definition | Current line | Role |
| --- | ---: | --- |
| `block_autocast`, `recurrent_helper`, `recurrent_fp32_state` | 81–106 | Explicit replay/eager/precision selection |
| `RotaryEmbedding` methods | 330–390 | Existing ordinary split-half RoPE, with length-based global cache |
| `SwiGLU` | 432 | Value-first/gate-second split; `silu(gate) * value` |
| `PreAttentionBlock` | 968 | Separate Q and KV projections, optional norms, head layout |
| `PostAttentionBlock` | 1017 | Residual addition, normalization, MLP and residual output |
| `OLMoRecurrentBlockBase` | 1039 | Canonical split Q/KV ownership and shared helper views |
| `OLMoRecurrentAutogradBlock._real_forward` | 1153 | Sequential ordinary-autograd recurrence |
| `block_attention_add` | 1257 | Compiled dyadic historical forward update |
| `recompute_alphas`, `recompute_atts` | 1287, 1305 | Materialized backward attention reconstruction |
| `mlp_batched_body` | 1316 | Compiled batched final MLP recomputation |
| `OLMoRecurrentBlockTiledFunction` | 1328 | Tiled forward and custom reverse recurrence |
| `OLMoRecurrentBlockTiled` | 1649 | Public tiled guards/entrypoint |

Both recurrent implementations currently reject RoPE. The ordinary RoPE helper
is useful source authority, but calling its one-token API naively would reset
the token position; the port must provide the actual position's precomputed
table slice. Stage A's immutable FP32 tables already express that contract.

## Execution strategy to preserve

The author implementation already writes only K/V permanently. Temporary inputs
use two projections (`q_proj`, `kv_proj`), whereas the native model owns one
packed `[3D,D]` parameter and ordinarily computes temporary QKV in one GEMM.
Preserving separate author Q/KV GEMMs with views of the packed weight is a
recordable execution difference, not an architecture or parameter change.

Forward uses position-major `[T,B,H,Hd]` running state and permanent K/V storage.
After each token finishes, a dyadic tile updates the following queries. The
attention helper uses compiled matrix products/pointwise operations; this RT
path does not call Flash Attention. Backward retains `x` and completed outputs,
then reconstructs attention in full `[B,H,key,query]` arrays. Its causal triangle
is therefore the transpose of a conventional `[query,key]` interpretation.

The major backward difference from our native implementation is important:

- Author code reconstructs each permanent writer with a graph-connected output
  leaf and writer weights. Reverse-time `torch.autograd.backward` accumulates
  writer parameter gradients **per token**, while supplying the completed-state
  adjoint needed by recurrence.
- Local MLP input VJPs use chunks selected by `bwd_mlp_chunks`. The config default
  is **1**; earlier project precision runs explicitly chose **4**. More chunks
  reduce retained MLP activations but can reduce autocast-cast reuse. Record the
  setting and change it only as a labeled memory/runtime choice.
- Temporary Q/K/V and final MLP parameter VJPs are computed in batched operations.
  The function returns only `x.grad`; real module parameter `.grad` writes occur
  as nested backward side effects.
- Native RT instead freezes local writer weights and computes a final batched
  permanent-projection parameter VJP. Substituting this into the port would be
  an additional algorithmic adaptation, not a minimal author baseline.

`PreAttentionBlock._compiled_forward` and `block_attention_add` use
`torch.compile(dynamic=False)`; `recompute_alphas`, `recompute_atts` and
`mlp_batched_body` use `torch.compile(fullgraph=True)`. Whole-model compilation
is not the source strategy. Preserve these bounded helpers where practical;
warm every shape before graph capture and record compilation/fallback behavior.
Local MLP reconstruction shares autocast contexts to reuse weight casts.

`scripts/train.py:163–168` can additionally call `block.compile(...)`, but the
author's root `sweep_recurrent_512.yaml` explicitly sets `compile: null` and
`cuda_graph: whole`, overriding the base recipe's compilation default. No
external compilation of per-token post-attention MLP or permanent writer
helpers was found in `olmo/` or `scripts/`. The port should not quietly enable
whole-block/local-MLP compilation and call that the original execution strategy.

The author already raises Dynamo's `cache_size_limit` to **13** in
`scripts/train.py` (both the upstream pin and current fork). T512 visits nine
power-of-two tile widths, 1 through 256, before counting distinct dtypes,
strides or grad modes; an unmodified limit of eight can silently benchmark a
fallback instead. The existing project helper
`scripts/stage_a_common.py::configure_compiled_helpers` sets
`recompile_limit=64` and `fail_on_recompile_limit_hit=True`; the existing
`scripts/stage_b_train.py::compiler_audit` rejects graph breaks/unimplemented
paths and records unique graphs and accumulated limits. Reuse that bounded
policy or implement an equally explicit audit, including actual runtime option
names/values. Fresh processes per benchmark shape prevent unrelated experiments
from exhausting the same helper cache. Limits are runtime metadata, not a
throughput optimization to leave unrecorded.

Functional compiled helpers should take Tensor operands plus immutable scalar
dimensions/epsilon/precision flags. Avoid constructing a new proxy `nn.Module`
inside every forward/backward or closing over invocation-specific tensor leaves;
those object identities can force new compiler guards. Keep reusable helper
functions at module scope, compile only the intended boundaries, and build
native-order RoPE tables outside their compiled bodies where practical. Warmup
must cover grad-disabled custom forward and grad-enabled reconstruction, not
only the easiest forward shape. An observed new specialization is not itself a
failure; a graph break or fallback is.

## Native block mapping and precision boundaries

Map native weights without changing values or registering new parameters:

| Native parameter/behavior | Author-derived counterpart |
| --- | --- |
| `att_proj.weight[:D]` | Q projection |
| `att_proj.weight[D:]` | K/V projection |
| `attn_out.weight` | Attention output projection |
| `ff_proj.weight`, shape `[2M,D]` | Value/gate projection |
| `ff_out.weight`, shape `[D,M]` | MLP output projection |
| Nonaffine LayerNorm, epsilon `1e-5` | Ordinary LayerNorm with affine disabled |
| No Q/K normalization | `attention_layer_norm=False` |
| Native pre-norm and zero dropout/bias | `norm_after=False`, all dropout/bias disabled |

For D2048 and native M8192 **per branch**, the author's `mlp_hidden_size` must
be **16384**, because its SwiGLU output multiplier is 0.5. Passing 8192 would
halve the MLP relative to the checkpoint. Model tying remains owned by the
native wrapper and is irrelevant to isolated block timing.

Rotate input-derived Q, temporary K and permanent output-derived K in native
FP32, restoring projection dtype afterward; never rotate values. Perform the
same rotation in graph-connected backward reconstruction so input and parameter
adjoints pass through it. For the author arithmetic, rotate Q **before** its
existing query prescaling.

Legacy author mixed precision prescales Q in projection dtype before the dot,
while native RT scales the completed score product. Its reconstructed softmax
uses FP32 internally, then casts probabilities to Q dtype; the native backend
retains additional probability/error arithmetic in FP32. Temporary-diagonal
pointwise arithmetic and low-precision attention adjoints differ as well.
These boundaries can affect BF16 gradients independently of the RoPE port.
Start with the unchanged native FP32 scan reference, then compare actual BF16
settings. Any arithmetic-aligned diagnostic must have a separate label rather
than silently replacing the author policy. FP32 agreement does not clear BF16.

## Required repairs and proposed adapter contract

Retain the local fork's canonical module ownership/legacy-alias validation,
complete norm/activation conversion semantics, replay of actual dense autocast
dtype, unsupported-input guards and whole-training-graph ownership discipline.
The optional `bf16_fp32_state` protection is a precision policy, not a universal
correctness repair. See the existing
[change audit](../rt-precision-alignment/change-audit.md) for this distinction.
Do not revive `make_graphed_callables` around the unchanged nested-gradient
function or silently claim shared-FBT/distributed compatibility.

A small project-owned functional adapter is preferable to importing the entire
modified author model and exposing unrelated CDRM/configuration paths:

```python
author_recurrent_reference(layer, x, rope_tables, *, precision_policy=...)
author_tiled_recurrent_layer(
    layer, x, rope_tables,
    *, compiled_helpers=True, bwd_mlp_chunks=..., precision_policy=...,
)
```

Initially require full-strength recurrence, equal Q/K/V head counts, one dense
unpadded document per row, no exported/prefix cache, no learned attention bias,
no FBT or NextLat, and explicit supported dtypes. Return hidden outputs only.

The custom autograd entry should receive the native `x`, packed QKV/output/MLP
weights and fixed RoPE tensors explicitly. A bounded way to retain the author's
reverse schedule without modifying real parameter `.grad` is to reconstruct
invocation-local detached Q/KV leaf views, accumulate nested writer gradients
only on those local leaves, then concatenate and return the full packed-weight
gradient to outer autograd. Other native weight gradients are returned likewise.
This is a proposed ownership adaptation requiring tests, not a proven shortcut.
Save the real input weights for normal version/lifetime checks, and never
register split copies as independent parameters.

Repeated `F.linear(weight[D:])` on an ordinary nonleaf view can lose the author's
leaf-weight autocast cache benefit. Local leaf views or explicit invocation-owned
casts must avoid repeatedly converting full matrices. Preserve intended cast
reuse and document its mechanism; capture changed-weight checks must prove that
casts reread updated parameters. Reusing stale detached BF16 snapshots is invalid.

Custom `autograd.Function.forward` runs without recording autograd operations;
the original module parameters still carry their leaf/gradient metadata. Capture
and backward-replay scopes must be audited explicitly: do not carry a cached
no-grad forward cast into gradient-enabled reconstruction, and do not enter
capture with BF16 weight casts created by uncaptured warmup still cached.
Use bounded fresh autocast scopes and keep gradients on invocation-local leaves.
Compare eager/captured raw gradients and post-update outputs to detect a stale
or disconnected cast; checking finite losses alone is insufficient. This is a
required port validation concern, not a demonstrated fault in the author path.

Minimum new checks: native FP32 scan versus adapted author scan/tiled outputs and
raw gradients; packed gradient mapping and unchanged parameter ownership;
dyadic boundary lengths; intended BF16 arithmetic; exact same-candidate captured
replay and a few changed-input/changed-weight Adam updates. The quadratic
workspace and per-token writer gradients must be included in measured memory
and timing. Only advance to shared FBT or broad wrapper replacement after a
repeatable block/stack performance reason and its own integration validation.

## Initial implementation and CPU review

The adapter in [`olmo_author.py`](../../../cdrm/pretrained/olmo_author.py)
implements both functional entrypoints, the author's helper-compilation
boundaries, private leaf gradient accumulation, native table rotations and
explicit packed-gradient returns. The author-style FP32-state policy remains
a separately named diagnostic. Original sequential references are unchanged.

Preserve the author's tensor lifetimes as well as its operations: release
temporary K/V after forward state initialization; release the full attention
matrix/permanent-key working storage after final dQ; release temporary
Q/K/V/adjoints after their VJP; and release the per-token adjoint list after
concatenation, before final batched MLP reconstruction. Retaining those unused
tensors until function exit would inflate measured peak memory. At B512/T512/H16,
one BF16 square attention matrix alone occupies 4 GiB.

The 57 focused CPU-container checks in
[`test_olmo_author.py`](../../../tests/test_olmo_author.py) pass at this initial
review. They compare author scan/tiled against the unchanged native FP32 scan
at lengths 1/2/3/5/8/17 and MLP chunk counts 1/4; cover arbitrary raw cotangents,
offset/nonconsecutive positions, two recurrent layers, shared calls, frozen
inputs/weights, zero cotangents, causality and checkpoint/gradient ownership;
and reject unsupported metadata. Two tiny CPU BF16 checks exercise finite
private-leaf replay with autocast caching on/off and backward outside the
forward autocast context. Those checks do not validate CUDA BF16 numerics.

Independent dispatch counts of executed eager `mm`/`bmm` at lengths 3/5/8 and
MLP chunks 1/4 confirm, per full-trainable RT invocation with `N=BT`:

- Dense matrix arithmetic: **`50ND² + 36NDM`**, versus the native KV-only
  implementation's `58ND² + 36NDM`. The author avoids the native implementation's
  additional separate batched writer VJP/reconstruction, while performing its
  parameter VJPs sequentially; fewer FLOPs need not mean better throughput.
- Attention matrix arithmetic: **`10BD·E + 6BD·T² − 6BD·floor(T/2)`**, with
  `E=T(T−1)/2`. The subtraction matters: singleton reverse tiles use pointwise
  multiply/reduce instead of three matmuls, so that work is outside a strictly
  matrix-only ledger. Forward singleton tiles still use matmuls.

These accounting checks validate the CPU operation ledger, not compiled
hardware instructions, fused-kernel FLOPs or observed GPU efficiency.
