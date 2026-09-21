# Native OLMo tiled RT: API and execution contract

This is O2's tiled implementation of the recurrence established in the
[O1 native/sequential usage guide](olmo1b-native-rt-usage.md). Its source is
[`cdrm/pretrained/olmo_tiled.py`](../cdrm/pretrained/olmo_tiled.py). The
[O2 protocol](reports/olmo1b-o2/protocol.md) defines the bounded correctness,
precision and profiling scope. This guide documents behavior and reproduction
commands; GPU measurements and acceptance belong in the selected validation
reports, not inferred from this API description.

The model remains original OLMo-1B at step60000, approximately 252B pretraining
tokens: 16 blocks, width 2048, 16 full-MHA heads of dimension 128, SwiGLU
intermediate 8192, non-affine LayerNorm, FP32 split-half RoPE and no Q/K norm.
All 65 native parameter tensors and the single tied 50,304-row embedding/readout
are preserved. Default recurrence selects layer 0; the other 15 blocks remain
ordinary. The top layer is index 15.

## Load the existing pretrained artifact

Use the already verified O1 artifact directory. The following code performs
strict loading; it does not download or create a checkpoint. Run it only inside
the GPU container after confirming `/workspace/cdrm-w-latent` and successful
`nvidia-smi` there.

```python
import torch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, load_native_tokenizer
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode

artifact_root = ".runtime/olmo1b-step60000/artifacts"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = OLMoTiledRTForCausalLM(
    OLMoConfig.native_1b(),
    attention_backend="math",
    attention_precision="fp32",
    device="meta",
    dtype=torch.float32,
)
state = load_native_state_dict(
    artifact_root,
    expected_shapes={name: tuple(value.shape) for name, value in model.state_dict().items()},
)
model.load_state_dict(state, strict=True, assign=True)
model = model.to("cuda").eval()
del state
assert model.readout_weight is model.transformer.wte.weight

tokenizer = load_native_tokenizer(artifact_root)
ids = torch.tensor([tokenizer.encode("A running total starts at zero.")], device="cuda")
mode = RTMode((0,), 1.0)
with torch.no_grad():
    result = model(ids, mode=mode)
    ordinary = model(ids, mode=RTMode((), 0.0))
    tiled_ordinary_limit = model(ids, mode=RTMode((0,), 0.0))
```

Meta construction avoids random full-model storage before materialization.
Construct any future optimizer after strict loading and device conversion, and
count the tied parameter once. O2's validation and profiling commands perform
no optimizer updates and produce no newly trained checkpoint.

The forward interface accepts exactly one of `input_ids` and `inputs_embeds`,
plus `mode`, `attention_mask`, `position_ids`, `past_key_values`, `use_cache` and
`return_logits`. Outputs expose `logits`, `last_hidden_state` and
`past_key_values`. With `return_logits=False`, logits are `None` and the final
normalized state is returned without the vocabulary projection.

`RTMode` is frozen and contains selected layer indices plus scalar alpha in
[0, 1]. Empty selection executes ordinary blocks. Alpha 0 still executes the
selected tiled recurrence, making the ordinary limit a checked property.
Different calls can share weights with different modes before one backward.
This interface performs one stack pass; it adds neither FBT nor NextLat.

## Recurrence and exact tiling

For a selected block, let A be its native input LayerNorm, x_t its input and
z_t the completed attention-residual plus MLP-residual output. Queries and the
temporary self entry come from A(x_t). After finishing position t, write:

\[
m_t=(1-\alpha)x_t+\alpha z_t,\qquad
\bar{k}_t=W_K A(m_t),\qquad v_t=W_V A(m_t).
\]

Attention uses `RoPE(t, bar_k_t)` for historical keys and the input-derived
temporary key/value at the current position. Persistent keys remain unrotated
in exported caches. The write uses the full block output before final model
LayerNorm. There is no additional key normalization or embedding bypass.

The forward precomputes queries and temporary entries, completes positions in
causal order, and merges completed memory into future attention using dyadic
tiles and online softmax statistics. This evaluates the same causal recurrence;
"exact" does not promise bitwise equality across matrix shapes, reduction orders
or precision settings.

The custom backward reconstructs projected memory and attention in parallel
from saved x and z, then propagates recurrent adjoints in reverse order through
local block/write Jacobians. It does not regenerate z through another sequential
recurrent forward. A final batched VJP returns input and weight gradients.

The four native block weights are explicit custom-function inputs. Parameter
gradients are returned to PyTorch's autograd engine; the implementation does not
accumulate hidden parameter `.grad` fields. This supports parameter-only
`autograd.grad`, frozen parameter subsets and multiple attached calls. At
fractional alpha, the write-source adjoint contributes `(1-alpha)` to the input
branch and `alpha` to the completed-output branch. Exported cache cotangents
are also included, even for a terminal write read only by a later chunk or by
a cache-only loss.

For a single block, the low-level diagnostic interface is:

```python
from cdrm.pretrained.olmo_tiled import tiled_recurrent_layer

z, (keys, values) = tiled_recurrent_layer(
    model.layers[0], x,
    alpha=0.37,
    past=None,                    # Or unrotated native-head (past_key, past_value).
    query_positions=positions,    # int64 [B, T].
    key_positions=positions,      # int64 [B, P + T] when a prefix of length P exists.
    key_valid=valid,              # bool [B, P + T].
    attention_precision="fp32",
)
```

Here x is `[B,T,2048]` for the selected native model, and returned K/V have
shape `[B,16,P+T,128]`. This low-level API expects valid block-shaped arguments;
use the model API for cache provenance validation and input/mask preparation.

## Precision and backend controls

| Control | Meaning |
| --- | --- |
| FP32 parameters, autocast disabled | FP32 semantic reference; the validation drivers also disable TF32. |
| `attention_precision="mixed"` | Tiled attention matrix operands/results follow the projected Q/K/V dtype. Under BF16 autocast, these matrix operations use BF16; online accumulators and specified reductions remain FP32. |
| `attention_precision="fp32"` | Tiled attention matrix arithmetic uses FP32. Dense projections and MLPs retain the caller's autocast behavior. |
| `attention_backend="math"` | Forces math SDPA for ordinary layers. Selected tiled layers use their own attention arithmetic. |
| `attention_backend="sdpa"` | Lets ordinary layers use supported PyTorch SDPA dispatch. It does not switch the tiled layer to FlashAttention. |

Native RoPE keeps its FP32 internal rotation under both tiled attention
policies. Stored parameters and accumulated parameter gradients remain FP32
when using the prescribed FP32 model with BF16 autocast. "Mixed" does not mean
all state or reductions are BF16; "FP32 attention" does not mean the entire
model executes FP32 under autocast.

The constructor defaults to `attention_precision="mixed"`. The example above
selects FP32 attention explicitly for semantic inspection. To exercise mixed
attention, construct/select that policy and use:

```python
model.attention_precision = "mixed"  # Begin a fresh cache/history for this policy.
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(ids, mode=mode)
```

Precision, alpha, configuration and coordinates are captured per invocation;
backward restores the invocation's autocast mode. A finite BF16 diagnostic is
not a BF16 training-quality or long-context clearance. Reconstructed batched
operations can round differently from tokenwise operations; consult the
recorded same-state FP32 and sequential-reference comparisons separately.

## Cache and gradient ownership

`use_cache=True` returns an `OLMoTiledCache`; supply it as `past_key_values` on
the next call. Continuations may contain several tokens. The cache is attached
to autograd, and its K/V storage keeps native heads with unrotated keys.
Sequence length includes the prefix; no implicit detach or truncation occurs.

`attention_mask` covers all cached and current keys, and cached prefix validity
cannot change. `position_ids` covers current tokens. Without explicit IDs,
valid tokens advance position; cached continuation starts after the maximum
valid cached coordinate. Padding-query outputs must be excluded from losses.
Independent packed documents require caller-managed resets; this milestone
does not supply document-boundary attention masks.

Cache reuse requires the same model, parameter versions, model conversion
generation, RT mode, attention policy, configured ordinary backend and
autocast/autograd/inference context. In particular, a `no_grad` prefix cannot
silently become attached training history. Ordinary, sequential-RT and tiled-RT
caches are distinct types and cannot be exchanged. Position/mask metadata is
owned by the invocation/cache; callers must not mutate cached K/V tensors or
write parameters through `.data` while a cache is live.

The custom function supports **first derivatives only**. Double backward,
Hessian-based use and higher-order differentiation are outside its contract.
Returning explicit weight gradients removes a historical hidden-accumulation
obstacle; it does not establish DDP/FSDP or multi-GPU correctness.

## Memory and runtime limits

Selected tiled blocks retain linear forward activation state, principally
inputs, completed outputs, metadata and any prefix, plus references to weights.
They do not retain every token's MLP graph or a quadratic attention history
from the forward scan. Other ordinary layers retain their normal autograd
state; this is not a claim about constant whole-model memory.

The current backward reconstructs full attention matrices of shape
`[B,H,T,P+T]`. Without a prefix this is **quadratic in sequence length**, despite
the linear retained forward state. The implementation also retains Python
loops for forward block completion and reverse local VJPs. It has no custom
FlashAttention/CuTE kernel and does not automatically enable `torch.compile`
or CUDA graphs. Compiler capture, graph replay, outer checkpoint wrappers,
multi-GPU execution and context-2048 capacity require separate validation.

## Reproduce the bounded checks and profile

Launch from the project directory. Check the container and GPU first:

```bash
bash scripts/docker_shell.sh bash -lc \
  'pwd && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

The drivers also require the project container and successful GPU detection;
they have no CPU fallback. Both reuse the prepared O1 artifacts, require a
fresh output directory, and log online to W&B under
`taylorbollman/pretrained-fbt-rt-nextlat`:

```bash
bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_tiled_validate.py --artifacts .runtime/olmo1b-step60000/artifacts --output-dir .runtime/olmo1b-step60000/tiled-validation-new'

bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_tiled_profile.py --artifacts .runtime/olmo1b-step60000/artifacts --output-dir .runtime/olmo1b-step60000/tiled-profile-new'
```

Validation covers the protocol's FP32 semantic comparisons, bounded BF16
observations, random block/cache cotangents and cache/causality checks. Profiling
measures forward plus backward with three warmups and five timed iterations,
reports wall and CUDA-event times separately, and records allocated/reserved
memory. Full-model runs select only layer 0 for recurrence. It does not update
the checkpoint, optimize maximum batch size or estimate sustained training
quality. Allocation/correctness limits in the protocol bound the expansion.

Intentional CPU tests use the same container without GPU passthrough:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_olmo_tiled.py'
```

Keep O2 results in new directories, preserving O1/OpenELM evidence. Retention
should include tested source hashes and full diagnostic reports, referencing
the verified O1 checkpoint object rather than uploading another copy. The
current research plan retains FBT, NextLat, adaptation and distributed training
as subsequent milestones; these commands do not start them.
