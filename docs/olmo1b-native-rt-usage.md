# Original OLMo-1B import and sequential RT reference

This implements the O1 milestone of the
[current plan](fbt-rt-nextlat-research-plan-v3.md). It preserves the selected
approximately 252B-token checkpoint's native computation, then exposes selected
blocks as attached sequential RT scans. See
[validation results](reports/olmo1b-o1/results.md) for completed scope and
[protocol](reports/olmo1b-o1/protocol.md) for the checks and tolerances.

## Selected model

Original `allenai/OLMo-1B`, `step60000-tokens252B`, revision
`81b71efbce6f4dada57c94860301af4298bcd351`: 16 layers, residual width 2048,
16 query/KV heads of dimension 128, SwiGLU intermediate 8192, non-affine
LayerNorm and RoPE. There is no Q/K normalization. One tied matrix retains all
50,304 embedding/readout rows, including rows outside the tokenizer's 50,280
vocabulary. The embedding has no `padding_idx`; explicit masks govern padding.

The native file's outer `model.` wrapper is removed when loading; all 65 native
parameter keys beneath `transformer.*` are retained. No projection splitting,
reinitialization, vocabulary cropping or untied-to-tied conversion occurs at
load. The separately published HF conversion remains an unvalidated secondary
artifact; this implementation uses the native source and bytes as authority.

## Prepare and inspect immutable artifacts

Use the project container even for intentional CPU inspection:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_prepare.py --artifact-root .runtime/olmo1b-step60000/artifacts --inspect'
```

Preparation uses `hf download` at the exact checkpoint SHA, then validates full
file sizes and hashes. Existing mismatched bytes are rejected. The native
tokenizer adds neither BOS nor EOS automatically; `encode(text, add_eos=True)`
is an explicit document-EOS option. The manifest records fixed Unicode/code/
special-token fixtures and the raw checkpoint config.

Do not execute the checkpoint's remote-code stubs through the installed local
`olmo` package: that would use our modified research fork. The independent
reference instead executes byte-identical upstream files at
`b3741bc21f1dd504838b7dbd9878ee077ded63bd`, isolated under
`cdrm.pretrained._loaded_olmo_reference`. See its
[source README](../cdrm/pretrained/_olmo_reference/README.md) for the limited
import adaptations. Its constructor's global SDPA flag changes are restored
so each comparison explicitly controls the attention backend.

## Model API

Inside the GPU container, after verifying `nvidia-smi`:

```python
import torch
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, load_native_tokenizer
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM, RTMode

root = ".runtime/olmo1b-step60000/artifacts"
config = OLMoConfig.native_1b()
model = OLMoRTForCausalLM(config, device="meta", dtype=torch.float32)
state = load_native_state_dict(
    root, expected_shapes={name: tuple(value.shape) for name, value in model.state_dict().items()}
)
model.load_state_dict(state, strict=True, assign=True)
model = model.to("cuda").eval()
del state
# Construct an optimizer only after materialization/device conversion if needed.
tokenizer = load_native_tokenizer(root)
ids = torch.tensor([tokenizer.encode("A running total starts at zero.")], device="cuda")

with torch.no_grad():
    ordinary = model(ids, mode=RTMode((), 0.0))
    recurrent = model(ids, mode=RTMode((0,), 1.0))
    # Alpha 0 still executes a scan, making ordinary equivalence testable.
    scan_limit = model(ids, mode=RTMode((0,), 0.0))
```

The ordinary-only `OLMoForCausalLM` has identical native state keys and no
`mode` argument. Constructors initialize fixture weights, but strict import
replaces every weight before use. Meta construction avoids allocating random
full-model tensors. Keep `model.readout_weight is model.transformer.wte.weight`
and count it only once in an optimizer. `return_logits=False` returns final
normalized states without materializing logits. `inputs_embeds` remains
attached; no internal detach is introduced.

## Recurrence and cache contract

For selected block input x_t and completed output z_t, historical memory is
written from `LayerNorm((1-alpha)*x_t + alpha*z_t)` through the native K/V
projections. The current token reads its input-derived temporary self entry
and earlier completed writes. Keys undergo native RoPE for attention and remain
**unrotated in the cache**. The full residual block output is used before model
final normalization; there is no Q/K norm or additional embedding bypass.

`RTMode` is immutable per forward, with selected layer indices and scalar alpha
in [0, 1]. Default selection is layer 0; top is 15. Different calls sharing weights
can keep different modes while their graphs remain attached. This is a one-pass
RT interface; FBT and NextLat are not implemented here.

Use `use_cache=True`, then pass `output.past_key_values` on continuation. Cached
chunks may contain multiple tokens. Explicit `attention_mask` covers cached
plus current keys; its prefix cannot change. `position_ids` covers current
tokens and may carry explicit offsets. By default, valid tokens advance RoPE
positions; padding queries must be excluded from losses. Document resets are
the caller's responsibility: packed independent-document attention is not an
interface in this milestone.

Recurrent caches reject another model's weights, changed parameters, a model
conversion (including dtype round trips), changed RT mode, autocast/gradient/
inference context or configured backend. Thus a prefix computed under
`no_grad()` cannot silently become attached training history. Metadata is
copied; do not mutate cached KV tensors. Unsupported `.data` weight writes are
outside the contract. Ordinary and recurrent cache types are distinct.

The implementation uses ordinary autograd and a token loop. It is a correctness
reference, not the planned exact tiled backend. It does not enable `torch.compile`,
CUDA graphs, custom tiled backward or activation checkpointing.

## Reproduce the bounded GPU checks

Each invocation requires a new output directory and logs online to W&B:

```bash
bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_validate.py --artifacts .runtime/olmo1b-step60000/artifacts --output-dir .runtime/olmo1b-step60000/ordinary-validation-new --max-length 64'

bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_rt_validate.py --artifacts .runtime/olmo1b-step60000/artifacts --output-dir .runtime/olmo1b-step60000/rt-validation-new --length 16'
```

Drivers require the container working directory and successful `nvidia-smi`;
there is no CPU fallback. BF16 checks use FP32 parameters with autocast. Source
equivalence and BF16-versus-FP32 differences are separate report sections;
the recurrent BF16 scope is finite forward/backward and complete gradient
ownership, not training clearance. Saved elementwise error screens in BF16
rows are diagnostic, even when an outer `passed` means the finite smoke check.

Scoped CPU tests are `tests/test_olmo_*.py`; run them explicitly with
`CDRM_DOCKER_GPUS=none`. Historical OpenELM files and reports remain unchanged.

## Retain and restore

`scripts/olmo_retain.py` requires completed ordinary and RT reports, unchanged
validated source hashes and intact prepared artifacts. It uploads the native
checkpoint separately from a small source/evidence archive, creating new GCS
objects only and verifying remote size, server MD5 and SHA256 metadata. Use
one timestamp below `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-step60000/`.
The archive's `RESTORE.md` and `evidence-members.json` describe restoration.
See the result's storage receipt for the actual retained prefix and generations.

Use a new lineage for subsequent tiled/adaptation experiments. No newly trained
checkpoint was produced by this milestone.
