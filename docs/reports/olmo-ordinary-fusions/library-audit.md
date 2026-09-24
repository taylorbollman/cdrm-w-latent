# Ordinary OLMo fusion library audit

2026-09-24. Read-only installed-source and upstream-source audit. No TE GPU
execution, import smoke, installation, numerical comparison or speedup is
claimed. The candidate scope is ordinary blocks and the existing CE objective;
RT tiles, FP8, normalization architecture and parameter ownership stay separate.

The recommendation is to prefer small functional replacements that retain the
native parameters and precision boundaries. **Do not replace our blocks with
TE `LayerNormLinear` or `LayerNormMLP` in this milestone.** These modules would
change normalization precision and require parameter/state adaptation. The
already-audited Dao CE is a smaller optional candidate, but its limited profile
share makes it reasonable to defer if the first few improvements suffice.

## Installed environment and source pin

The audit used the required project container with `CDRM_DOCKER_GPUS=none`,
reading files and distribution metadata without importing TE or executing
CUDA. Distribution versions are:

| Distribution | Installed version |
| --- | --- |
| `transformer-engine` | `2.16.0+4220403e` |
| `torch` | `2.13.0a0+8145d630e8.nv26.6.54250401` |
| `flash-attn` | `2.7.4.post1+git5231d95fe1.54157505` |
| `flash-attn-4` | `4.0.0b20` |

TE is installed under
`/usr/local/lib/python3.12/dist-packages/transformer_engine`. Both
`libtransformer_engine.so` and
`transformer_engine_torch.cpython-312-x86_64-linux-gnu.so` are present there.
There is no separate `transformer-engine-torch` distribution record; that does
**not** mean the bundled torch extension is absent. Presence is not proof that
its ABI imports or GPU kernels execute in our process. The FlashAttention
distribution version is also not a statement of which Python source wins the
project's import path: its vendored Dao modules are separately pinned in the
[earlier CE audit](../olmo-ce-integration/dao-ce-audit.md).

The TE version suffix resolves to upstream commit
`4220403e831d29e93868f7793693ea83f6b8b05b`. The installed LNLinear, LNMLP and CE
files were downloaded at that commit and their full-file SHA256 values matched.
Other inspected installed files are recorded below; no claim is made that all
package bytes have been compared with upstream.

| Path relative to installed TE root | SHA256 |
| --- | --- |
| `pytorch/module/base.py` | `744ff22d8936fd817b533197cc00fc5bdc8939d27f1f692e556b4108d9abb29b` |
| `pytorch/module/layernorm_linear.py` | `c5335a2ff859983fb50a1622f1f532fe761b9039f3e2d75c41ef5b1bb965ebeb` |
| `pytorch/module/layernorm_mlp.py` | `c68733ffc8fcc3e6140ec498b6ec152e09bb84eb23cc779841fb3c198afd823b` |
| `pytorch/ops/basic/layer_norm.py` | `166fbf43226c3c94ce2a9472651dd3795767dbb5054094d7c59d8ce135f7c2a6` |
| `pytorch/ops/basic/activation.py` | `06b39d68c40bf128dc24112fae7723c8627246bd11f4de3cd4b0fa882910c154` |
| `pytorch/cross_entropy.py` | `6e58d905ff32967876ae7a30b29a343325bec498027f1b4895d2bff1e2c582ba` |
| `pytorch/triton/cross_entropy.py` | `3e09310351e89a03e8b9f0ea83a44e9ec3012a50a2c1d6373b2483db964c8543` |
| `common/triton/cross_entropy.py` | `7c57cb8f31788f8d1dacc2bc2a3a190b7d0bb024bf4332fd0aa51cd5691d6a62` |
| `common/include/transformer_engine/activation.h` | `25d83b26e47351901e5c3eb4bc85f30ea0720d898774eea5e22dd56a44fed7d3` |

## Native contract and TE differences

Our checkpoint has nonaffine, mean-subtracting LayerNorm with epsilon `1e-5`;
FP32 parameters, residual stream and norm; BF16 dense projections under
autocast; and packed SwiGLU weights ordered `[value, gate]`. The activation is
`silu(gate) * value`. The readout uses the original token-embedding Parameter,
including all 50,304 vocabulary rows. The ordinary helper operates on existing
block parameters also used by RT. Preserving those identities, state-dict keys
and tied-gradient accumulation is part of a valid optimization.

| Component | Compatibility finding | Bounded disposition |
| --- | --- | --- |
| TE `LayerNormLinear` | Autocast casts the residual to BF16 **before** normalization; owns affine LN and projection parameters. | Defer full-module replacement. |
| TE `LayerNormMLP` | Same norm issue; gate-first packing differs; owned FC1/FC2 and norm parameters. | Defer; this is a larger adapter and precision experiment. |
| TE standalone LayerNorm | Also honors autocast by downcasting its input; always owns gamma/beta. | Only a later functional adapter with autocast disabled and fixed affine constants could preserve the intended contract. |
| TE SwiGLU | Gates the first half; a native packed input would compute the wrong function. | Do not directly substitute; retain our tested rounded compiled activation unless a separately mapped kernel is useful. |
| TE/Dao CE | Operates on logits, does not fuse the tied vocabulary projection. | Optional isolated loss candidate; prefer the existing out-of-place Dao plan. |

In installed LNLinear, lines 171–174 cast the norm input and affine tensors to
`activation_dtype` before calling normalization. LNMLP does the same at lines
357–360. The base module sets that dtype from CUDA autocast. Disabling autocast
around the whole module would retain FP32 normalization but would also select
FP32 GEMMs, defeating the intended mixed-precision substitution. These are
input-rounding differences even if the normalization reduction internally uses
FP32. [Pinned LNLinear](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/pytorch/module/layernorm_linear.py),
[pinned LNMLP](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/pytorch/module/layernorm_mlp.py),
[dtype selection](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/pytorch/module/base.py).

The module constructors have no nonaffine-LN option. `bias=False` removes the
linear bias, not the LayerNorm gamma/beta. `zero_centered_gamma` changes the
parameterization to `1 + gamma`; it does not remove the affine transform.
Freezing gamma at one and beta at zero would preserve that part of the formula,
but add registered state and possibly needless gradients/work unless wrapped
carefully. TE's base modules also serialize an `_extra_state` entry, even with
FP8 disabled. A direct replacement would therefore not preserve native state
keys. Standalone TE LayerNorm has the same affine ownership and autocast issue
in `ops/basic/layer_norm.py:70–113,198–215`.
[Standalone implementation](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/pytorch/ops/basic/layer_norm.py).

TE's activation API explicitly gates the first half; its C header documents
`Act(input[:H]) * input[H:]`. A TE MLP would need FC1 row reordering or an
activation reorder. Reordering live native weights would also affect RT;
maintaining a second copied parameter would break shared ownership and optimizer
state; concatenating activation halves adds materialization overhead. None is
an automatic drop-in. Fused activation can also remove intermediate BF16
rounding: our previous unrounded compiler experiment already showed why the
bounded gradient/loss check matters. The known-good compiler path retains
those casts. [Pinned activation contract](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/common/include/transformer_engine/activation.h).

Plain TE Linear is potentially adaptable, but does not remove the norm boundary
by itself. Keep `fuse_wgrad_accumulation=False` / `accumulate_into_main_grad=False`:
the alternate mode writes `.main_grad`, changing our optimizer and graph
gradient contract. Do not introduce FP8, persistent cast caches or a second
readout Parameter while pursuing ordinary fusion.

## CE: installed behavior, graph and checkpoint constraints

Installed TE `parallel_cross_entropy` accepts three-dimensional logits and
matching targets. Its Triton forward computes and **writes the input gradient
over the logits storage**; backward then scales that buffer. For our existing
FP32 logits, a possible adapter would pass fresh disposable logits reshaped to
`[1, positions, vocabulary]`, set `reduce_loss=False`, sum the returned FP32
per-position loss, and leave the valid-target denominator outside. Keep zero
label smoothing, no distributed group and the existing full vocabulary. This
preserves the mathematical objective and tied readout parameter, but the
installed implementation's buffer mutation is an additional integration
constraint. [Installed-version CE wrapper](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/pytorch/cross_entropy.py),
[installed-version CE kernel](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/common/triton/cross_entropy.py).

For this installed version, `is_cg_capturable=True` is required to avoid
`torch.equal` on CUDA tensor data in backward. That flag does not itself prove
our full checkpointed graph is correct. `nextlat._ce_chunk` has fresh local
logits and non-reentrant checkpointing, so there is a plausible integration
route; verify reconstructed saved tensors, hidden/readout gradients and changed
input/weight graph replay before accepting it. Do not reuse these logits for a
second objective or repeated backward. Retain input dtype FP32 initially;
passing BF16 directly would be a separate precision change.
[Graph-sensitive backward](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/pytorch/triton/cross_entropy.py).

**Upstream main differs materially.** At audit-time main
`21a066d257afeaf8039325f58df9a62587517976`, CE defaults to preserving its input,
has an `overwrite_input` option affecting backward, and treats
`is_cg_capturable` as deprecated/unused because capture is always supported.
Do not follow that API description when running the installed version. No TE
upgrade is recommended just for this small candidate.
[Newer CE API](https://github.com/NVIDIA/TransformerEngine/blob/21a066d257afeaf8039325f58df9a62587517976/transformer_engine/pytorch/cross_entropy.py).

Dao's existing audited FP32-logit, out-of-place loss option avoids TE's installed
destructive-forward behavior. It can leave projection, checkpoint chunk2048,
shared denominator and tied Parameter untouched. Our earlier profiles put the
named CE kernels at only a few percent of device time; replacing them does not
remove the vocabulary GEMM. A brief test is sensible if convenient, but there
is no demonstrated need to add two loss backends or prolong this milestone to
chase a small difference. See the
[Dao CE audit and scoped test proposal](../olmo-ce-integration/dao-ce-audit.md).

## Minimum acceptance if a candidate is pursued

Use one opt-in ordinary-only candidate at a time, keeping the current backend
as reference. Check unchanged parameter identities/state keys and scalar loss
counts; compare forward, raw hidden and all native parameter gradients using
the established budgets, with a strict FP32 isolated check where inexpensive.
Exercise both eager and existing non-reentrant checkpoint paths. Then validate
same-candidate graph replay with changed tokens and weights and a few Adam
updates. Warm forward and backward before capture. Benchmark complete steps
only after those operational checks, and retain numerical misses as measured.
A small operator speedup without a useful full-model gain is sufficient reason
to defer adoption and move on.
