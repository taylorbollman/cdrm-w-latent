# Ordinary fused RoPE and AdamW usage

These are explicit execution options. Native RoPE and the previous AdamW
dispatch remain defaults. Consult [results](results.md) for measured scope and
qualifications; this page is not authorization for a learning run.

```python
base = OLMoTiledRTForCausalLM(
    config,
    ordinary_activation_checkpointing=True,
    ordinary_checkpoint_layers=None,  # every ordinary layer
    ordinary_attention_backend="sdpa",
    ordinary_pointwise_backend="compiled",
    ordinary_rope_backend="dao",
    reuse_rope=True,
    # retain the existing RT settings separately
)
optimizer = build_adamw(
    model, lr=1e-5, betas=(0.9, 0.95), eps=1e-8,
    weight_decay=0.1, foreach=False, fused=True,
)
```

RoPE uses installed `flash_attn.layers.rotary.apply_rotary_emb`, with native
FP32 inputs/tables and an out-of-place result restored to Q/K dtype. Compact
tables preserve actual positional coordinates and are prepared once, outside
capture. This initial Dao option requires CUDA and supports cache-free ordinary
calls; ordinary prefix/export-cache calls fail explicitly. Native RT layers
retain their existing RoPE implementation. Runtime numerical/performance
evidence here is ordinary-only; other objective/backend combinations require
their own bounded integration checks.

Use `CDRM_FLASH_ATTENTION_SOURCE=installed` with the project launcher to select
the measured installed Dao source. Do not assume a different upstream version
has the same dtype/gradient-clone behavior. See [RoPE audit](dao-rope-audit.md).

Fused AdamW retains FP32 parameters, gradients and moments. It runs outside the
CUDA graph, as do clipping and the scheduler. `fused=True` cannot be combined
with `foreach=True`. It changes the optimizer descriptor used for exact resume;
do not silently alter that flag when resuming an old scalar-Adam run. The
fixed-gradient comparison permits bounded FP32 implementation differences;
same-candidate eager/graph complete updates require exact agreement.

Reproduce the primary candidate check from the project root:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_ordinary_fusions.py --stage correctness \
   --arm dao-rope-fused-adam --batch-size 8 --length 512 \
   --output-dir .runtime/olmo-ordinary-fusions/NEW-UNUSED-NAME'
```

Capacity uses `--stage capacity --batch-size 64 --length 512`, optionally
`--profile`. The profile adds one untimed optimizer update, yielding nine
physical updates rather than eight. `--optimizer-probe` is restricted to
correctness/fused-adam/B8/T512 and adds six fixed-gradient updates to the six
own eager/graph updates. Reports preserve all physical counters. Every run
requires a fresh output directory and logs online to W&B.

For future learning/save-resume integration, store the execution options in the
run/checkpoint configuration and reconstruct them explicitly: backend flags are
not tensors in `model.state_dict()`. The generic checkpoint validates the
caller-supplied configuration; it cannot infer omitted backend settings. The
current diagnostic saves source/configuration evidence, not trained weights.

Reference settings are all-layer activation checkpointing, rounded compiled
ordinary SwiGLU, deterministic PyTorch Flash SDPA, CE position chunks2048,
full valid CE supervision, BF16 mixed/FP32 residuals and norm, TF32 off and
autocast weight cache off. The graph captures forward/loss/backward. Rates are
complete training updates with input-copy/validation, clipping and Adam; they
are not inference rates or claims of learning quality.

TE full-module replacements were deferred after inspecting precision,
parameter ownership and packed-half differences. See [library audit](library-audit.md).
The selected path adds no learned parameters or logical matrix FLOPs.
