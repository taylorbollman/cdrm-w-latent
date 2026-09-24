# Ordinary OLMo execution options

These are explicit options on `OLMoTiledRTForCausalLM`, including when its
selected RT-layer tuple is empty and it executes the ordinary model. They do
not add parameters, change native checkpoint keys or select a new RT backend.
Read [results](results.md) for the tested scope and recommended configuration.

```python
model = OLMoTiledRTForCausalLM(
    config,
    attention_backend="sdpa",
    ordinary_attention_backend="sdpa",  # optional: "fa4", qualified below
    ordinary_pointwise_backend="compiled",  # default: "eager"
    ordinary_activation_checkpointing=True,
    ordinary_checkpoint_layers=None,  # all; alternate layers only with headroom
    reuse_rope=True,
)
```

The example keeps the established attention and all-layer checkpointing,
adding only the validated rounded SwiGLU helper. The B32 alternative uses
`ordinary_checkpoint_layers=tuple(range(0, config.num_layers, 2))`.
Consult results for the tested shapes; new defaults remain off.
Record the execution choices alongside the training configuration: they are
not model weights and are not reconstructed from a weights-only state dict.
Set them before creating a prepared layout or CUDA graph; changing them
invalidates that layout and requires preparing/warming/capturing a new one.

- `ordinary_attention_backend="sdpa"` retains the existing ordinary SDPA/math
  choice. `"fa4"` cannot override the explicit math reference.
- FA4 initially accepts CUDA BF16/FP16 Q/K/V for dense, unpadded full-sequence
  causal attention. No prefix/cache export, arbitrary mask, padded sequence or
  FP32-projection fallback. Prepared all-valid layouts prove and remove their
  redundant padding mask before calling attention; public explicit masks are
  rejected. Native RoPE executes first and is unchanged.
- `ordinary_pointwise_backend="compiled"` compiles only the ordinary
  value-first/gate-second SwiGLU expression. Its compiler options preserve
  intermediate BF16 precision casts rather than eliminating their rounding;
  it is independent of graph replay and FA4. Fullgraph compilation must be
  warmed for both forward and backward before capture.
- During grad-enabled training with checkpointing enabled,
  `ordinary_checkpoint_layers=None` checkpoints
  all ordinary calls; `()` checkpoints none; a sorted unique tuple selects
  layer indices. Selected RT calls use their own recomputation. Disabling the
  boolean requires `ordinary_checkpoint_layers=None`, preserving the old API.

## Container and evidence harness

Use the matching installed FA4 wheel, not the historical vendor source:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
  test -f /.dockerenv && nvidia-smi &&
  python scripts/olmo_ordinary_efficiency.py \
    --stage correctness --arm fa4 --batch-size 8 --length 512 \
    --output-dir .runtime/olmo-ordinary-efficiency/NEW-UNIQUE-RUN
'
```

The harness requires an unused output directory, logs online to W&B, reads
the pinned local pretrained artifacts and snapshots its sources/protocol.
It sets deterministic execution before importing CuTe. `--stage capacity`
accepts physical B16/32/64 at T512 and B8/16 at T2048; it performs only bounded
preparation/timed updates, not a learning experiment. `--profile` records an
additional untimed CUDA trace.

Arms: `control`, `fa4`, `compiled`, `checkpoint-alternating`,
`checkpoint-none`; separately verified combinations are available as
`fa4-compiled`, `fa4-compiled-checkpoint-alternating`, and
`fa4-compiled-checkpoint-none`. Combinations retaining SDPA are
`compiled-checkpoint-alternating` and `compiled-checkpoint-none`.
Correctness defaults to comparing against
control; `--reference-arm` selects a different explicit reference.

For a finite numerical-only miss, `--continue-after-compatibility-miss`
can complete the remaining correctness diagnostics. It does not relax any
threshold: the failed compatibility check remains, the overall report stays
failed, and the process exits unsuccessfully. Structural, dispatch, graph and
optimizer failures still stop immediately. Any timing under such a qualification
is exploratory and does not clear the numerical screen.

Full CE2048/KL128, fixture masks, precision, deterministic settings and physical
batch must match when comparing timings. CUDA graphs capture forward/loss/
backward; the reported full step also includes copying, clipping and AdamW.
Changing batch size is a throughput/memory alternative, not an equivalent
learning comparison. Accumulation remains separate work.

Generate the final summary and run retention in the same project container
(`CDRM_DOCKER_GPUS=none` is appropriate). Retention checks derived summary values
exactly; Python3.10 and3.12 use slightly different floating-point summation,
which can change the last bits of descriptive profile totals. This does not
change GPU measurements. Every selected run must supply its recorded source
revision through `--runtime-commit` and, for earlier attempts, `--run-commit`.
