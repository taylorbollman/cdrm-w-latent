# F3d bounded-workspace RT backward

Execution option on `OLMoTiledRTForCausalLM` and `tiled_recurrent_layer`:

```python
model = OLMoTiledRTForCausalLM(
    config,
    ordinary_activation_checkpointing=True,
    cast_weights_once=True,
    tile_backend="triton",
    backward_tile_backend="triton",
    backward_memory="recompute",
)
```

`backward_memory="materialized"` remains the default and the F3c reference.
The option is execution metadata, not a new checkpoint tensor. Record it in
run/resume configuration; changing it invalidates prepared static plans and
existing caches. No parameter count, forward recurrence, RoPE, normalization,
loss or gradient ownership changes. GPU flags require CUDA explicitly.

The new backward stores FP32 row maximum/denominator and temporary-self
probability. Reconstruction and final dQ use at most32 query rows against all
keys, preserving each full key reduction. Historical dK/dV are recomputed in
fixed16-key/32-query Triton tiles with FP32 accumulation over all query chunks,
then one BF16 result rounding. The eager fallback chunks output key columns,
retaining the complete query reduction. Prefix and temporary-self gradients
remain separate. This bounds **RT backward attention scratch**, not every model
allocation or ordinary attention mask. Recomputation adds work.

Mixed CUDA Q/K/V in BF16 with head dimensions16/32/64/128 and historical
rectangle sides1..2048 use the recompute kernel. Other supported precisions or
shapes use bounded eager reconstruction. Forward retains its existing F3b
fusion limits; a long-context forward rectangle may fall back to eager even
while its recomputed backward is fused. This is Triton, not FA4. Ordinary
deterministic Flash uses PyTorch SDPA in the validated static layouts.

Run only inside the project container after verifying GPU availability:

```bash
bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_f3d_probe.py --output-dir .runtime/olmo1b-step60000/f3d-probe-NEW'
bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_f3d_validate.py --case combined --variant recompute --stage correctness --batch-size 8 --length 512 --output-dir .runtime/olmo1b-step60000/f3d-combined-NEW'
```

Use fresh output directories. Each run logs W&B, snapshots sources/protocol and
rechecks hashes. Keep the [protocol](reports/olmo1b-f3d/protocol.md) and eventual
results/assessment alongside the raw JSON. Native diagnostics select RT layer0;
do not infer all-layer, padded-graph or multi-GPU readiness. A capacity result
is a finite complete-update/timing check, not all-gradient parity at that batch.
