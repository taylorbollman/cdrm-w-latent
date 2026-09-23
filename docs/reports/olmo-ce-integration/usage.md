# CE position chunking

The shared language-model loss supports an independent CE chunk override:

```python
config = NextLatConfig(model_dim=2048, vocab_chunk_size=128, ce_chunk_size=2048)
```

Each projection still uses the complete vocabulary. The numbers count selected
positions, not vocabulary rows or sequence length. Larger chunks reduce repeated
weight conversion/gradient accumulation and launch work, but can increase temporary
logit memory. CE masks, denominators, parameter tying and loss weights are unchanged.
KL continues to use128 here. This applies to dynamic and prepared/static losses.

Leaving `ce_chunk_size=None` uses `vocab_chunk_size` for CE, preserving all old
defaults. `to_dict()` omits the optional field when unset, so legacy checkpoint
configuration comparisons remain exact; explicit overrides are serialized.
Change the configuration before constructing `StaticFBTTraining`; an existing
layout or captured graph rejects a changed configuration and must be rebuilt.
Checkpoint resume should use the saved configuration. A deliberate chunk change
is a new recorded execution configuration, not an exact old-run replay claim.
Use `config.to_dict()` for checkpoint identity rather than generic `asdict`,
which includes the unset optional field. Frozen historical factories without
this option will reject an explicit mismatched configuration on resume.

The option applies to shared training CE. Evaluation retains its separate
explicit `chunk_size` argument. Dense-matrix FLOP estimates are unchanged:
chunk regrouping does not change their arithmetic count, and the estimator
already excludes conversion, fill, accumulation and launch costs.

## Bounded validation commands

From the host project directory, after the container/GPU identity check:

```bash
bash scripts/docker_shell.sh python scripts/olmo_ce_integration.py \
  --stage correctness --case ordinary --batch-size 8 \
  --output-dir .runtime/olmo-ce-integration/ordinary-b8-correctness

bash scripts/docker_shell.sh python scripts/olmo_ce_integration.py \
  --stage capacity --case ordinary --batch-size 64 --ce-chunk 2048 \
  --output-dir .runtime/olmo-ce-integration/ordinary-b64-half-c2048
```

Use new output directories for reruns; existing evidence must not be overwritten.
The correctness case also accepts `nextlat` and `combined`, with two selected RT
layers in the latter. Capacity permits ordinary B64 only; `--ce-chunk 128` is the
control and `--supervision full` selects all511 targets
per row. Historical F1–F4/ordinary-throughput harnesses are left frozen.

The harness forces deterministic PyTorch Flash SDPA, ordinary checkpointing,
BF16 mixed with FP32 weights/Adam, no TF32 and no autocast weight cache. It logs
online under the `olmo-ce-integration` W&B group. It does not train a useful model
or save its few-step weights; the original pinned checkpoint is the source.

Dao CE, fused RoPE/SwiGLU, Q/K normalization, RT algorithm changes and long quality
runs are outside this integration. See the audit and results for next-step scope.
