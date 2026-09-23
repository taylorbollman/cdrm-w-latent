# Experimental author RT in native OLMo

This is an opt-in bounded backend. Read the protocol and results before use.
`OLMoTiledRTForCausalLM` keeps `rt_implementation="native"` by default. To select
the author-derived path, construct it with:

```python
rt_implementation="author",
reuse_rope=True,
author_precision="author_legacy",
author_compiled_helpers=True,
author_bwd_mlp_chunks=4,
author_autocast_cache=True,
```

Use the normal RT mode to choose layers, with alpha=1. The implementation
preserves packed parameters, tied embeddings, outer-autograd ownership and
checkpoint state keys. Ordinary layers and the FBT bootstrap continue through
the existing attention/checkpointing path. Author RT itself uses compiled
matrix operations, not Flash Attention. Its backward materializes attention;
native probability-recomputation flags do not change author arithmetic.

The supported author scope is first-order gradients, FP32 parameters/residuals,
FP32 execution or BF16 autocast, all-valid unpadded sequences, no imported or
exported prefix cache, and full recurrence on selected layers. It requires
prepared immutable native split-half RoPE tables. Unsupported padding, caches,
fractional alpha and precision are rejected. Static layouts/graphs guard all
backend options; rebuild after changing them, shapes, mode or parameter storage.
The checkpoint does not silently encode a deployment backend choice: record
constructor/execution options with any future run configuration.

Use the project Docker launcher for every GPU command. Example actual-checkpoint
verification (fresh output directory required):

```bash
bash scripts/docker_shell.sh python scripts/olmo_rt_author_integration.py \
  --stage verify --case rt --backend author --batch-size 8 \
  --output-dir .runtime/olmo-rt-author-integration/verify-rt-author-b8
```

Repeat with `--case combined` for K2 FBT+RT+NextLat. Verification compares raw
native/author gradients and losses, observes real ordinary Flash dispatch, and
checks exact author eager/graph gradients, overwrite, three-update Adam parity
and changed weights. Compatibility misses remain failed even when operational
checks pass. No budgets may be widened after observing results.

Only after operational checks, use `--stage capacity --case rt|combined
--backend native|author --batch-size 64` for matched complete-update timing.
It includes copies, graph forward/loss/backward, clipping, Adam and scheduling.
Compilation/capture and validation are excluded from timing, with setup and
steady memory reported separately. These are short synthetic token fixtures;
no quality or sustained-training inference follows. W&B group:
`olmo-rt-author-integration`, project `taylorbollman/pretrained-fbt-rt-nextlat`.

The prospective protocol does not cover save/resume, distributed execution,
padding/cache support, long-run stability or production-default selection.
Prior F4 RT+FBT and BF16/full-FP32 qualifications remain in force.
