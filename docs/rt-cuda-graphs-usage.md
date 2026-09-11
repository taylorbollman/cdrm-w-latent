# Standard RT: opt-in CUDA graph profiling

Use `scripts/rt_batch_profile.py --cuda-graph` to capture the complete native model forward, head-chunked cross entropy and backward. Clipping and Adam remain outside the graph. This is a fixed-shape, single-GPU training/profiling path; it does not enable the released trainer's old `cuda_capture_model` wrapper.

The model has 12 tiled recurrent layers, width 1024, FFN 4096 and 16 heads. It has 151,045,120 backbone parameters and 216,843,264 including its untied 32128-row input and output tables. Every recurrent layer receives the entire physical batch. Only the output head is microbatched in groups of 2 sequences. The shifted loss uses 511 targets per sequence and the released trainer's B×512 denominator.

The precision policy remains `bf16_fp32_state`: BF16 projection autocast with FP32 parameters, residual/recurrent attention state, gradients, loss and Adam state. This is our previously qualified local policy. CUDA graph validation compares captured and uncaptured execution within the same precision; it does not provide new clearance for differences between BF16 and FP32.

From `/home/taylorbollman/cdrm-w-latent`, use a fresh output directory:

```bash
bash scripts/docker_shell.sh bash -lc '
  test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L &&
  python scripts/rt_batch_profile.py \
    --batch 512 --updates 10 --cuda-graph \
    --output-dir .runtime/rt-cuda-graphs/NEW_RUN/profiles/b512-graph \
    --wandb-group NEW_RUN --wandb-run-name b512-graph
'
```

The profiler checks an uncaptured reference loss and every raw parameter gradient at the requested batch size before measuring captured Adam updates. It reports capture setup separately from the two warmup updates and measured updates. It requires online W&B in the `taylorbollman/recurrent-transformer-capacity` project.

For focused changing-input and optimizer-equivalence validation:

```bash
bash scripts/docker_shell.sh bash -lc '
  test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L &&
  python scripts/rt_cuda_graph_validate.py \
    --tier full --precision bf16 \
    --output-dir .runtime/rt-cuda-graphs/NEW_RUN/validation/full-bf16 \
    --wandb-project recurrent-transformer-capacity \
    --wandb-group NEW_RUN --wandb-run-name full-bf16-validation
'
```

`--tier tiny` uses 2 layers, width 64, length 16, batch 3 and includes an output-head chunk tail. `--precision fp32` is available for same-precision graph/eager diagnostics. Validation includes two repeated inputs without updates, three changing-input Adam updates, all raw gradients, model parameters and optimizer states, model-state preservation and misuse guards. Prefix causality is observed on native forward; capture calls that same forward path, but the validator does not separately expose captured hidden prefixes.

`CapturedRTBackward` preserves the model's module tree and parameter names. It owns persistent FP32 gradient buffers and zeroes them inside each replay. After constructing it, call `replay(ids)`, then clipping and `optimizer.step()`. Do not set gradients to `None`, replace parameter storage, change model/layer configurations, attach Python hooks or precision observers, or replay under evaluation/no-grad/enclosing-autocast modes. Inputs must retain the captured shape, dtype and device. Returned loss storage is reused on the next replay.

The graph captures weight-casting operations inside the normal autocast scopes, so in-place Adam updates are reflected in subsequent replays. Capture warmup makes no optimizer updates. The older `make_graphed_callables` wrappers remain disabled: the tiled custom backward's internal parameter-gradient accumulation needs the explicit complete-backward capture used here.

Activation checkpointing is already implemented inside the tiled custom backward: it saves layer inputs/outputs and reconstructs attention without replaying the original sequential forward. `--bwd-mlp-chunks` defaults to 4. Additional outer whole-layer checkpointing remains off. See [checkpointing analysis](reports/rt-cuda-graphs/checkpointing.md).

For memory comparisons, use graph-pool reservation and available device memory as well as setup peaks. PyTorch's replay-time `max_memory_allocated` does not trace each intermediate allocation again; interpreting that number alone as a reduction in model working memory would be misleading. Sampled post-update free memory is not a continuous minimum.

Random benchmark data and weights are discarded. Retain the resolved configurations, source snapshots, metrics and logs; current milestone results are in [the report](reports/rt-cuda-graphs/results.md).
