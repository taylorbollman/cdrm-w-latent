# Reproducing combined T2048 measurements

Use the project Docker container and retained OLMo step60000 artifacts. Never
run CUDA directly on the host or silently fall back to CPU. From this checkout:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent &&
nvidia-smi &&
timeout 3600s python scripts/olmo_two_gpu_single_reference.py \
  --case combined --ordinary-rope-backend native \
  --ordinary-attention-backend sdpa --length 2048 --batch-size 32 \
  --output-dir .runtime/olmo-combined-long-context/sdpa-b32-new
'
```

Use a fresh directory for each attempt; physical batch must be even. Use B2
for the initial operational check. K2 performs an ordinary bootstrap followed
by an attached-feedback pass with native RT at indices0/15. Each pass has CE,
NextLat SmoothL1 and KL. Ordinary attention is forced Flash SDPA; native RT uses
its own Triton/eager historical tiling and recompute backward. Combined FA4 and
Dao RoPE remain rejected by this harness. No attention or precision default
changes are implied by the benchmark.

The fixture has separate full-valid row documents, full CE/latent supervision
and response-half KL. It concatenates two deterministic rank-style shards for
compatibility with the saved one-GPU reference; no distributed process group
runs. There is one physical batch per update and no gradient accumulation.
Input throughput counts B×T once, although K2 executes two passes. Do not compare
it to a rate counting pass-tokens twice.

Each process loads the same pretrained checkpoint, performs three preparation
updates, eleven backward warmups and five timed complete updates. Forward,
loss and backward are graphed; validation/copy, finite checks, clipping, fused
Adam and scheduler are included in timing. Compilation, fixture generation,
logging, reference copies and evidence retention are excluded. Setup/capture
and steady memory are reported separately; sampled free memory is not a
continuous minimum.

Each row checks actual ordinary and RT dispatch, exact initial eager/graph raw
gradients and losses, changed-weight terminal parity, and source/dependency
integrity. This does not constitute a new independent full-Adam trajectory or
quality-training comparison. Prior precision qualifications remain recorded.

W&B group`olmo-combined-long-context` in`taylorbollman/pretrained-fbt-rt-nextlat`.
Evidence is under`.runtime/olmo-combined-long-context/`. Preserve every attempt
and retain completed stage evidence before moving to the next batch. Sources
and the prospective protocol must stay frozen while a stage runs. Source logs
belong in the stage before retention; receipts belong outside the retained tree.

The user requests Flash SDPA only and review afterward. Do not automatically
add FA4, optimize large RT tiles, start a quality run or rerun T512.
