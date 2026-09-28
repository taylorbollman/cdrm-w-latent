# Reproducing the ordinary T2048 measurements

Use the required project container, installed Flash-Attention namespace, and
the retained original OLMo step60000 artifacts. Do not run CUDA from the host
or fall back to CPU. The benchmark requires the frozen protocol in this folder.
Use a new output directory for every attempt; existing evidence is not overwritten.

From the project checkout on the host:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent &&
nvidia-smi &&
timeout 1800s python scripts/olmo_two_gpu_single_reference.py \
  --case ordinary --ordinary-rope-backend dao \
  --ordinary-attention-backend sdpa --length 2048 --batch-size 32 \
  --output-dir .runtime/olmo-ordinary-long-context/sdpa-b32-new
'
```

For FA4, change the attention selector to `fa4` and use a fresh directory.
The comparison keeps Dao native-FP32 RoPE, compiled rounded SwiGLU, fused AdamW,
BF16 mixed execution, all-layer checkpointing and full CE position chunks2048.
FA4 is restricted to non-tiny ordinary T2048 with Dao RoPE. Neither RT nor
combined-model T2048 is enabled by this benchmark extension. Defaults stay SDPA.

The small integration diagnostic additionally uses `--batch-size 2`,
`--check-attention-parity` and, for preserving a finite numerical-only miss while
finishing operational checks, `--continue-after-compatibility-miss`. This option
does not turn a failed numerical comparison into a pass; the process exits
unsuccessfully after the operational checks finish. It cannot bypass structural,
nonfinite, dispatch or own eager/graph failures. Diagnostic timing is excluded
from performance plots.

Each capacity process performs three preparation updates, eleven backward
warmups, and five timed complete updates with changed token values. The one-GPU
fixture preserves the prior paired-rank construction solely for comparability;
no distributed process group or communication is used. Physical batch must be
even. There is one microbatch per optimizer update, with no accumulation.

The timing excludes setup, compilation, CPU reference copies, fixture generation,
logging and checkpoint I/O. It includes batch validation/copy, graph replay,
health checks, clipping, Adam and scheduler. Input throughput counts all B×T
tokens; CE supervises B×(T−1). The full-valid, separate-document fixture does not
benchmark a real data loader or packed/padded-document execution.

The actual attention dispatch, sources, dependencies, checkpoint hash, W&B URL,
parameter/FLOP inventories, memory phases and checks are in each `report.json`.
Eager references are held on CPU before capture; the final eager check occurs
after releasing the graph. This avoids counting the artificial overlap of live
graph storage and a separate eager backward as the training memory requirement.

Local evidence: `.runtime/olmo-ordinary-long-context/`. See the storage receipt
for the verified GCS prefix. Audit scripts and regeneration instructions are
retained in the closeout bundle under `audit/`. Historical T512 is explicitly
selected from the previous milestone; no new T512 run is required or queued.
