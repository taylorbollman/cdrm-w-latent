# Reproducing the bounded numerical probes

Use the project container and the retained source snapshots. These probes are
single-process diagnostics, not campaign training or distributed qualification.
They perform no optimizer updates and need only GPU 0. Each output directory
must be new; do not overwrite an old result, including a failed attempt.

From `/home/taylorbollman/cdrm-w-latent`, the GPU launcher pattern is:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
  set -euo pipefail
  test -f /.dockerenv
  test "$PWD" = /workspace/cdrm-w-latent
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  CUDA_VISIBLE_DEVICES=0 timeout 900s python scripts/olmo_campaign_precision_bridge.py \
    --output-dir .runtime/olmo-precision-localization/bridge-NEW
'
```

Helpers configure determinism before CUDA initialization, record runtime and
source pins, and log to W&B project `pretrained-fbt-rt-nextlat`. Credentials
remain in the existing environment configuration. The native source checkpoint
defaults to `.runtime/olmo1b-step60000/artifacts`.

Use these arguments inside the same launcher for the subsequent probes:

```bash
# Exact fixed-hidden anchor from completed bridge-01.
CUDA_VISIBLE_DEVICES=0 timeout 900s python scripts/olmo_campaign_aux_cotangents.py \
  --fixture .runtime/olmo-precision-localization/bridge-01/auxiliary-fixture.json \
  --fixture-sha256 aeab58a88c7eba15448a1b7630c9af747e492b3760b2364da5cd53380e063b27 \
  --output-dir .runtime/olmo-precision-localization/auxiliary-NEW

# One crossed condition, plus recomputed exact-reference checks.
CUDA_VISIBLE_DEVICES=0 timeout 900s python scripts/olmo_campaign_backend_cross.py \
  --reference-report .runtime/olmo-precision-localization/bridge-01/report.json \
  --reference-sha256 39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b \
  --output-dir .runtime/olmo-precision-localization/backend-cross-NEW

# Actual ordinary-attention operands and common cotangent at eight sites.
CUDA_VISIBLE_DEVICES=0 timeout 900s python scripts/olmo_campaign_attention_local.py \
  --reference-report .runtime/olmo-precision-localization/backend-cross-01/report.json \
  --reference-sha256 97ced83fd037c907bd6a8ad34c377b96a0c1bc04dc424c21950cbf2194da9a65 \
  --output-dir .runtime/olmo-precision-localization/attention-local-NEW
```

References include source and runtime contracts. A changed source file, GPU
software version or fixture is a new experiment, not an interchangeable replay;
the pinned helpers intentionally reject mismatches. Updating a comparison
requires an explicit new reference and protocol rather than bypassing checks.

Run the focused tests in the explicitly CPU-only container:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
    python -m pytest -q tests/test_campaign_precision_bridge.py \
      tests/test_campaign_aux_cotangents.py tests/test_campaign_backend_cross.py \
      tests/test_campaign_attention_local.py
  '
```

After the writer has stopped, copy its launcher log into the stage, then use
`scripts/olmo_two_gpu_retain.py` in a CPU-only container to retain the explicit
stage. The helper's historical name and namespace do not imply that these
probes ran DDP. Use a fresh cloud stage suffix and receipt path, retain the
report and all declared source snapshots, and verify the returned receipt.
The [storage receipt](storage-receipt.md) lists completed evidence. No new
trained checkpoint is produced by this milestone.
