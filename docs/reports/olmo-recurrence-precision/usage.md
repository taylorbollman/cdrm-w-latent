# Running the bounded recurrence/precision matrix

Use the pinned source snapshot and the project container. The matrix has four
fixed arms and two precision paths; its CLI deliberately has no training-step,
length or arbitrary-arm sweep option. Use a fresh output directory for each
attempt, and retain unsuccessful attempts.

From `/home/taylorbollman/cdrm-w-latent`:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
  set -euo pipefail
  test -f /.dockerenv
  test "$PWD" = /workspace/cdrm-w-latent
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  CUDA_VISIBLE_DEVICES=0 timeout 900s python scripts/olmo_campaign_recurrence_precision.py \
    --reference-report .runtime/olmo-precision-localization/bridge-01/report.json \
    --reference-sha256 39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b \
    --output-dir .runtime/olmo-recurrence-precision/matrix-NEW
'
```

The existing native checkpoint defaults to
`.runtime/olmo1b-step60000/artifacts`. Determinism is configured before CUDA.
W&B records go to `taylorbollman/pretrained-fbt-rt-nextlat`, group
`olmo-recurrence-precision`; credentials stay in the existing environment.

The reference report and all shared source bytes must match their pins. Changes
to software, source, inputs or precision policy are a new experiment and need
a newly documented reference; do not bypass the exact NFR reproduction guard.
The recipe's T1024 field is campaign metadata, while this diagnostic uses T16.
Two virtual input records run on one GPU and do not imply distributed testing.

Run focused tests in the explicitly CPU-only container:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
    python -m pytest -q tests/test_campaign_recurrence_precision.py \
      tests/test_campaign_precision_bridge.py tests/test_campaign_precision_components.py
  '
```

After a GPU writer exits, copy its launcher log into the stage and retain that
explicit directory using `scripts/olmo_two_gpu_retain.py` in a CPU-only
container. Use a fresh cloud suffix and receipt file; the historical helper
name/namespace does not imply DDP. No trained checkpoint is produced. Check
the final receipt and preserve all declared source snapshots, case reports and
local evidence. Cross-precision error measurements remain descriptive until a
specific resolution is documented.
