# Running the fixed-boundary diagnostic

Use the pinned source and original NF reference. The helper's fixed scope is
two full-model anchors and twelve local VJPs; arbitrary model/length/precision
sweeps and training are deliberately absent. Use a fresh output directory.

From `/home/taylorbollman/cdrm-w-latent`:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
  set -euo pipefail
  test -f /.dockerenv
  test "$PWD" = /workspace/cdrm-w-latent
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  CUDA_VISIBLE_DEVICES=0 timeout 900s python scripts/olmo_campaign_boundary_precision.py \
    --reference-report .runtime/olmo-recurrence-precision/matrix-01/report.json \
    --reference-sha256 bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412 \
    --output-dir .runtime/olmo-boundary-precision/boundary-NEW
'
```

The source checkpoint defaults to `.runtime/olmo1b-step60000/artifacts`.
Deterministic controls are applied before CUDA. Graphable metrics go online to
`taylorbollman/pretrained-fbt-rt-nextlat`; credentials stay in the environment.
This is one process on one GPU, not distributed training.

Run CPU tests with GPU passthrough explicitly disabled:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
    python -m pytest -q tests/test_campaign_boundary_precision.py
  '
```

The boundary export retains small tensor values and exact call/layout metadata
in JSON/base64. It is evidence, not a trained checkpoint. Use the captured full
cotangents for replay; valid-token selection is only for reporting. A successful
operational status requires controls and captured-output equality, not a new
cross-precision error threshold.

After the writer exits, copy its launcher log into the stage and retain that
directory with `scripts/olmo_two_gpu_retain.py` in a CPU-only container. Use
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T061000Z/`
with a fresh `boundary-STAGE` suffix and local receipt. The legacy namespace
does not imply DDP. Verify the retained tensor export, report and all declared
source snapshots. Keep unsuccessful attempts and completed sources immutable.
