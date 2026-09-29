# Crossed-state diagnostic usage

Read [protocol](protocol.md) before rerunning. This runner uses the original
prepared checkpoint and saved adapted checkpoint locally; path overrides do not
change the expected identities. Use a fresh stage directory and retain failures.
Do not modify completed helpers/tests/protocols to accommodate a later experiment.

Focused CPU checks in the GPU-disabled project container:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
    set -euo pipefail
    test -f /.dockerenv
    test "$PWD" = /workspace/cdrm-w-latent
    python -m pytest -q tests/test_campaign_crossed_precision.py \
      tests/test_campaign_position_geometry.py \
      tests/test_campaign_adapted_precision.py \
      tests/test_campaign_adapted_import.py
  '
```

After review and source freeze, root launches GPU0 inside the project container:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
  set -euo pipefail
  test -f /.dockerenv
  test "$PWD" = /workspace/cdrm-w-latent
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  CUDA_VISIBLE_DEVICES=0 timeout 900s \
    python scripts/olmo_campaign_crossed_precision.py \
      --output-dir .runtime/olmo-crossed-precision/crossed-01
' > .runtime/olmo-crossed-precision/crossed-01-launcher.log 2>&1
```

The runner stores independent CPU component copies and verifies all hashes before
strict in-place assembly. It performs four aggregate/eight physical backwards,
with no training updates or diagonal reruns. `adapted_import` identifies adapted
weight authority; assembly records identify the exact native and complete fusion
origin for each hybrid. Original construction metadata is not hybrid provenance.

Per-case reports are atomic. W&B logs scalar gradient metrics and compact
per-pass position summaries; detailed per-position observations stay in the
retained JSON report. Incoming-cotangent support is descriptive and never changes
the actual objective/backward. Stop after the four cases and assess.

Once the process and tracking writer have exited, copy the closed launcher log
into the stage and use the existing GPU-disabled `olmo_two_gpu_retain.py` workflow
for GCS evidence. No new model weights or full parameter-gradient vectors are
exported. A900s interrupted stage loses no training progress; preserve its partial
report/log and rerun only under a new stage directory after assessing the failure.
