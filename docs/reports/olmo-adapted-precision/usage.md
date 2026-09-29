# Adapted-state precision diagnostic usage

This is a bounded model-only numerical check, not a training launcher. Read
[protocol](protocol.md) and [baseline controls](baseline-and-controls.md).
The original O5c/O5d source guards remain unchanged. Do not rerun a completed
stage in its existing directory or alter frozen helpers after a result.

Run tests in the GPU-disabled project container:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
    test -f /.dockerenv
    test "$PWD" = /workspace/cdrm-w-latent
    python -m pytest -q tests/test_campaign_adapted_import.py \
      tests/test_campaign_adapted_precision.py \
      tests/test_campaign_recurrence_precision.py
  '
```

After reviewed source/test/protocol freeze, root launches inside the GPU container:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
  set -euo pipefail
  test -f /.dockerenv
  test "$PWD" = /workspace/cdrm-w-latent
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  CUDA_VISIBLE_DEVICES=0 timeout 900s \
    python scripts/olmo_campaign_adapted_precision.py \
      --output-dir .runtime/olmo-adapted-precision/adapted-01
' > .runtime/olmo-adapted-precision/adapted-01-launcher.log 2>&1
```

Defaults require the pinned cold reference, original prepared base artifacts and
saved O5c mixed update512 checkpoint at their recorded project paths. A file
override is a location override, not permission to change expected identity.
The runner verifies the cold construction without executing another backward,
then imports only adapted backbone/fusion tensors. `adapted_import` is the
authority for actual weights; `source_checkpoint` inside construction metadata
still identifies the original base artifacts. No optimizer/RNG history is loaded.

The report is atomically saved at setup/case boundaries. W&B uses
`taylorbollman/pretrained-fbt-rt-nextlat`. A 900-second termination may interrupt
a case, so preserve its partial report/log/source snapshot as a failed attempt.
No training progress is at risk. On completion, copy the closed launcher log
into the stage and retain it through `olmo_two_gpu_retain.py` in a GPU-disabled
container. Exact retention objects and result qualification belong in the
completed report's storage receipt, not in a mutable runtime assumption here.
