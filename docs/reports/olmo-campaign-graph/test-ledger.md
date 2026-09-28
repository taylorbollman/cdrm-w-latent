# Validation ledger

CPU tests run intentionally inside the CPU container (`CDRM_DOCKER_GPUS=none`),
with installed Flash package selection to avoid vendor import shadowing.

Final runtime preparation `0a074ea`: **867 passed in51.58s**. Command:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'python -m pytest -q --tb=short tests/test_campaign_*.py tests/test_olmo_fbt*.py tests/test_olmo_static.py tests/test_olmo_tiled.py tests/test_static*.py tests/test_resource_estimates.py tests/test_distributed_training_preparation.py tests/test_ddp*.py tests/test_olmo_f2_health_capacity.py tests/test_olmo_lm_training.py tests/test_olmo_lm_data.py tests/test_olmo_lm_schedule.py tests/test_olmo_f4_resources.py'
```

Coverage includes:

- Existing FBT/static/DDP/tiled/optimizer/data behavior and the PR35 campaign
  contracts.
- Explicit-mask versus valid-prefix fast path, all model/input gradients,
  K4 jitter, empty rows, checkpointing and forward storage ownership.
- Dense dynamic versus selected canonical CE/latent/KL values and raw gradients,
  stop-gradients, changing masks, constant tensor operation shapes, and no
  tensor-to-host scalar/nonzero operations in the loss execution body.
- All eight arms under unequal accumulation, changes to masks/tokens/noise and
  denominators, preserved buffers, rejection before mutation, token counters,
  and persistent-gradient checkpoint publication/restoration on save failure.
- Two CPU/Gloo ranks, rank-local jitter, `no_sync`, unequal and entirely empty
  local slots, full Adam reference, coordinated malformed input and proof that
  tensor payloads do not enter object collectives.
- Bounded GPU probe fixtures, snapshot restoration and comparison gates.

Ten warnings were expected: nine from the independent native oracle's tiny
unaligned vocabulary and one Google/grpc future dependency minimum. Earlier
development runs exposed test-fixture issues (invalid tiny vocabulary override,
missing fixture checkpoint hash, a misplaced fixture assertion) and a new
adapter helper name colliding with `nn.Module._buffers`; all were fixed before
the final pass. Raw development logs are retained alongside final evidence.
They are not failed GPU numerical qualifications.

The GPU protocol is in [protocol.md](protocol.md); numerical results and retained
attempts belong in [results.md](results.md) and [storage-receipt.md](storage-receipt.md).
CPU/Gloo tests do not certify NCCL, distributed CUDA graphs or distributed restart.
