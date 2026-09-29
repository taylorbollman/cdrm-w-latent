# Resource-ledger checks

2026-09-29. Focused command, explicitly inside the CPU container with GPU
passthrough disabled:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'python -m pytest -q tests/test_campaign_resource_ledger.py'
```

Result: **33 passed in2.76seconds**, no warnings. This is a new focused scope;
prior broad passing suites were not rerun. Checks cover actual tiny-module and
optimizer ownership for all eight arms, tied embedding uniqueness, dormant
fusion residency, actual stack invocations under all-pass RT, independently
intercepted sparse/dense CE/KL/predictor forward shapes, exact unchanged full-row
matrix estimates, mask-source unions, heterogeneous slot/rank sums, fully dummy
loss selections, rejected invalid counts and pinned input reconciliation.

Native-size CPU generation completed once at08:12:07–08:12:45UTC. All eight cards
and four final integrity checks pass. The original observed packed first update
was read from its immutable report; no data/model checkpoint was reread. All49
declared source files and49 retained source snapshots independently match their
recorded hashes. This is model ownership/arithmetic evidence; no actual GPU
operator trace, frozen-training estimate or optimizer update is claimed.
