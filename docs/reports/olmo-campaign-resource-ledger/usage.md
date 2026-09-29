# CPU resource ledger usage

The tool writes an analytic all-eight-arm ledger. It never loads a checkpoint,
runs a model forward/backward, updates an optimizer or requests CUDA. It builds
one randomly initialized native CPU backbone and reuses it while constructing
each arm's actual wrapper and fresh optimizer ownership. Allow roughly5GiB for
native parameter storage plus temporary initialization/auxiliary storage; this
is unrelated to the reported GPU operating points.

From the project root:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'python -m pytest -q tests/test_campaign_resource_ledger.py'
```

Use a new persistent output directory for each generation:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_campaign_resource_ledger.py --output-dir .runtime/olmo-campaign-resource-ledger/ledger-01'
```

The default retained packed report is byte-pinned. `--reference-report` permits
relocating identical bytes, not selecting another workload without a reviewed
pin change. No token shards or checkpoints are read. The report retains source
snapshots and exact input provenance. Per-arm JSON distinguishes:

- `parameters`: actual unique CPU module/optimizer ownership and active/deployed
  architecture counts; inactive fusion remains resident in non-F wrappers.
- `ranks`: allocated rows/slots, useful valid tokens, padded/dummy rows and
  useful CE/latent/KL/predictor-union selections.
- `sparse` / `dynamic_dense`: alternative selected-loss versus fixed-capacity
  loss matrix work at the same padded backbone footprint. This is not a timing
  or numerical-equivalence comparison between trainers.
- `totals`: sums across ranks, input/pass-token work, executed loss positions
  and estimated matrix FLOPs. World size and accumulation are applied once.

The ordinary attention/checkpoint range is an accounting convention. Neither
endpoint is a hardware-counter measurement or a rigorous issued-instruction
bound. No FLOPs-per-second, MFU or throughput claim is made. These fully trainable
cards must not be used for the separate frozen-backbone fusion-only warmup.
