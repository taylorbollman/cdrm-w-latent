# NFR KL continuation storage receipt

The original NFR32-to64 KL1/KL0.1 pair was activated after the F128 and
post-F diagnostic review. This receipt preserves **initial static authority**,
not a completed training run. The active training directories, live queue
state/events and training logs are deliberately excluded.

Evidence prefix:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/`

| Stage | Members | Archive generation / SHA256 | Receipt generation / SHA256 |
|---|---:|---|---|
| `nfr-kl-initial-static-authority-01` | 226 | 1790764674883619 / `cb6fc6f9698582304d27148450774f3e4dccbba1bef82ca2c2eff89b8626babd` | 1790764675387523 / `6ec028728e6ebbdefe08c4fbe485cab21ac2085016760623e9a7369b14e5e861` |
| `nfr-kl-status-v2-validation-01` | 2 | 1790764740859220 / `fb61d1aceeef8a09235497c605b885f4a5be346a24a9d9770b7bd600972f3ba0` | 1790764741382537 / `969b4aa2f9afd0eb9cb1d2eeacc38ee7b6471a2653fc238ab191a4302b473773` |

The 226-member static authority archive comes from
`.runtime/olmo-nfr-kl-continuation/initial-static-authority-01`. It includes
`activation-01.json` (SHA256
`c9b209280ac92c7b1a86655e90a52bfffe6a47f155c3e9cfb86586a7eb75d2e3`), scope,
dry resolution, preparation records, pinned host launcher, sequential queue,
original status helper and the complete 215-file runtime source snapshot.
Every activation authority and runtime source pin was verified before copying.
The preparation records retain their historical `prepared_not_launched` labels;
activation is separately and explicitly recorded afterward.

The optional read-only status helper is now:

```bash
python3 .runtime/olmo-nfr-kl-continuation/live_status_v2.py \
  --queue-name queue-after-f128-01
```

Version 2 skips the evaluator's temporary `panels: {}` placeholder, retains the
last completed development evaluation, and reports `dev_pending_at_update`
while a newer one is running. Four bounded metadata fixtures and the actual
live update32 placeholder passed. The original prepared utility remains
unchanged at SHA256
`12fa8792b56606330f848af079f9e500bceb70ce8d5d9572483b781bac30557c`.
Version 2 SHA256 is
`1de47a94a253e0f19c715f2e7246a963e3e90bc4ae594773dc7dbdfb83b32b18`.
The separate two-member archive preserves that helper and its validation
receipt. No training, queue or optimizer code was changed.

Both stages used the existing `scripts.olmo_two_gpu_retain` in a CPU-only
container with `env -u GOOGLE_APPLICATION_CREDENTIALS`, without
`--checkpoint-dir`. Archive/manifest/receipt size, server MD5, SHA metadata and
downloaded SHA256 checks passed. Local receipt/result files are under
`.runtime/olmo-fbt-stability-retention/` using the stage names above.
The static-input file inventory records exact source/destination paths and
hashes. No local files were deleted, no model state was loaded or rehashed,
and no GPU work was launched by retention.
