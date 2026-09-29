# Resolve a campaign draft without launching training

Run from `/home/taylorbollman/cdrm-w-latent` through the project container with
GPU passthrough disabled. The example below re-resolves the completed readiness
manifest into a new directory; it performs local byte hashing and CPU metadata
planning only. The output directory must not already exist.

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
    python scripts/olmo_campaign_manifest.py \
      --manifest .runtime/olmo-campaign-manifest/actual-01/manifest.json \
      --manifest-sha256 458d449eb72e2daea3682999b7e094b3d79bb0564cd9019249e911cc05069ca1 \
      --output-dir .runtime/olmo-campaign-manifest/review-copy-01
  '
```

The manifest uses container paths for the retained original OLMo-1B artifact
directory, readiness corpus, packed train index and ownership ledger. Those
local authorities must exist and match their pinned bytes. This command does
not download missing artifacts. The corpus resides on local SSD; restore its
verified retained bytes before resolving on a replacement VM. A missing or
different corpus, index, tokenizer, model manifest, checkpoint or source pin
fails closed. Source changes require an explicit review and newly pinned draft.

Inspect `resolved.json`, `plan-card.md` and `source-snapshot/`. The JSON contains
the declared original-weight startup, all eight resolved arm recipes, native
and NextLat configurations, physical partition, logical chunk membership/cursors,
objective counts, schedule prefix, parameter ownership, analytic matrix work,
retention/evaluation declarations and unresolved review items. The exact input
file hash and canonical JSON semantic hash are recorded separately.

A successful status is **CPU structural validation only**. Both launch
authorization and numerical clearance remain false. In particular, a BF16 NFR
declaration can be internally coherent while its numerical qualifications remain
open. The example's three updates, B8 per rank and two-rank partition are not
production or capacity recommendations. No model is constructed, no W&B SDK is
called, and retention/evaluation declarations cause no external writes or
evaluation. There is no `--train`, automatic backend fallback or implicit cycling.

Use a distinct reviewed JSON draft to change supported declarations. All recipe
fields are explicit. The current resolver supports only native T1024,
continuous-stream packing, RT layers 0/15, original pretrained weights with
fresh Adam, and the documented FP32 math/eager or BF16 Flash/Triton paths.
Adapted-startup checkpoints, inherited optimizers, hidden startup training,
cooldown, SFT and generation are rejected rather than silently substituted.

The focused CPU tests can be run independently:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
    python -m pytest -q tests/test_campaign_manifest.py
  '
```

Frozen implementation pins for the completed example:

| File | SHA256 |
| --- | --- |
| `scripts/olmo_campaign_manifest.py` | `12db2fe64daccee49acca5255c2d97cca2b1fcc4330d2287e974562a8798f0bd` |
| `tests/test_campaign_manifest.py` | `28af7751f35946f0601d863bd2f7068fcf2881f32e076dcae0982b663462baf5` |
| `docs/reports/olmo-campaign-manifest/protocol.md` | `86828d2f156b69a8f181481dddb308cdfcc326857a4c8f123399121803b38516` |

The resolver also pins the 49 inherited ownership-ledger/model sources. See
[results](results.md) for the completed example's counts, evidence and limits.
