# Retention and recovery authorities

The tiny control, reduced-KL and reduced-KL child-resume stages have completed
and published their terminal update 3 checkpoints. Their small evidence is
retained separately. Native NF control/reduced continuations are not included
in the completed-tiny inventory below; append their final inventory after both
stages, asynchronous retention and evidence writers close.

Checkpoint root:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/kl-continuation/20260930-pair01/`

| Completed stage | Suffix below checkpoint root | Verified updates | Final manifest SHA256 |
| --- | --- | --- | --- |
| Tiny control | `control/tiny-control-02/` | 2, 3 | `d99477ac5ca9ae0d5b95d2a62f842d71cabef4a059ec52ec877ed0c1bfe7bd32` |
| Tiny reduced KL | `reduced/tiny-reduced-01/` | 2, 3 | `c264b3e75471d023f92cc4a3c8412c5a257f2bc002ef7cec864731f618885bb7` |
| Tiny reduced child resume | `reduced/tiny-reduced-resume-01/` | 3 | `37bd9543e05f6f9a8a15283a5bd48419dc8581514f04e6e01a279575158b5e1f` |

Each update directory contains `state.pt` and `manifest.json`. Each tiny state
occupies 20,069,204 bytes. Resume preserves tensors and optimizer state, but a
reserialized checkpoint's complete file hash need not match another segment's
file hash. Exact state equality is established by the accepted restart audit.

Small evidence root:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T043200Z/`

The prefixes `kl-tiny-control`, `kl-tiny-reduced`, `kl-tiny-resume` and
`kl-inventory-tiny01` are verified. Their local receipts are respectively
`.runtime/olmo-kl-retention/tiny-control-02.json`, `tiny-reduced-01.json`,
`tiny-resume-01.json` and `inventory-tiny-01.json`.
The tiny-resume receipt SHA256 is
`f2650f23e4ed4112dedf63c2db9467ec8ae57fafff0f954a85ec37e70722b8b1`.

Inventory `.runtime/olmo-kl-retention/inventory-tiny-01/report.json` has SHA256
`2aa805fc4a49a1f6a4eb420d048b6ecea1ae51e883ee484a350d5d98e437e3ea`.
It accounts for five distinct checkpoint payload objects totaling 100,346,020 bytes,
their five manifests totaling 504,848 bytes and three small evidence archive/manifest
pairs totaling 3,852,963 bytes: 16 counted objects totaling 104,703,831 bytes.
Its later archive and acceptance-audit bundle are outside that earlier snapshot.

The inventory is metadata-only: it checks final reports, journal entries,
latest publications, asynchronous completion receipts, source inventories and
retained archive membership. It performs no state-payload read or new cloud
readback. Producer receipts record the earlier generation-pinned download
verification. Repeated references to the same URI/generation count once;
different stored URIs count separately even if content happens to match.
The remote `storage-receipt.json` self-publication objects are excluded from
object totals because their generation metadata is absent from the local
receipt they serialize. Exact local receipt bytes are preserved as inputs.

Original publication receipts under `checkpoint-publications/` are excluded by
the unchanged small-evidence archiver. The inventory copies those exact bytes
under neutral `input-snapshot/authority-NNNN.json` names, with an explicit
original-path/hash mapping. This preserves their bytes without modifying the
completed stages. It also snapshots every currently present top-level small
retention receipt and its own producing helper. A later inventory includes
earlier inventory receipts; it cannot include its own future upload without
a reference cycle.

The paired/restart acceptance reports are also copied into the separate
`.runtime/olmo-kl-retention/acceptance-audits-01` bundle with all thirteen exact
input/source authorities, including the original parent report and six auditor
sources. Bundle report SHA256:
`3b39f00426a504bee5679710cca1c1a62791a63c3ec88509d12fa8217a9a2926`.
Its verified cloud prefix is `kl-acceptance-audits01`; the corresponding local
`acceptance-audits-01.json` receipt has SHA256
`b1545fa5e118e5634dd9045457f1ac6e7f7d6b37dfdebb5243a072f19a04fdf2`.

After both native stages and their small-evidence uploads complete, run the
following metadata-only command from the repository root, substituting a fresh
inventory suffix if the destination already exists. The expected native stage
receipt filenames below should be used when retaining those stages.

```bash
python3 .runtime/olmo-kl-retention/build_inventory.py \
  --scope full-cohort \
  --stage tiny-control-02 tiny-control-02.json \
  --stage tiny-reduced-01 tiny-reduced-01.json \
  --stage tiny-reduced-resume-01 tiny-resume-01.json \
  --stage native-nf-control-32to64-01 native-nf-control-32to64-01.json \
  --stage native-nf-reduced-32to64-01 native-nf-reduced-32to64-01.json \
  --output-dir .runtime/olmo-kl-retention/inventory-full-01
```

For each closed native stage, use the unchanged retainer in a CPU-only project
container. This example assumes its report is final, W&B is synced and the
checkpoint worker drained. No checkpoint payload argument is needed because
the runtime already published those objects.

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
  test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent &&
  env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_two_gpu_retain.py \
    --input-dir .runtime/olmo-kl-continuation/native-nf-control-32to64-01 \
    --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T043200Z/kl-native-control \
    --receipt .runtime/olmo-kl-retention/native-nf-control-32to64-01.json'
```

Use the analogous reduced-stage paths and `kl-native-reduced` prefix, then
retain the completed inventory directory as `kl-inventory-full01` with receipt
`inventory-full-01.json`. Do not archive the whole live milestone directory.
Unsetting the stale credentials-path override lets the VM's existing metadata
credentials work; no credentials are copied into evidence.

Keep the original NF update 32 checkpoint and report available: the current continuation
launcher authenticates that parent even for a child resume. Child recovery
loads the exact branch configuration and does not apply the migration again.
The original parent and previously retained corpora/weights are references,
not newly counted payloads in this milestone. SSD states may disappear with
the VM. Recover from a verified cloud publication; a 600-second checkpoint
trigger is not a guaranteed ten-minute maximum loss interval.
