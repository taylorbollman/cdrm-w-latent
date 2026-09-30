# Retention and recovery authorities

The tiny control, reduced-KL and reduced-KL child-resume stages have completed
and published their terminal update 3 checkpoints. Their small evidence is
retained separately. Native NF control and reduced-KL continuations are also
closed, synced and retained at update 64. The full inventory includes both
native branches, the tiny acceptance stages and all eleven main evidence
receipts present after summary, audit and tracking uploads finished.

Checkpoint root:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/kl-continuation/20260930-pair01/`

| Completed stage | Suffix below checkpoint root | Verified updates | Final manifest SHA256 |
| --- | --- | --- | --- |
| Tiny control | `control/tiny-control-02/` | 2, 3 | `d99477ac5ca9ae0d5b95d2a62f842d71cabef4a059ec52ec877ed0c1bfe7bd32` |
| Tiny reduced KL | `reduced/tiny-reduced-01/` | 2, 3 | `c264b3e75471d023f92cc4a3c8412c5a257f2bc002ef7cec864731f618885bb7` |
| Tiny reduced child resume | `reduced/tiny-reduced-resume-01/` | 3 | `37bd9543e05f6f9a8a15283a5bd48419dc8581514f04e6e01a279575158b5e1f` |
| Native NF control | `control/native-nf-control-32to64-01/` | 40, 49, 58, 64 | `0340858a2495e084186779662c6d7f0552a63d39818fa41fc90b4440192c431a` |
| Native NF reduced KL | `reduced/native-nf-reduced-32to64-01/` | 40, 49, 58, 64 | `8b56a520321a2da7b9afcd7d217aea5eeee0382c3d2076a42350536d2254c88d` |

Each update directory contains `state.pt` and `manifest.json`. Each tiny state
occupies 20,069,204 bytes. Resume preserves tensors and optimizer state, but a
reserialized checkpoint's complete file hash need not match another segment's
file hash. Exact state equality is established by the accepted restart audit.

Native control's final state is 15,214,972,545 bytes with SHA256
`e86b6b6485bfa65f03a94d4bcf2e340bc8478c6f2842229179b6884165758fb9`.
Its state/manifest generations are respectively `1790746010650071` and
`1790746089767498`. The final publication receipt SHA256 is
`963c44ea2fc8fe35e46056fdac6ad06e41d2944dc616c6b3e9217f20d5f89ac2`.
The journal records updates 58 and 64 retained locally, with 40 and 49 pruned
only after cloud verification. All four remain cloud recovery authorities.
The completed control report SHA256 is
`d0b9c32c0f3465d8a9950915cf8a5582c47ee16c6df16347e8c2ed50879708b2`;
its independent 8,009-check audit passed, with report SHA256
`aeadf1b741795b5fe140fb6ac38ea7a588006c713acf0827b31bd90592bedb9f`.

Native reduced-KL's final state is also 15,214,972,545 bytes with SHA256
`3aaad70b5c62ad2895f7577f30d0ae6673d13bcdd64e9c4e9e345abdbb76bc02`.
Its state/manifest generations are respectively `1790749271321378` and
`1790749337641540`. The final publication receipt SHA256 is
`6f9ecbd8680573b33755a6f121ddb7363cd78618eac1fd790b2f2127ab67c72d`.
Its journal also retains 58 and 64 locally and records verified pruning of 40
and 49. The completed reduced report SHA256 is
`bd7fbc420ad5472d3f032b006a4df3bb0c87a4bdbceab14c61bed3761bc67e72`.

Small evidence root:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T043200Z/`

The prefixes `kl-tiny-control`, `kl-tiny-reduced`, `kl-tiny-resume` and
`kl-inventory-tiny01` are verified. Their local receipts are respectively
`.runtime/olmo-kl-retention/tiny-control-02.json`, `tiny-reduced-01.json`,
`tiny-resume-01.json` and `inventory-tiny-01.json`.
The tiny-resume receipt SHA256 is
`f2650f23e4ed4112dedf63c2db9467ec8ae57fafff0f954a85ec37e70722b8b1`.
Native control's 234-member evidence archive is verified under
`kl-native-control`; its local receipt
`.runtime/olmo-kl-retention/native-nf-control-32to64-01.json` has SHA256
`a13c2e7bc44ea5701df414ca7d7cafb30f03a182677e301f283f224296c925e6`.
This small-evidence upload did not reread or duplicate its checkpoint payloads.
Native reduced-KL's 234-member evidence archive is likewise verified under
`kl-native-reduced`; its local receipt
`.runtime/olmo-kl-retention/native-nf-reduced-32to64-01.json` has SHA256
`8a3a77d9158693ec840a5869a76986df0fdb21ab04a0fcc5c1f2e538dae1f075`.

Inventory `.runtime/olmo-kl-retention/inventory-tiny-01/report.json` has SHA256
`2aa805fc4a49a1f6a4eb420d048b6ecea1ae51e883ee484a350d5d98e437e3ea`.
It accounts for five distinct checkpoint payload objects totaling 100,346,020 bytes,
their five manifests totaling 504,848 bytes and three small evidence archive/manifest
pairs totaling 3,852,963 bytes: 16 counted objects totaling 104,703,831 bytes.
Its later archive and acceptance-audit bundle are outside that earlier snapshot.

The completed full inventory is
`.runtime/olmo-kl-retention/inventory-full-01/report.json`, SHA256
`be70e73b248b001fd5d2bb5c58208f126270b4eb06fbc7c58e5bb63ec37bce0c`.
It contains thirteen checkpoint states: five tiny states and eight native
states, with exact publication receipt bytes copied under neutral filenames.
Its accounting is:

| Object category | Distinct objects | Bytes |
| --- | ---: | ---: |
| Checkpoint state payloads | 13 | 121,820,126,380 |
| Checkpoint manifests | 13 | 5,479,936 |
| Small evidence archives | 11 | 30,205,787 |
| Small evidence manifests | 11 | 355,634 |
| Total counted | 48 | 121,856,167,737 |

The full inventory's own 41-member archive is subsequently verified at
`kl-inventory-full01`. Receipt
`.runtime/olmo-kl-retention/inventory-full-01.json` has SHA256
`dc2e938c830491707d466b49c4d81f4735b002c01fe6e1c0a2f0ac79bbcda92d`.
That later upload and subsequent closeout/admin artifacts are outside its
snapshot counts.

The eleven included small receipts cover five completed execution stages,
tiny acceptance audits, the earlier tiny inventory, native paired audit,
paired summary, chart logging and the preserved failed tiny preflight. The
last four use prefixes `kl-native-audit01`, `kl-summary`,
`kl-tracking-summary` and `kl-failed-preflight` under the small-evidence root.
Their receipt SHA256 values are respectively
`aac970f66cfee3698bad3c2bfe1a4a4dfd368ed762ca07101ae557c1c8a0e862`,
`64c61f90b6767281a6c28ff5be7d8a621550be4b7523cba3dd81d6bdbb3faa0d`,
`759d684e2de5873828423004527d1efaca08a2a1d996963fb11a4fc4e2369810` and
`66a406e5d9e92faecdf974061f2b4f0784c16fc65d43e4fece2945de377f4e56`.
The native audit bundle preserves twelve authorities and the 16,139-check
passed audit; paired chart run `72jc2qi3` is synced. These objects preserve
the reported evidence; retention itself is not a training-quality assessment.

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

The following metadata-only command produced the full inventory. It is
recorded for reproducibility; reruns require a fresh output suffix and describe
a later receipt snapshot, which may include later closeout artifacts.

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
container. Both native uploads have completed; the control command below is
recorded for recovery and as a future retention template. Its report must be final,
W&B synced and the checkpoint worker drained. No checkpoint payload argument
is needed because the runtime already published those objects.

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
  test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent &&
  env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_two_gpu_retain.py \
    --input-dir .runtime/olmo-kl-continuation/native-nf-control-32to64-01 \
    --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T043200Z/kl-native-control \
    --receipt .runtime/olmo-kl-retention/native-nf-control-32to64-01.json'
```

The analogous reduced-stage upload and completed inventory upload
`kl-inventory-full01` have also completed. Do not archive a whole live milestone directory.
Unsetting the stale credentials-path override lets the VM's existing metadata
credentials work; no credentials are copied into evidence.

Keep the original NF update 32 checkpoint and report available: the current continuation
launcher authenticates that parent even for a child resume. Child recovery
loads the exact branch configuration and does not apply the migration again.
The original parent and previously retained corpora/weights are references,
not newly counted payloads in this milestone. SSD states may disappear with
the VM. Recover from a verified cloud publication; a 600-second checkpoint
trigger is not a guaranteed ten-minute maximum loss interval.
