# Evaluation acceptance storage

All GPU stages are closed and W&B is synced. Checkpoints and small evidence were
retained with create-only writes, exact generations, byte counts, SHA256, server
MD5 and downloaded SHA256 verification. No local files were deleted. Both H100s
were verified idle at closeout. The boot disk has approximately 91 GiB free.

Runtime: `.runtime/olmo-campaign-evaluation/`. The pre-closeout inventory
`inventory-01/report.json` crosschecks **16 small-stage receipts, 32 evidence
objects (9,725,288 bytes), and 28 checkpoint objects including 14 state files
(20,488,663,589 bytes total)**. It deduplicates objects by URI/generation, not
content hash. Inventory, later closeout and the separate optimizer-history
receipts are excluded from these counts.

Inventory SHA256:
`4f974d286bab859396233caa8d090f996914325dfcb69f084009955b1ad4132c`.
This is a local consistency check of already verified producer receipts and
reports; it is not another full cloud readback. The tiny stopped checkpoint
was additionally downloaded from pinned GCS generations and used for both the
fresh-process continuation and evaluation-only resumed-boundary check.

## Small evidence

Every closed stage has `evidence.tar.gz` and `retention-manifest.json` under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T155500Z/campaign-evaluation-<stage>/`.
Exact pins and readback flags are in `receipts/<stage>.json` and the inventory.

Included stages cover CPU scopes, declarations, tiny reference/insertion/stop/
restore/resume/evaluation-only runs, native insertion and independent audits.
Superseded preflight declarations-01 and the auditor-only failed terminal audit
are retained explicitly. No GPU run used the superseded declaration; no model
failure is inferred from that audit's missing-empty-field assumption.

## Native checkpoint authorities

Directory root:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T161500Z/campaign-evaluation/NFR/native-insertion-01/`.
The historical storage subtree name does not change the actual declared model
startup or execution lineage.

| Boundary | State bytes | State generation | Manifest generation |
| ---: | ---: | --- | --- |
| Update 0 | 5,071,746,819 | `1790699020344823` | `1790699040683771` |
| Update 3 | 15,214,852,353 | `1790699881131039` | `1790699951813406` |

Each boundary uses `update-NNNNNN/state.pt` and `manifest.json`.
Manifest SHA256 pins, respectively:

- `5db2b45bf82f4fe57bf478de3fc162f4717e0cc67f1b4a4336cabc054dd93d21`
- `1df150e15d822929e0ff8683c5fffc6eddfaaa6ab5a174a6f1eac265631e4d1a`

The final state SHA256 is
`b46d2f501521a34e517c93f0e6946d0e6a2f5a1edec3d0cfdb654324d9b0423b`.
It includes the new declared identity and cannot be substituted for PR46's file
hash even though the scientific training state compares exactly. The finite
three-update plan is complete; same-lineage recovery does not extend it.

Tiny cloud-restored update-2 manifest SHA256:
`2cfb6fefafc953df50af18f413f9529993a6be74172deebd32fff422904f2acf`.
The verified local download is `tiny-restored-01/checkpoint`.

The companion [optimizer storage record](../olmo-optimizer-history/storage-receipt.md)
holds the independent numerical probe and public original-loss reference.
Final PR/closeout receipts are recorded in [progress](progress.md).
