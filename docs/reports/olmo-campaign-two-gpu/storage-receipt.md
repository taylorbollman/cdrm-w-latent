# Campaign two-H100 storage receipt

Recorded 2026-09-29 from verified stage receipts. The durable root is:

```text
gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T235700Z/
```

Each stage below has an `evidence.tar.gz` and `retention-manifest.json` under
`<root>/campaign-<stage>/`. Machine-readable local receipts are in
`.runtime/olmo-campaign-two-gpu/retention/<stage>.json`. They retain object
generations, sizes, SHA256 values, MD5 checks and verification outcomes.
All listed stage receipts report `status: verified`; none deletes local files.
All three completed B8, B12 and B16 capacity stages are included.
See [test ledger](test-ledger.md) for numerical and operational outcomes:
successful retention does not make a failed experiment pass.

## Verified evidence archives

The SHA256 values and generations below identify `evidence.tar.gz`, not the
larger checkpoint state. Each archive was downloaded and SHA256-checked in
addition to server size, MD5 and SHA metadata checks. Its manifest is also
download-verified. Source snapshots contain the exact source bytes pinned by
the corresponding report; different attempts are preserved separately.

| Stage | Members | Archive bytes | Archive generation | Archive SHA256 |
| --- | ---: | ---: | --- | --- |
| `nccl-01` | 3 | 3,754 | `1790639938831208` | `321df594bd1fd9aada44c523bfa1ce6928ce8865d6591b985642dc0229511733` |
| `data-restore` | 3 | 4,421 | `1790639942418202` | `f3e2116c60359f9481d52c33e63dd2e5437a0232f05e96b962ff43332757d5cb` |
| `tiny-eager-01` | 72 | 368,804 | `1790640155790436` | `8f8f9d682f7e7facbe82fdf4bc38d99227e6d729e14b42ac2ba0453d9608fc9b` |
| `tiny-graph-01` | 72 | 368,039 | `1790640159645528` | `84db218a5d2bbffaa80d89c93fdffbc5c58eb11096e07426349c473e886afc10` |
| `tiny-write-01` | 74 | 406,732 | `1790640210471442` | `796b8b167b4208e0de90c8153a9dfc65f8dec7afcf2e56152b4985e4107eca57` |
| `tiny-resume-01` | 74 | 290,766 | `1790640320818347` | `2f4e69dad93af17b08e3b9db0d7b50bde78246bf86a07ca286ba431d6b731620` |
| `tiny-write-02` | 74 | 406,987 | `1790640471412514` | `105f7c9864d178eeacb55dcd24ad7d88f15de7f86c877008974696e5bb8ffcc7` |
| `tiny-resume-02` | 74 | 355,658 | `1790640738624608` | `31f402cf63598d886d2cb7ad867809aa832b206e933f8e5c093496b14fe69bfa` |
| `pretrained-eager-01` | 72 | 321,542 | `1790640666751940` | `beeaa7c556b1a11d79a095d00d9bea76afd83d5c83e37c0c37963c3a5192d3f3` |
| `pretrained-prepared-eager-01` | 72 | 321,730 | `1790640970603502` | `1c7f1177e37ea47fcc477d13f528d1ba2fa24ece103b1c702aa1074dd18cb1cd` |
| `pretrained-prepared-graph-01` | 72 | 357,804 | `1790641175719911` | `bec2a25ad8f4c5e8cb52d407617094edb8f3ab9b841968862ff557423a9b96b1` |
| `pretrained-fp32-01` | 71 | 273,371 | `1790641248244882` | `737920877bd58caa2ba5359c9d11eddac5f493f18018b48695f055049bb8eef8` |
| `pretrained-write-01` | 74 | 491,305 | `1790641615118844` | `c7658d059699142539077356cccb452793b11fa0686dfdc991b8b59c16b9ce3a` |
| `pretrained-resume-01` | 74 | 400,794 | `1790641976505501` | `a5219559a06a164063517e29474400da459b7a7c8b149a7d5f0132297a903764` |
| `checkpoint-restores` | 5 | 3,675 | `1790641980097684` | `8ec824f885667d4fceec72a672c60b92de25a3a25324e540855f371641236798` |
| `pretrained-fp32-02` | 71 | 273,431 | `1790642118172202` | `cb1d827229d9962e14c6a2f8abc38ca85a93b1e8f43bd4c742781c8c090104b1` |
| `capacity-b8-m2-01` | 74 | 304,278 | `1790642663041190` | `dc0b09ee8546c2d1c2345fa65a262a103283c6ca38fac270cbf50107082ba14e` |
| `capacity-b16-m2-01` | 74 | 304,523 | `1790643313491107` | `8d9c7d7f5b336abe7132aa57d1d283a7939306c3b36dade15fd6bf1e6e7e73fb` |
| `capacity-b12-m2-01` | 74 | 304,921 | `1790643911208455` | `ff3be54621c26e8ddc89f99110f2b57425cf229f487b82bc8cc5be6f99123b79` |

The failed `pretrained-eager-01` numerical comparison and failed
`tiny-resume-01` safe-load attempt are retained in this inventory, along with
their reports and source snapshots. The corrected FP32 metadata repeat is
`pretrained-fp32-02` ([W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/627qt1c0));
the original FP32 report remains retained rather than overwritten.

## Full pretrained recovery checkpoint

The checkpoint prefix is:

```text
gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T235700Z/campaign-pretrained-write-01/checkpoint/
```

| Object | Bytes | Generation | SHA256 |
| --- | ---: | --- | --- |
| `state.pt` | **15,214,756,865** | `1790641614604389` | `bd06a69aa9d53bf5da074afd86b8167083f69c142ee7b89935fc50b6384fef08` |
| `manifest.json` | 68,151 | `1790641614873473` | `37db51846e5b6c344d9e42f2c047df0ba2bab96c0e93ad7be0e34f2e1f808a67` |

This is the full NFR campaign model plus initialized optimizer and recovery
state after one accumulated update, not merely the original OLMo weights.
It is a recovery-test checkpoint trained on the operational fixture, not a
quality-trained model or the starting point for a production comparison. New
campaign arms should use the agreed pinned pretrained weights and initialization.
Its source model remains pinned to OLMo-1B revision
`81b71efbce6f4dada57c94860301af4298bcd351`, original weight SHA256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.

The upload receipt deliberately records `download_sha256: false` for the two
checkpoint objects: its initial verification used server size/MD5 and SHA
metadata. **A separate complete download then verified both SHA256 values.**
The subsequent restore record is
`.runtime/olmo-campaign-two-gpu/checkpoint-restore-evidence/pretrained-01.json`,
retained inside `campaign-checkpoint-restores/evidence.tar.gz` above. It records
generation-pinned downloads to:

```text
/mnt/localssd/cdrm-checkpoints/campaign-two-gpu/pretrained-01-restored
```

Both downloaded objects have `download_sha256: true`, matching the exact bytes,
generations and SHA values above. Fresh torchrun processes used this downloaded
directory, not the original local save. The
[pretrained resume run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/u3pnp0xi)
passed all nine checks; its next update matches the original live-graph
continuation bitwise on both ranks, including raw gradients, complete model and
Adam state, metrics, scheduler/counters, cursor and RNG draws/state. Thus the
recovery evidence goes beyond upload metadata or an unexercised checksum.

## Tiny checkpoint lineage

Both tiny checkpoints are retained to preserve the serialization failure and
its correction. Their separate read-back records are included in
`campaign-checkpoint-restores/evidence.tar.gz`.

| Stage checkpoint object | Bytes | Generation | SHA256 |
| --- | ---: | --- | --- |
| `campaign-tiny-write-01/checkpoint/state.pt` | 733,716 | `1790640210027553` | `c02b498e4fc02e3613190092da3b0fc6f9814d7df2841b2159124229db3bfa0d` |
| `campaign-tiny-write-01/checkpoint/manifest.json` | 46,082 | `1790640210243949` | `bcdec0cf751bb95fcd8aec8508f0e70026a64c9c8657fe3a866b666bf231390c` |
| `campaign-tiny-write-02/checkpoint/state.pt` | 733,716 | `1790640470955835` | `bd65fd3a505048100fedeeb966a1427158f5062dec6700aeb4860a39bef9d53e` |
| `campaign-tiny-write-02/checkpoint/manifest.json` | 46,082 | `1790640471185253` | `a372f8e20563b38935432f361ebc9b3c21a2856b648954309ba00cecef796718` |

The first cloud copy is byte-correct but its safe load rejects a `TorchVersion`
metadata object; do not use that pair as restart qualification. The corrected
pair canonicalizes metadata scalars, retains `weights_only=True`, and its
[fresh resume](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/7du9uw9g)
passes bitwise continuation. Source hashes distinguish these attempts.

## Data and interruption boundaries

The existing document-token corpus was restored separately from its Dolma
storage prefix, without retokenization. `campaign-data-restore` records all 86
object checks and corpus verification: 28 shards, 12,512 unique documents and
7,054,230 tokens. This remains a source-coverage fixture, not a production
mixture or multidocument-packing acceptance.

Local SSD paths are disposable. The verified GCS checkpoint and evidence
prefixes above are the durable recovery references. The completed stage
receipts do not claim that an active capacity run's unsaved optimizer state is
recoverable; capacity probes are bounded disposable tests reconstructed from
the retained source checkpoint.

## Completed capacity retention

`capacity-b8-m2-01` completed successfully with all 12 stages passing and eight
complete Adam updates; [W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/kkmsgtfg).
Its verified evidence archive is included above. The companion manifest is
18,839 bytes, generation `1790642663356014`, SHA256
`ef8e1c52b5dbb2cda6550840500be91fdfe1e0039bed859c78f087a85ab72748`.
Both objects passed full download SHA256, server MD5, size and SHA metadata
verification. No full capacity checkpoint was uploaded and no local files were
deleted. Timing and memory results are in the [test ledger](test-ledger.md).

`capacity-b16-m2-01` also completed successfully with all 12 stages passing and
eight complete Adam updates;
[W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/cbl8xn7l).
Its verified archive is included above. The companion manifest is 18,840 bytes,
generation `1790643313758825`, SHA256
`7294a188e8a7cf6d4d0e6008a8cac171b280179f634ca1933bdcd5f9cc0539f4`.
Both objects passed full download SHA256, server MD5, size and SHA metadata
verification. No capacity checkpoint was uploaded and no local files deleted.

`capacity-b12-m2-01` completed successfully with all 12 stages passing and eight
complete Adam updates;
[W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8yx9sar0).
Its verified archive is included above. The companion manifest is 18,840 bytes,
generation `1790643911526332`, SHA256
`07734c3b49528ab831594790be19c82842125d999221c3627005218d73a746e2`.
Both objects passed full download SHA256, server MD5, size and SHA metadata
verification. No capacity checkpoint was uploaded and no local files deleted.

B12 is the recommended comfortable next-work configuration, with B8 available
for additional headroom and B16 a tighter option. The [test ledger](test-ledger.md)
records measured throughput, memory and the remaining qualification limits.
B32 was deliberately skipped after observing B16 headroom; there is no B32 run
or receipt. All capacity stages have stopped and their evidence is retained.

## Verified final documentation archive

The closeout prefix is `<root>/campaign-closeout/`; local receipt
`.runtime/olmo-campaign-two-gpu/retention/closeout.json` is verified. Its 37
members include documentation at commit `36c07f2`, CPU logs, all 19 stage
receipts, the source audit and final idle-GPU evidence. The audit verifies
1,120 report/source pairs across 18 report-bearing stages. This archive was
created before appending its own receipt here; this final receipt is in Git.

| Object | Bytes | Generation | SHA256 |
| --- | ---: | --- | --- |
| `evidence.tar.gz` | 91,306 | `1790644055095603` | `cc4874a6b886751ebbc7ff323968fdc8eeff362f2e1b79870fdd72ddd0c660a4` |
| `retention-manifest.json` | 8,633 | `1790644055379261` | `fad9e3bf6e3d3b812073e3fede3f18860c7c993911f00c569ce3f45d565a7e99` |

Both objects pass full download SHA256, server MD5, size and SHA metadata
checks. Including closeout, **20 verified receipts cover 46 cloud objects**.
The code and reports are delivered in
[PR 38](https://github.com/taylorbollman/cdrm-w-latent/pull/38). No GPU job remains.
