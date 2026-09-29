# Crossed-state evidence storage receipt

2026-09-29. Independent exact-generation readback verified both objects,
all 111 inventory members, the completed report and 109 source snapshots.
No new trained checkpoint was created or uploaded; no local/SSD files were deleted.

Namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T072000Z/`.
This historical directory name does not imply DDP: GPU execution used one
device; independent verification used a GPU-disabled CPU container.

| Object relative to namespace | Generation | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `crossed-01/evidence.tar.gz` | `1790667100037654` | 583,038 | `913035d22c1a9f79df81b337b0485284496f1ffcb14db6616c4aa1be9b883e5e` |
| `crossed-01/retention-manifest.json` | `1790667100290730` | 28,160 | `315b1c2432f4964ff5be685f9942fbcdada109365dfef6f47e3171b11591c45a` |

Receipt: `.runtime/olmo-crossed-precision/retention/crossed-01.json`.
Totals: **one receipt, two listed objects, 611,198 downloaded bytes and 111
inventory members**, plus two archive control files (113 tar files total).
Separate uploaded `storage-receipt.json` controls are outside the object count.
The archive contains the report, launcher log and 109 frozen sources.

The reused CPU audit/script report are under
`.runtime/olmo-crossed-precision/storage-audit-01/`:

| Artifact | SHA-256 |
| --- | --- |
| `audit.py` | `42acedf754d202764c6dd1c4051446f25faca07828cfc411305115f00ccd6366` |
| `report.json` | `135b51c1f0b55835a510fb691d305325db92b9a5595b798816fa74195531eba0` |

Verification covers exact generation, server metadata, full downloaded size/MD5/
SHA, local retained artifact identity and every manifest member. The archive is
inspected without filesystem extraction. The same bounded review verifies
hybrid group pins, four first-pass identities and 464 valid-position mask/support
records; no model or large checkpoint was read. Its own source snapshot and
receipt copy are saved for closeout.

Final report SHA:
`ba7989f586e19937ff6c64d7f3c65306eae9f61440b4fedb3942bdd3ff18e298`.
Cold/adapted weights remain under their existing pinned authority; this evidence
archive does not duplicate either checkpoint. Final documentation/audit/CPU-log
closeout is recorded below. Storage success does not change the diagnostic's
numerical qualifications, establish an additive causal decomposition or clear
the earlier adapted/adapted intermediate hidden-state discrepancy.

## Final closeout

The existing retention helper verified upload metadata and exact-generation
download SHA for two closeout objects containing24 inventory members. This is
a separate scope from the independent stage readback above; no second audit
framework or GPU run was added. Receipt:
`.runtime/olmo-crossed-precision/retention/closeout-01.json`.

| Object relative to namespace | Generation | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `crossed-closeout-01/evidence.tar.gz` | `1790667647053644` | 96,798 | `8bdf6b532b9597d49a9da4ac3d6781ae3142de0d167c0d9609c04a7536109aef` |
| `crossed-closeout-01/retention-manifest.json` | `1790667647365654` | 6,338 | `0c11c71b6dab398e19b6e0f2dc69d42df2b1917a9eb719e8be42787c4e576d7d` |

Closeout includes reviewed docs/handoff, new runner/helper/tests, final CPU and
environment logs, stage receipt and the independent audit plus its source.
It excludes downloaded archive duplicates and large weights. The immutable
closeout predates its own receipt and final PR metadata, retained here in Git.
No new trained checkpoint was produced and no local files were deleted.
