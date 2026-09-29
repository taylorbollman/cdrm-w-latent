# Checkpoint and evidence retention

All completed checkpoints and selected evidence are retained in `gs://fast-chunks`.
Large states were written only under `/mnt/localssd/cdrm-checkpoints/pilot-async`;
SSD persistence is not assumed. Persistent project receipts are under
`.runtime/olmo-pilot-async/retention`, and each execution stage retains its
storage journal, latest publication and individual checkpoint receipts.

## Inventory snapshot

`.runtime/olmo-pilot-async/inventory-01/report.json`, SHA256
`2b8d949260f64df6a6691c55de9dc74df937b153fab954d600ae09a8bba1a1ee`:

| Scope | Objects | Bytes |
| --- | ---: | ---: |
| 12 new checkpoint state/manifest pairs | 24 | 35,656,946,054 |
| 16 verified small-stage evidence receipts | 32 | 17,681,818 |
| Total snapshot | 56 | 35,674,627,872 |

Four execution stages completed: tiny blocking, tiny async, tiny cloud-resume,
and native accumulated NFR. Restored objects are not counted twice. There are
no new corpus objects: the PR49 real corpus and PR50 tiny synthetic corpus
remain separately retained. This is an independent local receipt/metadata
consistency inventory, not an additional cloud payload readback. Producer
retention already performed generation-pinned readback for every object.
Inventory and later closeout/administrative uploads are outside these counts.

Small-stage root:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T204500Z/`

Stage prefixes are `pilot-async-<stage-name>`, each with `evidence.tar.gz` and
`retention-manifest.json`. Individual checkpoint receipt bytes are preserved
under neutral `inventory-01/input-snapshot/authority-NNNN.json` filenames,
with exact original-path mappings and hashes in the inventory. This avoids
the unchanged generic stage archive's checkpoint-prefixed-directory filter.
The final closeout uses explicit safe-name mappings for the same reason.

## Native checkpoint authorities

Root:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T204500Z/pilot-async-native/NFR/native-nfr12-accum-01/`

| Update | Manifest SHA256 | Persistent publication receipt SHA256 |
| --- | --- | --- |
| 0 | `02e60e1a94a661626d71a03c01f3549a5a9b60116a8c7069b941b41e2f326fa3` | `e4cbdc8e5bf0026abb05e53d9ad2b333047a1e2c77fd6e94acae1d363d6ce35b` |
| 2 | `592d79ddb60e34f34c44f5c99a27022e3763a9270f997f120a6bf80bcd57a6b6` | `68399c3af2033d15525f1e1812bb8aed1d10c50850b90c4d17db1cc6e8c9285c` |
| 4 | `06146db4d53dd0088cb44fbb65d50983ed667d685ef9660a99741aeee90cdb44` | `641f9334dd29a3fb2c09a86d8eb971ba867e8a89c8ce95d497ae0fefb2db92e3` |

Each checkpoint is in `update-NNNNNN/{state.pt,manifest.json}`. Publication
receipts are in the native stage's `checkpoint-publications/update-NNNNNN.json`;
the three original receipt snapshots are respectively inventory authorities
0000, 0001 and 0002. Each receipt specifies exact object generations and hashes.
The populated state payload is 15,214,863,297 bytes. New owned segments keep
the newest two local checkpoints only after verified publication; historical
checkpoints and separate restore sources are not pruned by this policy.

Use `scripts/olmo_pilot_execution_restore.py` in the GPU-disabled project
container with the pinned publication receipt and a fresh SSD destination.
The new async executor accepts the restored manifest only within its bound
declaration, source and transport identity. This diagnostic is already at its
four-update finite limit; it is not the origin of the proposed learning cohort
or authorization to silently extend that limit.

The tiny cloud checkpoint-2 restore used publication SHA256
`3fe28b85eb91ed5602403da188aae4a8579823ea88aeaa20033f3c218aa6f082`
and manifest SHA256
`3c5b4057420d21aa4d8a6244d938487539f1e1aa9e55d3a9da45bc659a6231b2`.
Its fresh-process update-3 continuation passes 2,427 exact checks. Native
byte retention is verified, but no additional new-runtime native cloud-resume
GPU run was performed.

## Summary tracking and recovery qualifications

Native report SHA256:
`f2e4065c7167cad0cfa22f24d3803bcdd286f9792554b0bd9f126f0a7ed9e05d`.
Independent summary SHA256:
`86142ab9f44b6cd70b6da61d378b4bdffd49980e0306d940c233fe0cde154667`.

The chart run `1ofs0x3r` synced. Its optional immediate original-run metadata
readback failed; later independent read-only confirmation passes with SHA256
`4c6866684312fcef54f973d4e44391dbf3530c14c53dac510113e61377ec7d92`
in `tracking-summary-recovery-01/report.json`. Both the first attempt and the
successful confirmation are retained. No chart upload or training was rerun.

A completed SSD checkpoint can be newer than the cloud recovery authority.
If the VM and SSD disappear during publication, use the previous verified cloud
checkpoint. A 600-second save trigger is not a maximum rollback interval.
Normal termination drains the final worker; this run ends at verified update4
with no pending checkpoint. Final PR/closeout receipts are recorded in
[progress](progress.md).
