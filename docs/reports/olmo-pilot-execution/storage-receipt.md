# Durable storage and recovery authorities

Small evidence is retained with create-only GCS objects, exact generations,
server size/MD5/SHA metadata checks and downloaded SHA256 verification.
Persistent receipts are under `.runtime/olmo-pilot-execution/retention/`.

Small-stage prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T190649Z/pilot-execution-<stage>/`.
Each stage archive preserves its declarations, source snapshots and reports.
The reports embed the complete checkpoint publication records. The generic
retainer excludes paths beginning `checkpoint-`, including the individual
`checkpoint-publications` directory. The final inventory therefore snapshots
those receipt bytes under neutral filenames with explicit original-path/hash
mappings. Closeout likewise renames checkpoint-prefixed metadata paths and
records the mapping. Large checkpoint states are stored separately, not
duplicated in these archives.

Checkpoint prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T190649Z/pilot-execution/`.
Native declarations add `native-b32`, `native-b64` or `native-nfr12`, followed
by arm, segment and `update-NNNNNN`. The historical `olmo-fusion-startup` subtree
is required by the unchanged retainer; it does not imply that the ordinary base
run uses adapted weights. Exact object URIs and generations are in each segment's
`checkpoint-publications/update-NNNNNN.json`.

The synthetic tiny corpus, including its binary token files, is retained through
the document-corpus retainer at
`gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/pilot-execution-20260929/synthetic-fixture-corpus`.
Its 62 objects are separate from the fixture's small metadata/index archive.
Actual Dolma data and acquisition authorities remain in the PR49 receipts.

## Tiny restore authority

The stop-at-update-2 checkpoint manifest SHA256 is
`02ee7dc9e2aa6975c406a557bcc51bd820f45d9c8f08d84bef457be851b193f1`.
Its producer receipt SHA256 is
`2fbbdfd7496709c197af50d82c2f77f6912b3eb9c08f44824913df7f7e3c4788`.
Cloud restore downloaded and verified 20,150,587 bytes in 1.9168 seconds to
`/mnt/localssd/cdrm-checkpoints/pilot-execution/tiny-restored2-01`.
The actual resumed and terminal-only GPU runs consumed that restored source.

Both original failed v1 JSON audits and the passing v2 audits are retained;
passing reports do not replace failed evidence. Runtime sources stayed frozen.
The revised auditor and its 27-test CPU result have their own retained source
snapshot, outside the 192-file runtime inventory.

## Local retention and closeout

Each new segment retains its latest two completed local checkpoints only after
verified cloud publication, persistent receipt and journal updates. Historical
checkpoints and restore sources are not pruned. Large new states live under
`/mnt/localssd/cdrm-checkpoints/pilot-execution/`; source/evidence and receipts
remain on the persistent project disk. Checkpoint states on the SSD are not
the sole durable copy.

Final inventory counts and closeout pins will be recorded after all selected
native capacity runs and evidence publications finish. The inventory is a
verified-receipt snapshot, not an additional full cloud download. Its own and
later closeout/publication receipts are necessarily outside that snapshot.
