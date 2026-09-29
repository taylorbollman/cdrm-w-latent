# Pilot data storage and recovery

Large files are staged at
`/mnt/localssd/cdrm-data/olmo-dolma-v1_5-pilot-20260929` and independently recovered
at the same path with `-recovery` appended. Both SSD locations are disposable.
The persistent project evidence root is `.runtime/olmo-pilot-data/`.

## Cloud authorities

Data root:
`gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/pilot-20260929-v1`.
It contains 37 `raw-<source>` stages, `config`, 37 `shard-NNNNNN` stages,
`root-summary`, and `ordered-round0`. Each stage has immutable generation,
size and digest records. The ordered suite publishes its manifest last.
Raw stages retain exact extracted JSONL prefixes and line maps, not complete
upstream gzip objects. Their authorities map back to pinned URLs and ETags.

Small-evidence root:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T174435Z`.
Stages are prefixed `pilot-data-`: `declaration`, `acquisition`, `raw-recovery`,
`partial-recovery`, `preparation`, `raw-token-audit`, `ordered-build`,
`ordered-audit`, `complete-recovery`, `tracking`, and `closeout`.
Local receipts are in `.runtime/olmo-pilot-data/retention/`. The closeout stage
also preserves the data receipts, operational scripts, test logs and reports.

| Authority | SHA256 |
| --- | --- |
| Declared source plan | `5954f4480268e50c9041ffc06651a6539efe93ec54f6eb36a2977a4d5e2a9707` |
| Canonical recipe | `d826a3d005c3dfa56b8f1dbe882cd91f84d0d41922cac97563ec640cb5f93ba6` |
| Native tokenizer | `9ad33b4b39a9f83973c3f8c42a01948dd5b877a28ac9a5356956c4ff4ed0b714` |
| Complete corpus manifest | `f3206c360dd412ccbaf08c65b30a7ba7f1b30b56db7ffbb9559522831c797e06` |
| Upstream-to-retained-source mapping | `4d5749eea9711765c38b3985501d4fab299888b003f6013526fd1bda528e709e` |
| Ordered suite manifest | `080402225e24350cc377b720bdf39a55b87e71e4a6af5fea3cda7460eb169215` |
| Ordered suite identity | `3f10d09e5694a132a31742a1cd72886d8fc0e4a7e549ba9d2a4c9ceef318abe4` |
| Training panel manifest | `e890bdb6721cdf454251a45b9376d66622d62e6c164feeae3d624cd4194ea76d` |
| Training panel identity | `ee7e75c99412ffa11513c5d3737ebb5ed61226818dbf1bcbd367f7618b6ca3db` |
| Index retention receipt | `334aed24644fc407673b8136970c6e61edc3fd0eeeef736c6e5801ce0b3ebd59` |

## Completed recovery evidence

- `raw-recovery-summary-01/report.json`: all 111 raw objects, 2,086,379,363 bytes;
  summed transfer time 96.71 s, excluding orchestration and waiting.
- `partial-recovery-comparison.json`: four restored token shards plus four
  newly prepared shards equal the original eight-shard preparation exactly,
  including config and all 24 shard files.
- `complete-corpus-recovery.json`: remaining 29 token shards and root summary
  restored; complete semantic verification and manifest equality pass. This
  combines restoration and exact regeneration, rather than claiming all 37
  token shards were downloaded directly.
- `index-recovery-01/report.json`: all 44 suite objects, 792,408,198 bytes,
  restored in 39.48 s. Receipt and manifest digests are listed above.
- `ordered-audit-01/report.json`: semantic interval/count, mask, membership,
  sampled literal payload and cursor checks pass on the recovered corpus/indexes.

Use `olmo_pilot_data_restore.py` for individual raw/config/token/root-summary
stages and `olmo_pilot_index_retain.py restore` for the complete index suite.
Require the separate receipt SHA seal, use fresh destinations, and verify the
corpus before opening the reader. Do not fabricate a complete root manifest
over partial shards. See [operator notes](operator-notes.md).

No model checkpoints were created or deleted in this milestone. Existing model
and optimizer checkpoints remain governed by their earlier storage receipts.
