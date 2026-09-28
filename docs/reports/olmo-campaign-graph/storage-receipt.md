# Durable evidence receipt

2026-09-28. Both completed one-H100 attempts are uploaded and verified in
`gs://fast-chunks`. The existing retention helper uses an `olmo-two-gpu`
namespace; that path does not imply these probes used two GPUs.

Root:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T212000Z/`

| Stage | Members | Evidence bytes | Evidence SHA256 |
| --- | ---: | ---: | --- |
| `campaign-graph-01` | 73 | 254182 | `58de69f58292f57ac134fba0859456e00bfd3616155999ad3bc908898d272477` |
| `campaign-graph-02` | 72 | 259283 | `955275a439cac57a490d2a9a127fcd24b491fde3971faa1c0cf35e50e80b7bca` |

Each stage contains `evidence.tar.gz`, `retention-manifest.json` and
`storage-receipt.json`. Each archive includes its report, launcher log and all63
declared runtime source snapshots; additional CPU logs and protocol/report
documents are included. The first attempt preserves development logs; the
second preserves final CPU/lifecycle logs and closeout report/usage/progress.

Manifest SHA256:

- First: `365787969e71ff00a47fc9ec4eb83ba07962558b2ef850fc29a89b603ab092b6`
- Final: `f2d6bc14702e66d3dcc06ff26f357c5eec54cefdfed790f5c672b81d9d29cc49`

Remote receipt SHA256:

- First: `4454f0575fb640fefa219d223a68e4f320d6fc6c1c7ffd407e1c9a6d83dab8ce`
- Final: `85daff98830954cc1abd9e9b9c1ffcd1e939d9fdaec946a9d5d9df418385f4b1`

Server size, MD5, SHA256 metadata and downloaded SHA256 were verified for all
six objects. All declared source snapshots were checked against report hashes
before archival. Local receipts are
`.runtime/olmo-campaign-graph/retention/gpu-01.json` and `gpu-02.json`.
No local files were deleted. Archived evidence directories must not be mutated;
later additions require a new stage.

No new full model checkpoint was needed: each approximately55-second probe
starts from the retained pinned source checkpoint and uses disposable CPU RAM
snapshots. Distributed restart will create and verify an actual checkpoint in
the next milestone. Code/report changes are committed and pushed independently.
