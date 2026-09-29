# Progress and interruption handoff

2026-09-29: work in progress on `feat/olmo-pilot-data`, starting from `b9b0cb4`.
The user authorized proceeding with the bounded pilot-data milestone. No GPU
work or training is part of the current stage.

- Plan/recipe: 30 CPU tests passed. All 37 selected upstream objects passed HEAD
  inspection. The earlier readiness metadata yields exactly 12,512 exclusions.
- Source plan: `.runtime/olmo-pilot-data/declaration-01/source-plan.json`, SHA256
  `5954f4480268e50c9041ffc06651a6539efe93ec54f6eb36a2977a4d5e2a9707`.
- Canonical recipe SHA256:
  `d826a3d005c3dfa56b8f1dbe882cd91f84d0d41922cac97563ec640cb5f93ba6`.
- Streaming generation-pinned stage restore: 16 CPU tests passed. Actual cloud
  restore is pending.
- Preparation and ordered-reader helpers are undergoing peer review and tests;
  first acquisition has not started at this checkpoint. Freeze preparation code
  and tests before first acquisition because their hashes are in the authority.

Evidence: `.runtime/olmo-pilot-data/`. Proposed SSD root:
`/mnt/localssd/cdrm-data/olmo-dolma-v1_5-pilot-20260929`.
Every source/shard must be retained before more work. Keep old sources frozen;
new source files are versioned separately. Do not infer a quality campaign from
this data-preparation work. See [protocol.md](protocol.md) and preceding
[next steps](../olmo-campaign-storage/next-steps.md).
