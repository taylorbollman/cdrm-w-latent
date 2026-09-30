# FBT stability progress

2026-09-30 07:07 UTC: User authorized approximately6.5h autonomous work, starting
with FBT-only K4/beta1 and frequent Figure3-style curves. Further useful
milestones may proceed without review. No gate ramp/pass change planned.

07:13 UTC: Both H10080GB devices verified idle inside required project
container. SSD has about1.1TiB free; boot has about89GiB. Large states stay on
SSD plus verified GCS. Created branch feat/olmo-fbt-stability from d0d6b67.
Parallel implementation: explicit F-only fusion128 startup, safe streaming
pass diagnostics, independent audit. Historical sources stay unchanged.

Current authority: [protocol](protocol.md). No new GPU training launched yet.
