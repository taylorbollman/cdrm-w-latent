# FBT stability progress

2026-09-30 07:07 UTC: User authorized approximately6.5h autonomous work, starting
with FBT-only K4/beta1 and frequent Figure3-style curves. Further useful
milestones may proceed without review. No gate ramp/pass change planned.

07:13 UTC: Both H10080GB devices verified idle inside required project
container. SSD has about1.1TiB free; boot has about89GiB. Large states stay on
SSD plus verified GCS. Created branch feat/olmo-fbt-stability from d0d6b67.
Parallel implementation: explicit F-only fusion128 startup, safe streaming
pass diagnostics, independent audit. Historical sources stay unchanged.

Current authority: [protocol](protocol.md).

07:11 UTC: Started exact saved-Adam ordinary B control32→128 while new F-only
helpers are implemented. Host launcher `.runtime/olmo-fbt-stability/launch_baseline.py`,
log `native-b32-to128-01.log`, report directory of the same name. Parentmanifest
5bbf559a248cc16fff3369d4641a832d2e5ca2c8e24c4162ffab393af64d5411;
original B128 declaration and200-source runtime unchanged. No concurrent GPU
jobs may start until root confirms this stage finished.
