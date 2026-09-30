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

07:30 UTC: Versioned F execution/probe/online/independent-audit helpers committed
and pushed as52d9ac3. Execution+probe52 CPU checks, independent audit55 and
online23 pass. Final208 source inventory is runtime-sources.json; final native
resolved SHA50d147f5f4f36aebefbc594018e31ff0575ddaef756f195bdd3427419f3df2b2.
Origin evaluation publishes one merged W&B record with collective participation
on both ranks. Historical200 sources unchanged. CPU acceptance is separate
from pending two-GPU insertion/cloud-resume acceptance.

07:32 UTC: Sequential acceptance_queue.py waits for B128 terminal success, then
runs tiny-old-control-01, tiny-stability-01, cloud restore of update2 and
tiny-resume-01. Root audits before native F launch. Queue session70511;
B session82005. Optional separate saved NF/NFR curve and explicit NFR KL
continuation helpers are being prepared only; neither is launched or allowed
to modify the frozen live F sources. B has reached128; terminal publication
still pending. W&B ordinary control37uu86ip.

07:40 UTC: B128 completed_plan, cloud128 verified and W&B synced. Its final
CE2.687618 is a modest deterioration from32, with finite/unclipped updates.
FinalreportSHA9b8f44ce52163eca6679f78f4603fb26e023d6354273f2369749fd5b4798f7f3.
The old W&B checkpoint-summary fields lag terminal drain; report and publication
receipts are the checkpoint authority. Baseline evidence retention underway.

07:41 UTC: Tiny insertion passes8,531 independent checks; fresh-process verified
cloud restore2→3 passes10,150. Exact model/Adam/RNG/cursor/input/rawgradient and
repeated evaluation/probe outputs agree. Queue02 completed; queue01 was stopped
while waiting, solely to correct its terminalstatus spelling before any launch.
No training was interrupted by that host-queue change.

07:42 UTC: Native F launched as native-f12-to128-01, session60881; host launcher
launch_case.py, fixed stop128 inside192 ceiling, lean observation,14400s bound.
CPU curve_queue.py/session31926 publishes matched completed0/32/64/96/100/128
figures while training continues. GPU concurrency remains one two-rank job.
Conditional NFR KL215-source scope committedc1a764e; saved-component curves
committedabf269e. Both prepared only, no additional training authorized by their
mere existence; root selects useful follow-up using this F study's evidence.
