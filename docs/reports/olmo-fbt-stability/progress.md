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

07:45 UTC: F origin K32 probe complete and CPUsummarypublished (W&Bsocosavi;
training32sqvp7e). Already empiricalsettling atorigin: tailrelativehiddenchange
.108499 atpass4→about1.57e-6 at32, butCE2.673866first→7.239747at32. Thus initial
problem is stablepoorfeedback, not persistentpassinstability on thispanel.
Regular64roworiginCE2.642404/7.399125/7.143659/7.171813 is a differentpanel.
Rootfigures copiedinto reportfigures and resultsupdated. No recipe change.

07:53 UTC: F update8 probe: smallpanelCE pass1/4/8=2.6733/7.2986/7.3087;
tailrelativechange pass4/8=.017374/.000470. Settling is faster, laterCE remains
poor. First8updates finite/clipped; first7rawgrad8.50–51.82. Origincheckpoint0
alreadycloudverified. Training stays unchanged. Optional input-scale/fusion
paper-compatibility read-only assessment delegated; no new testgrid or mutation.
Acceptance/curveorigin archives allretainedGCS in storage-receipt.md (5772ee5).

07:59 UTC: Draft PR55 opened. Read-only paper/code scale assessment found no
raw25x fusion imbalance: normalizedembeddinggate, normalizedproduct calibrated
toOLMoembeddingRMS. Paperintentionallyomitsadditiveidentitypath. See
fusion-scale-assessment.md for sourcecitations and actual adaptations. No
normalization/gate change. Update14 finite, rawgrad4.00, trainCE4.8213;
firstregularposttraining64roweval16 pending. ConditionalNFRindependentpreflight
passes685checks/22CPUtests (61b3575), remainsunlaunched.

08:00 UTC: F16regularCE2.645652/6.909640/6.899153/6.993049, firstpassnearstable
andlaterpassmodestimprovement. SmallprobeK8CE7.1965 vsK4CE6.9288;
tailrelativechangeK8=.035253, largerthanupdate8(.000470). Do notconflatetrained
passimprovementwithconvergence; deepK32atupdate32 remainsnext. Checkpoint16
cycleactive; unchangedtrainingcontinues. See results.md earlytrainingsection.

08:18 UTC: F32deepcurve published j2qipybb; training32sqvp7e. RegularCE
2.651994/6.241516/6.326875/6.368114; smallK32CE6.389008,
taildelta6.91e-7. FBT learnswhilepreservingfirstpassandempiricalsettling;
laterpassesstillmuchworse. LatepreLN RMS1.323→3.764 vsorigin, entropy6.832→6.021;
recordedratherthancallinglowdeltaoverallhealth. ComparematchedNF32CE
2.960392/.../7.435141 andNFR32 3.043012/.../7.063815. No recipechange;
continueto128. Checkpoint32pending, previouscloud16. Figurescopiedtodocs.
Newtraining-semantics-assessment.md recordsattachedgradientmatch, overallhalf
lossnormalization anddeliberatepaperFigure3pass-mixture/startupdifferences.

08:24 UTC: Independent F32 interpretation agrees: allquartiles/unsettledsuffix
settle~7e-7; preLN RMS isnotmonotonicescalation (already~4.23at8vs3.76at32).
Provisionalfollowuppriority ifF128remainshealthy: boundedisolatedonline+saved
NF/NFRcurves, thenexplicitpairedNFRKL1/.1 saved32→64, ratherthanautomaticF192.
F-vsNFremovesbothauxlosses; itdoesnotisolateKL. ExistingmatchedNF64experiment
suppliesnarrowKLmotivation. Revisitbasedon64/96/100/128 andremainingtime.
Initialwindowtarget13:37UTC; pairmayneedsmalloverruntofinishbothendpoints
dependingactualruntime. No followup GPUlaunchyet. UseF64,notF128,formatched
contextwhenthepairedNFR64resultsareavailable.

08:58 UTC: F64 deep curves complete/synced1tbe2qdx. Regular CE
2.675040/4.536458/4.688323/4.765664; small K32 CE4.751938,
tail relative state change2.12e-6. No observed binary stabilization transition;
origin already settled, while training improves its poor feedback prediction.
First pass remains best. Current training continues unchanged to128. Physical
GPU0/1 follow-up launcher prepared, not launched; source and terminal-F128
binding guards checked. Independent helper review found no blocker (42 tests).
New NFR summary helper c720207 requires completed/audited pair, raw losses and
matched F64 context only; no weighted-objective quality comparison.

09:35 UTC: F96 complete, deep curves synced yg1f732m. Regular CE
2.680980/3.279866/3.328161/3.349725; small K32 CE3.294059; tail
delta at16 is6.87e-6 and at32 is1.51e-6. Settling faster than64 and
feedback prediction still improving, with no useful refinement over pass1.
Continue unchanged to100/128; no reason yet to change gate/pass design.
F64 exact context+curve32/64 summaries retainedGCS, see storage-receipt.
The optional NFR summary F64 mode guard was fixed against real metadata
(2ff12d8;14 focused tests); training/source208/215 remain unchanged.
