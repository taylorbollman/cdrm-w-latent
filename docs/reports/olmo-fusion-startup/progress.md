# Overnight fusion startup progress

2026-09-29 08:05UTC. Active on`feat/olmo-fusion-startup`, basebaf33fb.
Read overnight-plan.md for user authorization and scope. Work window ends around
14:00UTC; user requested no review stops while useful technical work remains.

Preflight inside correct project container:2xH10080GB idle0MiB/0%,399GiB persistent
disk free. No model run started. Prior runtime helpers/protocols unchanged.
Runner/data implementation delegated; root building numerical probe; independent
reviewer assessing useful nonblocking readiness gaps. Compact checkpoint design
should preserve original backbone authority plus trained fusion/optimizer/cursor.

Next: freeze data and runner interfaces, CPU tests, disposableGPU preflight,
source freeze, checkpoint0/32/128 warmup and numerical observations. Record all
new commands, report identities andGCS receipt paths here as stages complete.

## Frozen implementation and initial execution

Runtimefb153a8; focused finalCPU39passed3.84s, existingtrain+data36 anddata14
scopes overlap. Data manifest2316978559b8db357c8d4adf706e41c94f809922171e8fb0ba50dc17ad66cbc0;
freshfixture5e55ee7bab67bcffb9fb01de48b9a971bae6ecbee59b8b7ebdf189c52600918f.
Data prepared indata-01, retained via retention/data-01.json.

preflight-01 (GPU0,W&B a36mugu3) FAILED before any optimizer update. Initial
compactcheckpoint0 saved/verified inGCS. Newtrainer forced math-SDPA during
forward but exited its context beforebackward; checkpoint recomputation then
selected another backend, causing tensor-metadata mismatch. No core model
change; runneragent making a narrow scope correction and regression. Preserve
failedstage/source snapshots, newattempt required. AllGPUstages exited before
anysourceedit, so source-integrity records remain valid.

initial-probe-01 (GPU1,W&B z8jjlhi2) PASSED fouraggregate/eightphysicalbackwards
in72.05s. Originalfixture exactly reproduces coldNF60.8698%backbone/65.2122%fusion
gradienterrors. FreshdevT16 gives24.3337%/21.9682%. It is smaller but remains
substantial; this is not merely an anomaly limited to the oldsyntheticfixture.
Probe already held backendcontext acrossbackward and is unaffected bytrainer
correction; no reason to rerun it. Final source/firstpass/RNGcontrols pass.

Independent resource ledger complete:33CPUtests, all8 native-sizecards,49sources;
T1024two-rankfixture524288valid vs540672allocatedpositions. NoGPU. Report/evidence
under.runtime/olmo-campaign-resource-ledger/ledger-01, retention/resource-ledger-01.json.
Read docs/reports/olmo-campaign-resource-ledger/results.md. Lifecyclefault-handling
helper/tests being implemented separately; no existingruntimefileschange.

## Corrected preflight and main training

Runtimec8c96bd freezes corrected backend context through backward; focused40CPU
tests passed3.91s (overlap with prior suites). preflight-02 passed two logical
updates plus exact replay,113.15s wall including construction/retention.
W&Bpuuc00h4. Update1/2 compute6.008/5.031s. All checkpoints0/1/2 retained and
verifiedGCS; update1 SHA09f4be2b4378a8f2da3dbdfffd0e239373ce3f6c38cb2d6607eaa93bcbf7dbb0.

fresh-resume-01 independently resumed update1 in a new process. Complete
fusion/Adam/scheduler/RNG/cursor boundary, all update metrics except measured
time/memory, input/noise pins and serialized update2 checkpoint bytes match
uninterrupted preflight-02 exactly. Independent comparison JSON in that stage.

Maintrain-01 launchedGPU0 at08:28UTC, max32 then strictresume to128. W&Bxyk0qmbq,
https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/xyk0qmbq .
Initial update0 retainedGCS. Compute estimate~12min for128 excludes data/hash/I/O;
provisionalwall15–20min may be revised with actual progression. No source edits
to frozen runtime. Maintrain firstthree updates finite.

GPU1 adapted-position-01 completed. Independent all-pass forward fingerprints
and every comparable oldAA gradient/geometry summary reproduce exactly. The
12.4366% record0/pass1 hidden discrepancy is0.659623% on19 union-supported
positions. Two terminal positions with zero incoming cotangent in both
precisions account for99.7455% of squared hidden error. This localizes the old
fixture spike; it does not establish general BF16 safety. Supplemental findings
being saved separately.

Independent CPU lifecycle helper completed eight actual two-rankGloo cases;
3focusedtests,126casechecks plus5reportchecks. No productionrunner integration
yet and noNCCL/dead-peer/hung-callback claims. NewT128devfixture probe and
conditional same-Adam FP32/BF16 update comparison being prepared separately.

## Update32 and longer cold fixture

At08:33UTC train-01 completed32updates in252.4s wall, including initial loading
and two checkpoint retentions. Its update32 SHA is
4b7f82b5c4cf3c4d12566227c7a31845a0874fc8e19963544a69fa358af9dc82.
GPU0 now runs train-02 via strictresume to128; GPU1 probe-32-01 completed96.6s.
Original backbone/fusion relativegradienterror1.2156%/2.1943%; freshshort
2.1168%/5.5207%. Absolute backboneerror falls to0.934/1.441; improvement is not
only denominator growth. No schedule change.

Newlong-data-01 exports four additionaldevprefixesT128 fromC4/CommonCrawl/Pes2o/
Reddit, independent oftrain andfreshshort. SHA830920f60c687f667baee7f7d6f137b521b22a35604f02e2a2e3b038586e55f9.
Longhelper36focusedCPUtests passed4.53s, preparation5.50s. ColdlongGPU1stage
long-cold-01 passedoperationalchecks in57.3s, but numericalbackboneerror144.02%,
fusion167.45%; cosine.3235/.2497. This reinforces need forlongpostwarm comparison.
No acceptancebudget change; precise128comparison stillpending.

New independent helpers underdevelopment: counterfactualsameAdam FP32/BF16
fusionupdate withzero-gradientmomentum/decaycontrol; packedT1024heldoutfixture;
actualrepeatedcampaignlifecycleloop withtinyDDPgraphstop/recoverychecks. All new
files stay underscripts/tests/docs because addinganycdrm/pretrained/*.py would
change active source inventories. Rootalone launchesGPUstages.

## Update128 saved and preliminary broader observations

train-02 completed128 withallintegritychecks;669.61s segmentwall, totaltrainsegments
922.0s(~15.4min).128uniqueupdates=1,048,576CEtargets, allfrozenstateunchanged.
EndpointSHA892ff2fdcdeec89e3008a16a12e91158250ebe05adfe0e9efce8f153409b8cfc,
GCSgeneration1790671431225622 verified. W&Btrain02=2v0bwsxf. Short128probe passed:
originalbackbone/fusion.823%/1.365%; fresh1.983%/3.663%.

long-128-01 FAILED beforebackward/import because livearm_contract rt_layers tuple
comparedwithJSONlist. SavedJSONcontracts exact andallothercoldguards passed.
Oldlonghelper remainsfrozen. Newsharedcontextprobe (runtime98fd9c2) canonicalizes
JSONcontractrepresentation withoutdroppingfields; usesunchangedstrictoldlongcold
loader andsourcepins. Focusedpackeddata/context26CPUtests3.79s. Corrected
long-128-02 completed80.3s: backbone1.6269%,fusion2.7115% vs cold144.02%/167.45%;
allcold/state/firstpass/source/RNGguards pass. ReportSHA
1f387076f276a604e08530f72e0850980db4c72275f9a94afac98a3d8787f7b9.

packed-data-01 frozenSHA4932410f9fd370d9dae20a1075bf2a191c642e1563eb8557a6fe02fd7e83a975:
2xB1/T1024,C4/CommonCrawl,2048inputs2046CE2040latent2032KL,6trueboundaries,
8uniquedocuments. Excludesall8priorprobe devdocs, alltrain/confirmation.
packed-cold-01 completed80.6s, reportSHAc1cfb4a0af1524872033828e8a802d26fb534543331785d11f30b42e9726ae02;
backbone19.970%/fusion25.406%. GPU1 nowpacked-128-01 startupmatch.
GPU0 counterfactualupdate-128-01 (exact3Adamcalls,no realtrainingprogress).
Newcomponentprobe readyforreview (62affectedCPUtests7.47s) andnewtiny2GPU
lifecycleloop ready (25CPU/Glootests10.20s), bothfrozenpendingrootlaunch.
No oldcore/sourcechanges; noBF16productionclearance. Longfixtureimprovement
supports continuingtheseexplicitboundedfollowups, notextending128warmup.

## 09:15 UTC: broader gradients and lifecycle recovery

The saved fusion128 checkpoint now has paired T128 component observations in
`components-128-01` (report SHA
`59e850755c25d3e585b1787b3b7a70ec67a43a8a7edfc97ceacd0d6f75a0653d`).
NF combined backbone/fusion gradient errors are 0.735%/0.888%; NFR CE gives
32.442%/47.421%, and NFR combined 12.059%/20.389%. Predictor errors remain
0.763%/0.820% in the combined cases. No numerical clearance for native RT.
New alpha0/.25 combined probes and an ordinary N-only packed T1024 CE baseline
are running separately on GPUs0/1. Both helpers have independent source review
and CPU checks; all old math/probe sources remain frozen.

The tiny captured two-GPU lifecycle reference and stop-after-update1 passed.
The first fresh-resume gate failed exact comparison and then required external
container teardown after its failed report and W&B sync. A second fresh resume
with observation-only recording completed all updates. Its model, gradients,
Adam, metrics, data, rank1 RNG and counters are bitwise identical; only rank0
Python RNG index differs by two draws, consistent with checkpoint retention
consuming randomness after checkpoint capture. The failed gate remains retained.
A new runner adapter will preserve RNG around retention and improve failure
teardown, followed by fresh reference/stop/resume/failure stages. No old source
will be silently patched.

Counterfactual Adam, packed128 and both independent audits have verified GCS
evidence under retention/update-128-01.json, packed-128-01.json,
components-128-01.json, warmup-audit-01.json and update-packed-audit-01.json.
The generic small-evidence retainer rejected the T16 preparation report's richer
source-metadata schema before uploading it; that index will be retained via an
explicit manifest/database upload. This is a retention-adapter issue, not a data
validation failure. Current committed code is6ffe5c6, pushed.

## 09:38 UTC: continuations complete, full-model updates active

Both fusion-only 16-update continuations128→144 completed, and a fresh BF16
process replayed136→144 exactly (all8 updates, final state/Adam/RNG/cursor,
common-FP32 evaluations). W&B FP32=17ty29w3, BF16=2l8pyh82, replay=7dogvmmi.
Final held-out CE5.137168103/5.137339194, gap0.00017109 nats per target.
Cumulative fusion-update vectors differ18.4122%, cosine0.98300; full weights
differ0.36445%, Adam first/second moments9.9456%/2.3627%. Read
continuation-results.md; do not conflate close losses with equal trajectories.
Final comparison55 checks passes, artifact continuation-comparison-02.

Guarded lifecycle reference/stop/resume/log-failure/failure-resume are complete.
Both fresh resumes exactly match their reference, deliberate log failure exits
1 promptly and retains completed update1. Root orchestration script and log are
in.runtime/olmo-campaign-lifecycle/run_guarded_remaining.py and corresponding
guarded-remaining-launcher.log. Read loop-results.md. Verified evidence and
index retention are complete. A scoped CPU/cloud operator recovery bundle is
being tested; it downloads exact generations and emits a conditional command,
without launching GPU or weakening runtime checks.

Root started nfr-updates-01 on GPU0 at09:32UTC, W&Brmz59xy0, frozenaebd17a.
Four matched full NFR updates per precision, fresh all-component Adam and
original token warmup; train indices144..147,32768CE targets/33638 valid inputs
per trajectory,78 physical backwards total. CPU snapshots/geometry are expensive
and not throughput data. At09:36 process alive, CPU206%,RSS38GB, GPU21GiB;
no completed row yet. One-hour external bound; checkpoint cadence10minutes at
paired boundaries, endpoint checkpoints about15GB each. Both other GPU jobs
are finished. Root will run abrupt-rank-exit reference/failure/fresh-resume once
both GPUs are free (new helper11 CPU tests, independently reviewed). This tests
loss of a rank before update2 backward after retained update1, not mid-Adam
rollback or an actual whole-VM power loss.

After full NFR endpoint4, prepare one saved BF16-state packedT1024 bridge:
FP32 sparse/prepared, BF16 sparse/prepared, BF16 captured replay. Five measured
configurations, separately counted setup/capture backwards, no optimizer. This
addresses the old sparse/prepared training-path qualification directly. No
core model or original frozen helper changes. Current committede0adc06.

## 09:49 UTC — full-model first pair and recovery assets

NFR first paired update completed with matched initial state/data/LR/counters.
Backbone raw/clipped/delta differences13.4465/13.4162/17.4687%; fusion delta26.5810%.
Both finite. Heavy CPU FP64 geometry and full model/Adam snapshot/hash work account
for much of elapsed time; these diagnostic timings are not throughput results.
Do not edit the frozen running helper. GPU0 active, GPU1 reserved idle until root
can run the three tiny abrupt-rank stages together. New root orchestration file
`run_rank_failure.py` is prepared but not launched.

Recovery-bundle helper now explicitly separates host checkout and container paths.
Fresh real cloud restore02 verified six exact-generation objects/23,851,490bytes,
88 source snapshots and86 corpus files; assets_verified_launch_pending only.
Manifest09b5dc49f94bab3af06343d1c0fc5486845ff5e122eb89661f5fe4a4f140c247.
First asset-only success with wrong outer command path is preserved. Agent is
assessing remaining concrete campaign readiness gaps without starting more jobs.
