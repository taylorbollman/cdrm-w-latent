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
