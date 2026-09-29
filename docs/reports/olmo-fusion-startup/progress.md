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
