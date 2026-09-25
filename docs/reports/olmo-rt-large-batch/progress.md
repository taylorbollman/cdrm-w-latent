# Native RT large-batch milestone: in progress

Updated 2026-09-24 after interruption recovery. This is not the final assessment.
Branch `feat/native-rt-large-batch`; current runtime code `508920c`; report helper commit `3fdad1e` has identical
runtime sources and is recorded by subsequently launched processes.

Completed actual-checkpoint integration:

- `rt-optimized-b8-check-01`, runtime `56f2dd1`: Dao ordinary RoPE plus compiled
  SwiGLU/fused Adam has a CE-only numerical miss. CE difference 0.0006894
  nats/target, reference about 2.12, relative 0.0003252 versus budget 1e-5.
  Global gradient L2 0.0042023; all gradient/output budgets pass. All five own
  operational gates pass, including exact three-eager/three-graph Adam updates.
  Failed loss remains failed. Six physical updates. W&B `v6znjt7h`.
- `rt-compiled-native-b8-check-01`, runtime `ff35047`: all six gates pass;
  forward outputs/losses bitwise versus control, raw-gradient L2 0.00290366,
  exact graph/full-Adam checks. Six physical updates. W&B `xhf63nea`.
- `combined-compiled-native-b8-check-01`, runtime `ff35047`: all six gates pass;
  forward outputs/losses bitwise, raw-gradient L2 0.00544145, exact graph/full-Adam
  checks. Six physical updates. W&B `y9llge5g`.

Use `compiled-native` as the primary capacity candidate: native ordinary RoPE,
rounded compiled ordinary SwiGLU, fused Adam; native RT optimized tiles/recompute
remain unchanged. The Dao attribution is isolated by these arms without changing
the acceptance thresholds. Do not transfer this finding into a claim that prior
ordinary-only Dao results are invalid. Conditional memory candidate is
`fa4-native`, differing only in ordinary attention from `compiled-native`.

At recovery the H100 was idle and all three reports had finished. The first
capacity queue now runs sequential fresh Python processes: RT B64 control, RT
B64 compiled-native, RT B128 compiled-native, default transient cleanup off.
Outputs are `.runtime/olmo-rt-large-batch/<run-name>/report.json`; shell logs are
under `.runtime/olmo-rt-large-batch/logs/`. Inspect reports and the container GPU
before launching replacements; an interrupted conversation may leave jobs alive.
Sources/protocol are frozen per run; never change them while a run is active.

Next: adaptively increase RT physical batch, inspect phase memory, compare the
explicit cleanup option if needed, then combined capacity and conditional FA4
near a failed boundary. Repeat best/neighbor points, profile a useful operating
point, summarize/plot/retain all successes and failures. Do not claim a validation
OOM beside a live graph pool proves a replay-only training limit.

Current tests: 284 distinct scoped runtime/harness tests (129 static/capture,
119 ordinary/RT routing and 36 new harness); 46 of those rerun after the reporting
endpoint synchronization/partial-counter change. Evidence helper initially56
tests, with independent review in progress. No new long quality run/checkpoint.
Each bounded correctness run is six updates; capacity eight, or nine if profiled.

First capacity evidence (single measurements; repeats pending):

| Run | tokens/s | Setup allocated / reserved peak GiB | Status |
| --- | ---: | ---: | --- |
| RT control B64 | 22,319.5 | 32.076 / 50.311 | passed8updates |
| RT compiled-native B64 | 23,402.8 | 32.077 / 52.225 | passed8updates |
| RT compiled-native B128 | 28,010.0 | 45.746 / 78.059 | passed8updates, tight setup |

B128 ended with only1.795GiB free after eager-with-live-graph validation; do not
call this a comfortable point yet. B192 default setup is running next. If an OOM
is confined to validation, a CPU-reference-before-capture helper is being prepared
without changing the active runtime. Optional cleanup may need to include a
pre-warmup phase as well as the existing post-warmup phase. Native RT full-sequence
MLP backward and saved FP32 layer inputs remain genuine memory costs.

B192 default setup failed during validation_initial after capture succeeded.
It had three successful preparation updates; the failed allocation requested
1.5GiB while42.6GiB was held in private graph pools. This is retained as a
validation coexistence OOM, not an intrinsic replay limit.

Runtime a2bc709 adds optional validation-order before-capture and extends cleanup
to both before and after side-stream warmup. It preserves CPU all-gradient refs,
checks graph replay exactly, then releases graph/results before terminal eager
validation. CPU tests include37validation-helper cases,16cleanup cases and142
overlapping integration/regression tests. GPU queue: B128/B192/B256 compiled-native
with --release-transient-cache --validation-order before-capture, stopping on
failure. Report helper changes are separate and may create later Git revisions
with identical frozen runtime source hashes. Conditional FA4 and combined
capacity remain outstanding.

The first corrected B128 run passed all five gates/8updates at28,008.55tokens/s
versus28,010.05 in old ordering. Setup allocated peak remains45.746GiB, but
setup reserved falls78.059→48.377GiB and device free1.795→29.684GiB. This is a
preparation/validation improvement, not a reduction in the model's live tensor
requirements. Exact pre-capture-reference and terminal graph-release checks pass.
B192 corrected run is active; B256 follows only if it passes.

## Capacity continuation, 18:42 UTC

B192 with reordered validation passed all five gates and eight full updates:29,743.65 tokens/s,setup peak allocated59.417GiB/reserved62.926GiB,current reserved60.918GiB/device free17.145GiB.

B256 passed dispatch,three eager preparation updates,two CPU eager references and all warmup backward calls,then OOM during graph capture in native RT reconstructed weight VJP (`olmo_tiled.py:342`). Peak allocated before capture reached73.088GiB and reserved78.227GiB. Capture attempted another2GiB with1.70GiB free. This is distinct from the earlier B192 validation coexistence failure; no successful B256 replay or throughput claimed.

Combined controlB64 and compiled-nativeB64/B128 now queued sequentially using reordered validation/cache cleanup. Next: conditional ordinary FA4 integration and matched boundary memory, then repeat/profile useful points.

## Combined B64 pair

Control passed all five gates/eight updates at11,198.02tokens/s,setupreserved40.871GiB. Compiled-native passed at11,707.59tokens/s (+4.55%),setupreserved41.713GiB. Both usebefore-capture validation/cleanup. B128 candidate isnext.

## 2026-09-25 plan update after interruption

Combined compiled-nativeB128 finished successfully:all5gates,8updates,12,410.94inputtokens/s,5.28050seconds/update,setupallocated58.095GiB/reserved65.971GiB,currentreserved65.113GiB,free12.877GiB. All13reportsfinished (10pass/1numericfail/2OOM),88totalupdates. No benchmarkprocess is running. FA4/repeats/profiles/finalretention remainoutstanding; preliminaryresults coveronlyfirst10reports.

User requested an updated plan for remainingoneGPU work and transitiontotwoGPUs. See ../../native-rt-single-to-two-gpu-plan.md and currenthandoff. No newGPUworklaunchedfor this planningrequest.

## 2026-09-25 authorized closeout

User approved the single-to-two-GPU plan. RT FA4-native B8 integration completed with a retained CE-only compatibility failure (relativeCE0.000573327; globalgradientL2=0.00455068, all output/gradientbudgets pass). All5 own operationalgates pass,6updates. QualifiedFA4capacity is authorized byprotocol; noautomaticadoption. CombinedFA4B8 check runs next. Rootbenchmark sources remain frozen at a2bc709 while adapter and graphrecovery work proceed in isolated worktrees.

Combined FA4 B8 completed6updates:own5operationalgates pass, butcompatibility fails onlosses AND gradients (globalL2=0.04683667 vs0.015625budget). KeepFlashSDPA workingbackend. FA4capacity remainsqualifiedmeasurementonly,perprotocol. RT FA4B192/B256 conditionalcapacity queue started after bothchecks completed.

FA4 RT B192 passesowncapacitygates at29,888.96tokens/s (+0.49%singlepair), withidenticalsetupallocated59.417GiB/reserved62.926GiB and0.750GiB MORE currentreserved61.668GiB. RT FA4B256 fails samecaptureVJP allocation2GiB asSDPA after3eagerupdates/11totalwarmupbackwards. CombinedB8 FA4 additionallymissespass1outputL2budget (0.0182769),67/71gradienttensorL2budgets and15maxbudgets; noadoption.

GPUcloseoutqueue: combinedFA4B128; RTcompilednativeB192repeatwithprofile; RTB128repeat; combinedB128repeatwithprofile; combinedB64repeat. Allbefore-capture+cleanup. Newadaptercommit090ff544 and graphrecoverycommit6352410 existinisolatedworktreesonly; cherrypickafterfrozenGPUqueueends.

Combined FA4 B128 timing:12,503.52tokens/s (+0.75%singlepair),peakallocated58.095GiB unchanged,setupreserved64.113GiB vsSDPA65.971,currentreserved64.113vs65.113GiB (1GiBless). Compatibilityremainsfailed; keepSDPA. No B160refinementneededforworkingbackend: modestmemorydifference doesnotjustifyadoptingqualifiedFA4. ReverseSDPArepeats/profilesfollow.
