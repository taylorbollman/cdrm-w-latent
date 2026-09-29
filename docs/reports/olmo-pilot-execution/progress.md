# Ordered pilot execution progress

Current status: COMPLETE, PR50 merged `806c1be`; final administration on main. All three
native capacity runs and W&B synchronization finished. No active or queued GPU
run. Final inventory includes152objects/101,826,528,826bytes; see storage-receipt.
The entries below preserve the work timeline; earlier pending states are historical.

2026-09-29 initial entry: `feat/olmo-pilot-execution`, starting at `3d5c949`.
The user authorized ordered-runner integration, two-GPU restart/evaluation
acceptance, then bounded native T1024 capacity measurements. No quality
campaign or precision sweep is planned. Both H10080GB devices were verified
inside the required project container and idle at entry. SSD has ~1.4TiB free;
boot has ~91GiB free. All new large checkpoints belong on SSD and must be
retained to GCS; evidence stays under `.runtime/olmo-pilot-execution/`.

Root owns `olmo_pilot_execute.py`, runtime launch, native declarations, tracking,
retention and closeout. Agent pilot_contract owns new declaration/resolution
module/tests; pilot_eval owns named dev evaluation module/tests; pilot_acceptance
owns tiny fixture and independent audit modules/tests. No pre-existing runtime,
model, vendor or test file may change. Shared engine is PR48's unchanged SSD
engine. Native construction reuses its accepted model constructor. Source pins
must be frozen before GPU acceptance and kept stable through resume.

New protocol: [protocol.md](protocol.md). Tiny schema is
`olmo-pilot-execution-tiny-acceptance-v1`; native schema is separately versioned.
Tiny T16 / B2 per rank / NFR / three updates of80inputs includes true document
boundaries and uneven rank slots. Compare reference with named FP32 dev panels,
then stop2/cloud restore/resume3 and terminal evaluation-only acceptance.

Evidence namespace for this milestone:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T190649Z`.
Checkpoint namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T190649Z/pilot-execution`.
No GPU run started yet. Runtime/CPU code and test development are ongoing.


19:24 UTC: runtime is frozen at `be74dde`, 192 source pins with canonical digest
`4d6f9fd0d3e61e45e13236d2c2804913b9a4790275b3e71c37657df2da83a7cf`.
PR50 is open in draft. New source/test suite passed149distincttests in58.75s.
CPU evidence retained in `pilot-execution-cpu`; synthetic fixture corpus retained
separately with `olmo_document_retain.py`. Native B32/NFR12/B64/NFR8 CPU resolution
completed in native-declarations-01; these are not launched yet.

Tiny reference and evaluated runs completed. Insertion audit passed2063checks,
all3updates rawgradients/completeboundaries exact on bothranks. Evaluated dev-main
and books separately at update2 with independent B1/rank FP32 evaluation. Tiny
stop2 completed/retained; `tiny-restore2-01` is restoring its exact cloudgenerations
for resume3 and terminalstop2 checks. AllGPUlaunches are rootowned and sequential;
inspect launch-result JSON and report before starting another. Commands and logs
are in .runtime/olmo-pilot-execution; launch.py enforces external timeout.

Small additional adapters were needed for the newidentity: checkpoint storage
subclass overrides only two metadata validation methods; restore uses explicit
newidentity validators with unchanged streaming and bytepublication helpers.
Old outer cursor/checkpoint schemas remain shared, ordered inneridentity is new.
No preexisting source/test changes. Follow sourcefreeze through all resumes.


19:31 UTC: tiny integration/recovery acceptance is complete. Runtime stayed frozen;
a separate auditorv2/test commit `2d9fb70` adds27CPUtests for wall-time exclusion and
empty terminal segments. Original v1 false failures preserved. Final resume audit
passes2136checks, terminal1914. Both exact, including update3 gradients/finalstate.
Native B32 is running via native-b32-01 (torchrun2, BF16 graph,8updates, lean);
checkpoint0/4/8 retained automatically. Next NFR12; conditional largerbase orNFR8
only after capacity review. No concurrent GPU job. Analysis helper
summarize_capacity.py validates original192sourcepins independently; its timing
regions exclude logging/health/checkpoint/eval costs and must be labeled so.


19:49 UTC: native-b32-01 completed all8updates, final namedFP32 evaluation and
verified GCScheckpoint0/4/8; W&Boh4sdigb synced. Independentcapacity-b32-01
summary passed, SHA14e8a1fa9a4c3ea2a8b8d09f8ab118a2f4adbf98ab91df47c2cd6773cf47fb0b.
Updates4–8:69,494.1validinputs/s in compute regions,67,278.9 includingrecorded
materialization;42.402GiBreserved/35.215GiBsamplefree. Fullsegment1049.27s is
checkpoint-dominated, not a production throughput estimate. Native-nfr12-01
started (session99090), W&Bx8u0vz4c, no concurrentGPUjob. B64 remains planned
bounded followup; NFR8 onlyif12 lackscomfortablemargin. Runtime192pins unchanged.
CPUauditorv2 evidence retained; nativeB32 andsummary retention queue running.


20:05 UTC: native NFR12 finished442.81s graphpreparation with exactinitialboundary,
then4finite updates; checkpoint4 is being retained. Reserved59.0625GiB with
12.7833GiBsamplefree onbothGPUs afterAdamallocation. Earlycompute-regions~3.7–3.8k
inputs/s; heavyclipping observed(norm222.5 down45 at4), notnewBF16failureorquality
claim. FinalFP32dev stillpending. External1800sec deadline around20:18UTC; current
progress shouldcompletebeforeit, but inspect actualfinalreport. B64nextonlyafter
thisprocesscloses. W&Bx8u0vz4c. Newcheckpoint-cost.md distinguishes selectedsave
regions fromfullwalland recommends600s+terminalcadence forlaterpilot (clockresets
afterverifiedpublication). Runtime192sources stillunchanged. Allworkpushed;
retention/inventory/trackinghelpers preparedbutfinalrollupawaitsallnative runs.


20:17 UTC: native-nfr12-01 completed8updates/finalFP32eval/verifiedcheckpoints0,4,8;
W&Bx8u0vz4c synced. Stage1613.63s, no timeout. Independentcapacity-b32-nfr12-01
summary passes(SHA41de9b41f88a5307068684a36a464f41e7bc3f5f0ed3df01d4ecdf2f88b8766b).
NFR12:3,758.9compute-region inputs/s,3,578.0withmaterialization;59.06GiBreserved,
12.78GiBsamplefree,FP32eval30.47s. Allfinite, substantialclipping remains(norm24.14
at8), laterdevpassesworse thanfirst; notquality/refinementwin. NFR8notneeded.
Native-b64-01 nowrunning(session56203), W&B72bvkgwm; finalplannedGPUfixture.
Devprefixcoverage CPUauditretained:5k/16k/65k prefixescover4/6/7of9strata;allomit
books/wiki. Sourceoutcomesnotusedforselection. Inventoryhelper nowusesneutral
receipt filenamesbecausegenericretainer excludescheckpoint-publications dirs;
fullpublicationrecords alreadyembedded inreports, individualbytes willbeinfinal
inventoryarchive. No model/runtime/sourcefreezechanges. Preparefinalcapacity
summary,W&Bsummary,inventory,docs/PRcloseoutafterB64completes. Allpriorworkpushed.


20:35 UTC: final native-b64-01 completed all8updates, finalFP32evaluation,
checkpoint0/4/8 verified, W&B72bvkgwm synced. Fullstage1052.09s. No furtherGPUjob.
Final three-case capacity summary SHA507e43a97436ecaa667cb8dd0ab580bd65e85092c7ca00a986e5df7fc456e0ff;
all192runtimepins unchanged. B64:71,268.9compute-region inputs/s,69,001.2including
materialization,60.75GiBreserved,16.865GiBsamplefree. About2.6% fasterthanB32 for
18.35GiBadditionalreserved memory; B32remainsordinarydefault. NFR12comfortable.
W&Baggregate48jhxju3 synced, charts/regions explicitlyscoped. NativeB64 andfinal
summaryretained; finishingtrackingretention/inventory/closeout. Noqualitycampaign.


Final acceptance/retention closeout (2026-09-29): independent read-only report
review found no substantive blocker. Historical model/kernel/engine/test sources
are unchanged. All149+27distinctnewCPUtests passed; no redundantGPUrerun. Both
H100s verifiedidle/0MiB/0% insidecontainer; final-gpu-idle-01.log records this.
Inventory report SHAa31fec8d479ee31a930e84f2718c73c0ec8842e5befb2fa1b58ec51f85d9bdce;
retention receipt SHA946d8291544de5f9a6cbde6f939a1944f6e499cf112cb8abee875267fcc35e25.
Closeout archive contains2,501members with22explicit safe-name mappings;
117,550,865rawbytes before its manifest. It preserves metadata/source/docs,
including individual checkpoint receipts under neutral/safe names; large state
bytes and synthetic corpus were retained separately. Closeout report SHA
7ea9ddf64ad9140a53b739467ae028556e25529a95be1b990e2735f86de56bb4.
Closeout receipt SHA345c84c75f6fc5cf24ff11ed803ad0793424c17b2fdd24e5bffc94fd8c1cf89e.
Archive `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T190649Z/pilot-execution-closeout-01/evidence.tar.gz`
generation`1790714283791758`, SHA`73e47d68a0027dc1a8332405b60ad50c79125dea5225b2d7ce2475100eba4a9e`.
Full readback/server verification passed. Own and later administration receipts
are outside earlier inventory/closeout snapshots. No learning run queued.


PR50 merged as806c1bec5c39aabec6b0bdaa81eb8aa2cfbd14be fromexacthead
211fdc359a75e98481d6fdfc4d46668d1d34bdd6 afterready/mergeable/clean confirmation.
No configured CIchecks were reported; local/container tests and GPUacceptance
are the validation evidence above. All192runtime source hashes recheckedbefore
merge. Localmain fast-forwarded; no additional GPUjobs or numericalchanges.
Finalpublication snapshot retains merge response, PRbody, closeout/inventory
receipts and these finaldocs; its own receipt is recorded below afterverification.

Final publication verified: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T190649Z/pilot-execution-publication-01/evidence.tar.gz`
generation`1790714412209957`, SHA`4a7871edd4cb2fe3d71051c4c579ddead51f5762f49e3860c311908654b6377d`.
Receipt SHA`2d2678818149854054e8374afe85bd36e9abf04558be661b84932bce62cb0788`; 18archivedmembers.
The generic filter omits checkpoint-cost.md from this small administrative
archive; its identical final bytes are retained under the explicit safe-name
mapping in closeout-01. Final administration receipt and this annotation are
committed onmain, outside earlier retained snapshots. Allselectedworkcomplete.
