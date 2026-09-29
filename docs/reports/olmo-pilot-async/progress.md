# Background checkpoint progress

2026-09-29: user authorizes background upload/verification after completed SSD
snapshot and proceeding with sensible next steps. Branchfeat/olmo-pilot-async,
basefd21eca. GPUpreflight insidecontainer: see.runtime/olmo-pilot-async/preflight.log.
No GPUtraining launched yet. Newengine andexecutor retain originalmath/save
callbacks; newworker/loop have separate CPUtests. Existing192runtimepins frozen.

Design: onebackgroundthread owns purefile validation/publication/pruning;
cloudSDK onlyin freshCPUchild withnoCUDAvisibility/distributedenv. No shared
preserve_local_rng context whiletraining; thiswouldrewindtrainingRNG. Drain
before nextsave; pollsafeupdateboundaries; normalterminaldrains; unknownfailure
abortsworker withoutnewcollectives. Fulloldverificationretained. Jobclock starts
on localsubmit, cloudcompletedcounterseparate. Nativeworker childtimeout480sec;
localhash/fsync hasnoharddeadline. User'sriskacceptanceisrollbacktopreviouscloud.

Rootownsnewengine/entrypoint/protocol/audit/GPUlaunch. Agentsownnewasync loop,
worker andtheirtests; thirdagentpreparescostedpilotplan andCPUdeclarations.
Pilotproposalcommon524288inputs/update,128ceiling/firststop32; sharedfusion128
NF/NFR vs originalBcontrol, separateancestry. Thismilestonecancompleteasync
acceptance andnativeaccumulation/overlap beforelaunching32-updatecohort.
Heavyclipping andworselaterpasses remainopencriteria, notforgotten/cleared.


Runtime frozen85e5f78:200sourcepins, canonicaldigest
d84006ede4e5ad488880da2af387355ca82fc76d186cf87ba16cb007d2279fc5.
76CPUtests pass in6.00s (twoGoogleSDKfuturedependencywarnings); old192pinsunchanged.
PR51draft open. CPU/declarationevidence retained. Tinyblocking-01 complete40.64s,
W&B1d1nr010; tinyasync-01 complete25.12s,W&B2aq814on, allcheckpoints0/1/2/3
verified. DirectJSONinspection finds exactall3input/gradient/update/finalboundaries;
independentauditor/testsstillbeingcompleted. Timecomparisonisnotnativebenchmark.

Daemon restarted afterasynccompleted. Its outerlaunch-resultwasnotwritten, so
we recordcompletedreport/W&Bsync/idleGPUs ratherthan inventprocessexitcode. Both
H100sverified0MiB/0% insidecontainerafterrestart. No rerunofcompletedfixture.
ContinuingCPUcloudrestoreofasynccheckpoint2 thenfreshprocessupdate3. Allsources
recheckedunchanged. Separate4updateNFR524288batchnative declarationCPUpreparation
restartedaftercheckingitwasnotyetcreated. Nolearningcohortlaunched.


21:19UTC: freshcloudrestore2/update3completedexactly (directallgradient/state
comparison); W&Bp3r7luod synced. Tinytransportauditor preliminary2379checksPASS;
mutationtests/pinnedCLIreportsforthcoming (separatefromfrozenruntime). Native
NFRaccumulatedfixture declaration/preflightpassed, sourceinventoryunchanged.
Launchednative-nfr12-accum-01:4updates,T1024,B12/rank,524288inputs/update,22slots,
fusion128/freshAdam,checkpoint0/2/4 plus600sec,commonFP32dev65536at4. Timeout2700s.
Report .runtime/olmo-pilot-async/native-nfr12-accum-01/report.json. RootownsGPU;
do not launchanother untilcompleted. Expected~30min, checkpointsretainedbackground.
Thisis2.097M-inputexecution/overlapfixture, notstartof32-updatecohort.


Tiny independent final audits pass: transport2847checks
(SHA4fbe5234396892fc9436cb5d1c43993a4b7a1ce25e5904f36a0ab52dcef5616f),
resume2427checks (SHAbd14f9c5aa4451ef7452bd998af6cb0293d14a0b040a21f2b9fd525ae52ad7eb).
Auditor60newCPUtests pass1.99s; separatefrom76runtime tests, total136distinct.
Newauditorcommitae3b7c3; acceptedruntime200pinsunchanged. Bothauditstagesretained.
NativefixtureW&Bosqidkgt; initialization/graphpreparation underway. Openadaptation
issue separatelyrecorded inopen-issues.md andpilotreviewcriteria.

21:46 UTC: all four native updates are finite; 2,097,152 inputs processed.
Graph preparation took 448.69 seconds; selected compute plus materialization
rate is 3,567.69 inputs/s. Reserved memory is 59.06 GiB/GPU, sampled free
12.78 GiB. Populated local save regions take 76.18/76.41 seconds. Checkpoint 2
is verified in GCS after 346.45 seconds of background work overlapping updates
3/4; checkpoint 4 is locally complete and its worker is draining. Do not launch
another GPU job before checking completion. Final dev evaluation took225.45s,
preserved training state exactly, and returned CE3.18058/7.87717/7.71814/7.74559.
Norms211.12/76.50/73.80/52.42 remain heavily clipped. Open issue and pilot cost
estimates updated; no useful-refinement or BF16-clearance claim. Native final
summary/retention/inventory and PR closeout remain pending.

Native run completed with exit0, reportstatus completed_plan, W&B synced and
both GPUs verified idle. Final checkpoint4 is cloud-verified; no pending worker.
Runtime200pins unchanged. Final reportSHA
f2e4065c7167cad0cfa22f24d3803bcdd286f9792554b0bd9f126f0a7ed9e05d;
independent summarySHA86142ab9f44b6cd70b6da61d378b4bdffd49980e0306d940c233fe0cde154667.
Stage1886.27s; terminal background364.83s, loop blocking364.69s. Scope details
are inresults.md. Native stage and summary retained with verified receipts.

W&B charts1ofs0x3r synced. Optional immediate metadata readback of original
runosqidkgt failed after the requested summary update. Later independent
read-only confirmation found all four requested fields correct and verified
the existing chart run. Consistent with delayed read visibility; no training
or chart upload rerun. Firstfailure and recovery stages retained. RecoverySHA
4c6866684312fcef54f973d4e44391dbf3530c14c53dac510113e61377ec7d92.

Inventory passes:16small stage receipts/32objects plus12checkpoints/24objects,
total35,674,627,872bytes. Original individual publication receipts are preserved
under neutral snapshot names with mappings. InventorySHA
2b8d949260f64df6a6691c55de9dc74df937b153fab954d600ae09a8bba1a1ee.
Own/latercloseout receipts excluded fromcounts; no newcorpus. Documentation and
pilot cost updated. PR ready/merge and final closeout receipt follow below.

PR51 merged successfully at `039fa96b82027d2a945ccb5bf4db268e1338ea6f`, after
ready state, clean mergeability and exact-head check against
`141e0709d4659834dd76128ceb2e277e5ebc03aa`. GitHub had no configured check runs
or commit statuses; the local CPU/GPU acceptance is recorded in the ledger.
Main was fast-forwarded. No model/runtime source changed after freeze.

Final closeout01 contains 2,303 members (93,282,298 bytes before manifest).
Its report SHA256 is
`d86bc5d2bba9f21f2a9c6368fcd17ac40324d11db45aaa6465b3c50b23542285`;
verified retention receipt SHA256 is
`0fdfa1b6ea30969c3aa6ac4e7577b880a7acedf2ed5af70fe1214e88bc84b8d4`.
Prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T204500Z/pilot-async-closeout-01`.
The final administrative receipt, produced after this documentation commit,
is `.runtime/olmo-pilot-async/retention/closeout-admin-01.json`; it retains the
merge response, prior closeout receipt and final documentation snapshot.

Both GPUs are idle, no job is queued, and the proposed learning cohort remains
unlaunched. Next review: the bounded B/NF/NFR first32 segment and its roughly
two-hour NFR cost, with clipping/later-pass CE still open. Do not restart the
completed four-update diagnostic or silently turn it into the cohort origin.
