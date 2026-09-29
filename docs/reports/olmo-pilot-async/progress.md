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
