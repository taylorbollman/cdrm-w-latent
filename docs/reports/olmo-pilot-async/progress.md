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
