# Ordinary OLMo T2048: active work / interruption handoff

2026-09-28. User authorized one-H100 ordinary T2048 throughput/memory versus
savedT512, with and without FA4 atT2048. User explicitly says T512 need not be
rerun. Frozen protocol in this directory; branchfeat/olmo-ordinary-long-context
based on mergedPR31 (739080c). No qualitytraining, RT, FBT or NextLat scope.

Preflight confirms one idle H10080HBM3 (81559MiB), driver580.178.04,
PyTorch2.13.0a0+8145d630e8.nv26.06/CUDA13.3, FA4 4.0.0b20,
CuTE DSL4.6.0.dev0 and installed Dao2.7.4.post1. These match the saved single-GPU
T512B128 reference44,041.286tokens/s. Native model checkpoint manifest verified.
Use CDRM_FLASH_ATTENTION_SOURCE=installed to avoid vendor shadowing. No install.
Host persistent disk~19GiBfree; disposable throughput runs need nofullcheckpoint.

Root owns all GPU work. Runtimeedf3d0b is frozen/committed and pushed;111focused
CPUtests pass in3.23s (67installedJITwarnings). FirstGPUstagefa4-check-b2-01 finished8updates: strictloss-only failure retained,
all5operationalchecks pass; globalgradientL2.010833/outputL2.006477.
AbsoluteCEdifference7.2306e-5nats/target. Primaryqueue rootexec58032 runs
SDPAB16,FA4B16,SDPAB32,FA4B32 sequentially and retains each stage in GCS.
Inspect reports/processes/launcher logs before restarting.
Agentordinary_long_context_audit is preparing new.runtime audittools.
No T512runqueued. Source changes require a new revision/directory.

Queue: B2T2048 FA4-versusSDPA bounded numerical check (sameDao/compiled/fused
settings), then SDPA/FA4B16andB32. OptionalB48/B64onlyifmeasurementjustifies;
repeatselectedpairreverseorder. Preservefinite-onlynumericalfailures while
completingoperationaldiagnosis; stopforstructural/nonfinite/ownparityfailures.
Use existing11warmups,3prep+5timedupdates, CPUreferencesbeforecapture,
releasegraphbeforeterminaleagerchecks. No broadnumericcampaign.

Evidence `.runtime/olmo-ordinary-long-context/`; logs/preflight.log confirms
containerandinstalleddependencies. GCSexistingper-stagehelperprefix must begin
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/`; use a fresh timestamp
for this milestone:20260928T163200Z. CPUretentioncontainer with env-uGOOGLE_APPLICATION_CREDENTIALS
selectsworkingmountedADC. W&Bpretrained-fbt-rt-nextlat/newlongcontextgroup.
