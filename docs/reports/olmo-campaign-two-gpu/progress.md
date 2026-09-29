# Campaign two-H100 progress

2026-09-28: recovered current handoff and verified two idle H10080GB GPUs in
the required project container, NV18 connectivity, NCCL2.30.5, enlarged boot
disk and empty local SSD. Branch `feat/olmo-campaign-two-gpu` starts at `dde3240`.

Runner, distributed comparison probe and fresh-process restart harness are
being implemented independently. Root owns GPU launches. Initial NCCL sanity
stage is `.runtime/olmo-campaign-two-gpu/nccl-01`; see its launcher log/report.
No quality-training run has been launched. Planned protocol is recorded before
campaign GPU acceptance. Update this file and the results after each gate.

Initial NCCL sanity passed all5message sizes (4bytes through256MiB), exact sums;
W&Bci14andb. At100MiB median rank-max0.39675ms, at256MiB0.91291ms. These are
collective measurements, not model throughput.

Restored all86 finaltokenized objects fromGCS to the emptySSD, using recorded
generations and SHA256/size validation. Corpus verifier confirms28shards,
12,512unique documents and7,054,230tokens. Restore evidence is in
`.runtime/olmo-campaign-two-gpu/data-restore/`. No retokenization or packing.

2026-09-29, runtime `79fc75b`: scoped CPU suite60passes. Tiny actual NCCL eager
and CUDAgraph probes eachpass88gates acrossall8arms, M1/2/3, changing masks,
whollyempty rank and empty finalsync slots. Maximumrawgradient relativeL2
3.69222e-7; maximum3-updateparameter-update relativeL2 5.78202e-6. Both use
independent canonical objective asreference. W&Baxo1gal6/km5w3fog.

TinyNFR writephase passes; checkpoint saved afterfirstupdate and nextupdate
recorded onoriginal livegraph (W&Bedayyrx2). Checkpoint retained inGCS, then
downloaded to a freshSSDdirectory and SHA256/size verified; resumephase now
qualifies that restoredcopy in separatelylaunched torchrun processes.
RootGCSnamespace `.../olmo-two-gpu/20260928T235700Z/`; perstage receipts under
`.runtime/olmo-campaign-two-gpu/retention/`. Do notmutate published reports.

Tinyresume01 FAILED before modelmutation: safe `torch.load(weights_only=True)`
rejected `torch.torch_version.TorchVersion` saved in configuration.runtime.torch.
Shared `_plain` accepted this `str` subclass but returned it without casting to
builtinstr. Fixcanonicalscalar/key normalization rather than relaxing safeload;
add regression and repeat write/resume under newsourcepins. Oldwrite/checkpoint
and failedresume retained; oldpair is not restartqualification.

Runtime `0c77166`: metadatafix passes70scopedCPUtests; broadcampaign/checkpoint/
DDP suite389passes. Tinywrite02 and cloud-restored fresh tinyresume02 PASS,
including bitwise next gradients/Adam/model/RNG/cursor; W&Bo34f5ijq/7du9uw9g.

Pretrained eager01: ordinaryB passesall11gates; NFR fails independentcanonical
rawgradient atupdate1 (globalL2 .0340224273) despitepassinglosses. Read
[localization plan](qualification-plan.md). Preservefailure. Addpreparedlocal
reference toseparate dense/sparseBF16 differencefromDDP; no relaxedbudgets and
no generalprecisionclearance. Capacity waitsforpreparedexecution/restart gates.

Runtime `61d6d2b`: prepared-reference eager NFR passes all11operationalgates.
The separatelyrecorded canonical-vs-prepared local qualification reproduces
exactly .0340224273 gradientL2 BEFORE DDP; the discrepancy is not introduced by
distributed synchronization in this fixture. Independentreference status stays
FAILED, operationalstatus PASS. Graphed B/NFR comparison follows, then one
bounded actualpretrained FP32 loss-layout diagnostic and fullcheckpointrestart.

Pretrained preparedgraph01 passes22operationalgates (B/NFR,3Adamupdates each);
NFRindependentBF16qualification remainsfailed. W&B7kkx637i. FixedstatefullFP32
sparse/dense diagnostic passes at7.38198e-7 gradientL2; W&B5gv7m5wl. Runtime
`28d6a93`. Fullpretrained NFR writephase is underway in `pretrained-write-01`;
checkpoint SSDpath `/mnt/localssd/cdrm-checkpoints/campaign-two-gpu/pretrained-01`.
Aftercompletion retain+restorefromGCS beforefreshresume. No qualitytraining.

Fullpretrained write01 passed in187.75s. Canonicalstate15,214,756,865bytes;
SHA256 `bd06a69aa9d53bf5da074afd86b8167083f69c142ee7b89935fc50b6384fef08`.
Uploaded under `campaign-pretrained-write-01/checkpoint/`, then generation-pinned
downloaded to `pretrained-01-restored` and bothfiles SHAverified/manifestinspected.
Freshprocess `pretrained-resume-01` is running against this actualcloudcopy.
FinalCPUregression445tests pass60.03s; includeshistoricalDDPgraphscopes.

Fullpretrained freshresume01 PASS in129.57s (W&Bu3pnp0xi).
Bothranks match original-livegraph nextupdate bitwise: inputs/rawgradients,
metrics/model/Adam/counters/cursor, RNGstates and actualRNGdraws. Fullrestart
qualification is B2/T16, sameworldsize/hardware/runtime; notT1024coldcapacity.
NextboundedT1024capacity changesordering to prepareDDP→3eagerAdamupdates→
capturewithresidentAdam→untimedreplay/discard→5timedupdates. This includes
actualmomentmemory (~9.45GiB/rank) duringcapture; coldDDPconstructatT1024 remains
futurequalification. No core math/runner changes required.

2026-09-29 00:44 UTC: T1024 capacity B8/rank, two microbatches, passed all
stages in 524.64 s. Five measured complete updates: 3,428.23 global valid input
tokens/s, median 9.564 s/update, peak reserved 50.35 GiB and final sampled free
22.97 GiB per GPU. W&B kkmsgtfg. Launcher log copied after process exit;
verified retention receipt is `retention/capacity-b8-m2-01.json`.
B16/rank is running in a fresh process, bounded to 900 s; session evidence is
`capacity-b16-m2-01/` plus its external launcher log. No other GPU jobs.

00:54 UTC: B16 passes all 12 stages, 4,969.84 input tokens/s, median 13.187 s,
69.97 GiB peak reserved and 3.30 GiB final sampled free/GPU. W&B cbl8xn7l.
Too little headroom for recommendation; skip B32. Before any next timing,
amend protocol/CLI to allow a single B12 candidate with unchanged setup/math.
This is a measured middle option, not an extrapolated memory claim.
