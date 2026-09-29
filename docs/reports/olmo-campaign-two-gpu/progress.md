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
