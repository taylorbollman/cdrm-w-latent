# Progress

2026-09-30 15:24 UTC: began the user-authorized endpoint probes and conditional
KL0.1 NFR64→128 milestone. Both H100s verified idle in the required project
container; no prior queue/training process remains. The completed NFR64 pair,
its populated checkpoints and cloud receipts are available. SSD has about
1 TiB free; persistent boot disk about 88 GiB. Large new states remain on SSD
with verified cloud retention.

Branch: `feat/olmo-nfr-stability-128`, from main `4b15854`. Implementation is
being prepared in separate, explicit diagnostic and continuation scopes.
See [protocol.md](protocol.md). GPU execution has not yet started.

15:39 UTC: both saved-state endpoint probes launched in isolated one-GPU
containers after 44 focused CPU tests, authenticated real checkpoint preflights
and independent review. Control uses physical GPU0, reduced uses GPU1.
Host sessions 2942/66539; read-only monitor 25914. Both are online:
[KL1](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/y8xlyhtw),
[KL0.1](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/sf5dwyaz).
Reports are `.runtime/olmo-nfr-endpoint-curves/result-{control,reduced}-01/report.json`;
launch receipts/logs are adjacent `launch-{control,reduced}-01` directories.
Scope SHA `e8d52df40aa5456639c919ff32d58cde12a65a3fcf49df07d91778f218d48880`;
launch plan SHA `a4001afe27885330c777db707e0eae6884472730e7271a0ddf4c4685efc05cee`.
Each has a 30-minute timeout. No training is running beside them.

The continuation passed 36 focused CPU checks and authentic metadata dry
resolution. Its scope is
`.runtime/olmo-nfr-stability-128/continuation-prepared-01/scope.json`, SHA
`563c6bbb3b5491bffae75879fbfc9299b94021a25c8c391a91d451a1622d4523`.
The 222-source inventory and protocol are frozen. The adapter preserves the
ten accepted model/update/save/graph callbacks, strictly loads old64 state,
then records only the new execution/observation authority. Activation awaits
the completed probes; no further user permission is required for the agreed
conditional continuation. PR56 is a draft; code and handoff are being saved.

15:48 UTC: both probes completed all eight rows, preservation passed, W&B
synced and host launchers exited0. Summary3p91owsx is published; plot inspected.
Control report SHA `6a80643b3855829e4ad444a0a42dc188181348e7e9899879c40fe1d4ae276ac9`;
reduced `97fe4e9029f809c7bc46956b8fee989e590a6f9839959e8bef64f72cd4378a0b`.
Both show bounded/decaying changes. Reduced K4/K8 tail residuals to finite32
are2.1792%/.1766%; more passes slightly worsen CE5.412258→5.431623. See results.

15:49 UTC: activated unchanged reduced64→128 after that assessment. Both
GPUs verified released inside the new training container before launch.
Hostsession83327; read-only monitor68687. W&B
https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/2zu5jloq .
Activation receipt `.runtime/olmo-nfr-stability-128/activation-01.json`, SHA
`9a85cd4f6b2d48e95a0e301c92f6536bee302acb3b5af0496da15c1247c95f71`.
Run report is `.runtime/olmo-nfr-stability-128/native-nfr-reduced-64to128-01/report.json`;
host log/launch receipt are adjacent `native-nfr-reduced-64to128-01.log` and
`native-nfr-reduced-64to128-01-launch.json`. SSD root is
`/mnt/localssd/cdrm-checkpoints/nfr-stability-128/native-nfr-reduced-64to128-01`.
The external timeout is6hours; STOP.json in the runtime root requests a
checkpointed stop. Do not start another GPU job while this two-rank run lives.

After terminal128, the final saved state will receive the same one-GPU K32
observation, using a new explicit authority outside the live training pins.
No additional intermediate deep-pass grid is planned absent a specific concern.
