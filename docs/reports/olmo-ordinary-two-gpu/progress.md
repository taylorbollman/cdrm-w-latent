# Ordinary base model: current work and interruption handoff

2026-09-25. User authorized a bounded ordinary-only batch/memory/throughput sweep
on the two H10080GB GPUs, T512. No quality training, bucket-view adoption or dynamic
layout work is queued. Branch `feat/olmo-ordinary-two-gpu`, based on merged PR30.
Read the prospective protocol and execution-options note in this directory.

The unchanged native-RoPE ordinary DDP B64/rank control is running as
`.runtime/olmo-ordinary-two-gpu/ordinary-native-ddp-b64-01`. Its source is frozen.
Agent is adding explicit ordinary-only Dao RoPE and proper no-RT selection to
existing single/ZeRO-1 benchmark harnesses in an isolated worktree. Root owns all
GPU commands. Do not cherry-pick runtime changes during a live probe.

After control: integrate/test helper; actual ordinary Dao graph complete-update
check B1/rank, Dao B64/B128, then adaptiveB192/B256 if memory supports. ZeRO-1 only
if useful; matched single reference and repeat selected DDP point. Preserve
fixed full CE and precision/optimizations; no new broad numeric campaign.

The wrapper retains an unused frozen32MiB fusion module; report its resident
bytes separately from1,176,764,416 active ordinary parameters. No RT/FBT/NextLat
executes. Source originalstep60000OLMo, nativeFP32RoPEvariant Dao, FlashSDPA,
BF16mixed/FP32trainingstate, compiledroundedSwiGLU,fusedAdam,checkpointing,
CUDAgraphs with actualDDP/NCCL. Prior ordinary DaoT512 qualification applies;
T2048 and RT Dao qualifications are not cleared by this work.

New evidence root `.runtime/olmo-ordinary-two-gpu/`; existing retention helper
uses prefix `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T195200Z/<stage>`.
Keep every completed attempt/log and additional ordinary protocol. CPUonly
container for retention, `env -u GOOGLE_APPLICATION_CREDENTIALS` for ADC.
Persistentdisk~19GiB free. Disposable eight-update performance probes need no
new fullcheckpoint; originalweights alreadyretained. W&Bexistingproject/group,
ordinary-prefixedruns. GPUcontaineronly;NCCLasyncerrorhandling0+externaltimeout.
