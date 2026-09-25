# Ordinary base model: current work and interruption handoff

2026-09-25. User authorized a bounded ordinary-only batch/memory/throughput sweep
on the two H10080GB GPUs, T512. No quality training, bucket-view adoption or dynamic
layout work is queued. Branch `feat/olmo-ordinary-two-gpu`, based on merged PR30.
Read the prospective protocol and execution-options note in this directory.

Native-RoPE ordinary DDP B64/rank control passed78,183.60tokens/s at source
`0c6ace4`, with44.22GiB sampled free/rank. It is retained in GCS. Runtime
extension `467bde7` is integrated and pushed;93focusedCPUtests pass and an
independent review finds no material issue. Production defaults are unchanged.

Active root exec32463 runs ordinary DaoB1 full-update correctness, then Dao
DDPB64 andB128 capacity sequentially with external timeouts. Source is frozen.
Each stage is in `.runtime/olmo-ordinary-two-gpu/ordinary-dao-*`; inspect latest
report/rankprogress before restarting. Do not edit runtime until the queue ends.
After B128, inspect memory/gain before B192/B256; then conditionalZeRO1,
matchedsingle reference and repeat selected DDP point. No broad numerics.

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
