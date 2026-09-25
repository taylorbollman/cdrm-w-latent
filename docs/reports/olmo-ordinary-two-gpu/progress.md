# Ordinary base model: completed work and interruption handoff

2026-09-25. The ordinary-only batch/memory/throughput sweep on two H10080GB
GPUs at T512 is complete. No GPU or quality-training job is queued. Both GPUs
were verified idle after the last repeat. Read results.md, execution-options.md,
zero2-nextlat-clarifications.md and storage-receipt.md in this directory.

Recommended operating point: Dao RoPE DDP B64/GPU, global B128, approximately
85.2k input tokens/s and44.9GiB sampled free/GPU. Independent repeats differ
0.14%. B32 gives80.3k/s; B128/B192 give85.8k/86.3k/s with35.9/25.8GiB free.
The matched single-GPU B128 run gives44.0k/s, about1.93x scaling on two GPUs.
Native RoPE B64 control gives78.2k/s. All use one microbatch/update, no accumulation.

Runtime467bde7 adds explicit ordinary Dao selection and ordinary-case support
in shared benchmark harnesses, preserving production defaults.93focusedCPUtests
pass (3.26seconds;67installedJITdeprecationwarnings). Eight execution stages
pass: Dao B1 correctness, native B64 control, Dao B32/B64/B128/B192, a fresh B64
repeat and single B128. All completed stages have source/evidence retention;
1,303 source pairs match.55distributed optimizer executions plus8single updates
are118rank-level optimizer calls, not118distinct distributed updates.

The actual original step60000 OLMo has1,176,764,416 active parameters. Wrapper
retains an unused frozen32MiB fusion module; no RT/FBT/NextLat executes. Ordinary
DaoFP32RoPE, FlashSDPA, BF16mixed/FP32trainingstate, compiledroundedSwiGLU,
fusedAdam, checkpointing and CUDAgraphs including actualDDP/NCCL are active.
Own eager/graph checks pass; older RT/combined and longer-context precision
qualifications remain. No Q/K normalization change.

ZeRO2 was deferred, not judged incapable of improving throughput. Ordinary DDP
already has ample memory at its throughput plateau. RT/FBT/NextLat may benefit
from gradient sharding; compare at the intended physical/global batch and
accumulation schedule. Current graphs do not support accumulated replay.
Current OLMo NextLat uses CE+SmoothL1+KL with auxiliarycoefficients1.0;
ordinary throughput disables it, and historical A5 recipes differed.

Local evidence: `.runtime/olmo-ordinary-two-gpu/`.
GCS prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T195200Z/`.
Retain final stage receipts, audit and plots. Original pretrained checkpoint
already has O1 retention; disposable short capacity runs created no new full
checkpoints. Keep all prior RT milestone evidence separate and unchanged.
GPU work remains container-only; NCCL graph launches need asyncerrorhandling0
and an external timeout. CPU retention uses GPUdisabled container and
`env -u GOOGLE_APPLICATION_CREDENTIALS` to use working mounted ADC.

No next experiment automatically queued. Choose intended model/data/training
batch before further optimization. Fresh-process recovery with its actual data
cursor remains advisable before a long training run. User permits future
standalone one-GPU work on a one-GPU VM; this reference intentionally used the
same two-GPU host (one GPU active) for a controlled scaling comparison.
