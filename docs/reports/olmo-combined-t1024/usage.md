# Reproducing the combined T1024 benchmark

Use the project Docker launcher and retained OLMo step60000 artifacts. Never
run CUDA from the host or fall back to CPU. From the checkout:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent &&
nvidia-smi &&
timeout 3600s python scripts/olmo_two_gpu_single_reference.py \
  --case combined --ordinary-rope-backend native \
  --ordinary-attention-backend sdpa --length 1024 --batch-size 64 \
  --output-dir .runtime/olmo-combined-t1024/sdpa-b64-new
'
```

Use a new output directory for every attempt and an even physical batch.
B32 is the first bounded capacity/check row; B64 matches the saved T512/B128
and T2048/B32 input tokens per update. This is one GPU with no accumulation or
DDP, despite reuse of the historical paired-rank fixture construction.

K2 performs an ordinary bootstrap and one attached-feedback pass with native
RT at layers 0 and 15. Both passes use CE, NextLat SmoothL1 and KL. Ordinary
attention is Flash SDPA; native RT retains its existing Triton and larger eager
forward tiles plus recompute backward. Combined FA4 and Dao RoPE remain rejected.

Each process starts from identical pretrained weights. It performs three real
Adam preparation updates, eleven requested capture warmups plus one gradient-
preparation backward, and five timed complete updates. Graphs capture forward,
loss and backward; clipping, Adam and scheduler remain outside. Timing includes
batch validation/copy and finite checks, excludes compilation, fixture creation,
logging, reference copies and evidence I/O.

Five operational gates check actual dispatch, exact initial and changed-weight
terminal eager/graph loss/raw-gradient parity, dependencies and sources. This
does not establish new independent BF16/full-Adam trajectory equivalence.
Setup and steady memory are separate; free memory is sampled, not continuous.

W&B group `olmo-combined-t1024` under `taylorbollman/pretrained-fbt-rt-nextlat`.
Local reports, snapshots and receipts are in `.runtime/olmo-combined-t1024/`.
Retain a completed stage before proceeding. Do not edit pinned runtime sources
or protocol while a stage is active. No new T512/T2048 execution, FA4, quality
training or kernel optimization is part of this milestone.
