# Ordinary OLMo optimization investigation

2026-09-24. User requested investigating ordinary-model efficiency while
considering the RT-backend decision. This is a bounded investigation, not a
production backend change or training experiment. Main code remains unchanged.

## Main finding

Installed FlashAttention-4 runs causal BF16 forward and deterministic backward
on this H100, including CUDA graphs. With native OLMo attention-input strides,
the isolated attention region is **1.13x faster at B64/T512** and **1.47x faster
at B16/T2048** than the current forced PyTorch Flash SDPA backend. These are
not full-model speedups. FA4 is compatible in principle with the pretrained
ordinary model: it consumes the same already-RoPE-rotated Q/K and unrotated V,
with the same causal softmax attention, head geometry and scaling. No new
weights, Q/K normalization, positional encoding or checkpoint conversion is
needed. Actual pretrained-model loss/gradient/update compatibility remains to
be tested before adopting it.

## Environment and scope

- One H100 80GB HBM3, confirmed inside `/workspace/cdrm-w-latent` container.
- Torch `2.13.0a0+8145d630e8.nv26.06`, CUDA13.3.
- `flash-attn-4==4.0.0b20`, CUTLASS DSL `4.6.0.dev0` and CUDA13 extra already
  installed. No dependency installation or container rebuild was needed.
- Select `CDRM_FLASH_ATTENTION_SOURCE=installed` on the project launcher;
  the historical vendor source shadows the matching package otherwise.
  See [environment note](../../olmo-fa4-environment.md).
- Ordinary production code still uses PyTorch SDPA, forced to Flash by the
  performance harness. Installing FA4 does not change that dispatch.
- Standalone FA3 was not importable in this image and was not benchmarked.
  It is a sensible optional Hopper comparator; the version number alone does
  not establish which implementation is fastest at a particular shape.
- BF16 Q/K/V,16 heads of128 dimensions, causal full attention, dropout0,
  deterministic algorithms and explicit FA4 deterministic backward, TF32off.
- Synthetic packed projection outputs receive native RoPE outside the timer.
  There are no checkpoint weights, projections, CE, RT, FBT, NextLat or
  optimizer updates in these measurements.

Official references checked2026-09-24:
[FA4 installation and Hopper support](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md),
[FA3 Hopper implementation](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release).

## Isolated measurement

Ten forward/backward warmups before capture. Ten timing samples per backend,
five in each ordering (SDPA/FA4 then FA4/SDPA);20 graph replays per sample.
CUDA-event timing includes forward, backward and persistent Q/K/V-gradient
zeroing. Setup, compilation, RoPE, logging and input generation are excluded.
Each backend's eager/captured output and all three input gradients match
bitwise on every tested shape. There is no cross-backend bitwise requirement.

| Batch | Length | PyTorch Flash ms | FA4 ms | Attention-region speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 0.05385 | 0.03422 | 1.574x |
| 64 | 512 | 2.43620 | 2.15385 | 1.131x |
| 16 | 2048 | 4.86996 | 3.30289 | 1.474x |

Primary evidence: [native-layout report](native-layout.json),
[W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/g0b6gfy7).
Executed sources/logs remain in
`.runtime/ordinary-fa-investigation/native-layout/`.

Both arms receive identical native-style layouts: tensors have logical
`[B,H,T,D]` shape, Q/K use dense BTHD storage after the actual native rotation,
and V retains packed-QKV strides. At B64/T512 the strides are
`(1048576,128,2048,1)` for Q/K and `(3145728,128,6144,1)` for V.
The probe uses `empty_strided` plus copying for independent gradient leaves,
preserving V's non-dense layout. It does not time contiguous conversion of V
as a prerequisite to either backend.

An earlier generic-layout probe was fair across its two candidates but used
BHTD-contiguous Q/K and dense BTHD V; it measured1.34x/1.61x at the larger
shapes. Those rates should not be substituted for the native-layout results.
It is retained separately in [generic-layout report](generic-layout.json),
[W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/xbywzrwd).
The original generic probe's inline comment claiming native strides was
incorrect; retaining the source does not endorse that comment.

Two setup/API attempts are also retained: attempt00 stopped before attention
because importing CuTe initialized CUDA before the determinism guard;
attempt01 stopped on the pinned FA4 API returning an output/LSE tuple rather
than a tensor. Import ordering and output unpacking fixed these harness
issues. They are not attention numerical failures. No optimizer updates
occurred in any attempt.

## Bounded numerical observations

On native-layout B64/T512 and B16/T2048, FA4 versus PyTorch Flash output
relative L2 is0.000900/0.000996; the largest Q/K/V-gradient relative L2 is
0.001152/0.001174 (about0.12%). Values are finite. At B1/T64 both outputs are
bitwise equal, and both implementations differ from a full-FP32 math-SDPA
reference by approximately0.18% in output and at most0.27% in input gradients.

These are descriptive random-fixture checks, with exact own-backend graph
checks. They do not clear complete pretrained-model numerics, loss/update
parity, changed-input replay, padding, cache decoding, shared FBT calls or
multi-GPU execution. No previous RT qualification is changed.

## Other concrete opportunities

The current native16-layer full-CE reference is about36.6k input tokens/s at
B64/T512 with CE2048, ordinary checkpointing and CUDA graphs. The historical
31.1k rate used CE128 and half supervision; it is not the current full-CE
baseline. [CE integration](../olmo-ce-integration/results.md)

1. **Checkpoint and microbatch policy.** The matched six-layer B32/full-CE
   test improved75.0k→88.9k tokens/s (+18.5%) by disabling ordinary block
   checkpointing, while allocated memory grew11.19→21.17GiB. A16-layer
   measurement is needed; do not assume the same gain or fit. Try fewer
   checkpointed layers/selective recomputation or a smaller microbatch where
   appropriate. Preserve effective batch through validated accumulation when
   comparing learning. Graph accumulation remains separate unfinished work.
2. **Compile/fuse pointwise work.** Ordinary blocks still run separate
   operations for SwiGLU, residual additions, normalization and RoPE. CUDA
   graphs reduce host launch overhead but do not fuse those memory accesses.
   Preserve nonaffine LayerNorm, native SwiGLU split order, FP32 residuals,
   and split-half FP32 RoPE. Table reuse is already available and gave only
   +0.31% ordinary throughput; remaining work is application/cast fusion.
3. **Readout/loss traffic.** CE2048 already removed substantial redundant
   work. Dao CE optimizes the loss after logits exist; it does not fuse the
   vocabulary projection. Consider a separately validated fused linear CE
   path if a refreshed profile supports it. Preserve tied readout gradients,
   the full vocabulary, masks, reduction denominators and separate NextLat KL.
   [Liger](https://github.com/linkedin/Liger-Kernel) and
   [Cut Cross-Entropy](https://github.com/apple-aiml-research/ml-cross-entropy)
   provide relevant implementations. CCE's default gradient filtering is an
   additional approximation; begin any comparison with its unfiltered option.
4. **Invocation-scoped cast reuse.** Ordinary checkpoint/readout execution
   may repeatedly cast FP32 weights. The global autocast cache is deliberately
   disabled around captured training to avoid stale/disconnected copies.
   Use explicit lifetime and changed-weight tests before adopting reuse.
5. **Optimizer and clipping fusion.** Current AdamW/clipping use conservative
   non-fused/non-foreach paths. Full-CE B64 spends about4.4% of total step time
   outside captured forward/loss/backward, including more than optimizer;
   this bounds the likely standalone opportunity at that operating point.

Evidence for prioritization is the earlier
[six-layer CE2048 profile](../olmo-ordinary-throughput/profile-audit.md):
Flash3.57%, GEMMs42.6%, copy/casts11.1%, additions7.5%, fills6.3%, SiLU4.4%,
LayerNorm3.1%, CE4.1% of summed device-event duration. These are neither
disjoint optimization promises nor measurements of the16-layer model.
For illustration, doubling an operation accounting for3.57% of time saves
only1.785% of that time, absent secondary effects. A fresh16-layer profile at
T512 andT2048 should guide implementation priorities.

## Recommended next bounded milestone

1. Refresh the16-layer ordinary CE2048 full-update/operator profile.
2. Add an opt-in ordinary FA4 route for dense, unpadded full-sequence attention,
   leaving checkpoint parameters and RoPE unchanged. Test actual checkpoint
   losses/raw gradients and a few eager/graph optimizer updates, then measure
   full-step B64/T512 and a memory-safe T2048 point. Expose unsupported
   mask/cache cases explicitly; causal alignment differs across some APIs.
3. Compare checkpoint policy and one ordinary pointwise-compilation/fusion
   change at a time, maintaining fixed exposure/effective batch for learning.
4. Pursue loss/cast/optimizer work according to measured time and memory,
   keeping FA3 an optional comparator if obtaining it is inexpensive.

This is a recommendation, not an automatically started implementation queue.
No production defaults, RT path or pretrained parameters were modified; GPU
work is complete and no learning run was launched.

## Retention

Both successful reports and exact diagnostic sources, failed attempts and
this assessment are retained in GCS; see [storage receipt](storage-receipt.json).
The user-owned repository/home files also persist independently of local SSD.
