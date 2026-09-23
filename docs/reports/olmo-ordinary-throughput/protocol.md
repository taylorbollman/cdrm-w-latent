# Ordinary OLMo throughput: bounded comparison protocol

Frozen before GPU measurements, 2026-09-23. This diagnostic answers the user's
question about the 31.1k input-token/s ordinary F4 result versus the RT paper's
153k token/s six-layer ordinary result. It measures a smaller native ordinary
model and isolates avoidable execution overhead. It does not change RT, add
FA4, perform a learning run, or establish an exact reproduction of paper timing.

## Fixed model and execution

- Random native OLMo, six layers, width 2048, 32 attention heads of width 64,
  SwiGLU intermediate width 8192 **per branch** (packed up/gate output 16384),
  tied input/readout vocabulary 50304, native RoPE and nonaffine LayerNorm.
  The layer/head bridge described below changes heads to 16 explicitly.
- No RT, FBT or NextLat execution. The existing wrapper may retain its frozen
  fusion parameters; report those separately from active native parameters.
  Native six-layer active parameters: 505,675,776. Initial weights and random
  tokens are reproducible from recorded seeds; no pretrained weights required.
- Sequence length 512, one unpadded independent document per row. Use a fixed
  structural mask and changed token values between optimizer updates. Initial
  CE mask supervises the final 256 target positions per row, matching F4.
  A separate full-CE arm supervises all 511 next-token targets per row.
- BF16 autocast, FP32 parameters, gradients and Adam states; TF32 disabled;
  autocast weight cache disabled. Force deterministic PyTorch Flash SDPA and
  record the PyTorch/CUDA/GPU versions and actual attention dispatch. Use the
  existing AdamW, clipping and schedule policy. No `torch.compile` in this
  first native measurement; CUDA graph capture is not compilation.
- Ordinary activation checkpointing is initially enabled. CUDA graphs capture
  forward, loss and backward; copies, validation, clipping, AdamW and schedule
  remain outside capture and are included in complete-update timing.
- Run only inside the project GPU container after verifying container identity
  and `nvidia-smi`. No host CUDA work and no CPU fallback.

## Bounded sequence and adaptive arms

Each arm runs in a fresh process and records its complete configuration,
source/protocol hashes, seeds, stages, failures and W&B URL. Keep runtime source
fixed during this queue. The F4 reference is existing evidence, not a required
rerun. Do not silently replace a physical batch by gradient accumulation.

| Arm | Physical batch | Heads | CE selection/chunk | Purpose |
| --- | ---: | ---: | --- | --- |
| Six-layer bridge | 64 | 16 | Half / 128 positions | Change only depth/random initialization from native F4 architecture and retain its loss/execution policy |
| Requested large batch | 512 | 32 | Half / 128 positions | Directly test the requested six-layer, true B512 setup |
| CE chunk diagnostic, if needed | Same feasible batch as its paired arm | Same | Half / 2048 positions | Change only CE chunk size when the 128-position implementation is slow or memory-heavy |
| Full-CE diagnostic | Same feasible batch as its paired arm | Same | All 511 targets / recorded chunk | Measure the additional work of pretraining-style supervision |
| Checkpoint-off diagnostic, if needed | 32 | 32 | Matched masks and chunk | Bound memory while separating checkpoint recomputation from ordinary throughput |

The head change is explicit; this queue is a directional comparison, not a
factorial attribution of every difference. If B512 fails, preserve the OOM and
its failing stage. A fresh, explicitly labeled lower-batch attempt may establish
a safe measurement, but cannot be reported as successful physical B512. The
2048-position chunk arm is a performance-only regrouping of the same CE sums,
not a vocabulary truncation, target-count change or new loss. Floating-point
summation order can change. Before interpreting its speed, verify finite loss
and gradients and a bounded same-weight loss comparison; do not claim bitwise
equality across chunk sizes.

Do not launch a broad optimization sweep. A global-B512/micro-B32 accumulation
arm is optional follow-up only if it resolves a remaining comparison question;
it must be labeled as 16 microbatches, not physical B512. No need to reproduce
the authors' complete training stack merely to finish this diagnostic.

## Timing, memory and checks

For each successful arm, initialize Adam with three complete eager updates,
then perform ten backward warmups and graph capture. Record any additional
backward used to initialize persistent gradient buffers separately. Measure
three complete graph-backed optimizer updates and, separately, three graph
forward/loss/backward replays without optimizer. Report each observation and
the median, wall and CUDA-event seconds, input tokens/s and CE targets/s.
Do not mix capture/setup time with steady-state speed.

Measure setup and steady-state peak allocated/reserved GPU memory separately,
plus current reservation and physical device capacity. Inspect finite losses,
gradient norm, changed trainable weights and finite Adam/model state. These are
execution checks, not convergence or full numerical-clearance claims. A small
ordinary dispatch trace may confirm Flash execution; do not time under the
profiler. Compare full-update and captured-region times before attributing
cost to Python validation, synchronization, clipping or Adam.

Record CPU-side layout-validation and input-copy work as part of the complete
step. Loss logging must not occur inside the timed region beyond the canonical
optimizer-step metrics already included in F4. Log graphable results to
`taylorbollman/pretrained-fbt-rt-nextlat`, retain local raw reports, and retain
small reproducible evidence in `gs://fast-chunks`; random disposable weights
need not be retained when the source and initialization seeds are preserved.

## Why the paper comparison needs qualifications

The [RT paper, Appendix D.1](https://arxiv.org/pdf/2604.21215) reports ordinary
153k and RT 49k tokens/s for six layers at width 2048, MLP width 8192, 32 heads,
length 512. Section 6 identifies single-H100 experiments. The appendix does not
fully specify the exact timing interval or identify the saved run behind each
rate. Do not import the separate Section 6 per-layer timing exclusions into the
Appendix D.1 throughput claim.

The pinned authors' recipe differs materially from native F4:

| Property | Native diagnostic / F4 | Authors' six-layer recipe |
| --- | --- | --- |
| Depth | Six here; F4 had 16 | Six |
| FFN | Three-matrix SwiGLU8192, packed projection16384 | Two-matrix GELU8192 |
| Vocabulary/readout | 50304, tied | 32100 vocabulary padded to32128, untied |
| Position/norm | RoPE, native nonaffine LN, no Q/K norm | ALiBi; affine layer and attention normalization |
| Ordinary batch | Physical64/512, no accumulation | Global512, physical microbatch32 |
| Ordinary execution | Full-block checkpointing, graphs, no compiler | Checkpointing null in base recipe, `torch.compile` default |
| Supervision | Half-target F4 bridge; full-target arm separate | Full next-token training |

The RT recipe uses physical512, whole CUDA graphs and logit microbatch2; it is
not evidence that the ordinary model used physical512. F4's sixteen-layer
native body has about3.56 times the dense block parameters of the paper's
six-layer GELU body, before checkpoint recomputation or vocabulary costs.
Hence the raw31k-versus153k ratio alone does not diagnose a FlashAttention
problem. Conversely, performance should be measured rather than excused by
parameter count: small checkpointed CE chunks repeatedly process the large
readout matrix, and native unfused RoPE/pointwise execution may add avoidable
cost. These are hypotheses to test, not measured bottleneck claims.

## Source lineage

- Native/F4 baseline: project commit
  `f44455a6ed1c4afe984d4ce29c492998b526acda`; historical F4 runtime
  `397885b`, as recorded in its reports. Reuse the canonical
  `StaticFBTTraining`, `PreparedFBTLayout`, prepared CE and optimizer helpers;
  add an isolated benchmark entry point rather than editing frozen F4 reports.
- Authors' upstream pin:
  `a21b42d2bc292edb86ed1b62cee4bcab809a9d21`; local vendor HEAD
  `824767f9a6f8e29959a0c486d169ce10b7194d41`. Relevant recipes are unchanged
  from that upstream pin:
  [base C4](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/configs/kempner/base-c4-t5.yaml),
  [six-layer model](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/configs/kempner/models/300m_6.yaml),
  [ordinary sweep](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/sweep_olmo_512.yaml),
  [RT sweep](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/sweep_recurrent_512.yaml).
- Preserve existing RT+FBT precision qualifications. This ordinary-only
  throughput diagnostic neither clears nor worsens their numerical status.
