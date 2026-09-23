# CE chunk integration and pretrained ordinary baseline

Prospective bounded protocol, 2026-09-23. Follow-up to PR21's ordinary throughput
diagnosis; no learning or quality comparison, Q/K/RoPE/RT algorithm change, or
production default change. The optional `NextLatConfig.ce_chunk_size` controls
selected CE positions per full-vocabulary projection. `None` preserves both
historical behavior and serialized checkpoint configuration. KL remains on
`vocab_chunk_size=128` in this queue. No vocabulary truncation or mask change.

## Fixed execution

Original pretrained native OLMo-1B step60000 (~252B tokens):16 layers, D2048,
H16, SwiGLU8192 per branch, tied50304 readout, native RoPE/nonaffine LayerNorm,
no Q/K normalization. Use the validated local artifact manifest/tokenizer and
record its checkpoint pin. BF16 mixed with FP32 parameters/gradients/Adam;
TF32 off, deterministic PyTorch Flash SDPA, autocast cache off, ordinary block
checkpointing on. RT where selected uses Triton forward/backward, cast reuse,
recompute backward workspace, layers0 and15; FBT uses K2; NextLat stays horizon1.
Seed20260922 and the same real-text fixtures/rotations as F4. One unpadded
independent document per row, T512, half-target CE unless explicitly full.
One physical batch/update, no accumulation, one H10080GB. GPU only in container.

## Checks before timing

CPU: full-vocabulary dense CE loss/hidden/readout gradients; masks and partial
chunks; unchanged auxiliary values/gradients/detach ownership; old serialized
configs and changed-config layout rejection; all eight RT/FBT/NextLat switches
using tiny dynamic/reference and static/candidate execution. Use small chunks
to actually cross boundaries, rather than a vacuous tiny128-vs2048 comparison.

Native B8/T512 cases: ordinary, NextLat-only, combined RT+FBT+NextLat. Each starts
at the identical pretrained weights/new-branch initialization. Compare CE128
against CE2048 with KL128 fixed. Require exact counts/gradient ownership and
bitwise unchanged auxiliary losses. CE relative loss error<=1e-5, global raw
gradient relative L2<=1/64, each tensor L2<=1/32 and max-error normalized by
reference tensor maximum<=1/16. These are prospective engineering screens,
not paper-derived thresholds. Preserve any failed gate and investigate; do not
silently widen thresholds. Check zero norms explicitly. This comparison changes
only CE grouping; it cannot clear the earlier F4 RT-backward qualification.

For each CE2048 candidate:ten warmup backwards, graph capture, exact eager/graph
loss and all participating gradients, changed tokens, repeated replay overwrites,
three eager versus three graph AdamW/clipping/scheduler updates with exact final
model/moments/counters, then changed-weights graph check. Six physical updates
per successful correctness report. Finite state is required. All eight switches
are covered on CPU; native GPU is deliberately representative, not exhaustive.

## Native throughput

Fresh process for each ordinary B64/T512 arm: half CE128 control, half CE2048
candidate, full CE2048. Half supervises256 positions/row, full511. The first
pair changes only CE chunking. Full supervision changes the workload and is
reported separately. Three preparation optimizer updates, ten warmup backwards
plus capture, three timed complete graph-backed updates, then three separate
forward/loss/backward replays. Median of three synchronized wall and CUDA-event
samples. Include input validation/copies, clipping, AdamW and schedule in full
step timing. No W&B/file I/O or profiler in timed closures. Record setup versus
steady allocated/reserved peaks and current reservations. Six physical updates
per successful timing report. Expected queue: six reports,36 physical updates.

The B64 control remeasures F4's ~31.1k/s using current hardware/runtime; do not
claim gains solely against the older timing. A result does not reproduce the
paper's six-layer recipe or imply a learning advantage. No checkpoint of these
few diagnostic updates is needed; preserve the native starting checkpoint by
reference, sources, token-generation code, metrics and protocol snapshots.

## Dao CE and scope

Audit the Dao FlashAttention demo loss and Triton source at pinned versions.
Its loss receives materialized logits; linear readout fusion is a separate
question. In the prior chunk2048 trace named CE kernels occupy ~4.09% of device
time; this does not justify blocking the primary integration on a kernel swap.
Do not adopt it merely from import success. Optional future probe must preserve
FP32 logits, sum reduction, label validation and non-inplace backward, and test
checkpointing/graphs and gradients. Fused RoPE/SwiGLU remain separate work.

## Provenance and retention

Freeze runtime/protocol in Git before GPU queue. Each report carries source and
protocol hashes with exact snapshots, configuration, W&B URL, stage/failure,
runtime/GPU identity and update counts. Online W&B group `olmo-ce-integration`
under `taylorbollman/pretrained-fbt-rt-nextlat`. Preserve explicit selected
reports/logs/sources, CPU test log, summary, audit and handoff in a small verified
GCS evidence bundle under `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/`.
Retain failures as well as successful final selection; no hidden retry/sweep.

The F4 RT+FBT maximum-coordinate gate miss and ~18% BF16-vs-FP32 initialization
gradient difference remain qualified. Graph recovery/accumulation, padded graphs,
online readiness, genuine multi-GPU and later bounded precision follow-up remain
on the broader V4 plan. This task does not authorize long quality runs.
