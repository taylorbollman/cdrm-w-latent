# F3e: multiple selected native RT layers

Prospective protocol, 2026-09-23. The user authorizes this next functionality
and resource milestone and clarifies that the main experiment will certainly
use more than one RT layer, but need not use all layers. Layer selection remains
an experimental choice. No quality training or architecture selection is implied.

## Fixed implementation and primary layouts

Reuse the F3d validated implementation: BF16 mixed, FP32 parameters/gradients/
Adam, TF32 off, ordinary activation checkpointing, deterministic ordinary Flash,
per-invocation weight-cast reuse, Triton historical forward/backward tiles and
`backward_memory="recompute"`. Materialized backward is the comparison reference.
Both arms use identical forward operations. Original OLMo-1B step60000 (~252B
tokens), native RoPE, nonaffine LayerNorm, no Q/K normalization, model and loss
semantics remain unchanged. Only execution selection changes. No new RT weights.

Layer indices are zero-based in the native 16-layer model:

| Label | Selected RT layers | Role |
| --- | --- | --- |
| single | 0 | Fresh resource anchor; not the intended main architecture |
| adjacent2 | 0, 1 | Direct recurrent-block handoff |
| spread2 | 0, 15 | Recurrent blocks separated by ordinary blocks |
| spread4 | 0, 5, 10, 15 | Several recurrent blocks distributed through depth |
| all16 | 0 through 15 | Separate stress case, not an assumed experiment default |

RT-only executes one stack. Combined means FBT K2 (ordinary bootstrap then one
feedback stack with selected RT) plus training-only NextLat. A K3 check adds a
second feedback pass with the same weights. Actual RT block calls must equal
selected-layer count times feedback-pass count (or once for RT-only). Count
dispatch per layer during the initial eager comparison, including supported
Triton tiles and explicit eager fallback; keep observers outside captured timing.

## Checked stages

1. Small CPU four-layer independent sequential references for adjacent, spread
   and all-layer selections. Check complete FBT/NextLat objectives and gradients,
   shared K3 calls, parameter ownership and prepared execution guards.
2. Actual native checkpoint correctness, staged from small to longer layouts:
   - Combined spread2, B1/T32.
   - Combined adjacent2 and spread2, each B8/T512.
   - RT-only spread2, B8/T512.
   - Combined spread4, B4/T512.
   - Combined K3 spread2, B1/T32.
   - Combined spread2, B1/T2048 (native context limit).
   - Combined all16, B1/T32 as a separately labeled stress test.
3. Complete graph-update resource screen after relevant correctness passes:
   - Combined single/adjacent2/spread2/spread4 at common B64/T512.
   - RT-only spread2 at B128/T512.
   - Combined spread2 at B8/T2048.
   - If its small stress check passes, all16 combined at B8/T512.
   These are comfortable operating points to test, not maximum-batch searches.
   If setup approaches the device limit or OOMs, retain the attempt and retry
   that layout at half batch; report any unmatched shape explicitly. Do not
   squeeze the final GiB. An all16-specific failure need not invalidate primary
   two/four-layer evidence; stop expanding that stress arm and explain its scope.

Each correctness run compares all initial loss terms/parameter gradients with
materialized backward, then same-candidate eager/CUDA-graph original inputs,
changed tokens, gradient overwrite and changed weights. Three eager versus
three graph complete AdamW updates must match model, moments, scheduler,
counters and metrics. Capture includes forward/loss/backward only; batch copy/
validation, clipping, optimizer and scheduler stay outside and in full-step time.
Use ten warmup backwards before capture. Capacity uses three preparation
updates plus three timed changed-input/weight graph updates and finite state
checks. Capacity health is not an all-gradient comparison at those batch sizes.

Primary mixed engineering budgets stay: global gradient relative L2 <=1/64,
per-tensor <=1/32, max absolute error/reference-tensor-max <=1/16. Zero references
require exact zero. Forward losses should remain bitwise identical. Preserve
stricter diagnostics. Same-candidate graph checks retain relative L2 1e-5 and
max absolute 1e-6+1e-5*reference-max; complete Adam/state comparisons are exact.
No threshold is widened after observing results. Localize material discrepancies
at the smallest failing layout before continuing dependent measurements.

## Resource and provenance record

Add recompute support to the analytic estimator with unchanged materialized
default: extra no-prefix QK work is 2BD(E+T²) per RT invocation, E=T(T-1)/2.
Report exact unique architectural, resident, trainable, gradient-participating,
optimizer-owned and deployable parameter counts. Use actual CE/pair/KL/predictor
union selections, pass multiplicity and batch in matrix-FLOP estimates. Excluded
pointwise, optimizer, kernel-padding and launch costs stay explicit. FLOPs are
not measured device utilization or a prediction of speed.

Record input tokens/s, CE targets/s, update time, peak allocated, peak reserved
and current reserved separately. Three-update timings are directional. Use fresh
single-layer anchor and common batches where feasible. Do not infer quality,
the best layer placement, scaling to several GPUs or native FA4 integration.
Long forward rectangles above256 still use eager fallback; backward recompute
supports rectangle sides through2048. Q/K math stays native unless a specific
health issue appears. Existing stream-mismatch warnings remain visible.

All GPU work requires verified project container/H100. Freeze source/protocol
snapshots before each run, verify hashes afterward, log online to W&B, retain
raw JSON and small evidence in GCS, including failures. Reuse the pinned native
checkpoint; disposable few-update model states need not be uploaded. No long
learning run is queued. Genuine two-GPU correctness needs a second GPU and is
separate from this milestone.
