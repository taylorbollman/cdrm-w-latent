# F2: scale attribution and physical-batch execution

Authorized 2026-09-22. This is a bounded functionality/efficiency milestone,
not a learning or quality comparison. Base: F1 merge 0e64ca0.

Keep native OLMo-1B step60000 (~252B tokens), sixteen width2048 blocks,
RoPE, tied vocabulary, native LayerNorm and no Q/K normalization. RT selects
index0 only. FBT uses K2 except the named K3 scale case; NextLat remains the
shared training-only predictor. Preserve all F1 objective weights, detachments,
pass coefficients and valid-target denominators. FP32 parameters, gradients and
Adam moments; BF16 autocast, TF32 off. One H10080GB; no distributed claims.

## Bounded numerical health

At fixed original weights and matched B2/T32 operational real-text fixtures,
observe ordinary, RT, FBT, NextLat, combined K2 and combined K3. Capture actual
layer inputs/outputs and projected Q/K for indices0,1,7,15, including selected
RT's temporary and permanent entries. Reconstruct detached score/entropy
diagnostics; fused SDPA does not expose its literal probabilities, and this
reconstruction is not a claim of identical rounding to every dyadic tile.
Exclude invalid rows/keys and self-only rows from attention saturation summaries.

Differentiate each correctly weighted pass/CE/latent/KL component through the
attached cross-pass graph. Stream summaries, compare group norms/directions,
and report decomposition closure. BF16 repeated backwards need not decompose
bitwise; tiny FP32 closure has a meaningful correctness test. No optimizer step
or model-math change is part of attribution. Require observer neutrality on
tiny fixtures and one actual BF16 combined case. Close observation before any
backward/checkpoint replay. Escalate only for a concrete failure or scale concern.

## Physical batch and checkpointing

Measure full changing-input/changing-weight canonical AdamW steps at T512 for
standalone RT and combined K2, initially B1,8,16,32,64,128,256,512. This is the
physical batch; gradient accumulation is one. Stop each sweep at a capacity
failure or measured allocated peak above65GiB; do not optimize the last GB.
Use three warmup and three timed updates per cell; report wall and CUDA-event
times, all warm/timed metrics, peak allocated/reserved memory, target counts and
parameter accounting. Exclude fixture generation, observers and reporting from
timing. A few repeated prompt fixtures are not representative corpus learning.

Add default-off non-reentrant activation checkpointing for ordinary blocks only.
Selected RT retains its existing x/z reconstruction/custom backward; never wrap
an entire recurrent pass and replay the slow sequential forward. Check tiny
full-gradient parity, cache contracts, selected-RT execution count, and one exact
actual-checkpoint BF16 complete update before the checkpoint-enabled sweep.
Backward still materializes quadratic RT attention intermediates; this change
does not eliminate that memory cost. Timed complete-step results remain eager.

## CUDA graph feasibility

Current full training is not graphed or compiled. Prototype only a fixed-shape,
unpadded native stack forward/hidden-cotangent backward, with persistent gradient
buffers and changed token/weight replay validation. This intentionally excludes
CE, auxiliary losses, FBT and the optimizer. Report it as a feasibility
microbenchmark, never as combined training throughput. Keep graph-pool memory
separate from the eager capacity measurements. A failed capture is a specific
implementation blocker to localize, not evidence that RT is uncapturable.

Section6 of the [RT paper](https://arxiv.org/html/2604.21215v1#S6) motivates this
order: physical batch raises recurrent MLP arithmetic intensity; accumulation
does not. Retained x/z permits cheaper recomputation, and graphs address launch
overhead. Its B512 experiments used substantially smaller models, so this
1.18B-parameter model needs an actual capacity measurement.

## Evidence and stopping point

Log graphable metrics online to taylorbollman/pretrained-fbt-rt-nextlat. Persist
case-level progress JSON after every result; these short diagnostic updates do
not require large disposable training checkpoints. Retain small reports, source
snapshots and tests in gs://fast-chunks, referencing already retained native
weights. Record source hashes and hardware, and leave a handoff across compaction.
Assess Q/K math and next execution priorities at this PR review point. No long
training or automatic normalization adaptation is authorized by this protocol.
