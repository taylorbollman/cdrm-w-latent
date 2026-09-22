# F3: static-layout canonical training and CUDA graphs

Authorized 2026-09-22 following the completed F2 review. This is a bounded
functionality and execution milestone. It does not authorize a quality run,
normalization change, all-sixteen-layer RT claim, or a new RT Flash/CuTE kernel.
Read the [F2 assessment](../olmo1b-f2/assessment.md) and
[current handoff](../../fbt-rt-nextlat-handoff.md) first.

## Model and objective

Reuse the original OLMo-1B step60000 checkpoint, approximately252B pretraining
tokens: sixteen width2048 blocks, sixteen heads, native RoPE, SwiGLU8192,
50,304 tied embedding/readout rows, non-affine LayerNorm and no Q/K normalization.
RT selects index0 only. Standalone RT is one recurrent stack pass; combined
uses the existing ordinary bootstrap plus an RT/FBT pass at K2. A named K3
case adds one extra shared-stack pass if the small diagnostic is feasible.
NextLat remains a training-only predictor, without a separate latent rollout.

Preserve canonical CE, NextLat regression and KL weights, pass coefficients,
target detachments, attached conditioning states and embeddings, tied readout,
document-boundary rules and valid-target denominators. Preserve the existing
pass0 + gamma * mean(extra-pass losses) objective. Graph integration must not
change pretrained or recurrent model math to make capture easier.

Prepare static layout metadata outside capture: validated shapes, masks,
positions, document boundaries, CE indices, same-document latent pairs and KL
triples. Capture the equivalent differentiable operations using fixed-shape
tensor work. Token values may change between replays; incompatible layout,
mode, shape, precision or parameter topology requires preparation/capture again.
Keep the current one-document-per-training-row contract. Tiny tests cover
irregular padding and boundaries; operational graph cells explicitly record
their exercised layout. Packed training and cache continuation are outside F3.

## Correctness gates

1. On tiny FP32 models, compare the existing canonical implementation against
   the prepared static path: component losses, denominators and every owned
   parameter gradient. Cover ordinary, RT and combined K2/K3, target detachments,
   masks/boundaries, frozen/inactive ownership and tied parameters. Verify that
   changing supported token values preserves the layout contract and invalid
   layouts fail before capture.
2. On the pinned actual checkpoint, begin with B1/T32 standalone RT and
   combined K2, each with ordinary checkpointing off/on. Include the named K3
   case if practical. Compare prepared eager forward/loss/backward against
   graph replay using changed tokens and persistent gradient buffers. Check
   all gradients, finite state, inactive parameters, gradient ownership and
   gradient overwrite rather than accidental accumulation.
3. Continue representative RT and combined K2 checks to B8/T512 with
   deterministic Flash SDPA. Compare complete changing-input/changing-weight
   AdamW updates, including clipping, model state, first/second moments,
   scheduler and counters. Record the original and replay update counts
   separately. Shorter checks do not clear this shape or the optimizer path.

Use FP32 parameters, gradients and Adam state, BF16 mixed compute, TF32 off,
and the deterministic Flash settings established in F2. Configure deterministic
algorithms and cuBLAS workspace before CUDA initialization. Disable autocast
weight caching in both compared paths. Ordinary SDPA backend selection does
not fuse native RT's eager tiled operations. Trace actual dispatch rather than
inferring it from the requested backend.

For graph versus prepared-eager numerical comparisons retain the original F2
budget: relative L2 <=1e-5 and maximum absolute error
<=1e-6 + 1e-5 * reference maximum absolute value. Report bitwise equality
separately; do not silently widen budgets. Finite outputs alone do not pass a
comparison. Localize a concrete failure before broadening tests; preserve
failed diagnostic evidence and distinguish unsupported capture/backend cases
from a completed comparison that disagrees. The retained F2 default-cuDNN
variability is not a passing reference for this strict deterministic check.

## Capture and activation checkpointing

Capture prepared forward, canonical loss and backward. Keep input staging,
gradient clipping, AdamW and scheduler outside the CUDA graph. Allocate and
retain persistent gradient buffers; capture must overwrite their contents
while preserving storage identity. Warmup/capture are disposable preparation,
not counted as successful training updates. Real diagnostic updates change
weights and optimizer moments outside the captured region.

The default-off ordinary_activation_checkpointing option remains ordinary
blocks only and uses non-reentrant checkpointing. Selected RT retains its
existing input/output reconstruction and custom backward. Never checkpoint a
whole recurrent pass and replay its sequential forward. Check checkpointed
and uncheckpointed behavior with the static path and capture; do not infer
compatibility from the earlier eager-only F2 check. Cache use during active
training checkpointing remains rejected.

## Bounded physical-batch measurements

After the smaller correctness gates pass, profile RT and combined K2 at
T512 with ordinary checkpointing on, considering B32,64,128 in that order.
The allocated-memory comfort budget is65GiB. Stop increasing batch on OOM,
capture failure or a measured peak above that budget; a smaller completed
subset is a valid bounded outcome. CUDA graph private pools can consume more
memory than F2 eager execution. Record allocated and reserved peaks, capture
and steady-state memory, and relevant pool retention; do not assume F2's B128
fit predicts graph capacity. Gradient accumulation remains one.

Use ten graph warmup iterations and three timed complete updates per selected
cell. Measure matched prepared-eager and graph executions. Report complete-step
wall/CUDA-event times including input copies, replay/forward/loss/backward,
clipping, AdamW and scheduler; exclude fixture preparation, W&B and reporting.
Also report an independently timed prepared-forward/loss/backward scope so
graph-region speedup is not mislabeled as complete training speedup. Every
scope records precisely which input copies, zeroing and optimizer operations
it contains. Include all timing samples and valid input/CE/pair/triple counts.
Use brief profiler observations only where needed to establish actual dispatch
or a leading bottleneck. No broad compiler sweep or FLOP estimate is required.

## Evidence and stopping point

Use only the project GPU container after verifying its working directory and
nvidia-smi. Log graphable metrics online under
taylorbollman/pretrained-fbt-rt-nextlat. Persist case-level progress and a final
olmo-f3-graph-training-v1 report with exact configuration, model/checkpoint pins,
runtime source hashes, frozen protocol hash, W&B identity and limitations.
Retain bounded reports, tests, source snapshots and plots in gs://fast-chunks;
verify and reference the existing immutable native checkpoint rather than
uploading disposable model or optimizer states from fixture updates.

Pause at this PR review point with explicit implemented/validated/untested
scope and practical batch guidance. No long learning, full-layer RT, distributed
execution or RT attention-kernel rewrite follows automatically.
