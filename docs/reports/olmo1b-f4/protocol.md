# F4 prospective training-resource protocol

Frozen before GPU execution, 2026-09-23. User authorizes functionality and
resource measurements, not quality training. Base: PR19 / main `deecfe6`.

## Question and fixed execution

Measure all eight independent RT / FBT / NextLat switches on one H100 80GB.
RT means selected layers `(0,15)`, a representative two-layer execution layout,
not a placement recommendation. FBT means K2 (ordinary bootstrap followed by
one feedback stack); selected RT runs in that second stack only. No FBT means
one stack. NextLat is a training-only predictor, with no autonomous recurrence.
Keep native OLMo-1B step60000 (~252B tokens), RoPE, tied weights and no Q/K norm.
Keep F3e's BF16 autocast/FP32 parameters, gradients and Adam states, TF32 off,
no autocast weight cache, deterministic ordinary Flash, ordinary activation
checkpointing, RT weight-cast reuse and Triton historical forward/backward
with bounded recompute workspace. No kernel or model-math changes.

Names: ordinary, rt, fbt, nextlat, rt-fbt, rt-nextlat, fbt-nextlat, combined.
All start from the same pinned native checkpoint and seed20260922. Auxiliary
branches follow their existing initialization. LR1e-5, AdamW(.9,.95), eps1e-8,
weight decay.1, two-update LR warmup and canonical clipping are unchanged.
This is a few-update execution fixture, not a learning-rate recommendation.

## Checks and measurements

1. Six newly covered graph combinations: ordinary, nextlat, fbt and
   fbt-nextlat at B1/T32; rt-nextlat and rt-fbt at B8/T512. Also combined
   B1/T32 for a fresh operator audit. Reuse F3e RT-only B8/T512 and combined
   B8/T512 correctness explicitly; do not count those updates again.
2. Each correctness run compares optimized backward to materialized reference,
   then same-candidate eager/graph initial and changed tokens, repeated replay
   overwrite, three eager versus three graph Adam updates and changed weights.
   Require full active gradient/loss inventories, finite model/moments and
   matching state/counters. Existing numerical budgets remain unchanged:
   global gradient relative L2 <=1/64; per-tensor <=1/32; peak-coordinate
   error/reference-tensor-peak <=1/16; zero references require exact zero.
   Initial forward losses must be bitwise; graph/Adam parity must be exact.
   This is not a new full-model FP32 versus BF16 validation campaign.
3. All eight common capacity cards: physical/global B64, T512, accumulation1.
   Three eager complete updates initialize Adam, ten backward warmups precede
   capture, then three changed-input/weight complete graph updates are timed.
   Timing includes input copy, validation, forward/loss/backward, clipping,
   AdamW and scheduler, but excludes tokenization, W&B, tracing and artifact I/O.
   Report all three wall/device times and their medians; no significance claim.
4. After the common table, allow one larger B96/T512 check for each cell whose
   B64 peak reserved memory is below65GiB. This is bounded directional scaling,
   not a maximum-batch search. Label >72GiB peak reservation as tight setup
   headroom even if successful. On OOM retain the failure; keep the B64 card.
   Stop on numerical/parity failures and investigate before dependent work.
5. Untimed B1/T32 ordinary and combined eager backward traces audit selected
   operator FLOPs, shapes, ordinary Flash and device kernels. Retain compressed
   traces. These are incomplete operation counters, not measured hardware FLOPs:
   fused Flash/Triton and custom/recomputed work need analytic accounting.
   Independent small CPU matrix-shape oracle tests already cover the estimator.

The real-text fixture has one full unpadded document per row. At B64/T512:
32,768 input tokens and16,384 CE targets/update; when NextLat is enabled,
32,704 latent pairs and16,384 KL triples, with32,704 predictor positions
(the union, not the sum). FBT executes two stacks but does not double data
exposure. Each term and pass retains its actual denominator/coefficient.

## Accounting and retention

Report registered/resident, active/trainable, gradient-participating,
optimizer-owned and deployable parameter counts, deduplicating tied/shared
weights. The harness retains frozen8,388,608 fusion weights even when FBT is
inactive. NextLat82,726,912 weights are absent at deployment. RT adds none.
Report analytic matrix FLOPs/update as a range including checkpointing, RT
permanent KV and recomputation, K passes, fusion, predictor, CE/KL readouts.
Exclude pointwise, optimizer, communication, launch and hardware-padding work.
Do not divide by K when reporting data-token throughput.

Split setup and timed steady memory: record allocated/reserved current and
peaks for each, plus their maximum. Do not equate reserved with live tensors.
Three-update medians are directional; successful capture does not establish
long-run stability, maximum safe batch, or future larger-model headroom.

Freeze all transitive runtime sources and protocol, snapshot per run, verify
hashes afterward. Online W&B: taylorbollman/pretrained-fbt-rt-nextlat,
group olmo1b-f4-features. Retain small evidence at
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f4-features/`.
Keep the pinned native checkpoint, not disposable few-update model copies.

## Explicit limits and reused evidence

F3e provides native T2048, K3, two/four/all-layer checks with its original
shapes, hashes and qualifications; those are not a new all-eight long-context
matrix. Its RT B128/T512 succeeds but reserves76.80GiB during setup; B64 is
our common conservative reference. No native FA4 RT integration claim: RT
is Triton, ordinary attention is Flash. Native Q/K stays unchanged.

This milestone completes the **training** resource comparison. Finite prefill
and exact-online/decode resource cards remain separate readiness work, with
four deployment routes because NextLat disappears. Graph recovery,
accumulation, padding and genuine two-GPU tests also remain. Only one GPU is
available. Training throughput is not an inference-throughput estimate.
