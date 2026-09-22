# O5b: first matched FBT-only recovery pilot

Frozen before comparative training, 2026-09-22. The user reviewed O5a and asked
to proceed. This authorizes one bounded paired pilot and its capacity/evaluation
preflight, not an automatic 100M+ continuation or an eight-arm interaction sweep.

## Question and arms

Can a gentle transition to the new feedback input recover code-continuation
quality without unacceptable general-language loss? This approximately21M-token
pilot measures adaptation direction and operational health, not FBT efficacy.

Both arms start from original OLMo-1B step60000 (~252B tokens), the same pinned
native weights, tokenizer and new fusion initialization validated in O5a.
Keep all native architecture details, including no Q/K normalization. **RT and
NextLat are off in both arms.** Do not load any O4-adapted weights.

- `ordinary`: two ordinary stack passes, beta0 throughout.
- `fbt`: ordinary pass0 plus one feedback pass, beta gradually increasing to1.

Both use K2/gamma1: **CE(pass0) + CE(pass1)**. Passes share parameters and
cross-pass gradients remain attached. Counts normalize each CE by the same
valid target positions; input exposure is counted once. Report pass0/pass1
losses separately; their sum is an optimization objective, not final-pass NLL.
Ordinary executes both passes explicitly. O4's single-CE result is contextual;
it is not substituted for this matching control.

Fusion is the O5a source-pinned asymmetric gate product, fixed native embedding
RMS scale0.0370765589, explicit epsilon1e-5 and FP32 RMS reductions. Beta0
bypasses it exactly. Only document starts remain plain at beta1. No sampled
prefix mixin or hidden-state jitter: this first mechanism/recovery diagnosis
deliberately differs from those author-reproduction training choices. No
separate fusion-only pretraining, latent auxiliary objective, compiler, CUDA
graphs, distributed wrapper or new normalization conversion is added.

## Data, optimizer and exposure

Reuse the exact O4 prepared stream, manifest hash and pinned datasets:
CodeSearchNet Python train as text; code dev and WikiText dev for evaluation.
Official tests remain reserved. Unknown original-pretraining overlap, independent
windows and nonstandard WikiText tokenization mean these are adaptation/retention
metrics, not published benchmark reproduction or generated-code correctness.

T512, effective batch32. Preflight selects physical32 if complete beta1 steps
stay below60GiB allocated; otherwise physical16 accumulated twice. Both arms
use the same selection. No batch-size tuning after comparative training begins.
The actual highest-memory full-length real windows are used in profiling.

FP32 master parameters/AdamW state, BF16 mixed forward/backward, ordinary SDPA,
TF32 off. Backbone LR1e-5; separate new-fusion LR1e-4. AdamW betas(.9,.95),
epsilon1e-8, matrix weight decay0.1, global norm clipping1.

Freeze the existing deterministic O4 exposure schedule at effective32:

1. 100 updates with beta0; backbone LR warms linearly to1e-5. This is **not a
   fusion warm start**, since beta0 has no fusion gradients.
2. Beta ramps over at least10M valid input tokens AND200 updates, using the
   smaller actual token/update progress fractions. First ramp update101 still
   uses beta0; first nonzero beta is update102. Fusion LR starts at1e-6 then
   warms over its first100 potentially active updates to1e-4 at update201.
3. Continue at beta1 for at least the actual ramp token AND update exposure.

The prepared stream resolves this to2,634 updates /20,855,799 valid input tokens
per arm. Warmup ends100; ramp ends1,367; beta1 exposure ends2,634. Both arms
see identical84,288 windows in identical order, with20,771,511 CE targets.
Compute uses two complete stack passes; report41,711,598 stack-input-token
positions separately from data exposure. No cycling or uncounted warm start.
The schedule helper retains historical `alpha` field names internally; O5b
explicitly interprets those fields as **beta**, with RT selection empty.

## Evaluation and decision rules

- At initialization, update50, warmup/ramp/end boundaries, and each1M-token
  threshold: fixed128-window code and retention prefixes, **each pass** scored.
- At endpoint: fixed512-window prefixes with per-window/original-document IDs
  for paired document-bootstrap comparisons. Keep these separate from curves.
- At initialization, ramp end and endpoint: first32 windows from each dev split,
  truncated to at most64 tokens, scored as pass0, finiteK2 and exact-online
  feedback on **identical tokens and masks**. Online is teacher-forced next-token
  prediction with freshly computed previous-token state, not generated-code
  task evaluation. Do not compare its short-context NLL to512-window NLL.
- Record finite loss, pre-clip global norm, post-clip native/fusion group norms,
  both LRs, beta, per-pass CE, input/CE/stack token counts, timing and peak memory
  online in `taylorbollman/pretrained-fbt-rt-nextlat`.

Nonfinite objectives/gradients fail before the optimizer step. Save only known
completed boundaries and verify parameters/moments before checkpointing. A
coarse catastrophic gate stops if the same pass exceeds its own initial NLL by
>1.5 nats in **both domains** at two consecutive scheduled evaluations. Modest
recovery gaps and ordinary code/retention tradeoffs do not trigger it. Report
the triggering pass/metrics; do not tune the recipe opportunistically mid-arm.

Save full optimizer checkpoints at warmup/ramp/end, every30minutes and requested
stops; verify immutable GCS copies before removing an older local checkpoint.
Both arms use one source/config/data inventory. Resumptions preserve configuration,
groups/schedulers, RNG and exact cursor; abandoned post-checkpoint evaluations
stay in audit history while authoritative curves refer to the restored lineage.

Final interpretation compares the matched new control, pass0 retention, feedback
pass and exact-online behavior, including document intervals and source/runtime
records. One seed and this limited exposure cannot establish architectural
efficacy. The author-reproduction training used much larger budgets. Review
recovery direction and cost before extending toward50–100M tokens or adding RT,
NextLat, prefix sampling or noise.
