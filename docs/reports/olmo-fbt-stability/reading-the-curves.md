# Reading the FBT-only curves

The horizontal axis is the total number of model passes over the same held-out
tokens. Pass 1 is ordinary OLMo; training uses passes 1 through 4. Each colored
curve holds the model weights fixed at one optimizer checkpoint. The later
passes on a curve are additional inference computation, not additional training.

Three questions should be read together:

1. **Does repeated feedback settle?** Relative hidden change is
   `RMS(h[k] - h[k-1]) / RMS(h[k-1])`. Falling values indicate empirical settling
   on this panel. A plateau, oscillation or increase motivates closer inspection.
   Neither a single low value nor a shrinking curve proves global contraction.
2. **Does it settle to useful predictions?** Compare CE at passes 2, 4, 8 and 32
   with the same checkpoint's first pass. A stable but much worse later-pass CE
   is not useful refinement. Compare the first pass with the ordinary trained
   control to detect a retention cost from feedback training.
3. **Has the representation merely changed scale or become less informative?**
   Read pre-final-normalization RMS, final hidden RMS, stack-input RMS and
   predictive entropy alongside CE and state change. Scale changes or reduced
   position variation require interpretation; these alone do not prove collapse.

Because feedback uses preceding positions, more passes necessarily settle an
increasing prefix in exact causal arithmetic. The main change curve excludes
that guaranteed prefix. The fixed last-128-position curve provides a check with
unchanged positional membership across pass counts. Report both, since the
unsettled-suffix membership itself changes with the pass number.

The small curves use eight fixed development rows and noiseless FP32 evaluation.
Training uses BF16 mixed precision and feedback jitter. The regular development
evaluation covers 64 rows. Their CE values have different sampling variation and
must not be spliced into one apparent learning curve. No final confirmation set
is used for this diagnostic.

The completed F128 exact-online comparison uses two explicitly recorded,
isolated document crops. Within each crop, finite-pass and sequential execution use the
same tokens and reset context. Its CE is not directly comparable with packed
1024-token development CE. It answers whether finite passes approximate the
sequential process on those crops, not whether generation quality improves.
At K32, hidden and logit relative errors are around 1e-6. At K4, nearly equal
mean CE still accompanies 1.08% hidden error. See [post-diagnostics](post-diagnostics.md).

## What the completed observations mean

The origin already settled while predicting badly. Repeating feedback was
therefore not visibly diverging on this panel; it was arriving consistently
at a poor predictor. During F-only training, the main improvement was in
**what it predicted after settling**, rather than a transition from an unstable
iteration to a stable one. This origin includes 128 earlier fusion-only updates,
so we cannot say when settling first emerged. It also does not reproduce the
paper's Figure 3 training distribution.

Lowering KL weight in both completed NF and NFR comparisons improved every
pass's CE, while raw latent and KL losses became worse. At NFR update 64,
first/fourth-pass CE improves by 0.155 / 1.127 nats relative to control, but
the fourth pass still trails its own first pass by 2.745 nats. See the
[paired curves](../olmo-nfr-kl-continuation/figures/development-raw-losses.pdf).
This is a tradeoff among training objectives: better next-token prediction,
less agreement with the auxiliary targets. The lower weighted objective cannot
itself establish improvement because its definition changed. Lowering KL also
changes gradient scales and clipping, so these results do not identify a single
mechanism or prove that the auxiliary objective is generally harmful.

A transition between saved checkpoints is localized only to that interval.
Eight-update short probes help identify where to inspect; retained resumable
states permit a targeted replay if a sharper transition is worth resolving.
F stopped at 128; no extension to 192 is queued. The current
[next-step criteria](next-steps.md) prioritize bounded NFR endpoint dynamics
before considering further training.
