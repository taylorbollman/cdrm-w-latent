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

The optional exact-online comparison uses two explicitly recorded, isolated
document crops. Within each crop, finite-pass and sequential execution use the
same tokens and reset context. Its CE is not directly comparable with packed
1024-token development CE. It answers whether finite passes approximate the
sequential process on those crops, not whether generation quality improves.

A transition between saved checkpoints is localized only to that interval.
Eight-update short probes help identify where to inspect; retained resumable
states permit a targeted replay if a sharper transition is worth resolving.
An unchanged 128-to-192 extension can examine behavior beyond the learning-rate
warmup, but a short run without recovery does not establish that FBT cannot work.
