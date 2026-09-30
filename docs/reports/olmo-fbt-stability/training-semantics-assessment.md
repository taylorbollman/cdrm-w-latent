# FBT training semantics: current study, paper, and Nanochat

2026-09-30. Bounded read-only review. The current F-only run continues
unchanged. This study uses Figure 3-style observations; it does **not** reproduce
Figure 3's training distribution or backbone.

The [current declaration](protocol.md) runs four total passes on every update:
one ordinary pass and three feedback passes. Its objective is

\[
L_F=\tfrac12 L_1+\tfrac16(L_2+L_3+L_4).
\]

Feedback remains differentiated through the earlier passes. Beta is fixed at
1, with independent keyed `Uniform[-0.02,0.02]` hidden-state noise before
fusion during training; probes remove noise. There is no prefix sampling or
fusion-strength ramp. The 100-update LR warmup changes optimizer step size,
not feedback strength. Source: [CE policy](../../../cdrm/pretrained/fbt_training.py#L44),
[finite-pass recurrence](../../../cdrm/pretrained/olmo_fbt.py#L363), and the
immutable F execution declaration.

The paper's relevant statements are:

| Topic | Original paper |
| --- | --- |
| Cross-pass gradients | Explicitly not detached (§3.3). |
| Objective | Ordinary CE plus the mean feedback CE; lambda1 (Eq.12). |
| Figure3 mixtures | Green: 75% K1/25% K2. Blue: 75% K1/22% K2/3% K3. |
| Exposure | Feedback introduced mid-training, first K2, then some deeper batches. |
| Prefix mixin | Ordinary prefixes train the boundary between ordinary prompt prefill and fused generation; an extra fused prefill is the alternative. |
| Jitter | Magnitude0.02; Appendix C adds noise to hidden states before fusion. |
| Ramp | No continuous fusion-strength or gradient-exposure ramp specified. Pass-frequency scheduling is distinct. |

[Paper §3.3, §4 and Appendix C](https://arxiv.org/html/2608.08888v1).
Its simplified pseudocode sums pass losses, whereas Eq.12 specifies averaging
the feedback losses. The retained implementation below follows Eq.12. The
paper's inference-time detached vLLM state is not a training stop-gradient.

The retained **author Nanochat implementation** at revision
`7037c60924870aca6e30fac95212b0c7caee052d` confirms the practical semantics:

- Its [multipass loop](../../../cdrm/pretrained/_fbt_reference/gpt.py#L849)
  carries attached hidden tensors through all passes and returns
  `L1 + mean(L2,...,LK)`. No intermediate optimizer step or gradient truncation
  appears. The [training driver](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/scripts/base_train.py#L765)
  backpropagates that complete loss; its detached values are logging copies.
- The [K2 continuation launch](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/runs/train_d20_lf_k2_modes_from40k_a100.slurm#L34)
  resumes update40000 with optimizer state, activates K2 immediately, and
  explicitly disables prefix mixin. There is no continuous fusion ramp in
  that path. This differs from the original paper's mixture.
- Even with prefix sampling disabled, Nanochat's
  [mask helper](../../../cdrm/pretrained/_fbt_reference/gpt.py#L117) keeps each
  BOS/document-start token ordinary. Our declared packed continuous-stream
  policy allows feedback across document boundaries. Thus “no prefix mixin”
  does not make boundary policies identical.

For K4, our CE objective is exactly half the retained implementation's
objective: the ordinary-to-total-feedback ratio is unchanged. Multiplying
the entire loss by one-half preserves its stationary points but halves raw
gradients. It changes clipping thresholds expressed in objective units and
can interact with Adam's epsilon/moment history; it is not automatically
equivalent to changing the LR. Report raw gradient norms with this convention.
There is no reason here to mistake a different loss scale for an absent
cross-pass gradient.

Our startup is also deliberate adaptation rather than paper reproduction:
[fusion128](../olmo-fusion-startup/warmup-results.md) trained only the two
fusion matrices in FP32 for 128 updates/1,048,576 CE targets, with the backbone
fixed and no auxiliary losses. The current study imports those weights with
the original OLMo backbone, then creates fresh Adam for active parameters.
It does not inherit either OLMo pretraining moments or the fusion warmup
optimizer. The Nanochat branch instead inherits its trained optimizer.

If later passes become unstable after useful short-pass learning, first
separate tail instability from first-pass degradation and from poor-but-settled
predictions. An agreed subsequent experiment could adjust pass exposure or
prefix coverage, one factor at a time. Neither detaching feedback nor adding
a gate ramp would repair a demonstrated source mismatch: both would be new
algorithmic choices. The present short run alone cannot attribute instability
to any one of exposure, optimization history, backbone, or data differences.

This review reuses the [retained source audit](../../../cdrm/pretrained/_fbt_reference/README.md)
and [Nanochat budget audit](../../fbt-nanochat-budget-audit.md). No GPU job,
numerical test grid, or frozen execution-source edit was performed.
