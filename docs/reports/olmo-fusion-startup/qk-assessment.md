# Q/K normalization: what the retained evidence supports

2026-09-29. **Keep native OLMo Q/K behavior for the current bounded checks.**
Existing observations identify sensitivity to feedback state and temporal
recurrence. They do not identify uncontrolled attention-score scale, or missing
Q/K normalization, as its cause. This is a qualified decision to preserve the
pretrained function, not clearance of attention health at every learned state.
This note reuses retained observations; it adds no GPU run or numerical probe.

The [F2 attention table](../olmo1b-f2/results.md#observed-attention-scale)
sampled actual post-RoPE Q/K at layers0/1/7/15 on original weights, B2/T32 and
B2/T128. RT selected only layer0; feedback used K2/K3 with the older
pass/objective policy. In the ordinary T128 layer0, Q/permanent-K RMS was
0.957/1.291. At feedback passes1/2, it was approximately0.463/0.460 and
0.442/0.445. Reconstructed normalized entropy was0.985/0.989, versus0.733
in the ordinary first pass; maximum-probability p95 was0.278/0.246, versus0.761.
Thus that sampled feedback attention was **flatter**, not more concentrated.
Other layers included concentrated heads, and the RT-only T32 layer0 p95 was
0.992; concentration by itself is not an error. F2 excluded padding, future
keys and self-only rows from concentration summaries. These are FP32 score
reconstructions from actual operands, not private fused-kernel probabilities.

The later [attention-local report](../../../.runtime/olmo-precision-localization/attention-local-01/report.json)
is closer to the current architecture: cold NFR K4, RT0/15 on every pass,
beta1/jitter0.02, CE-only gradient with zero auxiliary cotangents, two B2/T16
records. It captured eight ordinary-attention sites, layers1/14 and passes0/3,
excluding checkpoint recomputation. The following pairs give record0/record1:

| Ordinary layer / pass | Largest within-query logit range | Mean maximum attention probability |
|---|---:|---:|
| 1 / 0 | 14.578 /12.835 | 0.613 /0.811 |
| 1 / 3 | 7.018 /7.006 | 0.467 /0.672 |
| 14 / 0 | 7.827 /6.341 | 0.678 /0.741 |
| 14 / 3 | 8.615 /5.299 | 0.774 /0.758 |

All reconstructed scores were finite; sampled extremes were−11.877 and11.535.
Layer1 became less concentrated after feedback; layer14 did not uniformly do
so. **Unlike F2, these local summaries include self-only first queries.** Their
reported maximum probability1.0 is therefore expected and cannot establish
softmax saturation. The means are length-dependent too. No saturation rate or
adapted-state Q/K verdict follows from this table.

At those exact captured inputs and cotangents, local BF16 Flash versus promoted
FP32 math gave output relative L2 of0.160–0.186%, dQ0.247–1.501%,
dK0.229–0.684% and dV0.174–0.222%; all local Flash outputs reproduced the
production hashes. This did not reproduce the much larger whole-model split
locally. It does not clear unobserved attention calls or identify a specific
source of amplification. See the [local interpretation](../olmo-precision-localization/results.md#fixed-input-attention-local-differences-are-much-smaller).

The [F4 roundoff assessment](../olmo1b-f4/roundoff-assessment.md) likewise found
roughly18% BF16/FP32 gradient disagreement shared by materialized and recompute
RT+FBT, B8/T512/K2 without NextLat. FP32 changed several forward/backward
arithmetic paths together; that evidence did not localize a Q/K-scale fault.
Exact graph/Adam repeats established execution consistency, not Q/K health.

Today's [fusion startup](warmup-results.md) sharply reduced NF precision
sensitivity without changing backbone Q/K parameters or adding normalization:
held-out T128 backbone discrepancy144.02%→1.63%, with absolute error falling too.
This demonstrates that a no-QK-normalization model can become much less
sensitive after adapting its feedback input. It does **not** show which
internal activation or attention statistic improved. The [RT-strength checks](rt-strength-and-baseline-results.md)
then found combined-loss backbone differences0.704%,9.912%,12.059% at
alpha0/0.25/1. Those results implicate the recurrent function/state transition
as a useful next boundary; changing alpha also changes hidden states and
Jacobians, so they cannot specifically implicate Q/K normalization.

We have **not** repeated these Q/K/score/concentration observations at fusion128,
the full-NFR update4 endpoint, later continuation checkpoints, or packed T1024
with all three components active. Nor have we measured every layer/pass or
actual RT temporary/permanent attention saturation in those adapted states.
If a concrete instability emerges there, a bounded comparison of actual Q/K
RMS, allowed score ranges and multi-key concentration against the matching
ordinary/state reference would discriminate a scale concern. A large gradient
percentage alone is insufficient reason to change the pretrained architecture.

Evidence pins: attention-local report SHA256
`aa105ea3d1678f787840dc84d85997ed9e8ba60c33ae007f007aa3723717f51e`,
retained attention fixture
`c08fa685da76f211fb50132db44af98e1d0c747d26c96480a5c082ee6b4e147c`.
F2 raw stages are `f2-health-01` and `f2-health-t128-01` under
`.runtime/olmo1b-step60000/`; their pins and sources are retained in the
[F2 summary](../olmo1b-f2/summary.json), SHA256
`1859a8616dba58d4605a4fbc716af83ca78c66f19cd823235374f925a804d1cb`.
The [F4 roundoff summary](../olmo1b-f4/roundoff-summary.json) is SHA256
`18688b9a356b754079b45bac82c6842a00847cdbb98593a648921cb805f9f140`.
