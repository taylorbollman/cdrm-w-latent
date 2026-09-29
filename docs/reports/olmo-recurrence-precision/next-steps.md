# Decision after recurrence separation

The matrix identifies substantial sensitivity with either recurrence mechanism,
with the largest difference in their combination. Ordinary OLMo is much closer
to FP32 on the same fixture. This is a useful separation of configurations,
but it does not identify one faulty operation. The existing attention-local
checks and exact eager/Triton RT comparison remain relevant.

The one conditional correction was unsuccessful under its written criterion.
FP32 fusion improves the NF backbone gradient difference from 60.87% to 56.16%,
but worsens final hidden-state agreement on both records. Fusion's own gradient
relative L2 also increases, although its cosine improves. Keep production
precision unchanged and retain this result. Do not continue promoting modules
without a new diagnostic reason.

## Recommended bounded continuation: replay fixed module boundaries

Distinguish two effects that the full-model comparison currently mixes:

1. A module receives different hidden states because earlier computation rounded
   differently; even an exact local derivative can then change substantially.
2. At identical module inputs and incoming gradients, changing precision or its
   backend changes the module's output and derivative.

Use NF first, since it shows a substantial discrepancy without RT. Freeze a new
protocol and helper; do not edit or rerun the completed source inventories as a
new experiment under their old names. The proposed scope is two exact NF anchor
cases (FP32 and production BF16, four physical backwards) to capture boundaries,
then **12 local vector-Jacobian products (VJPs)**:

- Record 0, first and final feedback transitions, entering passes 1 and 3.
- At each transition, inspect the fusion module and the following ordinary
  backbone stack. For the stack, capture its actual input after feedback mixing,
  jitter and masking, not an approximation reconstructed from a report.
- For each module/site, evaluate FP32 at its FP32-origin inputs (A), FP32 at its
  BF16-origin inputs (B), and production BF16 at those same BF16-origin inputs
  (C). Use one common incoming gradient captured from the production BF16
  execution, detached and unchanged in all three evaluations.

A versus B measures response to inherited input perturbations with arithmetic
held fixed. B versus C measures local precision/backend differences at a common
boundary state. Measure output differences and input/parameter VJP geometry;
report unused parameters explicitly. Preserve full tensor shapes, masks and
cotangents for the computation, even where summaries select only valid tokens.
Require C to reproduce its captured local BF16 outputs exactly, and require both
full-model anchors to reproduce the saved NF endpoints before interpreting it.

For an entire stack, B versus C still includes rounding differences in its
internal forward activations. It is therefore a module-level numerical
comparison, not a backward-kernel error detector by itself. Do not add relative
error norms or interpret this local decomposition as additive contributions to
the full-model gradient discrepancy. Sampling two sites cannot clear every
pass, record or RT configuration.

Stop after this diagnostic and choose the next action from the evidence. A large
matched-input local difference motivates inspecting that module internally. A
small local difference with a large inherited-input effect motivates examining
feedback sensitivity and initialization/transition strength, rather than
assuming a custom backward bug. Q/K normalization is not the default response.

## What remains before changing training policy

These measurements use one short isolated fixture, the pretrained ordinary
backbone and **untrained, full-strength beta=1 feedback**. They establish
sensitivity at this initial state. They do not establish that BF16 training will
fail, that a trained feedback model behaves similarly, or that an observed
scalar-loss improvement resolves gradient agreement. Relative L2 is a vector
distance divided by the FP32 gradient norm, not a fraction of incorrect
parameters or predictions.

Any actual remedy or change in acceptance needs separate confirmation with the
combined model, both NextLat losses and packed T1024 data. The earlier auxiliary
loss-layout qualifications (3.40224% isolated and 1.6953% packed) remain open.
Graph/restart/performance checks become necessary if the selected runtime policy
changes. No quality training, Q/K-normalization transition, H200 qualification or
additional GPU experiment is launched by this recommendation.
