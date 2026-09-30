# Decision criteria after F128 and the paired NFR continuation

Prepared 2026-09-30 while the already authorized NFR KL1/KL0.1 comparison is
running. This note proposes work for user review; it authorizes no additional
execution. Preserve the declared stop at update 64 for each branch. F means
FBT alone; NF adds NextLat; NFR adds both NextLat and native RT.

## What is established, and what is not

F128 demonstrates that the current feedback path can adapt without a gate
ramp or model redesign. On the regular development panel, pass 4 CE falls from
7.171813 to 2.942000 while first-pass CE ends at 2.687038, close to ordinary
B128's 2.687618. All 128 updates are finite; the final 12 no longer clip. This
supports functionality and first-pass retention on the measured workload.
It does **not** establish useful refinement: pass 4 remains 0.255 nats worse
than pass 1. See the [training results](results.md).

Settling is a separate property. Every measured F/NF/NFR checkpoint settles
by K32 on the eight-row panel, including poor predictors. NF32 and NFR32
settle faster than F32 despite worse CE. The two isolated F128 crops directly
agree with exact-online execution near 1e-6 at K32; K4 still has 1.08% hidden
error despite a tiny mean CE difference. Those observations cover neither
general generation behavior nor combined-model exact-online equivalence.
See [post-diagnostics](post-diagnostics.md). They add no general BF16 clearance.

## How to assess the NFR pair

Use the restored 32, intermediate 48 and terminal 64 evaluations on the same
regular 64-row panel. The branches share the complete NFR32 state, including
Adam, and differ only in KL weight. Inspect absolute first/later-pass CE,
their gaps, unweighted latent/KL losses, finite updates, preclip norms and
checkpoint/preservation results. A smaller weighted total is not a comparable
quality score. Heavy clipping alone is not an execution failure.

| Paired outcome | Interpretation and decision |
| --- | --- |
| KL0.1 improves later CE while preserving or improving pass 1 | The NF loss-balance result transfers to this combined setting over the measured interval. Treat 0.1 as an operational candidate, not proof that KL is generally harmful or RT beneficial. |
| Both improve similarly | Consistent with recovery from continued adaptation; this interval provides little evidence for a coefficient preference. Retain the declared control unless other measurements justify a choice. |
| Smaller pass gap comes from worse pass 1 | Do not count it as successful refinement or retention. Judge absolute losses separately. |
| Better CE accompanies higher raw latent/KL loss | Report the tradeoff, as in the prior NF pair. This is not improvement in every objective. |
| Lower gradient norm without better CE/retention | Optimizer behavior changed; predictive benefit remains unestablished. |
| One branch has nonfinite values or fails preservation | Treat it as an execution/health issue, retain evidence and localize it before extending. Do not diagnose an architectural cause from that outcome alone. |

The [NF pair](../olmo-kl-continuation/results.md) improved every pass's CE with
KL0.1, but worsened both raw auxiliary losses relative to control. F-versus-NF
removes both auxiliary losses; it cannot independently attribute that effect
to KL. NFR-versus-NF also has different learned trajectories. Neither
comparison establishes RT's causal value.

## Smallest sensible follow-up after review

If a candidate completes cleanly with improving feedback prediction and
acceptable first-pass retention, propose **one unchanged NFR64→128
continuation**, retaining its exact optimizer, schedule, data order and
architecture. Its question is whether the combined model remains healthy
through the existing update 100 warmup boundary and continues adapting.
Inspect saved 96/100/128 states using the existing measurement conventions.
This is a bounded functionality/adaptation study, not a quality competition;
the unextended branch's 64 endpoint is not a matched 128 control. A stronger
causal claim would require a separately agreed matched continuation.

If the pair is ambiguous, or a new concern appears, do less first: inspect the
relevant saved endpoint with a bounded no-update probe addressing that
specific concern. Examples are deeper-pass/tail behavior or per-loss gradient
balance, not another broad precision grid. Authenticating NFR64 in a new
diagnostic scope must be explicit; the existing saved-component helper admits
NFR32 only. Do not silently repurpose its checkpoint allowlist.

No outcome here automatically calls for a fusion ramp, detached feedback,
new Q/K normalization, altered RT placement or a pass-frequency sweep.
Poor feedback prediction and mathematical iteration instability are different
problems; the present evidence mostly shows the former recovering.

## Efficiency and recovery before spending more

First consolidate existing measurements into one scope-consistent ledger:
real input tokens/s, full wall time, memory, parameter ownership, and clearly
labeled estimated FLOPs where an existing validated estimate applies. Count
inputs once despite K4. F recorded 10,746 inputs/s for compute plus materialization, 10,207 in
broader timed update regions including scheduled development/probes, and
7,014 over its full diagnostic executor; those denominators differ. See the
[resource ledger](resource-ledger.md) for the exact timing scopes.
Checkpoint waits and local save regions account for substantial overhead.

A later storage-only milestone can coalesce near-adjacent named and wall-time
saves while retaining resumable checkpoints and asynchronous cloud upload.
That is an efficiency opportunity, not a reason to change live training or
weaken state correctness. Do not repeat batch/backend benchmarks merely to
confirm an already measured setting. Neither a larger training budget nor an
implementation optimization should be mistaken for evidence of model value.
