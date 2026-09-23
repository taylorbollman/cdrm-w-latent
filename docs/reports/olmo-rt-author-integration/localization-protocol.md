# Fixed-input, fixed-cotangent RT numerical localization

Prospective bounded diagnostic after actual-model native/author BF16 comparison
misses, despite exact own-backend graph and complete-Adam checks. No optimizer
updates or quality training. Do not change numerical budgets after observation.

Use original OLMo-1B step60000, the same deterministic RT-only B8/T512 full-CE
fixture at update0, selected RT layers0/15, CE2048, ordinary deterministic Flash
and ordinary activation checkpointing. Run exactly one native mixed forward and
mean-CE backward, intercepting the actual block0 input, completed output and
incoming output cotangent. Restore the interception even after failure. This
captures the true loss direction before clipping; never normalize this cotangent.

Free the rest of the model and hold block0 weights, input, positions and incoming
cotangent fixed. Evaluate local VJPs for native mixed, author legacy mixed,
author fp32_state with mixed dense layers, native full FP32 and author full FP32.
Native mixed retains both Stage A switches, Triton tiles and recompute backward;
author retains compiled helper boundaries, four MLP chunks and fresh private
cast caches. Full FP32 means no autocast and no TF32. Each candidate retains its
own recurrent forward trajectory. Fixed cotangents remove downstream-model
variation but do not by themselves distinguish local forward-state rounding
from local backward reconstruction rounding.

Also compare native and author legacy mixed with exactly the same actual input
and a deterministic Gaussian cotangent rescaled to the original cotangent's
global L2 norm. This changes direction without introducing a different overall
adjoint magnitude. It is not an additional Gaussian-input benchmark.

One explicitly diagnostic author legacy arm changes only backward reconstructed
attention: remove the permanent diagonal before PV and add temporary self
separately. Its forward must remain bitwise equal to legacy. Preserve BF16
dtypes and the original fullgraph helper boundary. Changed compiler fusion or
rounding may accompany this algebra change, so improvement would support this
reconstruction as a source, not uniquely establish an instruction-level cause.
No production/core source changes. At token0, reconstructed attention should
mathematically equal the temporary value; record that discrepancy directly.

Report each output/input-gradient/parameter-gradient comparison against native
FP32 and native mixed where applicable. Reuse existing FP32 and BF16 local
budgets as labeled screens, with no fabricated MSE/CE screen for a fixed external
cotangent. Mixed-versus-FP32 rows are descriptive. Completion means finite,
owned, unchanged fixtures and completed measurements; numerical screen misses
remain explicit and do not become a clearance claim.

Use torch.autograd.grad for local VJPs, preserving original parameter identities,
requires_grad flags and .grad ownership. Hash actual input/cotangents/positions,
block weights and source/fixture provenance; retain only hashes and small
metrics, never new weights or input tensors. Track online in the existing
pretrained-fbt-rt-nextlat project. Freeze this script, all imported runtime
sources and this protocol before GPU work. Root runs one bounded CUDA process
inside the required container. No localizer timing is a throughput benchmark.
