# Four paired NFR functionality updates

2026-09-29. The fixed-strength diagnostic found that alpha0 native scans have
about0.70% backbone BF16/FP32 gradient disagreement, whereas alpha0.25 andalpha1
have about9.91% and12.06%. The smallest next step tests optimizer impact with
all components active. It does not tune recurrence to reduce a numerical score.

Strictly import original OLMo backbone/tied readout, fresh predictor and saved
NF fusion128, then enable the existing NFR mode: layers0/15, alpha1 on every
pass, K4, beta1, jitter0.02, isolated documents. All original diagnostic parameters
are trainable. The checkpoint's fusion Adam history is deliberately discarded;
both trajectories start from identical **fresh campaign Adam** across backbone,
fusion and predictor. This is a new diagnostic optimizer clock, not continuation
of missing backbone moments or of the NF fusion-only optimization.

Use exactly prepared training selections144..147 from the immutable startup
data helper, after the separate fusion continuation's128..143. Each has8,192
CE targets at B8/T128. Report actual input tokens, latent/KL counts, windows and
padding; windows can share context across selection boundaries with complementary
target masks. This small diagnostic budget differs from the campaign's524,288
valid-input-token update. Do not change the production budget or retokenize data.

Execute four full-parameter updates for FP32/math/eager and four for production
BF16/ordinary Flash/native Triton, eight optimizer calls total. Reuse canonical
combined CE/latent/KL backward, original pass weights and stop-gradients. Use
`build_campaign_adamw` with campaign defaults: plateau LR2e-4, betas(.9,.95),
epsilon1e-5, decay.1 with existing tied-embedding/norm/bias exclusions, clip1.
Use the original52,428,800 valid-token warmup
and floor fraction0.1, hence first LR2e-5. The scheduler advances by actual valid
input tokens in each diagnostic update; it does not pretend each consumed a
production batch. No accelerated adaptation, LR sweep or precision tolerance.
Use the ordinary non-fused Adam implementation for this bounded measurement so
CPU tests and GPU observations share optimizer semantics; this is not throughput
qualification or a proposal to replace production fused Adam.

Keep two independent CPU snapshots of complete model, Adam, scheduler, RNG,
module modes and committed counters. Before each path, validate and restore its
own boundary without replacing parameter objects, then use identical tokens,
masks and keyed noise. No graph or live backward exists during restoration.
CPU oracle tests must show snapshots do not alias live optimizer tensors or each
other, exact uninterrupted-versus-restored next updates, unchanged parameter
identities and rejected invalid clocks/ownership/moments before mutation.

Compare raw/clipped gradients, actual master deltas, Adam moments and parameter
separation per parameter/group. Keep FP32 CPU snapshots only; generate FP64
differences per parameter during reduction. Discard temporary gradient/delta
observations after each pair and retain compact geometry and hashes. The first
pair starts from identical weights; subsequent differences include trajectory
divergence and are not same-state rounding measurements. Fresh Adam's first
update can suppress gradient-magnitude differences, so examine all four steps.

Evaluate the fixed held-out long development fixture at0 and4 using the same
read-only FP32 path for both trajectories. Record all three losses and their
combined objective. Preserve parameter values, flags, modes, RNG, input/noise
and absent gradients; no held-out optimization or quality claim is made.

Save both full generic training checkpoints at4 and at completed paired
boundaries if ten minutes elapse sooner. Upload each immutable file to GCS with
the existing generation-pinned download verification. These are about15GB per
trajectory; leave local files intact. The existing generic checkpoint schema and
strict source/configuration/ownership loader remain unchanged; the new helper
adds diagnostic counters/data-cursor checks and CPU restart tests. A paired
save is complete only after both retained receipts exist. No emergency save is
attempted after a failed optimizer update. Cadence is checked at safe completed
pairs; a long update or checkpoint I/O can exceed that interval.

Root alone launches this single-GPU diagnostic after source/tests/review freeze,
with deterministic setup before CUDA, TF32 and autocast caches disabled, forced
attention context through backward, online W&B and incremental persistent
reports. Set a bounded external timeout from a short preflight/observed update
time, with room for two full checkpoint uploads. No DDP, graphs, packed-context,
full campaign restart, long-run stability or BF16 production clearance follows.
