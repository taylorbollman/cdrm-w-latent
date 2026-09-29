# Assessment and next readiness milestone

2026-09-29. **A bounded BF16 functionality pilot from the explicitly adapted
startup is reasonable. A cold-start, long, all-arm quality campaign is not yet
cleared by these results.** The most useful next work is a manifest-driven
training entrypoint with recoverable execution, rather than another numerical
GPU sweep. Preserve the native model, losses, RT kernels and Q/K behavior unless
new evidence identifies a concrete problem.

Here NF means NextLat plus FBT feedback; NFR additionally activates native
temporal RT at layers 0/15. K4 means four feedback passes through shared weights.

## What the BF16 evidence supports

The [FP32 fusion-only warmup](warmup-results.md) substantially improves several
matched-state precision comparisons without modifying the pretrained backbone.
At its endpoint, NF combined-loss backbone gradients differ by 0.735% across
precisions, but adding RT gives 12.059% on the same T128 fixture; CE-only NFR
gives 32.442%. Different objectives and denominators prevent treating the smaller
combined percentage as proof that auxiliary losses repair the CE discrepancy.

Actual optimization is more informative than these percentages alone. The
[first four full-NFR updates](nfr-updates-results.md) are finite and improve
fixed held-out losses in both precisions, with a visible BF16 deficit. The
subsequent [16-update conditional comparison](nfr-continuation-results.md)
finishes with closely matched held-out CE: 5.669885 FP32 versus 5.667641 BF16,
evaluated through the same FP32 path. Both recover from a midpoint CE regression;
KL and latent losses improve throughout the measured held-out checkpoints.
First-pass CE remains slightly worse than the shared origin, and later-pass CE
is still much higher than first-pass CE. This is not evidence of useful
refinement or a quality win.

Close losses do not imply equal optimization. The conditional paths' cumulative
backbone changes differ by **16.27%**, cosine 0.986783, despite similar update
norms. Fusion/predictor changes differ by 1.40%/1.45%. The separate fusion-only
continuation also produced close losses with an 18.41% difference in cumulative
fusion updates. These are trajectory comparisons after weights and moments
diverge, not repeated same-state backward measurements. Four held-out documents
and short isolated-T128 training cannot establish long-run equivalence.

The [saved-state packed T1024 bridge](packed-bridge-results.md) confirms that
prepared BF16 and captured BF16 produce bitwise-identical losses and all 71
gradient tensors. FP32 sparse/prepared semantics pass. At that particular
adapted state, BF16-versus-FP32 prepared gradients differ by 1.0955% globally.
The separate BF16 sparse/prepared difference is 0.08792% globally, with
50/71 tensors failing the unchanged elementwise budget. Preserve this layout
qualification; graph equality does not erase it. This bridge performed no
optimizer updates and used a different fixture and document policy from the
T128 continuation, so it does not demonstrate packed training convergence or
attribute a numerical reduction solely to adaptation.

## Startup is part of the experiment

| Regime | Actual state and optimizer | What it establishes |
| --- | --- | --- |
| Original pretrained backbone + fresh fusion/predictor | No adaptation; fresh optimizer | Original authority and cold-state diagnostics, including large precision discrepancies |
| Fusion warmup 128 | Only fusion matrices adapted by FP32 CE; backbone/predictor frozen; fusion-only Adam | Improved NF precision sensitivity; its 128→144 continuation inherits those fusion moments |
| Full NFR updates 0→4 | Original backbone/predictor + fusion128, fresh Adam for all active components | Matched initial full-model FP32/BF16 update comparison; no inherited fusion-only moments |
| Conditional NFR updates 4→20 | Both precisions inherit the **same BF16 update-4 weights and full Adam history**; scheduler prefix extended | Subsequent precision sensitivity from that shared history, not independent FP32/BF16 training from a cold start |

The current [manifest resolver](../olmo-campaign-manifest/protocol.md) supports
only original pretrained startup with a fresh optimizer. It deliberately rejects
adapted imports and inherited Adam. A future launcher must not silently label
one of the other regimes as that original startup. Any quality comparison must
declare and account for adaptation tokens, objectives, trainability, optimizer
resets and schedule changes; an adapted NFR checkpoint is not an exposure-matched
baseline against untouched ordinary OLMo.

## Recommended next milestone: one manifest-driven execution path

The next authorized implementation is an **ordinary-B-only launch adapter**;
its code review and runtime acceptance are pending. It is gated on the current
pretrained B reference/resume acceptance. Use
the validated manifest to drive three small T1024 updates and an exact
cloud-restored continuation. Explicitly restrict this first adapter to original
startup, two ranks and the accepted BF16 captured path; reject unsupported arms,
startup and backend declarations. Derive the recipe, data membership, schedule,
allocation and checkpoint policy from the manifest, without monkeypatching the
frozen diagnostic runner or silently applying its hardcoded defaults. Keep the
manifest and launcher source identities in the new checkpoint lineage. This
adds real configuration-to-execution coverage; an all-arm startup abstraction
alone would leave that integration untested. The following steps describe the
broader extension after that bounded entrypoint works.

1. **Connect the resolved manifest to the common host loop.** Bind model/data/
   index/source pins, component ownership, per-arm physical allocation, exact
   masks and global loss counts, keyed feedback noise, committed cursor, token
   schedule, precision/backend policy and checkpoint lineage. Keep the same
   logical data exposure across arms while allowing different physical batches
   and accumulation. The resolver is a validated plan, not a training launcher.
2. **Make startup explicit before using adapted evidence.** First preserve the
   existing original/fresh-optimizer contract. Add a separately named, strictly
   validated adapted-start or resume path only where needed for the bounded
   functional pilot. Reuse tested imports and record any policy transition;
   never weaken historical checkpoint guards. Continuing saved NFR update20 is
   suitable for execution checks, but would carry its prior training history.
3. **Accept the new integration with small tests.** Exercise all eight component
   selections on tiny CPU models, including disabled losses and parameter
   ownership. Reuse completed two-GPU lifecycle, rank-failure and evaluation
   insertion evidence. Finish and assess the already planned ordinary pretrained
   B reference/cloud-restore/resume acceptance before calling that integration
   ready. Then check only the additional manifest dispatch and recovery behavior
   on representative B/NFR paths; expand coverage if another arm introduces
   untested ownership, graph or unused-parameter behavior. This is operational
   acceptance, not another broad precision sweep.
4. **Freeze the first actual pilot contract.** Select the training mixture,
   finite token budget, disjoint evaluation set and per-pass reporting before
   launch. Replace readiness-corpus defaults deliberately. On the target GPU
   topology, measure steady-state memory and input tokens/sec after warmup,
   with checkpoint/evaluation costs reported separately. Retain analytical work
   and parameter counts per arm. H200 capacity and throughput require a short
   hardware check; H100 evidence is not an H200 batch-size guarantee.

The resolver's current 49,152-token readiness example covers only two books
documents. It is appropriate for checking data/clock plumbing, not a selected
pretraining mixture or a representative learning pilot.

For an initial adapted BF16 pilot, keep finite-value/participation checks,
gradient norms and clipping, separate CE/latent/KL and per-pass held-out losses,
and verified completed-boundary checkpoints at most ten minutes apart. A small
common-FP32 observation at a predeclared boundary can anchor interpretation
without duplicating an entire training run. If losses or updates behave
unexpectedly, compare matched states and data before changing architecture.
Do not silently revise numerical budgets or call close combined loss a pass
when CE degrades. A longer quality or RT-value experiment is a subsequent
decision, informed by this functional pilot and actual resource costs.

## Q/K and remaining limits

The [Q/K assessment](qk-assessment.md) found no identified normalization-scale
fault. Fusion adaptation reduced sensitivity without changing Q/K normalization;
the RT-strength check points toward recurrence/state sensitivity, not a proven
kernel or normalization defect. Keep native OLMo Q/K behavior. Only a concrete
new concern warrants bounded adapted-state Q/K/logit/concentration measurements
and then a separately controlled intervention.

No further numerical GPU sweep is recommended on the present evidence.
Outstanding qualifications are cold-start sensitivity, the recorded BF16 layout
miss, unequal optimization trajectories, the shared-origin/short-fixture limits,
and untested longer packed training. Operational restart success, small held-out
loss gaps and a valid manifest do not resolve those qualifications. They do
support progressing toward a carefully specified, recoverable pilot.
