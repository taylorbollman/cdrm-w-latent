# Next PR: manifest-driven component execution and recovery

2026-09-29. Implementation handoff for the [next readiness milestone](next-steps.md).
The [ordinary-B manifest bridge](../olmo-campaign-manifest/run-results.md) is accepted.
This map proposes implementation and bounded acceptance, **not authorization to
launch a quality campaign**. Preserve the [numerical qualifications](results.md).

## Boundary and reusable APIs

Create a new versioned execution/resolution layer beside the frozen acceptance
scripts. Leave their source inventories, protocols and checkpoint loaders intact;
do not monkeypatch module globals or broaden historical source guards. Suggested
new modules are `scripts/olmo_campaign_execution.py` for shared construction and
contracts, and a separately invoked launch CLI. Names are provisional.

| Existing API | Reuse or necessary new boundary |
| --- | --- |
| `olmo_campaign_manifest.validate_manifest`, `resolve`, `plan_updates` | Reuse original/fresh resolution, exact byte pins and finite whole-chunk plan. Add a versioned startup/resume schema around it; current resolver accepts only a fresh source prefix. |
| `olmo_campaign_manifest_run.load_draft`, `construct_declared` | Preserve pre-CUDA re-resolution and explicit backend consumption. Extract/generalize that behavior in the new module; remove B-specific acceptance restrictions there only. |
| `CampaignRecipe`, `build_campaign_model`, `build_campaign_adamw` | Use actual recipes for `B/N/F/R/NF/NR/FR/NFR`; retain tied ownership, dormant fusion and predictor presence rules. Do not create eight separate model implementations. |
| `PackedCampaignData.peek_update`, `rank_batches`, `commit`, `restore_cursor` | Keep logical membership shared across arms; use arm-specific physical allocation. Pure planning must not advance the committed reader. |
| `CampaignObjective`, `CampaignDDPGraphTraining` | Reuse global per-loss normalization, dynamic mask/noise refills, local/synchronized graphs, clipping and optimizer stepping. |
| `CampaignTokenSchedule`, `feedback_noise_for_rows` | Schedule by actual global valid inputs; key jitter by logical update and row identity, independent of physical allocation. |
| `parameter_layout`, `optimizer_ownership`, resource ledger cards | Reuse counting primitives. Add a component-aware ownership assertion; the B-only `model_contract` is unsuitable, and the ledger's `parameter_card` intentionally requires empty Adam. |
| `Coordinator`, `LoopPolicy`, `StopRequest`, `run_loop` | Reuse completed-boundary ordering and rank-zero host-action coordination. Unknown collective/update failures still require external whole-job teardown. |
| `save_distributed_checkpoint`, `inspect_distributed_checkpoint`, `load_distributed_checkpoint` | Reuse committed manifest, tensor/optimizer ownership, topology, source/configuration, cursor and RNG checks. Graphs are rebuilt after restore. |
| `retain_without_rng`, `run_stage_releasing_failure` | Reuse RNG isolation and coordinated-failure graph cleanup by explicit callbacks, without installing the legacy monkeypatch scope. |

The new resolved execution contract must bind arm, actual model/mode, trainable
ownership, precision/backend, optimizer implementation, document policy, data
membership, target masks, pass weights, loss denominators, seeds and scheduler.
Keep these separate from observation verbosity and a segment's stop boundary.
`CampaignRecipe.to_dict()` retains historical fresh-optimizer defaults; actual
inherited startup must have explicit additional authority, not misleadingly
reuse that field as a claim of fresh Adam.

## Three distinct initialization and recovery contracts

1. **Original/fresh, all eight arms.** Load the pinned native OLMo weights,
   create branches from declared seeds, then create fresh Adam and the finite
   token schedule. Reject unsupported arm/backend combinations before CUDA;
   never silently fall back. Preserve original B first-update behavior.
2. **Adapted initialization, explicitly allowlisted.** First support only the
   exact adapted lineage selected for the functional pilot. Fusion128 import
   can reuse `olmo_fusion_startup_train.load_fusion_checkpoint` under its original
   NF construction and frozen backbone/predictor contract. Only afterward apply
   a separately checked RT activation or isolated-to-packed policy transition.
   Declare a fresh all-component optimizer, reset clocks and new data origin if
   that is the chosen fork; retain prior exposure and import pins separately.
   Do not promise this import for arbitrary predictor-free or non-feedback arms.
3. **Same-lineage recovery.** Restore a committed distributed boundary from the
   new launcher with unchanged model, optimizer, schedule, topology and source
   contract. Continue the saved data/token clocks and RNG. A stop, failed report
   or absent report must not invalidate an otherwise fully verified checkpoint.

Full-NFR update4/20 with inherited Adam is a **different adapted migration**.
Reuse the strict historical `load_endpoint`/`load_continuation` authority checks
if this origin is selected, then declare every permitted policy/runtime/topology
transition and optimizer decision. The single-process isolated/unfused checkpoint
is not already a two-rank packed/fused checkpoint. A schedule extension must
preserve its completed prefix and current LR, following the checked
`extend_schedule` pattern; a new corpus cursor needs separate provenance.
Do not broaden that import simply to make a candidate checkpoint load.

For any migration, save its new validated distributed origin before accepting
later generic resumes. Preserve original source/weight pins and the transformation
record alongside new execution sources. Exposure accounting must distinguish
historical adaptation from new pilot tokens; it is not an exposure-matched
comparison with untouched B unless the later experiment deliberately makes it so.

## Generic committed-boundary recovery

Replace the *new launcher's* dependency on manifest-B `load_reference`: that
acceptance function deliberately requires a passed three-update reference and
checkpoint1. Generic recovery instead takes an exact checkpoint-manifest SHA and
verified object generations, validates its immutable run contract and complete
boundary, and resumes at its committed logical update without an uninterrupted
oracle. Keep exact-reference comparison as an optional acceptance observer.

Preserve ordering: backward → clipping/Adam → scheduler/counters → cursor commit
→ safe-boundary observations → checkpoint/retention. A failure before that safe
boundary never triggers emergency serialization of uncertain state. Logging
failure after a successful update may save that boundary before coordinated exit.
Publish verified receipts only after state and manifest retention; never replace
an existing checkpoint directory or silently prune partial evidence. A failed
upload can be retried separately against exact committed bytes and create-only
cloud generations. Local pruning is separate work, not implied by a retention
count setting. Ten minutes is a completed-boundary checkpoint target; measure
update and I/O delays rather than promise a hard durable-wall-clock interval.

## Lean execution and acceptance observations

Extract callback-based observers from the acceptance driver. Operational mode
keeps finite losses, global counts, participation/ownership guards, gradient
norm/clipping, component losses, LR/token/cursor clocks and verified checkpoints.
Do not copy/hash every model, gradient and Adam tensor on every training update.
Acceptance mode retains full boundary/gradient digests, replica equality and
bitwise next-update oracles at explicitly selected boundaries. Both modes must
execute identical math and restore observation RNG/modes; test that directly.

Reuse `olmo_campaign_eval_insertion.evaluation_scope` and preservation checks
when implementing declared evaluation. Its existing `final_pass_ce`/`Observer`
are tiny-fixture/final-pass acceptance tools, not the promised all-pass evaluator.
Add per-pass CE and separate auxiliary summaries with exact global denominators;
keep evaluation data, jitter policy and precision explicit. Time preparation,
compute, hashing, evaluation, serialization and transfer separately. The new
streamed-readback probe has not changed the accepted retention implementation.

## Minimal meaningful acceptance

| Scope | Required evidence |
| --- | --- |
| CPU: all eight arms | Literal tiny-model objective/update oracle; correct inactive branches and tied ownership; same logical rows/noise across different physical partitions; true document boundaries, padding/dummy slots and nonzero global counts for every enabled loss. |
| CPU: authority and recovery | Original/adapted allowlists; reject changed pins, modes, optimizer, topology, schedule or cursor before mutation; fresh-object next-update recovery from stopped/failed-run checkpoints without a successful reference report. |
| CPU: observations/failures | Lean versus acceptance next-update equality; stop before/after an update; logging/retention/publication failure ordering; no save after uncertain partial update; preserve prefix/LR for any supported schedule fork. |
| Two GPUs: new shared path | Short tiny-NFR reference versus checkpoint/stop/cloud-restore/resume, including accumulation and a local empty slot. Reuse prior B, logging-failure, abrupt-rank and live-evaluation evidence where call paths are unchanged; add a focused case if a newly dispatched arm changes static participation or graph ownership. |
| Two GPUs: selected actual startup | A few packed T1024 updates from the explicitly chosen NFR origin, then fresh-process recovery; compare the next update within the same precision/configuration. Add pretrained evaluation insertion only if the new evaluator is included. This is not another FP32/BF16 sweep. |

Freeze and review contracts before GPU acceptance; retain progress and verified
completed boundaries. Stop at this integration milestone for the pilot decision.
The production mixture, total pilot/adaptation budget, held-out evaluation set,
RT selection, per-arm physical batches/accumulation and H100 versus H200 topology
remain unchosen. H200 needs a short actual capacity/steady-state throughput check;
the tiny diagnostic budget and two-books readiness prefix are not campaign defaults.
