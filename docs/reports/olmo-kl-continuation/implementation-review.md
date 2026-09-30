# Implementation review: explicit objective fork

This review concerns the approved NF update-32-to-64 continuation, not a generic
checkpoint migration framework. The
[protocol](protocol.md) fixes the two coefficients and comparison scope.

## Why a separate adapter is necessary

Three accepted contracts intentionally reject this intervention:

- `CampaignRecipe.to_dict()` serializes KL 1 and `optimizer_state="fresh"`.
- `scripts.olmo_campaign_execution.model_contract()` requires CE/latent/KL
  weights all 1 for NF/NFR, at both entry and final ownership checks.
- `scripts.olmo_campaign_eval_control.summarize()` accepts only 0/1 weights
  and requires equal latent/KL coefficients.

The distributed checkpoint loader also correctly requires exact configuration,
source identity, parameter/optimizer ownership and scheduler metadata. It must
not be weakened to load a changed objective as if it were the same experiment.

The smallest suitable change is a separately versioned branch adapter and
executor/engine entry path. Reuse the existing forward, losses, graph runner,
data materialization, optimizer, scheduler, async loop and storage. Keep their
200 frozen files unchanged. New helpers belong under `scripts/` to avoid the
historical `cdrm/pretrained/*.py` inventory glob changing old declarations.

## Restore and transition ordering

The root executor/engine owns two disjoint entry cases:

1. **Create branch:** authenticate the pinned parent and load it with its
   original configuration/fingerprint. Restore counters/cursors/RNG and
   validate original clocks. Record the complete parent boundary, apply the
   one allowed config override, and verify identical state. Construct truthful
   child metadata before any child save or graph preparation.
2. **Resume child:** construct the declared child objective before loading,
   then use the existing strict loader with that child's configuration and
   fingerprint. Restore its saved cursor and clocks. Do not reload the parent,
   reset moments, reapply startup adaptation or accept another branch's state.

The execution identity can retain the accepted
`olmo-pilot-execution-identity-v1` envelope so storage validators remain usable.
Its payload must explicitly include the new objective-branch version, complete
parent authority and effective objective/inheritance policy. This is a new
identity, not a reinterpretation of the old resolved declaration. Checkpoint
metadata equality continues to protect subsequent child resumes.

`set_kl_weight` should use `dataclasses.replace` on each existing immutable
NextLat config, preserving their other fields independently. The predictor
retains historical metadata that need not equal the wrapper's complete config;
only the intended coefficient must agree. No parameter, buffer or module is
recreated. No adapter/layout exists yet: prepared losses capture their config,
normalization coefficients and parameter/module ownership.

The new model-contract check preserves every prior ownership/alias/trainability
check, replacing only the expected objective coefficient. Likewise the new
effective recipe explicitly records inherited Adam and the coefficient, while
retaining the original architecture/schedule recipe as ancestry. Do not alter
historical recipe serialization to make a new run appear old.

## Exact invariants

At the transition, compare both ranks independently and verify replica state
agreement where appropriate:

| State | Required invariant |
| --- | --- |
| Model | Every named parameter and persistent buffer unchanged; same unique ownership, aliases, requires-grad and module modes |
| Adam | Same unique parameter/group order, hyperparameters, LR, step tensor, first and second moments; all owned parameters populated at update 32 |
| Scheduler | Complete original 128-update prefix, hash, warmup settings, base LRs, last epoch, step count and next LR unchanged |
| Counters | All seven existing counters unchanged; no zero reset or exposure subtraction |
| Data | Same ordered corpus/index pins and logical cursor; update 33 membership/noise unchanged |
| RNG | Python, NumPy, CPU, rank-local CUDA and explicit data/local generators unchanged |
| Live graphs | No graph/adapter before override; preparation subsequently preserves the full branch boundary |
| Permitted difference | Explicit objective/lineage metadata and `lambda_kl`, either unchanged 1 or changed to 0.1 |

Compare input and jitter identities between branches at the first new update
and through metadata accounting. Keep the same branch-independent logical row
keys. Full parameter equality between branches is expected at their origin;
after optimization their divergence is intentional.

## Evaluation adaptation

The new `scripts/olmo_kl_evaluation.py` subclasses the existing ordered
controller. The parent has no reducer-injection seam, so the subclass retains
the `run_if_due` body with only an explicit live-objective check and the new
reducer call. Scheduling, ordered panel loading, common-FP32 execution,
live-state restoration and publication remain the same. There is no runtime
monkeypatch of the accepted module.

The scalar reducer retains the old global sums/counts and reconstruction
checks; it is restricted to K4, latent 1 and declared KL 1 or 0.1. It reports
the actual weights and branch-weighted total. Raw pass and aggregate
CE/latent/KL statistics do not change merely because a coefficient changes.
The new schema is `olmo-kl-continuation-evaluation-v1`; downstream comparison
code must accept that explicit schema rather than falsify an old one.

At a common saved state, the two branches' forward losses/counts should match
before weighting. This is a particularly cheap test that catches accidental
mask changes or applying the coefficient inside a raw loss. It does not imply
their future losses should remain equal.

## Bounded tests and recovery evidence

The new CPU evaluator tests cover unchanged KL-1 reduction versus the accepted
reducer, KL-0.1 weighting with identical raw statistics, uneven denominators
and dummy rows, invalid weights and mutations to masks/counts/coefficient
accounting. Actual tiny NF/NFR evaluations preserve model, populated Adam,
gradient buffers, RNG, runtime and data cursor across both weights. Failure
injection must restore evaluation runtime and prevent publication.

The branch tests should reject an altered parent digest, arm, topology,
precision, partition, enabled objective, KL coefficient, scheduler prefix or
ownership before touching state. A control transition is a numerical no-op.
A reduced transition changes only config metadata; immediate raw losses remain
equal and the known KL gradient contribution receives the new coefficient.

Two-rank acceptance uses the existing tiny NFR plan with populated Adam at
update 1. Compare control continuation through update 3 with the unchanged
accepted path. Separately compare KL-0.1 uninterrupted update 3 with a fresh
process restored from its retained update-2 checkpoint. Verify data/noise,
gradient/update observations, model/Adam, scheduler, counters, cursor and RNG,
and reject loading that child under the control identity. Preserve the actual
manifest/publication pins and evidence; a local model-only reload does not
exercise cloud recovery or optimizer inheritance.

No large numerical grid or native duplicate training run is required. Native
origin/load/preparation checks and scheduled evaluation accompany the actual
authorized comparison. Report the distinction between tiny restart evidence
and any later native restart that actually occurs.

## Remaining interpretation and operational checks

The original finite plan already extends to 128, so no scheduler migration is
needed. The accepted evaluation policy repeats update 32 at a restored
boundary and then evaluates 48/64; keep that labeled behavior. Same dev prefix
does not create independent replication. Report first and later CEs and their
gaps, plus raw auxiliary trajectories, rather than comparing differently
weighted total objectives.

All Adam history is inherited from the current 32-update adaptation stage;
this does not add unavailable original OLMo pretraining moments. Lowering KL
may change global clipping and subsequent moment dynamics, so a local
gradient conflict is motivation, not proof of a future quality gain.

Each branch has fresh output and cloud namespaces. The immutable parent remains
available while child checkpoints upload asynchronously. Only verified
publications are recovery authorities. Preserve the distinction between last
local save, pending upload and last verified cloud update in W&B and reports.
