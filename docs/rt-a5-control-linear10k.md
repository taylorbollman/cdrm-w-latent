> **Superseded17:10UTC:** user stopped the control and deferred the linear run.
> Current task: [six-layer constant input injection](rt-a5-six-layer-input001-10k.md).

# Four-layer control and conditional linear input ramp

## Current authorization

On2026-09-15 the user requested stopping the conservative quadratic input ramp,
running a fresh four-layer L1R RT + NextLat control with no embedding modification,
and, if length36 whole-word exactness becomes positive by10k, trying linear warmup.
The latest correction sets the linear target to **lambda0.01 at50,000 updates**:
`lambda(s) = 0.01 * min(max(s,0)/50000,1)`. Initial linear budget is10k
(lambda0.002), subject to user steering. This replaces the earlier linear0.001 target.
Do not extend either pilot automatically beyond10k. No value-bypass run is queued.

Control has index0 window2 and indices1/2/3 full RT; original NextLat objective,
D512/H8/GELU2048/ALiBi/Mitchell initialization, B1024/T12, FP32, original seeds/data
order and optimizer. All43 shared initial parameter tensors match the input model;
the learned262,144-parameter projection is absent.13,702,656 parameters total.
Linear arm restores the same input projection at index1, with13,964,800 parameters.
Original NextLat conditioning on the embedding remains part of its original objective.

## Stopped conservative run

Lineage `.runtime/rt-a5/20260915T163300Z-input-conservative10k/`.
W&B https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/h5m6ufcn
Stopped cleanly at9,665 updates; full102400-row E36=0, M36=35.7375%,
L12 whole-word99.6553%. Lambda0.00003736489. Saved-state validation passed44
FP32 model/Adam states and all9,665 order hashes. Checkpoint
`train-conservative/checkpoints/step-009665.pt`, SHA256
`75644b2fbf8531c27a8b5d00f9fd2bc7e95ff7af9e174117f5929266181814ba`.
The saved state permits a later exact resume. No current resume is authorized.

Old conditional coordinator112526 and old retention-binding monitor113160 were
terminated on the new instruction. Their prospective frozen evidence is historical;
`diagnostics-superseded.json` records the change. Do not restart old coordinator.
Conservative checkpoint retention and terminal archive/readback are complete:
286 members verified; archive SHA256
`850d634244fa31a3d895a6958a2285b3445f5beae24a9e573ac1cd006b8865ba`.
Readback SHA256 `1df521a8aba7022beda92458b4286c6049cebdc72addcee53f192e5091a18461`.
Report `docs/reports/rt-a5/input-conservative10k/`, report W&B `yd98box9`.
Root reviewed all four figures. Do not modify archived conservative members.

## Control

Lineage `.runtime/rt-a5/20260915T164000Z-l1r-four-layer-control10k/`.
Existing `scripts.rt_a5_l1r_depth_train --n-layers4`, source58 SHA256
`04d3854620854130eb9a67093fd0538b5762ac870b9464291fde8a6899fbc78b`.
Runtime helpers were revised to explicit unconditional authorization; original
versions remain under `superseded-conditional-v1/`. Three discarded actualB1024/T12
GPU updates and causalT36 check passed. Control launched at16:59UTC, coordinator
PID123431, W&B https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/cu7yhsc2 .
Protocolv2 SHA256 `2201d5146265a7b606ef44643c16b53b07c74255b82c8b25feda72d0e37b3848`.
Run `python3 .runtime/rt-a5/20260915T164000Z-l1r-four-layer-control10k/status.py`
for latest local progress. Never duplicate launch or overlap GPU trainers.

## Reporting and retention

Log graphable runs online to taylorbollman/rt-a5-state-tracking. Retain checkpoints
and evidence under gs://fast-chunks/cdrm-w-latent/rt-a5/<lineage>/.
The conservative arm stopped before10k: use common retained5k endpoints for an
exactly matched comparison, and clearly label terminal9,665 versus10,000 results.
Use all102400 reused OOD development words for terminal decisions. No final
confirmation evaluation or autonomous latent rollout is authorized.

## Linear preparation

New lineage `.runtime/rt-a5/20260915T170000Z-input-linear10k/`, `train-linear/`.
New `scripts/rt_a5_linear_input.py`, `_train.py`,
`configs/rt_a5_linear_input/base.json`; source65 SHA256
`d24a0e8b89522c014f1483b53c1db536c17d51492e21116d944b489a977db89c`.
Nine CPU schedule/initialization/resume checks passed. Runtime preflight/launch
require a hashed positive control-success authorization record, not yet generated.
No linear GPU work or training has run.

Control retainer PID125409 is active and bound to protocolv2. The replacement
comparison reporter is `scripts.rt_a5_stopped_conservative_control_report`; it
requires closed control10k and actual saved-state pass. Output is
`docs/reports/rt-a5/control-vs-stopped-conservative/`. Common5k and unequal
terminal9,665/10,000 comparisons are explicitly distinguished.

Control CPU finalizer PID130380 (`finalize.py`) waits for actual10k completion,
checks saved state, creates the stopped/control report, and records whether any
full retained checkpoint had positiveE36 in `linear-decision.json`. It does not
launch linear itself. Root must review and, on a positive decision, write the
linear `authorization.json`, run its GPU preflight, bind implementation review,
and launch it. If user stops control early, stop this waiting finalizer before
changing closure scope.

Linear source/runtime review passed (`170000.../source-review.json`), including
all9 CPU schedule/init/resume checks and4 bounded authorization checks. Its gate
accepts any full positive retained control evaluation through10k after control
completes; it binds actual report/protocol/state/checkpoint evidence. Runtime
helpers/source65 are frozen. No linear authorization or GPU execution exists yet.
