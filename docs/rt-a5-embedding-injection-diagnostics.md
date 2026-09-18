# Four-layer embedding-injection diagnostics

## Execution handoff — superseded by quadratic warmup

**Latest user steering (2026-09-15):** end the fixed-coefficient run, defer the
value-bypass approach, and first run a fresh four-layer input diagnostic with
`lambda(s) = 0.05 * min(s / 50000, 1)^2`. See
[the new handoff](rt-a5-input-quadratic-warmup.md). No value diagnostic is queued.

Fixed input injection completed20k and passed its44-tensor saved-state check.
The value run started at15:53:39UTC and was stopped at the user's request.
Its history contains1593 completed updates, but **only the0/1000 checkpoints
were saved**. Its latest resumable state is therefore1k. The raw launch exit137
and stale running report are preserved with a separate user-stop disposition;
they do not indicate a numerical failure. W&B value run:
[8leiny8u](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/8leiny8u).
The matched input/value10k comparison is cancelled. Coordinators58475/68038
and the old pair-retention watcher39890 were stopped; do not restart them.
Earlier queue instructions below are historical and are superseded here.

Input20k finished15:53:34UTC: E36=0/102400, M36=36.9818%; cumulative exactness
at positions12/13/14/16 is99.7686%/70.4941%/17.6855%/0.1768%.
The [input10k→20k report](reports/rt-a5/embedding-input20k/report.md)
is generated and synced to W&B`2caw0d4k`. Root viewed all four original figures;
a separate `training-losses-readable.png/pdf` supplement fixes crowded x-axis
labels while preserving the original artifacts. Root viewed that supplement too.

The [projection-strength analysis](reports/rt-a5/input-bypass-strength/through-020000/README.md)
is complete and synced to W&B`ei1upnr1`. Across saved0/1k/5k/10k/15k/20k
checkpoints, the added/raw-embedding RMS ratio increased2.0076%→3.2787%.
Lambda remained fixed0.02; the learned object was P_e. No full-model forward or
task evaluation was used for this CPU-only saved-weight analysis. Root viewed
the initial and final figures. Its watcher has exited.

All saved checkpoints are retained or being verified under the corresponding
`gs://fast-chunks/cdrm-w-latent/rt-a5/` lineages. A separate combined closure is
being prepared at `.runtime/rt-a5/20260915T160000Z-fixed-injection-closure/`.
Use that closure, not the old archivers requiring a completed value10k run.
The stopped value run is an abbreviated attempt, not a matched completed arm.

## Extension history (historical instructions, superseded above)

**Latest steering:** while the input run approached10k, the user requested
continuation to **20k** because its boundary metrics fluctuate. The input10k
lineage completed successfully at15:33:44UTC and its44-tensor saved-state check
passed. Exact model/Adam/RNG/order continuation is now active in
`.runtime/rt-a5/20260915T153500Z-embedding-input20k/`, W&B
[zdjkfpmf](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/zdjkfpmf).
Coordinator56839/launcher56840 started15:35:10UTC, saving15k/20k and stopping20k.
The value variant remains queued for **10k after input20k**; it has not started.
The runtime `after20k.py` coordinatorPID58475 is waiting for input20k success;
it will run the child CPU saved-state checker, launch the original frozen
value10k command, and check that saved endpoint. Its status is
`after20k-status.json` in the input20k lineage. Child retention watcherPID57885
uploads15k/20k; the original pair watcher still handles both fresh10k lineages.

Root stopped only the old pair coordinator47770 before it could launch value;
the active input10k trainer was untouched. The original matched10k comparison
finalizer54357 was stopped while waiting. Preserve their frozen helpers and
supersession receipts. The matched10k comparison can still use the saved
input10k checkpoint after value10k; a separate supplement reports the input20k
extension and its adaptive budget. Both new diagnostics remain part of the task.
The inherited frozen driver's `checkpoint_selection` text still says fixed10k
in continuation reports; the explicit20k protocol, numeric endpoint, and
checkpoint records govern the extension. Preserve the source rather than
changing that historical label during a run.

The user also requested a graph of the learnable gate. **There is no learned
scalar gate in these runs: lambda is fixed at0.02; P_e is learned.** A new
CPU-only saved-weight analysis is preparing projection-weight RMS and actual
`0.02 P_e e_t` RMS, equally weighting all60 token embeddings at each checkpoint,
plus the ratio to raw embedding RMS. This ratio is not the fraction of the
upper block's hidden input. Preserve training unchanged and label this
distinction in plots; do not introduce a learned lambda mid-run.

Implementation, CPU checks, and bounded GPU checks are complete. The six-layer
run finished at20k, and the sequential diagnostic pair is running. The two
fresh run directories are:

- `.runtime/rt-a5/20260915T150000Z-embedding-input10k/`
- `.runtime/rt-a5/20260915T150000Z-embedding-value10k/`

The original pair coordinatorPID47770 launched the input run on2026-09-15 at15:14:57UTC;
its training W&B is
[f41rp000](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/f41rp000).
That original queue is superseded by the input20k extension described above.
Check current status before starting anything after an
interruption. Retention watcherPID39890 handles newly saved checkpoints.

Both models have **13,964,800 parameters in44 tensors**, including the original
1,049,600-parameter NextLat predictor and the new262,144-parameter projection.
The62-file training manifest is frozen at
`555f943367e4165565cebb20f52b8f729af61d57dd2ec133c15dd45761fae075`.
Historical58 source files remain unchanged. The separate implementation is in
`scripts/rt_a5_embedding_injection.py`, `scripts/rt_a5_value_bypass.py`, and
`scripts/rt_a5_embedding_injection_train.py`.

Validation:15 factory CPU tests,7 value-adapter CPU tests, and one exact
checkpoint/optimizer/RNG resume test passed. The three completed GPU comparisons
use D512/B2 and lengths3/12/36, comparing the changed tiled value write against
ordinary autograd for outputs and all block/input/raw-embedding/projection
gradients. All passed; the largest per-tensor gradient relativeL2 was
`4.9040341e-7`. Five discarded B1024/T12 updates per variant also passed at the
actual training shape with finite FP32 model/gradient/Adam state, active
projection and embedding gradients, and exact future-token causality atT36.
Both initial full-model tensor digests are
`d9c659b2161eb45e81e648666cc42a48fa110b76f836e2a2ecd5008ca7d60f56`.
The warm update time was about0.105s for each arm, with about1GB peak allocation.

The input lineage's `prepare_gpu.py` waits for successful six-layer completion
and runs these GPU checks sequentially. It stops at `ready_for_root_review`;
root binds the validation receipts to each implementation review before
launching `run_pair.py`. That coordinator runs input10k, checks its saved state
in a CPU container, then runs value10k and checks its saved state. The separate
`retain_pair.py` uploads newly saved0/1k/5k/10k checkpoints. No automatic retry
or continuation beyond10k is authorized. Status lives in `preparation-status.json`,
`pair-status.json`, and each arm's `launch-status.json`.

The combined reporter is `scripts/rt_a5_embedding_injection_report.py`, with
output `docs/reports/rt-a5/embedding-injection10k/`. Generate it after both
completed/synced endpoints and passed saved-state checks, visually review the
figures, then finalize each lineage's evidence archive/readback. Preserve the
two/six-layer context as a qualified depth comparison, not a matched four-layer
uninjected control.

## Requested scope

After the active **six-layer L1R RT + NextLat continuation finishes20k**, the
user authorized two separate fresh10000-update diagnostics. The user clarified
that these new models have **four total layers**: restricted window2 at index0,
then three full RT layers. Apply each design change **only at index1**.
Do not change the active six-layer run's architecture or stop its20k endpoint.

Run in this order, keeping the two changes separate:

1. Input injection at index1: `x2_t = u_t + 0.02 P_e e_t`.
2. Permanent-value bypass at index1:
   `k_perm_t = W_K h_t`, `v_perm_t = W_V h_t + 0.02 P_e e_t`.

In both cases `e_t` is the original token embedding, `P_e` is a learned
bias-free width-by-width projection, and lambda is fixed at0.02. Its seed is
independent of the base model/predictor so the two new arms can share identical
four-layer backbone/predictor initialization. A variance-preserving zero-mean
normal projection initialization with std1/sqrt(width) is the intended simple
implementation. The input variant uses the existing index1 normalization
after addition; do not silently add another normalization layer.

The value variant changes only the permanent stored values written from the
processed recurrent output. Contextual keys and temporary current-position
values retain their existing definitions. Later tokens attend to these stored
values normally. Both the projection and original embedding must receive the
corresponding gradients; a forward-only injection is insufficient. Recompute
and backward must reconstruct the same changed memory values. Keep states
and gradients attached.

Use the existing width512, GELU-FFN2048, ALiBi, Mitchell initialization at the
actual four-layer depth, original NextLat predictor/objective, FP32 runtime,
batch1024, training length12, dataset/order/seeds and AdamW recipe. Fresh10k
runs, not converted/resumed weights from the six-layer run. No simultaneous
GPU training jobs. Log online to `taylorbollman/rt-a5-state-tracking`; retain
checkpoints and final evidence in each new lineage under `gs://fast-chunks`.

The current six-layer continuation is
`.runtime/rt-a5/20260915T144415Z-l1r-six-layer-nextlat20k/`; see
[its handoff](rt-a5-l1r-six-layer-20k.md). It automatically resumed its10k parent
and uses W&B run`vmg2i60x`. Its20k stopping point is still active unless the
user explicitly changes it. Completion of that run does not complete the
new request: both four-layer10k diagnostics remain authorized afterward.

Factory ownership during preparation: `nextlat100k_report` is implementing
the input branch in new `scripts/rt_a5_embedding_injection.py`, config and CPU
tests; `rt200k_lineage_audit` is investigating/implementing the permanent-value
branch in a separate new module. Root owns training/protocols/GPU validation.
No frozen historical or current training sources may be edited.

Use bounded checks that exercise the actual new paths: coefficient-zero
equivalence, active projection/embedding gradients, causal forward behavior,
and for the changed tiled value write a small FP32 differentiable reference
for forward/backward plus a few actual-batch updates. Do not expand into a
mixed-precision/compiler campaign.

Evaluate the same length12/36 development roles and preserve E/A/M meanings.
These are rapid one-seed diagnostics. An unmodified four-layer control has not
been requested or run; comparisons to existing two-/six-layer results also
change depth, so a failure cannot by itself isolate the injection's effect.
If needed, recommend that matched control at review rather than silently
launching a third experiment. Confirmation and autonomous latent rollout
remain unevaluated.

## Retrospective retained evidence

The eight actual checkpoint objects (input0/1k/5k/10k, input15k/20k, and value0/1k) have local SHA and GCS checksum/generation verification. The stopped value W&B history was recovered through update1550; the preserved local history reaches1593. No unsaved endpoint weights or later remote metrics are claimed.

The combined closure is `.runtime/rt-a5/20260915T160000Z-fixed-injection-closure/`; its stable archive destination is `gs://fast-chunks/cdrm-w-latent/rt-a5/20260915T160000Z-fixed-injection-closure/evidence.tar.gz`. Closure scope and bindings are recorded in `closure.json`; `evidence-storage.json` and `evidence-readback.json` record upload and readback completion after final review. The new quadratic experiment is excluded from this closure.
