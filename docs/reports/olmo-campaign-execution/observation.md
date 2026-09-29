# Lean and acceptance observations

`scripts/olmo_campaign_execution_observer.py` is an opt-in observation component for the common campaign runner. It does not construct a model, choose an arm, run forward/backward, clip gradients, call Adam, advance the scheduler/counters/cursor, log to W&B, run collectives or save a checkpoint. The caller retains the same `CampaignDDPGraphTraining.backward` and `step` path in both modes.

## Integration contract

Construct `ExecutionObserver(mode, model=..., optimizer=..., scheduler=..., device=..., generators=..., max_grad_norm=...)` after any checkpoint restore. Pass the **current** counter object to `after_update`; it is not captured at construction.

1. Caller performs `runner.backward(...)` with its unchanged precision/backend context.
2. `raw_gradients = observer.after_backward()` reads preclip gradient hashes in acceptance; it returns `None` immediately in lean.
3. Caller performs the unchanged `runner.step(...)` and commits the data cursor.
4. `observer.after_update(metrics, counters=counters, cursor=plain_cursor, input_record=..., raw_gradients=...)` returns a JSON-compatible row. Pass `None` for both optional evidence arguments in lean. Acceptance receives caller-computed input evidence and the hashes returned in step 2.
5. Caller gathers/persists evidence before calling the optional acceptance-only `observer.assert_reference(row, expected)` gate. The gate excludes only `observation_seconds`; losses, counts, norm, LR, cursor, gradients, state and RNG remain exact comparisons.

The root runner owns coordinated errors and must invoke callbacks symmetrically across ranks. Observer exceptions are ordinary host-side validation failures, not permission to resume an unknown CUDA/NCCL failure. The observer never adds an internal collective.

## Lean scope

Lean copies only already-returned scalar metrics and the plain committed cursor. It validates finite scalar values, objective/count structure, nonnegative and cumulative-count agreement, LR group shape and the returned norm. It does **not** enumerate parameters, call `state_dict`, scan model/Adam/gradient values, hash inputs, compute another norm or touch RNG/model modes. It rejects full input/gradient evidence to catch accidental acceptance work supplied to a lean callback.

Existing runner loss/gradient finite checks, participation/storage checks, clipping and optimizer behavior remain active. Full checkpoint boundary/ownership/integrity checks remain the caller's responsibility and are independent of observation mode. Lean is a choice of telemetry cost, not weaker checkpoint authority or numerical clearance.

Loss means divide the existing global loss sums by the existing global per-term counts; zero-count inactive auxiliaries remain `None`. Counts are once per supervised position, not multiplied by FBT pass count. Clipping reports the returned preclip norm and configured threshold; its coefficient is explicitly a host floating-point estimate using the runner's `max_norm/(norm+1e-6)` rule, not a second clip operation or an exact measurement of the dtype-rounded device coefficient.

## Acceptance scope

Acceptance uses the frozen `tree_digests` and complete `boundary` helper to retain preclip gradients and completed model/Adam/scheduler/counter/cursor/RNG evidence. Its read-only hash operations can transfer tensors from GPU to CPU and are intentionally expensive. Local Python/NumPy/torch RNG, supplied named generators and heterogeneous module training modes are preserved, including on observation errors. No peer-rank CUDA RNG is enumerated. Model/optimizer tensor values and gradient storage are never restored or rewritten by this observer: the callback only reads them.

Reference comparison is explicit and only valid between acceptance rows from an independently pinned matching run; this helper does not authenticate reference files or make a source/topology compatibility decision. The common runner must supply that authority.

## Timing and validation

`observation_seconds.after_backward` and `.after_update` are host wall times around each callback. The lean timer does not synchronize CUDA or imply isolated kernel timing. Acceptance CPU transfers/hash operations naturally include their waits. The common runner separately measures model preparation, backward/step, save, retention and evaluation; never treat observer timing as total step time or sum concurrent rank durations into elapsed time.

Focused CPU tests exercise real tiny campaign models for all eight arms, including separate CE/latent/KL eligibility and unchanged global counts, and compare unobserved versus lean versus acceptance training results. They check complete resulting model/Adam/scheduler/counters/RNG, module modes and gradient storage, plus lean hash/scan prohibitions, scalar validation, exact-reference rejection and exception cleanup. CPU tests do not qualify CUDA capture, NCCL, BF16 numerics or pretrained scale; those remain common-runner acceptance stages.
