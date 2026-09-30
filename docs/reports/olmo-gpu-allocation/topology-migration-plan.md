# Next milestone: production rank-count migration and job isolation

Prepared during allocation benchmarking. This is the next proposed engineering
milestone, not an executed checkpoint migration or authorization to extend the
learning experiment beyond128.

## Preserve the distinction established by this milestone

The new allocation benchmark imports identical weights and populated Adam but
deliberately resets its benchmark cursor/counters and holds LR fixed. It proves
that the shared compute path can operate on one/two ranks and measures useful
throughput. It does not yet qualify a production continuation across rank counts.

Current production checkpoints are replicated, not tensor/optimizer-sharded.
The shared state file contains the complete model and Adam; rank-local records
contain RNG and cursor envelopes. No tensor resharding algorithm is required.
The existing loader explicitly rejects a changed world size and remains intact.

## New explicit migration receipt

Use a new versioned adapter, not an in-place edit of an old manifest:

- Authenticate source manifest/state, model parameter aliases, optimizer ownership,
  populated moments and Adam step, completed logical counters, finite scheduler
  contract, common ordered-data index and cursor, and old source fingerprints.
- Record parent checkpoint hash, old/new world size, destination identity,
  physical batch and accumulation layout, new runtime fingerprint and RNG policy.
- Copy exact model/Adam state and LR/scheduler/token clocks. Keep the canonical
  logical cursor; rebuild only the rank/world-size/batch envelope. Never reset
  the optimizer, warmup, completed exposure or document-boundary masks.
- Preserve historical cumulative counters. Future physical microbatch counts
  change with the allocation; real examples and valid tokens per update stay
  fixed. Do not reinterpret old physical microbatch counts as tokens.
- Keep jitter keyed to logical update and row occurrence, so changing rank
  ownership preserves the actual noise assigned to each example. Audit which
  other RNG consumers exist. The current dropout-free computation does not
  justify inventing a universal bitwise RNG migration rule for future models.
- Reuse old rank-local RNG where mapped; explicitly seed any additional rank
  streams under the destination lineage. Record the policy and qualification.
  Floating-point reduction/accumulation order changes across topologies.
- Reconstruct CUDA graphs/DDP in the new process group. Capture/setup must not
  change loaded weights, optimizer state, scheduler or data position.

## Bounded checks that belong on the current two-GPU machine

1. Tiny one/two-rank save, fresh-process load and topology changes, with a common
   global-objective oracle. Cover uneven partitions/dummy rows and distinct
   CE/latent/KL masks; both unchanged-topology and changed-topology controls.
2. One native NFR checkpoint close to the endpoint, with populated Adam and
   headroom in its *existing* finite schedule. Saved127 is a useful candidate
   if retained and authenticated: compare a disposable two-rank reference
   update128 against a migrated one-rank update128. Do not silently extend
   the completed128 schedule to create a fixture.
3. Same-topology restoration should retain the established reproducibility
   contract. Across topologies, first verify exact starting state, examples,
   masks, jitter, denominators and LR. Then measure same-precision loss, raw
   gradient and actual Adam-update differences. Compare learned updates, not
   just differences relative to the large pretrained weight norm. Report any
   material discrepancy; a finite loss alone is not mathematical acceptance.
4. Stop/fail one disposable independent job and verify that the other job and
   its outputs continue unaffected. Ensure separate GPU ownership, process
   groups, rendezvous, output directories, W&B runs and retention prefixes.
   Never inject failure into the saved learning experiment.
5. Check a committed checkpoint written after migration, restore it in a fresh
   process with its new identity, and validate the next data/update boundary.
   Preserve the normal asynchronous publication policy and terminal drain.
6. Close any remaining native campaign integration gaps for N/R/NR/FR with
   bounded updates at a small physical batch, using the same packed masks,
   loss accounting and execution core. Existing all-eight tiny tests and older
   resource sweeps are useful evidence, but do not automatically certify every
   native combination under the current runner. Do this once the general
   production entrypoint is ready, without a long learning or capacity sweep.

These checks need distributed execution but little scale. The native update
itself can be faster on more GPUs, while the small fixture, migration semantics,
failure isolation and storage validation do not need eight GPUs.

## Defer until the target node

- Actual eight-rank NCCL/graph startup and checkpoint acceptance; eight emulated
  CPU ranks would test metadata only.
- H200 memory calibration, first at the H100 physical batch, then larger batches.
  Revisit checkpointing only as a separate measured change.
- Eight singles/four pairs/one eight-rank comparison, with fixed effective batch,
  sufficient warmup, concurrent timing and actual shared CPU/storage contention.
- Cloud credentials, Docker/driver compatibility, source/data/checkpoint access
  and fresh-process recovery on that provider. Do not assume host paths or
  credentials transfer automatically, and never include secrets in run evidence.

Longer scientific continuation, schedule-horizon extension and learning-quality
comparisons remain separate decisions from this infrastructure qualification.

## Implementation map from the current code

This section is a read-only code audit for the next milestone. No migration
adapter or production continuation has been implemented by this allocation PR.

| Responsibility | Reusable implementation | Restriction or new wrapper needed |
|---|---|---|
| Authenticate a committed parent | `cdrm/pretrained/distributed_checkpoint.py:inspect_distributed_checkpoint` | Supply the independently retained manifest SHA and verify the full state SHA before importing. Checking local presence alone is insufficient. |
| Strict checkpoint validation/load/save | `load_distributed_checkpoint`, `save_distributed_checkpoint`; existing `_validate_payload` validation requirements | The loader deliberately rejects a different world size, configuration, source fingerprint and rank RNG device. Keep it unchanged. A new explicit migration loader must validate the original identity before installing a declared destination identity; later same-topology child resumes can use the existing strict loader. |
| Model and optimizer construction | `build_campaign_model`, `build_campaign_adamw`; authenticated configuration-driven construction in `olmo_allocation_benchmark.py` | The benchmark importer is a useful construction example, not a production loader: it omits scheduler/cursor/RNG restoration. Retain actual NextLat settings, module training modes, tied ownership, FP32 masters and populated moments. For reduced-KL NFR use `olmo_kl_branch.branch_model_contract`; the original generic contract assumes KL=1. |
| Ordered logical data and partitioning | `OrderedCampaignData.restore_cursor`, `peek_update`, `rank_batches`, `commit` | These already accept an arbitrary positive world size. Restore the identical inner `PackedCursor` into a fresh reader; replace only its outer rank/allocation envelope. Commit exactly once after a successful optimizer update. |
| Schedule and learning-rate clocks | `CampaignTokenSchedule.load_state_dict`, `checkpoint_contract`, `validate_next_update`; `olmo_campaign_execution.validate_clocks` | Reconstruct the original finite token prefix before loading. The scheduler rejects a changed plan hash, token prefix, warmup, base LR or exhausted horizon. Reuse its validation rather than holding LR fixed as the benchmark does. |
| Objective, graph and reducer | `CampaignObjective`, `CampaignDDPGraphTraining.prepare/capture/backward/step/checkpoint_boundary` | The core already uses the actual process-group size and global per-term denominators. No model math or reducer change is needed. Capture must preserve the fully restored boundary; explicitly release captured NCCL graphs before destroying the process group. |
| Per-example feedback jitter | `feedback_noise_for_rows` | Use the original row occurrence keys and `plan.start_cursor.next_update`. Never use a new segment-local update number or incorporate rank into the seed. Dummy rows receive zero noise and zero loss participation. |
| State-preservation evidence | `olmo_campaign_restart.boundary`, `olmo_lm_common.tree_digests` | Compare complete weights/Adam/scheduler/counters/cursor and each destination rank's assigned RNG before versus after graph preparation. Compare common logical state across topologies separately from intentionally different rank envelopes. |
| Retention and independent job ownership | `AsyncCheckpointRetention`, `SSDCheckpointStorage`, existing loop/CPU cloud worker | Keep create-only destinations, unique job/retention identities, one pending upload and terminal drain. The current production engine hardcodes two ranks; wrap its lifecycle in a new generic engine, not a change to the frozen engine. |

The important counter exception is `microbatches`. Existing
`olmo_campaign_execution.expected_counters` recomputes the entire historical
count using the current fixed world size. A migrated run instead needs an
authenticated origin counter plus increments under each destination allocation.
All logical counters remain exact: optimizer updates, real inputs, packed rows,
CE positions, latent pairs and KL triples. Record the origin allocation and the
new allocation explicitly rather than weakening the old counter check.

Keep new code under `scripts/` so the frozen historical
`cdrm/pretrained/*.py` inventories remain unchanged. A small implementation can
use three new files: `olmo_topology_contract.py` for declarations and counter/
cursor mapping, `olmo_topology_checkpoint.py` for authenticated import and the
new migration receipt, and `olmo_topology_execute.py` for the generic bounded
execution/lifecycle. Add focused contract/import tests and a separately run
acceptance auditor. Reuse existing async storage and numerical helpers; no new
model backend, optimizer, checkpoint format or broad precision investigation is
needed. The destination checkpoint can retain the existing replicated format,
with its new configuration/fingerprint and parent migration receipt.

### Available native fixture and smallest useful sequence

Saved NFR127 is present locally at
`/mnt/localssd/cdrm-checkpoints/nfr-stability-128/native-nfr-reduced-64to128-01/update-000127`.
Its manifest SHA256 is
`c9504decfe0e13fc6537979502a165ba73a0db984ed1ed5be997b659b01dd445`;
the referenced state file is 15,214,957,057 bytes. This audit checked manifest
bytes and state-file presence, **not** the full state hash or cloud recovery.
Authenticate those before making it the next milestone's source authority.

Its committed logical position is update 127, chunk 65024, with 66,584,576 real
inputs and 5,588 physical microbatches. The unchanged 128-update schedule has
exactly one update left. At B12, that final update adds 44 physical microbatches
with two ranks or 43 with one rank, yielding 5,632 or 5,631 respectively; every
logical token/loss counter should still agree.

1. CPU tests reject malformed parent identities, foreign or missing Adam
   ownership, counter/cursor disagreement, undeclared RNG mapping, changed
   schedule prefixes and destination collisions before state mutation.
2. Tiny GPU fixture: ordinary same-topology restart control, then 2→1 and 1→2
   migration, preserving a finite schedule with at least two updates remaining.
   Save the migrated boundary, restore it in a fresh process, and run the next
   update against its uninterrupted destination-topology control. Include
   uneven allocation, dummy rows and separate CE/latent/KL masks.
3. Native source 127: first prove exact imported state/data/jitter/LR and
   preparation preservation. Compare one disposable 2-rank update 128 with one
   migrated 1-rank update 128 using the existing BF16 path. Record raw-gradient
   and actual Adam-update differences alongside loss; do not interpret a small
   difference relative to total pretrained weight norm as equivalent updates.
4. Save the migrated 127 boundary **before** advancing, so a fresh-process
   same-topology resume can execute the final permitted update 128. A checkpoint
   saved at 128 can additionally be restored for boundary/evaluation checks,
   but must reject update 129. This avoids inventing a schedule extension just
   to test recovery. Use the tiny fixture for an additional post-save update
   beyond its first migrated update.
5. In a separate disposable pair of independent jobs, stop one job at a bounded
   boundary, then exercise one abnormal exit. Verify that the other job's GPU,
   process group, training progress, W&B output and retained checkpoint remain
   independent. A queue policy that cancels both peers on any failure is not
   evidence of process-group isolation.

The main refinement to the earlier outline is therefore small: write and
freshly restore the native **migrated 127** checkpoint, not only the terminal 128
checkpoint. This qualifies a real resumed update without requesting any extra
scientific training steps or changing the finite schedule.
