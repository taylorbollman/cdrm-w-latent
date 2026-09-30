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
