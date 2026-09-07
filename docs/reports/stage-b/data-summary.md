# Frozen Stage B data audit

The complete audit passed before research training. Its machine-readable record
is [data-archive-audit.json](data-archive-audit.json); source and task semantics
are documented in [task-sources.md](task-sources.md) and
[the ordered-state task](../../stage-b-state-task.md).

| Task | Training examples | Training input tokens | Training answer targets | Evaluation conditions | Dev examples | Test examples |
|---|---:|---:|---:|---:|---:|---:|
| MQAR | 128,000 | 16,384,000 | 1,024,000 | 4 | 4,096 | 16,384 |
| Noisy recall | 128,000 | 16,384,000 | 128,000 | 6 | 6,144 | 24,576 |
| Ordered state updates | 128,000 | 16,384,000 | 128,000 | 6 | 6,144 | 24,576 |

Each task's frozen training arrays contain exactly 2,000 batches of 64 examples
at length 128. Both architectures consume those same arrays in the same order.
Training-stream digests were first computed during preparation, then verified
by regenerating the entire stream during archival. The runner checks the archive
and every batch digest before training. These counts are per task dataset;
running both architectures doubles the examples and tokens consumed.

Every evaluation condition has 1,024 development examples and 4,096 test
examples. Delay controls intentionally reuse base semantic examples within a
split while increasing neutral-token separation. They are useful controlled
comparisons, not independent replications. No combined task benchmark score is
defined. Separate 64-example calibration fixtures are repeatedly reused for the
operational memorization checks and do not initialize the research runs.

## Checks and their limits

- Exact interpreters reconstruct every scored answer from earlier visible
  records. Labels are already aligned to the scored logit positions; random
  context positions are excluded from loss. The terminal answer is not inserted
  into the input.
- Input-only SHA-256 sets have zero train/dev, train/test, and dev/test overlap
  for all three tasks. This covers the full planned training horizon, not a
  small sampled prefix.
- Complete observed key/value mappings have zero cross-split overlap for both
  retrieval tasks. Mapping fingerprints were reconstructed from actual inputs
  and checked against metadata. Individual symbols and individual key/value
  pairs may still occur in multiple splits.
- Deliberately introduced input leakage, complete-mapping leakage, and mapping
  metadata corruption are rejected by the archive audit.
- State operations are fixed permutations shared across splits. Generator tests
  verify that changing any single relevant operation changes the oracle answer,
  and changing another entity's operation does not. These are generator checks,
  not learned-model counterfactual results.
- The state composition holdout excludes an ordered adjacent operation pattern.
  It does not guarantee novel resulting transition functions: 13 of its 17
  distinct four-operation functions also occur through allowed training strings.

Chance and shortcut baselines are retained separately for every condition.
MQAR's uniform guess among eight stored values reaches 12.5%, well above its
1/512 legal-value chance. For state tracking, ignoring early operations can
exceed 1/6 chance because permutations sometimes cancel. Thus learned accuracy
must be compared with the measured shortcuts as well as chance.

## Retention

The 77 fixture files occupy 150,012,639 bytes locally and are archived at
`gs://fast-chunks/cdrm-w-latent/stage-b/20260906T190223Z/fixtures/`.
[The checksum comparison](storage-fixtures-check.txt) found 77 local and 77
remote files and required no transfers. This bundle includes the exact training,
development, test and calibration arrays, held-out metadata, and manifests.
Training metadata is reproducible from the pinned sources and per-update seed
contract; training inputs and labels are retained directly.
