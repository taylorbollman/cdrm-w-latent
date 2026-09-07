# Ordered state updates: custom Stage B task

This is a project diagnostic, not an official MAD or Zoology task. It asks a
model to compose ordered transformations of one entity's finite state while
ignoring other entities' updates. It does not yet implement the later
retrieve–update–retrieve extension: an update here reads and overwrites the state
of its own entity, rather than retrieving a separate source entity.

The implementation is [state_tracking.py](../cdrm/synthetic/state_tracking.py).
`generate_state_tracking(config, split, seed, num_examples)` returns the shared
`SyntheticBatch`: integer `[N,T]` inputs, equally shaped labels with `-100` at
unscored positions, and JSON-compatible per-example metadata. `task_spec(config)`
returns the fully resolved settings, vocabulary boundaries, permutations, and
generator revision. Labels already align with logits and must **not** be shifted
again by a trainer.

## Interpreter and serialization

There are six states, three operations, and initially four possible entity
symbols. An operation maps old state `s` to `permutation[s]`:

| Operation | Images of states 0, 1, 2, 3, 4, 5 |
|---|---|
| 0 | 1, 2, 3, 4, 5, 0 |
| 1 | 2, 4, 0, 5, 1, 3 |
| 2 | 5, 0, 1, 2, 3, 4 |

Operations 0 and 1 do not commute. All three permutations are bijections and
give different outputs at every possible input state. Thus changing any single
relevant operation changes the final answer, even when further operations
follow it. This property is tested for the earliest relevant operations as well
as the latest one. Changing another entity's operation leaves the answer alone.

Sequences contain these atomic records, separated by random neutral tokens:

```text
BOS
INIT entity initial_state
... initialization of each selected entity, in shuffled order ...
UPDATE entity operation
... interleaved entity-specific updates ...
QUERY target_entity ANSWER
```

The sole scored position is the terminal `ANSWER` marker. Its label is the
target's final state token. No intermediate or final answer is inserted into the
input. Initial state tokens are necessary task data; their occurrences after
`INIT` are unscored. The exact interpreter parses the actual tokens and rejects
malformed records, uninitialized entities, and nonterminal queries.

The vocabulary has disjoint ranges for five structural tokens, entity symbols,
six state symbols, three operations, and eight neutral symbols. The default
vocabulary therefore has 26 tokens. There is no padding token: all unused space
is occupied by neutral tokens, and all examples are the configured length.

## Frozen condition definitions

The initial suggested condition is length 128, three entities, four relevant
updates, four other-entity updates, and no mandatory minimum delay. The generator
also supports the planned two to four entities and four to eight relevant
updates. Actual run settings and any development-driven changes must be stored
in the run configuration, not silently substituted in a result report.

The configurations explicitly distinguish the following conditions:

| Condition | `composition_partition` | `entity_partition` |
|---|---|---|
| Training / in-distribution development and test | `train` | `train` |
| Held-out operation composition | `heldout` | `train` |
| Held-out query role | `train` | `heldout` |

An entity's operation string is obtained after removing all other entities'
updates. The ordered adjacent pair **(1, 2)** is excluded from every entity's
operation string in training and in-distribution evaluation. It is required in
the queried entity's string for composition evaluation; distractor entities
still use the training partition. Conditional operation strings are sampled
uniformly with a dynamic program. All three individual operation symbols appear
in training, including operation 1 or 2 in other compositions. Because the state
space is finite, an unseen operation string can have the same resulting
permutation as a seen string. The claim concerns unseen ordered composition
patterns, not guaranteed unseen group elements.

For the frozen four-update condition, exhaustive enumeration gives **55 training
strings and 26 held-out strings**. Training strings realize 16 distinct
state-transition permutations; held-out strings realize 17, with **13 of those
17 already realizable by a training string**. Thus this condition does not test
entirely novel transition functions. It tests whether learned operation semantics
transfer to the excluded ordered adjacent pattern. This enumeration concerns
four relevant updates and these exact three permutations; it is not a bound for
other update counts, and the 13/17 fraction is over distinct functions, not the
sample-weighted fraction of evaluation examples. A high score is useful evidence
about this specified pattern holdout, not proof of general compositional reasoning.

Entity IDs 0, 1, and 2 can be queried in training. Entity ID 3 appears as an
initialized and updated nonqueried entity, and is reserved as the target for
the held-out query-role condition. Initial record order and selection of other
entities are randomized. This tests a new role for a known embedding; it does
not test an untrained symbol. With a different entity vocabulary size, its last
ID takes the held-out role.

The `split` argument names an independent deterministic random stream, such as
`train`, `dev`, or `test`; it never implicitly changes condition settings.
Per-example semantic and layout streams are separately hashed from generator
revision, seed, split, and example index. Changing only length from 128 to 256 or
512 preserves entity identities, initial states, ordered updates, and answers;
only neutral gaps change. A configured `minimum_query_delay` adds neutral tokens
between the final update record and query. The actual distance from both first
and last relevant updates to query is recorded in metadata, alongside entity
count and relevant/distractor update counts.

## Shortcut checks and interpretation

Initial states are sampled uniformly and independently of operations, target,
layout, and distractors. Every composition is a permutation, so the correct
answer is uniform over six states conditional on these variables. Its chance
accuracy is 1/6; uniform guessing across the full 26-token vocabulary is 1/26.
The augmented fixture manifest reports both values separately. Neutral tokens have no state meaning, and random entity and
record ordering cannot reveal the answer. The tests check answer balance and
the answer distribution conditional on the last operation on independent
fixtures; these are finite-sample audits in addition to the bijection argument.

`state_tracking_baselines` reports exact-interpreter accuracy and three generous
retrieval shortcuts: return the last assigned state (initialization), apply only
the last relevant operation to the known initial state, or apply only the last
two relevant operations. These heuristics know the correct target and initial
state and skip earlier updates. Cancellation of finite permutations can let a
shortcut exceed 1/6, so its measured accuracy is reported rather than asserted
to equal chance. None solves the generated task. Learned accuracy must be
compared with these baselines, not only with uniform chance.

Passing generator tests establishes answer and partition correctness (NUM), not
that a neural network has learned state tracking (SYN). A low learned score
requires a learnability/optimization check before drawing an architectural
conclusion. Per-example fixture metadata retains the semantic trajectory and
answer location so prediction failures can be analyzed by composition, role,
update count, and observed delay.

Run the generator checks explicitly in a CPU container:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'python -m pytest -q tests/test_synthetic_state.py'
```
