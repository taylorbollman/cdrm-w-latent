# Stage B retrieval task sources and label contract

These are symbolic synthetic task adapters for this project's exploratory pilot.
They are not replications of either source paper's complete training setup or a
combined MAD/Zoology benchmark score. Generators, oracle checks and baselines are
NUM evidence; only held-out learned-model results are SYN evidence.

## Pinned, licensed source snapshots

| Source | Pinned revision | Files retained |
| --- | --- | --- |
| [HazyResearch/Zoology](https://github.com/HazyResearch/zoology/tree/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb) | `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb` | `zoology/data/multiquery_ar.py`, complete Apache-2.0 license |
| [athms/MAD](https://github.com/athms/mad-lab/tree/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b) | `0f49a452b84ca0d13f8eb9c1ffa649032376fb1b` | `mad/data/instances.py`, noisy-recall baseline config, complete MIT license |

The unmodified files are under `vendors/zoology` and `vendors/mad-lab`. Each
`PROVENANCE.json` records the repository, full revision, exact raw source URLs,
retrieval date and SHA-256 of every retained file. These are minimal source
snapshots, not installed repositories or submodules. No source training framework,
tracking service, broad optional dependency set or model implementation is imported.

`cdrm/synthetic/retrieval.py` is our changed adaptation, version `retrieval-v1`.
MQAR ports the generator semantics to local NumPy arrays/RNGs. MAD dynamically
loads and calls the pinned, unmodified NumPy-only generator. The snapshot hash
checks prevent an unnoticed change to either upstream source.

## Shared interface and determinism

`generate_mqar(config, split, seed, num_examples)` and
`generate_noisy_recall(...)` return `SyntheticBatch` with `int64` arrays
`input_ids[N,T]`, `labels[N,T]`, and per-example metadata. Labels are **already
aligned to logits at the query-key positions**; the trainer must not shift them.
Only answer positions are scored, and other labels are `-100`. A source value
always precedes its query; the terminal MAD answer is never included in inputs.

Each example receives a local NumPy Generator derived with SHA-256 and
`SeedSequence` from adapter version, task, split, seed, example index and purpose.
No global NumPy/Torch RNG is changed. Larger calls preserve smaller-call prefixes.
Train, dev, test and calibration have distinct RNG domains even if callers reuse a
numeric seed. Repeating a task/split/seed call deliberately repeats its prefix;
streaming callers must assign distinct per-update seeds or generate one fixed pool.
The run configuration separately fixes training/dev/test numeric seeds.

Individual symbol IDs and individual key/value pairs are allowed in all splits.
The generalization claim is unseen sequences and within-example mappings, not new
untrained embeddings. `audit_retrieval_splits` rejects duplicate input arrays
within a supplied split or across supplied splits and reports whole-map overlaps.
Distinct seeds alone do not establish separation. Delay controls deliberately
reuse mappings within one evaluation split and must not be treated as independent
examples in uncertainty calculations or passed as separate train/test splits.

## MQAR semantics

The [pinned generator](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/data/multiquery_ar.py)
starts with `K` distinct key/value pairs, then queries every key once at distinct
even offsets sampled without replacement from a power-law distribution. Keys come
from `[1, floor(V/2))`; values are distinct draws from `[floor(V/2), V)`. Token zero
is initially a filler. With `random_non_queries=True`, fillers are replaced by
uniform full-vocabulary tokens and may accidentally equal a key or value. Such
accidental key matches are not designated scored queries, as in upstream.

The adapter supports one pass through the association prefix, preserving the
upstream feasibility conditions `4*K <= base_length` and `V > base_length`. We do
not reproduce the source's optional repeated-pass behavior. It uses local NumPy
random draws for fillers instead of source Torch-global draws; seeded examples
are therefore not bitwise copies of the source dataset, while sampling semantics
and label placement match. The upstream generator itself is exercised by a CPU
test against our independent query oracle.

Frozen proposed initial condition: `sequence_length=128`, `num_kv_pairs=8`,
`vocab_size=1024`, `power_a=0.01`, `random_non_queries=True`. The K=16 control changes
association count while keeping length and vocabulary fixed. With 512 possible
values, uninformed task-class chance is 1/512; a model that ignores the query but
chooses an observed association value uniformly achieves expected 1/K. The latter
is a stronger shortcut baseline and is reported separately.

## MAD noisy recall semantics and adaptation

The [pinned noisy generator](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/data/instances.py)
wraps its in-context recall generator. Each context slot is either a noise-token
pair or a key/value record. Keys are sampled with replacement, and each key gets
a fresh random value on first appearance within an example, reused on later
appearances. Different keys may share a value. A final key is chosen from observed
keys, and its associated value is the terminal target.

We use single-query mode. Its copy marker and terminal key give exactly the
requested T input tokens; the source multi-query mode yields T-1 input tokens for
its even `seq_len` argument and scores repeated-key occurrences as well. The
source official noisy baseline sets `multi_query=True`; this pilot's terminal-only
choice is explicitly different and is not an official MAD baseline reproduction.

The source training branch predicts all next tokens, including random context.
To honor this project's answer-only objective, the adapter passes
`is_training=False` on **all** splits. Thus labels remain source-native shifted
answer labels and no second shift is applied. A rare upstream all-noise draw can
occur because its guaranteed-record index includes the reserved final slot. The
adapter rejects that invalid draw, with a bounded 100-attempt retry limit, and
records the retry count. This conditions on the task having at least one record.

Proposed low/moderate conditions: `sequence_length=128`, `vocab_size=81`,
`noise_vocab_size=16`, `frac_noise=0.2` / `0.6`. This reserves 32 key IDs, 32 value
IDs, 16 noise IDs and one copy marker. Uniform-value chance is 1/32. A balanced
training mixture uses both conditions, with separate evaluation metrics. At fixed
length, changing the noise probability also changes the number of records and
potentially distinct associations; metadata records both and the realized noise
fraction. This is not described as changing distraction alone.

## Fixed-association delay controls

Use `sequence_length=256, delay_tokens=128` or
`sequence_length=512, delay_tokens=384` with the same task/split/seed as the T=128
base. `base_length=sequence_length-delay_tokens` remains 128.

For MQAR, extra fillers are inserted immediately after the association prefix and
before all queries. For MAD, dedicated-noise tokens are inserted immediately
before the final copy marker and key. Every base input token, mapping and record
is preserved, and every query's source-value distance increases by the specified
number of tokens. Extra filler/noise uses a separate RNG domain, so neither later
base examples nor association choices change. These controls explicitly adapt
the sources to vary delay at fixed record count; increasing the sources' raw
sequence length would also alter their query spacing or record count.

## Oracles and shortcut checks

`retrieval_oracle(batch, task, config)` reconstructs mappings from complete earlier
records. It never reads labels or saved answer values. MQAR needs metadata's
designated answer positions because random filler can equal a query key. The
oracle's optional `history_tokens` limit exposes which queries are solvable from
a recent token window, with unavailable answers left as abstentions (`-100`).

`retrieval_baselines` reports answer accuracy for the exact oracle, uniform legal
value chance, uniform full-vocabulary chance, the last observed record's value,
and expected accuracy for a uniform choice among distinct previously observed
values. It also reports restricted-history coverage, accuracy with abstention
counted as error, and expected accuracy using uniform-value fallback when the
record is outside the window. Chance CE is theoretical uniform CE, not a trained
model's CE. These baselines are separate, not substitutes for trained controls.

Per-example metadata records query positions, source-value distances, record and
association counts, task settings, input/label and mapping fingerprints, split,
seed, index, adapter version and source revision. Exact sequence and mapping
fingerprints support actual fixture audits; they are not presented as formal
proofs of statistical independence.

## Validation

The retrieval NUM suite runs in the explicitly CPU-only project container:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 python -m pytest -q tests/test_synthetic_retrieval.py --tb=short'
```

It verifies pinned source hashes, exact source agreement for MAD, source-native
MQAR alignment, valid vocabulary/masks, exact oracle answers, local deterministic
RNGs and stable prefixes, observed split separation and duplicate rejection,
T128/256/512 delay invariance, actual low/moderate noise rates, counterfactual
sensitivity to source records, and no future-token dependence in oracle answers.
It does not train a model or establish neural-network causality; Stage A's model
NUM tests provide the latter implementation gate.
