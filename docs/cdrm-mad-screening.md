# Fresh MAD data for SEQ-6 difficulty screening

The two initial settings and the declared optional third setting are exact
one-field difficulty changes from pinned MAD
revision `0f49a452b84ca0d13f8eb9c1ffa649032376fb1b`. The official
`make_benchmark_mad_configs` varies one baseline field at a time; these settings
follow that convention. They use the unchanged native generator, supervision and
special-symbol behavior described in [the data contract](cdrm-mad-data.md).

| Setting | Official change | Actual input and targets | Fixed dev baseline |
|---|---|---|---|
| `recall-v128-t128` | Recall vocabulary 16 → 128; configured length remains 128 | Input T127; keys 0–63 and values 64–127. Native training has 127 next-token targets. Retrieval targets equal `64 − distinct_context_keys`: dev has 15–32, mean 23.777 per example. | Uniform answer vocabulary 1.5625%; query-ignoring modal 9.663%; modal sequence exact 0%. |
| `copy-v16-t256-k96` | Selective-copy count 16 → 96; vocabulary 16 and length 256 unchanged | Input T256; source IDs 0–13, blank 14, marker 15 at position 159. Exactly 96 scored outputs at positions 160–255. | Uniform answer vocabulary approximately 7.143%; order-ignoring modal 12.196%; modal sequence exact 0%. |
| `copy-v128-t256-k16` (optional) | Selective-copy vocabulary 16 → 128; copy count 16 and length 256 unchanged | Input T256; source IDs 0–125, blank 126, marker 127 at position 239. Exactly 16 scored outputs at positions 240–255. | Uniform answer vocabulary approximately 0.79365%; order-ignoring modal 10.483%; modal sequence exact 0%. |

Increasing copy count also reduces the number of inserted blanks at fixed length:
96 source symbols and 63 blanks precede the marker. It is an established harder
ordered-output setting, not a pure memory-length intervention with every other
distributional property held constant. Recall's larger vocabulary reduces the
fraction of causally retrievable native training targets; dense native CE should
not be interpreted as answer-only CE or expected to approach zero on fresh data.

Primary task sources and their exact file hashes are:

- [Official recall YAML](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/configs/tasks/in-context-recall.yml):
  `7be6752039dac5934d2bf65fb645ba13037544e74f927c5e9f1f538b5578f24d`.
- [Official selective-copying YAML](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/configs/tasks/selective-copying.yml):
  `c342339f554abef90857e9c906fe16b2a848cb273edcd32d62873f8dc20f8ba4`.
- [Official benchmark construction](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/configs.py)
  and [generator](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/data/instances.py)
  are retained with hashes in `vendors/mad-lab/CDRM_PROVENANCE.json`.

Each setting has 12,800 training and 1,280 development examples. Seeds are 112345
and 123456, independent of the preceding 12-block pilot. Shuffle seed 45678 produces
the retained 200 shared epoch permutations. All examples passed independent
oracles; deterministic causal-prefix and counterfactual checks passed; there are
no exact input duplicates within or across the train/dev splits. The adapter/prep
test suite passes 29 CPU tests, including direct native-generator equality and
bitwise agreement with the original default corpus.

The roots are under
`.runtime/cdrm-naive/20260907T123830Z/data-screening-v1/`:

| Root suffix | Initial manifest SHA256 |
|---|---|
| `recall-v128-t128` | `dd7c67c8d91203de5d746c7ec975f1f7d17b5efe7c1b0c68606a3adf6d8afe09` |
| `copy-v16-t256-k96` | `a70282765db11bfc559a8af7a1048b474341c9defc7a95d23124ce846607a8e9` |
| `copy-v128-t256-k16` | `385c24dea422d63ec5d97676f3e4368b969714a04315c1d4391d4fc4aae6c85e` |

Use the native task name within each root:

```python
from cdrm.mad_data import load_dataset

data = load_dataset(
    ".runtime/cdrm-naive/20260907T123830Z/data-screening-v1/recall-v128-t128",
    "in-context-recall", "train")
# data.manifest['vocab_size'] == 128; config/task_overrides are authoritative.
# data.input_ids, data.labels, data.answer_labels remain int64 arrays.
```

**No final split was generated during screening preparation.** After freezing
the common epoch-45 endpoint and best-development checkpoint roles, this pilot
appended the selected copying final split: 1,280 examples, seed 134567. All
examples pass the oracle and no exact train/dev/final overlap was found. Its
supplemental manifest SHA256 is
`5f8254e36f3d93d938705c69bb1f7b095867506d926efb93a5a81a9774b53f69`.
The command below reproduces that completed operation; **do not rerun it into the
same output directory**:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/cdrm_prepare_mad.py --setting copy-v16-t256-k96 --splits final --append-splits --output-dir .runtime/cdrm-naive/20260907T123830Z/data-screening-v1/copy-v16-t256-k96'
```

The addition checks the existing setting/source/seed identities, rechecks overlap,
and writes `manifest.added-final.json`, referencing the original manifest. It leaves
every original split, epoch-order array and `manifest.json` untouched. Both
deferred settings still have no final split; append theirs only after a future
comparison is selected and frozen. Generating new
train/dev roots uses `--splits train dev` and omits `--append-splits`; existing data
cannot be overwritten.

The optional third setting, `copy-v128-t256-k16`, has also been prepared using the
same frozen script, train/dev counts and separate split seeds. Its data is available
if the initial screening motivates this already-declared alternative; preparing it
does not imply a model run or a new hyperparameter sweep. Its exact generation
command, exit status and log hashes are retained in
`.runtime/cdrm-naive/20260907T123830Z/execution/data-screening-copy-v128-prepare.json`.
Fuzzy recall remains deferred because its native training/evaluation motif
distributions and padding need a separate explicit oracle review. No SEQ-6 or
CDRM-6 learning claim is made by this data document.
