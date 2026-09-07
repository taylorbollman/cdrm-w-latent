# Official MAD data for the initial CDRM pilot

This lineage executes the original NumPy generators from MAD revision
`0f49a452b84ca0d13f8eb9c1ffa649032376fb1b`, with the baseline task YAML settings and
the relevant `MADConfig` defaults. The existing Stage B generators and corpora are
unchanged. The extra pinned source files and their SHA256 hashes are listed in
[CDRM_PROVENANCE.json](../vendors/mad-lab/CDRM_PROVENANCE.json); the original
[license](../vendors/mad-lab/LICENSE) is retained. MAD is introduced in
[Mechanistic Design and Scaling of Hybrid Architectures](https://arxiv.org/abs/2403.17844).
The recurrent-transformer paper's [synthetic experiment description](https://arxiv.org/html/2604.21215#A4.SS2)
motivates these tasks; these are local pilot runs, not a reproduction of unreported
author seeds or complete training settings.

| Task | Official configured length | Actual model input length | Vocabulary | Native training targets | Dev/final targets |
|---|---:|---:|---|---|---|
| In-context recall, multi-query | 128 | **127** | keys 0–7, values 8–15; no reserved symbol | Dense next-token targets at all 127 positions | Values for repeated keys, including the terminal query |
| Selective copying, copy 16 | 256 | 256 | copied symbols 0–13, blank 14, copy marker 15 | Sixteen copy answers at positions 240–255 | Identical mask |

The recall generator constructs 128 tokens and returns `inputs[:-1]`. Its official
caller passes `seq_len=128` unchanged. No padding or compensating extra token is
added here. `noise_vocab_size=0` comes from `MADConfig`, overriding the recall
function's unsuitable standalone default 16. Multi-query recall uses all 16 symbols,
so token 15 is a legitimate value. The adapter calls the pinned source directly.
See the [official generator](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/data/instances.py),
[configuration](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/configs.py),
and [dataset caller](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/data/dataset.py).

All labels are already aligned with the logits at the same position. **Do not shift
them in the trainer.** For recall, `labels` retain the original dense training
objective, while `answer_labels` retain only causally retrievable values. The
runner should report native objective CE separately from answer-only CE, answer
accuracy and sequence exact match. Dense recall loss includes random key choices
and the first presentation of random values, so its oracle is not claimed to
predict every native training token.

Recall is teacher-forced: previous key/value pairs, including previous probed
values, are visible ground-truth tokens. The current answer is predicted at its
key position, before that answer's following input token is visible. Subsequent
probes may therefore use ground-truth earlier answers. Selective copying has only
blank tokens after the marker and never feeds copied answers back. Its original
`np.insert` construction puts no trailing blank between the final source symbol
and marker; that distribution is preserved.

Each task has 12800 fixed training examples, 1280 dev examples and 1280 final examples.
Split seeds are 12345, 23456 and 34567 respectively. This is an explicit departure
from MAD's serial train/test consumption of one generator: the additional dev
split and separate RNG streams keep development independent from final evaluation.
The selective copier also calls legacy global `np.random.randint`. The adapter
seeds that stream with the same split seed as `default_rng`, as MAD's entry point
does, then restores the caller's global NumPy state. Generation is serial and
requires neither Torch nor MAD's trainer dependencies.

All native draws are retained. Exact duplicates are audited within and across
splits using input-only hashes and input-plus-native-label hashes; input-only
checks are essential because recall train/eval masks differ. No rejection sampling
or mapping novelty filter changes this small-vocabulary task distribution.
`manifest.json` records the actual collision counts. Final data may be read by
integrity oracles and frozen task baselines before endpoint selection; no model
evaluation on it should inform task, checkpoint or hyperparameter selection.

For each task, `epoch-indices.npy` stores all 200 zero-based epoch permutations,
each containing every training index exactly once. They use
`default_rng(SeedSequence([45678, epoch])).permutation(12800)` and are independent
of model RNG and architecture. Reusing these arrays makes paired runs consume the
same examples in the same order. No new examples are generated during training.

The oracle derives recall answers from completed earlier pairs and the current
query key. The copy oracle filters blank symbols from the prefix before the copy
marker and preserves order. Preparation checks every example against its native
evaluation mask; it additionally checks prefix-only retrieval predictions and
value/query/copy interventions on deterministic subsets. The test suite compares
adapter output byte-for-byte with direct calls to the pinned functions.

Uniform answer-vocabulary token chance is 1/8 for recall and 1/14 for copying;
uniform full-vocabulary chance is 1/16 for either. The query-ignoring recall baseline
takes the mode of values across distinct earlier keys, with smallest-value ties.
It ignores the current query key: repeated internal queries, conditional on being
scored, are uniform among previously seen keys; the terminal query is sampled
uniformly from distinct presented keys. For copying, a constant modal source symbol
ignores required output order. Both empirical token and direct sequence exact
accuracy are retained; conditional expected token accuracy is also reported.
Sequence exact match is counted on each example's complete answer mask, never
estimated by raising empirical token accuracy to a power. The separate analytic
independent-uniform-guess sequence chance is labeled as such.

The pinned optimizer wrapper constructs `CosineAnnealingLR(T_max=epochs,
eta_min=min_lr)`, but returns it under the key `scheduler` rather than the current
Lightning key `lr_scheduler`. The new runner explicitly steps once at each
completed epoch with a 200-epoch horizon and minimum LR 1e-6. This implements the
intended epoch schedule; it does not claim byte-identical behavior to every
historical Lightning environment. Source:
[pinned optimizer wrapper](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/model/pl_model_wrapper.py).

Run CPU checks from the project host directory:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 python -m pytest -q tests/test_mad_data.py'
```

Generate a new empty destination; preparation refuses to overwrite existing data:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 python scripts/cdrm_prepare_mad.py --output-dir .runtime/cdrm-naive/REPLAY_ID/data'
```

```python
from cdrm.mad_data import load_dataset, epoch_indices

data = load_dataset(".runtime/cdrm-naive/20260907T123830Z/data",
                    "in-context-recall", "train")
indices = epoch_indices(len(data), epoch=0)
batch = data.take(indices[:128])
# batch.input_ids, batch.labels, batch.answer_labels are int64 [128,127].
# Persist data.sha256 and data.manifest['manifest_sha256'] in checkpoints.
```

The initial corpus destination is
`.runtime/cdrm-naive/20260907T123830Z/data`. `.runtime` is under the persistent
project home checkout; SSD scratch is disposable. The root controller retains
this lineage under `gs://fast-chunks/cdrm-w-latent/cdrm-naive/20260907T123830Z` and
verifies uploaded artifacts separately. Preparation itself performs no upload.
Fuzzy recall is not included in this first data freeze: its official training and
evaluation motif distributions differ and deserve their own explicit oracle audit.
