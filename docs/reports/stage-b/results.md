# Stage B results: the first synthetic pilot

Stage B is complete: six paired research runs, their held-out evaluations, and
the numerical, operational, data and retention checks. This initial budget did
not establish a useful R3 retrieval or state-computation advantage. Both models
learned substantially less than the tasks permit. R3 showed a modest noisy-recall
cross-entropy improvement at shorter lengths in this seed, with much greater
update cost; this is worth preserving without calling it reliable retrieval.

The [full paired report](pilot/results.md) contains all 16 evaluation conditions,
learning curves, answer/sequence accuracy, cross-entropy, paired intervals,
measured shortcuts and artifact hashes. The
[development review](development-review.md) was recorded before final-test
evaluation; it explains why the original endpoints were retained unchanged.

## Main held-out endpoints

Each arm trained for 2,000 updates, batch 64, length 128, using the same examples
and corresponding initial weights. Models have 12 blocks, width 256, four heads,
and approximately 9.5–10 million parameters depending on vocabulary. R3 replaces
only block index 3, with rho=1. Parameters and computation are FP32. These are
task-native symbolic models, not pretrained language models.

| Main condition | SEQ accuracy | R3 accuracy | SEQ answer CE | R3 answer CE |
|---|---:|---:|---:|---:|
| MQAR, eight associations | 13.35% | 13.57% | 2.8198 | 2.8220 |
| Noisy recall, low noise | 7.30% | 8.18% | 3.2262 | 3.1714 |
| Noisy recall, moderate noise | 9.16% | 9.64% | 3.0936 | 3.0317 |
| Ordered state updates, in distribution | 20.34% | 20.07% | 1.7654 | 1.7673 |

Every condition uses 4,096 test examples. All four main accuracy-difference
intervals include zero. The 2,000-resample paired bootstrap resamples whole
examples, preserving the dependence between MQAR answers. Its intervals describe
example variation for this one trained pair; they exclude training-seed variation
and are not adjusted for the multiple reported conditions.

**MQAR:** both models remain close to the 12.5% shortcut of guessing among eight
stored values. Neither gets all answers correct in any test example. The
resulting zero-width empirical sequence-difference bootstrap interval is
degenerate; it does not establish population equivalence. With 16 associations,
accuracies are 6.34% and 6.32%, close to the 6.25% shortcut. These results are
compatible with weak candidate-value learning and do not demonstrate reliable
key-to-value binding.

**Noisy recall:** R3 reduces CE by 0.055 at low noise and 0.062 at moderate noise
at the training length. Their paired 95% intervals are respectively
[-0.069, -0.041] and [-0.079, -0.045], for R3 minus SEQ. CE also improves at T256,
but worsens at T512 by about 0.03. All six noisy-recall accuracy-difference
intervals include zero. Noise probability also changes the number of records;
the separate delay controls preserve the original records.

A separate [post hoc analysis](modal-value-baseline.md) found a stronger
query-ignoring shortcut: count
values across distinct observed keys, then predict the most common value,
breaking ties by smallest value ID. It reaches **11.25% / 15.21%** empirical test
accuracy at low/moderate noise, above both models. Its conditional expected
accuracy under the generator's uniform-over-distinct-keys query distribution is
12.18% / 14.35%. It never reads the terminal query when choosing its prediction.
This analysis was added after observing pilot metrics and does not modify the
frozen fixtures, model settings, or original baseline records.

**State updates:** both models remain below even the initial-state-only shortcut
(26.03%); the true initial state plus the last two operations reaches 38.65%.
Held-out composition accuracy is about 16.9%, near 1/6 chance, and eight-update
accuracy is about 17%. Isolated delayed-accuracy differences do not establish
state computation. The composition holdout concerns an ordered operation pattern,
not entirely new transition functions: 13 of its 17 distinct four-operation
functions can also be produced by training strings.

## Cost and validation

Recorded optimizer-update time totals **183.64 seconds for SEQ and 1,402.82 seconds
for R3: 7.64 times as much** for R3 at equal updates. All six runs total 26.44
minutes of update time. The full controller window through calibration, research
training, final evaluation and intervening uploads was **47.10 minutes**, within
the declared two-hour budget. Earlier data preparation and numerical probes are
outside that window. These are measured costs on one H100 80GB, not a formal
matched-time quality experiment. See [execution-summary.json](execution-summary.json)
and the [development curves against update time](pilot/development-update-time-curves.pdf).

- The complete CPU suite passed 70 tests. The final report also passed consistency
  checks against real checkpoint, batch, label, prediction and compiler records.
- All six separate 100-update fixed-batch calibrations reached 100% accuracy.
  This demonstrates fitting ability, not held-out task learning; those weights
  never initialize the research runs. See [calibration-summary.json](calibration-summary.json).
- The full frozen training streams and held-out data passed oracle, label,
  counterfactual and overlap checks. Both architectures consumed the exact same
  2,000 batches per task. See [data-summary.md](data-summary.md).
- The selected FP32 tiled path passed its declared unit-direction numerical
  fixtures. BF16 and raw unnormalized-cotangent failures remain recorded;
  BF16 was not used. See [backend-summary.md](backend-summary.md).
- Bounded midpoint recovery passed bitwise for both architectures. R3 update-zero
  resume is explicitly rejected, and optional R3 accumulation retains one
  parameter-bound failure. The pilot uses no accumulation. These are documented
  limits, not claims of arbitrary-size bitwise reproducibility. See
  [runner-validation.md](runner-validation.md), including its metadata correction.

## Retained artifacts and next work

The archive prefix is:

`gs://fast-chunks/cdrm-w-latent/stage-b/20260906T190223Z/`

The [checkpoint ledger](checkpoint-ledger.json) identifies all 24 research
initial/final/best-development/latest artifacts by path, SHA-256, update count,
lineage and verification scope. All six final checkpoints loaded successfully
for GPU test evaluation. The remaining roles were verified by file hashes;
their research artifacts were not separately replayed. The
[storage manifest](storage.json) records actual cloud verification for data,
checkpoints, predictions, calibration, numerical diagnostics and source.

Local artifacts remain under the persistent project's `.runtime` directory.
Local SSD caches are disposable. The frozen training source is retained as
`source/training-source-v2.tar.gz`; a later source snapshot also captures final
reporting and documentation. [The usage guide](../../stage-b-usage.md) covers
reproduction and recovery.

The next scientific milestone should establish useful task learning on
development data before interpreting architectural differences or paying for
large LM runs. A bounded paired MQAR continuation is a useful first optimization
control because its CE is still falling: preserve weights, Adam state and the
next data position, explicitly declare any extension schedule and budget, and
keep these 2,000-update endpoints immutable. That requires an explicit new
experiment lineage, not bypassing the current runner's identity checks. State
tracking also needs a difficulty/optimization regime that clears its shortcuts.
Once a useful regime is identified, replicate across training seeds and reserve
a fresh confirmatory test set. This pilot does not rule out CDRM or latent
objectives, which remain distinct hypotheses.
