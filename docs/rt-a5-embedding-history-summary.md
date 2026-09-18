# Historical A5-only embedding-injection results

All three embedding approaches preserved substantial A5 state tracking, but this
single-seed diagnostic did not establish a consistent winner over the baseline.
Their ranking changed across checkpoints.

The table reports **length-36 whole-word accuracy**: every predicted state in a
word must be correct. Every listed measurement uses the same full **102,400-word
development pool**, not the smaller routine evaluation subset. Models trained
on length-12 words.

| Model | 5,000 updates | 10,000 updates | 20,000 updates |
|---|---:|---:|---:|
| Baseline, no embedding injection | 67.9365% | 82.0771% | 80.9629% |
| Input addition, fixed λ=0.01 | 71.2295% | 73.7578% | 76.5264% |
| Permanent-value addition, fixed λ=0.01 | 71.5645% | 78.6465% | 82.1289% |
| Reassigned embedding-access head | 69.4609% | 78.2344% | 82.6738% |

The baseline and input-addition runs ended at 20,000 updates. Constant-value
addition reached **79.8447% at exactly 25,000**; its graceful stop completed one
additional update and scored **84.6934% at 25,001**. Those are different saved
checkpoints and should not be substituted for one another. The embedding head
ended at 25,000 with **82.4814%**. There is no 25,000-update baseline in this
comparison.

An additional permanent-value experiment used a linear ramp from zero to
λ=0.01 over 20,000 updates. Its length-36 whole-word accuracy was **63.8252% at
5,000**, **75.4678% at 10,000**, **78.6396% at 20,000**, and **83.8047% at
25,000**. This is a schedule variant of value injection, rather than a fourth
embedding mechanism.

## Model and comparison scope

These were **six-layer, width-512, eight-head** RT models: block index 0 had a
two-token attention window, followed by five full RT blocks. Only block index 1
received the embedding modification. They used NextLat training, ALiBi,
Mitchell initialization, GELU FFN width 2,048, batch size **1,024**, and full
FP32 eager execution. Training used the same finite 800,000-word A5 dataset,
constant AdamW learning rate 1e-4, and matched data order. Backbone/data-order
seed was 1234, predictor seed 1235, and added-projection seed 1236.

All 61 original baseline model tensors were shared exactly at initialization.
Input and value injection each added one learned 512×512 projection, increasing
parameters from **19,998,208 to 20,260,352**. The head approach reassigned the
eighth existing head's permanent values to projected raw token embeddings,
using that head's existing value-projection rows; it added no parameter or
scalar gate. Contextual permanent keys and temporary self key/value paths
remained unchanged.

At 10,000 updates, length-12 whole-word accuracy was already **99.3467%** for
input injection, **99.5703%** for constant-value injection, and **99.6035%** for
the embedding head. Their lower length-36 results measure generalization beyond
the training length, rather than failure to learn the training-length task.

These results are exploratory: one seed, repeatedly inspected development data,
and extensions selected after observing results. No untouched confirmation set
or autonomous latent rollout was evaluated. The small 20,000-update advantage
for value/head injection does not establish a reliable improvement, particularly
given the changing ordering and checkpoint fluctuations.

The current mixed A5/Fuzzy comparison instead uses **two layers, width 128,
16 heads, and 2,560 examples per task per update**, with NextLat. In that model,
one reassigned head occupies 1/16 of attention width rather than the historical
1/8. The historical A5-only results therefore establish feasibility and provide
context; they do not predict the mixed-task ranking.

## Saved sources

These numbers were checked against completed reports and their saved metric
JSON, rather than the earlier prospective queue instructions.

- Baseline through 20,000: [report](reports/rt-a5/l1r-six-layer-nextlat20k/report.md)
  and [summary](reports/rt-a5/l1r-six-layer-nextlat20k/summary.json).
- Input injection, including matched baseline through 20,000:
  [report](reports/rt-a5/six-layer-input001-20k/README.md) and
  [summary](reports/rt-a5/six-layer-input001-20k/summary.json).
- Constant-value injection:
  [report](reports/rt-a5/six-layer-value-constant50k-recovery/README.md) and
  [summary](reports/rt-a5/six-layer-value-constant50k-recovery/summary.json).
  The directory retains its earlier 50k plan name; the actual run stopped at
  25,001 after the budget changed.
- Embedding-access head: [report](reports/rt-a5/six-layer-head25k/README.md) and
  [summary](reports/rt-a5/six-layer-head25k/summary.json).
- Linear-ramp value injection:
  [report](reports/rt-a5/six-layer-value-linear25k/README.md) and
  [summary](reports/rt-a5/six-layer-value-linear25k/summary.json).

The value/head reports display their original baseline reference through 10,000;
the 20,000 baseline in the table above comes from the completed baseline
continuation, also paired in the input-injection report. Each variant's own
20,000 measurement is present in its saved full-evaluation metrics.
