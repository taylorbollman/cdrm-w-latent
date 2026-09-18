# Stored-value embedding route versus baseline at 15k

The stored-value route finishes with slightly higher A5 length-36 whole-word
accuracy and slightly lower Fuzzy T400 accuracy than the baseline. The learning
curves have broadly similar timing in optimizer updates; the value route used
about 11.2% more measured training time. These results do not establish a broad
advantage or show that the baseline already implements an embedding bypass.
The completed T1024 follow-up strengthens the tradeoff interpretation:
aggregate answer accuracy is almost equal, while the value route performs
worse on first-value and terminal first-value predictions and better on
teacher-forced value continuations.

The [paired training report](reports/rt-nextlat-fuzzy-a5/d128-b2560-embedding-reordered/after-value/report.md)
contains all checkpoint-bound curves, provenance and matched comparisons.
W&B training: [baseline continuation](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/qygud2ze),
[stored-value route](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/ucahrd6v).

Both models have two RT layers: layer 0 has window two and layer 1 has full
recurrent attention. They retain D128/H16, GELU FFN512, ALiBi, Mitchell
initialization, FP32 eager execution and NextLat training. Each of 15,000
updates combines 2,560 A5 T12 examples and 2,560 Fuzzy T400 examples, giving
38.4 million presentations per source. Shared initial tensors and ordered
training examples were verified exactly.

Only the value model adds the raw input embedding to the upper layer's
permanent stored values, schematically

\[
v_t = W_V h_t + 0.01 P_e e_t.
\]

Keys retain their contextual path. The projection is learned; the coefficient
is fixed. This adds 16,384 parameters: 496,000 versus the baseline's 479,616.

## Fixed endpoint results

| Metric | Baseline | Stored-value route | Value minus baseline |
| --- | ---: | ---: | ---: |
| A5 L12 token accuracy | 99.9993% | 99.9994% | approximately equal |
| A5 L12 whole-word accuracy | 99.9961% | 99.9951% | −0.0010 pp |
| A5 L36 token accuracy | 99.2369% | 99.3277% | +0.0908 pp |
| A5 L36 whole-word accuracy | 92.8535% | 93.4990% | +0.6455 pp |
| Fuzzy T400 answer accuracy | 99.9112% | 99.8682% | −0.0430 pp |
| Fuzzy T400 first-value accuracy | 99.9256% | 99.9084% | −0.0172 pp |
| Fuzzy T400 terminal-query accuracy | 99.5272% | 99.4090% | −0.1182 pp |
| Fuzzy T400 answer-motif exactness | 99.8397% | 99.7596% | −0.0801 pp |
| Fuzzy T400 whole-sequence exactness | 97.8906% | 96.8750% | −1.0156 pp |

A5 endpoints use 102,400 words per role. Fuzzy endpoints use the same 1,280
examples. All endpoint metrics within an arm come from that arm's same
15k checkpoint. These are development results from one paired model seed,
not a replicated estimate of an architectural effect.

## Learning timing

Both models still have zero observed full-pool A5 L36 whole-word accuracy at
10k. Their first positive full observations occur at 10.4k: baseline 1.4180%,
value 0.2490%. At the common 12.5k full evaluation, baseline reaches 74.1211%
and value 78.4893%. Both exceed 90% at the scheduled 15k full evaluation.
Intermediate 4,096-word monitoring subsets also show fluctuations; the full
evaluation times are observations, not exact crossing times.

| First observed full Fuzzy threshold | Baseline update / training hours | Value update / training hours |
| --- | ---: | ---: |
| 50% answer accuracy | 2,700 / 1.464 h | 2,700 / 1.644 h |
| 90% answer accuracy | 5,000 / 2.717 h | 5,300 / 3.211 h |
| 99% answer accuracy | 7,500 / 4.061 h | 7,300 / 4.405 h |

The 99% threshold is reached 200 updates earlier by the value route, but later
in measured training time. Total committed-update time is 8.126 hours for the
baseline and 9.036 hours for value, a difference of about 54.6 minutes. The
baseline clock correctly includes all three training segments. These totals
exclude evaluation, checkpointing and reporting; sequential runs may also
experience different system conditions. There is no consistent learning-speed
win across the listed thresholds.

## Longer-length follow-up

The original baseline probe found useful T1024 headroom despite near-ceiling
T400 results: answer accuracy 79.4008%, first-value accuracy 74.4914%, terminal
first-value accuracy 41.8750%, and accuracy 38.2993% for retrieval distances
above 512. The [baseline length report](reports/rt-nextlat-fuzzy-a5/d128-b2560-baseline-length-probe/report.md)
documents the shared T512/T1024 development pools.

The value model has now been evaluated on those same pools with its fixed 15k
checkpoint. The [paired length report](reports/rt-nextlat-fuzzy-a5/d128-b2560-value-length-comparison/report.md)
and [W&B run](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/etp92l88)
reuse the verified baseline measurements, bound to the same data, source and
checkpoint hashes. No training or A5 re-evaluation was performed.

| Metric | Baseline T512 | Value T512 | Baseline T1024 | Value T1024 |
| --- | ---: | ---: | ---: | ---: |
| Answer-token accuracy | 99.4959% | 99.5658% | 79.4008% | 79.4942% |
| First-value accuracy | 99.4326% | 99.4912% | 74.4914% | 70.5474% |
| Value-continuation accuracy | 99.5598% | 99.6412% | 84.3590% | 88.5301% |
| Terminal-query accuracy | 97.3277% | 97.8699% | 48.9054% | 46.5598% |
| Terminal first-value accuracy | 97.1875% | 97.1875% | 41.8750% | 31.7188% |
| Answer-motif exactness | 99.0812% | 99.1983% | 70.1713% | 67.2487% |
| Whole-sequence exactness | 82.2656% | 84.3750% | 0% | 0% |

At T1024, the aggregate answer difference is only +0.0934 percentage points
for value. It hides a −3.9441-point difference on first values and a
+4.1711-point difference on continuation tokens, where preceding value tokens
are already visible under teacher forcing. Terminal first-value accuracy is
10.1562 points lower with the value route. Whole-sequence exactness is at a
floor in both models, so it cannot rank this comparison.

Distance-conditioned results also show a tradeoff. At T1024, the value route
improves the 257–512-distance bin from 76.5485% to 79.3681% and the 513+ bin
from 38.2993% to 46.9002%. However, bins from 1–256 decline by approximately
3.56–5.23 points. These bins score all native answer tokens, including
teacher-forced continuations. They do not isolate first-token retrieval at
long distances; no claim of improved long-range first-value recall follows
from their aggregate improvement alone.

T512 gives a modest favorable result for the value route, while the more
difficult T1024 pool exposes mixed behavior rather than a broad benefit.
Both pools are development data from the same single trained seed. There is
no new training condition, replication or mechanistic intervention here.

Evaluation runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260918T175000Z-d128-value-length-comparison/`.
The output preserves complete count-based metrics, source snapshots and the
baseline-cache evidence hashes. GCS closeout is recorded in that runtime's
verified `retention/final-receipt.json`:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260918T175000Z-d128-value-length-comparison/final-evidence.tar.gz`,
generation `1789753840910428`, SHA256
`fce5476f3bce2068c7fae8b255b05cb8601a75cfce76da641488adec1030cbff`.

The other embedding arms have been cancelled by the user. The head run was
cleanly stopped at 368 updates; input remains stopped at 452. Those partial
runs are not substitutes for the matched 15k baseline/value comparison, and
no further embedding-arm training is authorized by this follow-up.

## What this does and does not say about embedding propagation

The baseline already receives information derived from the token embeddings
through its ordinary computation, including residual paths. Similar benchmark
performance does not demonstrate that it stores raw embeddings in the same
way as the explicit value route. It also does not establish that the added
route is redundant, heavily used, or ignored. This experiment changes one
specific route with a small fixed coefficient and a learned projection.
Resolving those internal mechanisms would require a separately scoped
intervention or representation study, which has not been performed here.

For now, compare the observed task tradeoff and learning curves, especially
the harder length-generalization measurements, without assigning an internal
explanation that the measurements do not support.
