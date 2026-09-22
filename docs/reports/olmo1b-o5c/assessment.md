# O5c assessment: general-text training repairs the feedback input

Completed and reviewed 2026-09-22. Both arms finished 512 optimizer updates and
4,194,304 supervised CE targets from the same O5b FBT endpoint. Only the two
fusion matrices trained. Every native parameter and the fixed fusion-scale
buffer remained byte-identical; the ordinary pass's evaluation scores stayed
exactly constant. No further training is queued.

See [results](results.md), [learning curves](learning-curves.pdf),
[full comparison](final-comparison.json), [optimization health](optimization-summary.json)
and the frozen [protocol](protocol.md). The independent review agreed with the
reported arithmetic, source identities and interpretation.

## Main result

Mixed code/general-text training strongly repairs the observed WikiText
feedback deficit. Continuing on code alone produces much smaller recovery.
The mixed arm gives up a small amount of code quality relative to code-only.

Final 512-window evaluation; lower NLL is better:

| Path | Code NLL | WikiText NLL |
| --- | ---: | ---: |
| Shared starting feedback pass | 1.737004 | 4.969719 |
| Code-only fusion adaptation | 1.715954 | 4.623289 |
| Mixed fusion adaptation | 1.720900 | 3.061443 |
| Frozen ordinary pass, identical throughout | 1.698216 | 3.183361 |

Relative to code-only, mixed improves WikiText NLL by **1.561846 nats**
(paired document 95% interval: improvement 1.426909–1.680952), at a code cost
of **0.004947 nats** (interval +0.003321 to +0.006483). That code difference
corresponds to approximately 0.50% higher perplexity. WikiText perplexity is
21.36 for mixed versus 101.83 for code-only; these are the project's recorded
preprocessing/development scores, not standard benchmark reproductions.

Mixed feedback also beats its own fixed ordinary pass on WikiText by
0.121918 nats (interval −0.135367 to −0.108459 for feedback minus ordinary),
about 11.48% lower perplexity. It remains **0.022684 nats worse on code**
(interval +0.019615 to +0.025522). WikiText next-token accuracy is 41.637%
for mixed feedback versus 40.738% for its ordinary pass; code accuracy is
63.847% versus 64.219%. There is no uniform improvement across both domains.

These intervals resample original evaluation documents, combining windows
before token weighting: 422 code documents / 127,850 CE targets and 59
WikiText documents / 246,910 CE targets. They do not quantify training-seed
variance. Both runs use one seed and development sets; reserved tests were
not scored.

## What this establishes, and what it leaves open

The large retention deficit is substantially repairable through training of
the new fusion input alone. Native-backbone damage or a mandatory backbone
architecture change is unnecessary to explain this particular repair. The
code-only branch controls additional fusion updates and total supervised
exposure; mixed's much larger WikiText improvement supports general-text
coverage as an important contributor. It does not identify the unique cause
of every O5b error or settle broader FBT design questions.

An FBT advantage over an ordinarily trained model is **not established**.
The fixed ordinary pass received none of the new general-text updates; all
of that learning occurred in fusion. A comparable ordinary-model continuation
could improve too. The two-arm comparison isolates training-data allocation
within this frozen-backbone FBT setup, not recurrence versus no recurrence
under equal additional training. Equal CE targets and optimizer updates also
do not imply equal compute.

The mixed arm has 2,097,152 code and 2,097,152 general-text CE targets; code-only
has 4,194,304 code targets. Shared code targets and context boundaries match
exactly, but the mixed arm reaches them at half the update rate. Thus its code
difference includes reduced code exposure and a different optimization history.
General training uses 624 unique WikiText training articles, with exact text
and token-hash exclusions against existing documents. This is a narrow domain;
no semantic near-duplicate guarantee or broad-language generalization claim is
made. Original OLMo pretraining overlap remains unknown.

The curve figures use the fixed **128-window** subsets, whereas the table and
paired intervals use **512 windows**. Their numerical levels differ because
the evaluation selections differ. In the small curves, mixed crosses the
ordinary WikiText NLL between updates 256 and 320 and continues improving.
It is still improving at the fixed endpoint; no asymptotic conclusion follows.

## Numerical health and operational completion

All recorded update/evaluation scalars and checkpoint parameter/AdamW checks
were finite. Code-only preclip gradient norm median/max was 0.2106/0.5716;
no updates clipped. Mixed median/max was 0.4823/9.2805; 63 of 512 updates
clipped, concentrated early (last clipped update 211). Its final norm was
0.3565. The large initial mixed gradients settled while retention improved;
there was no training health-gate stop.

138 distinct scoped CPU tests passed, covering existing FBT objectives and the
new freezing, gradient routing, data, reporting and recovery checks. Actual
H100 preflight zero-LR steps preserved every tensor. One disposable nonzero
step changed exactly the two fusion matrices and was restored exactly before
baseline evaluation. Its 512-window starting metrics reproduce O5b exactly.
The CPU future-update optimizer replay passes; an actual full H100 optimizer
replay was not added or claimed.

Median optimizer-update times were 0.492 seconds for code and 0.392 seconds
for mixed. Core update time summed to 252 and 205 seconds respectively;
checkpoint serialization, checksums and cloud transfers account for much of
the elapsed time. Individual arms took about 12.2 and 11.6 minutes including
evaluation/checkpoints. The queue took **27.6 minutes**, including loading and
a startup retry, so no multi-hour run was needed.

There was one environmental incident: the mixed arm's first `nvidia-smi` query
failed before model loading, W&B initialization, data consumption or any update.
A fresh project container passed the same checks; the unchanged queue skipped
the completed code arm and successfully started mixed. No driver reset, reboot,
training-code change or optimizer replay was needed. The cause is unconfirmed;
see [preserved incident](startup-incident.json). No training updates were lost.

Checkpoints at updates 128, 256, 384 and 512 for each arm have verified GCS
generation/size/MD5/SHA receipts. The latest full local checkpoint remains;
older local files were removed only after verified successors. The initial
archive retains data/source/provenance, and the final archive adds reports,
events, plots and this assessment. Read [usage](../../olmo1b-o5c-usage.md) for
paths and safe recovery. Completed queues must not be relaunched as new work.

## Recommended next review point

Run a bounded **evaluation-only** diagnostic at the unchanged repaired mixed
endpoint: fixed beta 1, identical windows and token positions, finite K2/K3/K4
versus exact sequential online feedback. Keep matched ordinary-pass and source
references and verify weights remain unchanged. Begin with short prefixes and
separately label any longer-context extension; do not mix their scores.

K2 training supplies ordinary-pass preceding states to fusion. Later passes
and online execution instead supply feedback-generated states. The earlier
O5b diagnostic at different weights cannot establish that this newly learned
repair transfers to that inference regime. If online quality falls while K2
holds, input-state distribution mismatch becomes a specific follow-up, with
the author's prefix mixing/jitter or rollout-consistent training as separately
controlled possibilities. If the repair persists, establish a comparable
ordinary-model additional-training control before claiming an FBT advantage.

This next diagnostic is a **recommendation, not launched**. RT/NextLat
interaction training and a longer adaptation budget remain separate decisions.
