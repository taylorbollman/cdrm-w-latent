# O5e assessment: ordinary training surpasses the fusion-only repair

Completed and assessed 2026-09-22. The single authorized control finished all
512 updates and its final evaluation in 23.8 minutes, without interruption or
retuning. All four recovery checkpoints are retained in GCS. No further run is
queued. See [results and plots](results.md), [protocol](protocol.md),
[machine-readable comparison](final-comparison.json) and [health summary](health-summary.json).

## What the control shows

Ordinary additional training reaches better WikiText quality than the
fusion-only repair, while essentially preserving its starting code quality.
It also beats both evaluated fusion execution paths on code and WikiText NLL.
The earlier advantage over a frozen ordinary pass therefore does not establish
an advantage over ordinary adaptation on the same data.

Final first 512 development windows per domain, maximum 512 tokens, batch 8,
BF16 mixed with FP32 parameters. Lower NLL is better:

| Model at evaluation | Code NLL | WikiText NLL | WikiText token accuracy |
| --- | ---: | ---: | ---: |
| Shared starting ordinary pass | 1.698216 | 3.183361 | 40.738% |
| O5e trained ordinary | **1.696348** | **2.720260** | **45.762%** |
| O5c mixed fusion, K2 | 1.720900 | 3.061443 | 41.637% |
| O5c mixed fusion, exact online | 1.723307 | 3.140935 | 40.597% |

Ordinary minus fusion K2 is -0.024553 code NLL and -0.341184 WikiText NLL;
paired original-document 95% intervals are [-0.028505,-0.020318] and
[-0.356701,-0.327663]. Against exact online, differences are -0.026960
[-0.031058,-0.022563] and -0.420675 [-0.440000,-0.403838]. All these intervals
favor ordinary. Its WikiText perplexity is 15.184, about 28.9% below K2 and 34.3%
below exact online.

Relative to the shared starting ordinary model, code NLL changes by only
-0.001869, with interval [-0.004935,+0.001339] including zero. Code token
accuracy is 64.217% versus 64.219%. Describe code as preserved, rather than
claiming a reliable code improvement. WikiText NLL improves by 0.463101,
interval [-0.484403,-0.444964] for trained minus starting, and token accuracy
rises by about 5.02 percentage points.

The matched 128-window learning curves tell the same directional story.
Ordinary WikiText NLL falls from 3.249787 to 2.886215 by update 50, then improves
gradually to 2.794104 at 512. Code stays close to its start, ending 1.594512 versus
1.599637. These are monitoring scores on a smaller selection; do not mix them
with the 512-window endpoint table or infer an online learning curve from the
previous fixed-weight endpoint diagnostic.

## What was matched, and what was different

Both adaptations start from the same O5b FBT update2634 native backbone and
use the exact O5c data02 mixed plan:4,194,304 supervised targets over 512 updates,
4096 code and 4096 general-text targets per update. They share target order,
segment and original-document identities, independent reset contexts, update
boundaries, physical chunks at most 16, seed, precision and evaluation selection.
There are 4,208,250 valid input tokens and 13,946 segments; no cycling or reserved
test access. O5e loads model weights only and creates fresh optimizer state.

O5e trains all 65 native tensors, including the tied embedding/readout:
1,176,764,416 parameters at native LR 1e-5 with 50-update warmup. FBT, RT and
NextLat are off; the two inherited fusion matrices and fixed scale remain
byte-identical and unused. O5c trained 8,388,608 fusion parameters at LR 1e-4 over
a fixed native backbone. Ordinary therefore has about 140.3 times as many
trainable parameters. The compute paths and memory costs also differ. Shared native weights do not
imply identical starting forward functions: feedback starts with its own larger
WikiText deficit.

O5e uses one ordinary CE. O5c's ordinary-pass CE was constant with respect to
the fusion parameters, leaving one trainable feedback CE. Doubling the ordinary
loss would add a gradient-scaling difference. These objective semantics were
checked with tiny CPU gradient tests and observed on the actual run.

This is a practical equal-data continuation control. It supports the conclusion
that the measured repair is achievable through ordinary adaptation; it does not
isolate FBT architecture from adaptation capacity or demonstrate that a fully
trainable feedback model cannot help. The earlier fusion result remains evidence
that a relatively small trained input path can repair much of its own deficit.

## Numerical health and costs

All 512 recorded objectives and gradient norms are finite. Model and AdamW state
passed finite checks at checkpoint boundaries, and all frozen fusion hashes
remained unchanged. Every update clipped at norm 1: preclip norms have minimum
2.4415, median2.8524 and maximum29.8041 at update 84. That spike is isolated;
neighboring norms are 2.8891 and 3.6630. Eight updates exceed 5 and two exceed 10;
postclip native norms remain below 1. The last 64-update median is 2.7974.
Describe training as controlled under clipping. There was no sustained divergence,
health-gate trigger, restart or late quality collapse. This provides no reason
to reopen the broader precision investigation.

Median actual update time is 0.3681s, summed update time 191.22s, peak allocation
34.76 GiB. O5c mixed was 0.3916s median,204.72s summed and 18.66GiB peak. These
measurements exclude evaluations and checkpoint work and are not a FLOP match.
The O5e report timer spans 23.7827 minutes from tracker startup through final
synchronization, including evaluations/checkpoints but excluding earlier model
loading. The separate preflight took 110.42s. Full 14.15GB checkpoint serialization and
verified retention dominate elapsed time. O5c mixed's full run took 11.61 minutes
and saved smaller 4.81GB checkpoints.

The H100 preflight exercised both the first update and largest-row update with
real AdamW state. A disposable scheduled nonzero step changed all 65 native
tensors, left fusion fixed, and was then restored byte-for-byte. Both initial
full-domain ordinary NLLs reproduced the shared source. 94 distinct scoped CPU
tests passed, including native-only gradients, exact tiny future-update recovery,
report validation and retention guards. Independent actual-run review confirmed
data quotas, single CE, warmup and stability; full-GPU optimizer replay was not
added. Inherited model, optimizer, evaluator and O5b/O5c/O5d runtime files remain
unchanged.

One seed, development subsets and a narrow WikiText proxy limit interpretation.
Intervals use 1,000 paired original-document resamples, seed 20260922, covering
422 code documents/127,850 targets and 59 WikiText documents/246,910 targets.
They describe evaluation-document variability, not training-seed variance or
unknown pretraining overlap. Exact-online FBT scores are teacher-forced, not
free-running generation. Multi-GPU execution, longer contexts and programming
execution quality remain outside this control.

## Recommended next decision

Keep this ordinary endpoint as the adaptation reference. Before attributing a
quality difference to feedback, the next focused comparison would train the
native backbone **and** fusion on the same mixed plan, from the same O5b source,
using the same native LR and budget. Its finite-pass CE reduction should be
explicit: averaging the two pass losses would reduce to one ordinary CE when
both passes are ordinary, unlike an unnormalized sum. Reuse this compatible
ordinary control and assess the newly trained FBT endpoint under its actual
online execution as needed. This would address the large trainable-capacity
difference before adding RT or NextLat interactions.

That is a recommendation for a separately specified next milestone. No joint
FBT training, new ordinary arm, precision campaign or budget extension is queued.

## Recovery and retained evidence

- [PR12](https://github.com/taylorbollman/cdrm-w-latent/pull/12); runtime/protocol
  commit `3d3f865`, retention implementation `07774f7`.
- Run `.runtime/olmo1b-step60000/o5e-pilot-01/ordinary/`, exec72097 completed0,
  2026-09-22T13:27:22.533253–13:51:09.497991UTC. Do not resume it.
- [Training W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ri76qycw);
  preflight `62kr0zjb`. Configuration SHA256
  `497f2dc70fcd0fefbf907a51306e7c8bb76208c8e44c928cbe289d9b214221cd`.
- GCS prefix
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5e-ordinary-control/20260922T132415Z/`.
  Complete boundaries 128/256/384/512 are retained; older local files were removed
  only after a successor was verified. The final local checkpoint remains.
- Final `ordinary/update-000512.pt`:14,154,906,281 bytes, SHA256
  `62b288c1e0e7e725ca9b56f65b52d018080b97eaaf28af3215c1613dba3eba45`,
  generation `1790085054588072`.
- [Initial receipt](initial-storage-receipt.json) and
  [final receipt](final-storage-receipt.json) identify evidence archives and
  verified parent checkpoint/data references. Parent weights and datasets are
  reused without duplicate uploads. Read the current handoff and usage before
  future work; this completed control has no pending training.
