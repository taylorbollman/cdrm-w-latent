# Completed mixed A5/Fuzzy learning-rate pilot

The higher learning rate substantially accelerated learning. At 5,000 updates,
the model achieved **83.4307% A5 length-36 whole-word accuracy** and **99.5960%
Fuzzy answer accuracy**. All recorded training values remained finite, but
late A5 performance oscillated enough that a further LR increase is not the
recommended next step.

[Training run](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/y0mjyvyb)
and [comparison figures](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/57mpiict).
The [original protocol](rt-nextlat-mixed-lr3e4-5000.md) records the matched setup.

## Results

Both arms use the same two-layer D128 restricted-first RT, NextLat, FP32,
initialization, data order, and 2,560 examples per task per update. The new
rate rises from 1e-4 at update 1 to 3e-4 at update 100 and then stays constant.
The reference uses constant 1e-4.

| Development metric | Original LR, 5k | New LR, 5k | Original LR, 15k context |
|---|---:|---:|---:|
| A5 length-12 whole word | 0.0000% | 99.9375% | 99.9961% |
| A5 length-36 whole word | 0.0000% | 83.4307% | 92.8535% |
| Fuzzy answer tokens | 90.5379% | 99.5960% | 99.9112% |
| Fuzzy first value token | 82.5510% | 99.6508% | 99.9256% |
| Fuzzy all-answer sequence exactness | 6.1719% | 91.1719% | 97.8906% |

A5 endpoint measurements use 102,400 words at each length; Fuzzy uses all
1,280 development examples. The new endpoint's A5 length-36 token accuracy
is 97.5973%; final-state accuracy is 90.2578%. Neither is whole-word accuracy.
The original 15k column is longer-budget context, not an equal-budget endpoint.

The first positive full-development A5 length-36 whole-word result moved from
10,400 to 2,400 updates. The new 2,400 result was only 41/102,400 exact words;
substantial accuracy developed afterward. Fuzzy first exceeded 99% answer
accuracy at 3,200 rather than 7,500 updates. These are first observed crossings,
not guarantees of sustained performance.

The new run took 162.83 minutes of training versus 163.02 minutes for the
original first 5k; training plus evaluation/checkpointing took about 168 minutes.
This improvement comes from fewer updates to learn, rather than faster updates.

## Stability assessment

All 5,000 history rows have finite numeric values, including losses and logged
gradient norms. Training completed and checkpoint retention verified. There is
no observed numerical breakdown or runaway loss trend.

Optimization is meaningfully uneven. On the same fixed 4,096-word monitoring
set, A5 length-36 whole-word accuracy was:

- 83.72% at 4,000, 49.22% at 4,100, and 86.11% at 4,200.
- 91.43% at 4,700 and 54.37% at 4,900.
- The separate full 102,400-word endpoint evaluation at 5,000 recovered to 83.43%.

These changes are too large to dismiss as small-set sampling noise; the first
two examples compare identical evaluation pools. They indicate sensitivity of
the learned behavior to parameter updates. They do not establish an arithmetic
precision defect. A single run also cannot attribute every fluctuation solely
to LR or distinguish task interference from other optimization effects.

Gradient clipping at norm 1 was active on 44.28% of updates, compared with
48.10% of the original first 5k. The new final 1k clipped on 33.3% of updates.
Learning stages differ, so clipping frequency alone is not a controlled
stability comparison. The largest pre-clipping norm was 49.46 at update 3,772;
the original entire 15k peaked at 21.33. At the new spike, A5 training CE rose
from 0.0101 at update 3,770 to 0.3327 at 3,772, then recovered to 0.0094 at 3,773.
Clipping remains useful; it does not eliminate the observed accuracy swings.

## Recommendation and proposed next architecture

Keep **3e-4 with the existing 100-update warmup** for the proposed three-layer
baseline: **restricted RT → full RT → full RT**, retaining NextLat and the
remaining recipe. Do not combine this depth change with another LR increase.
Depth has been consequential in earlier A5 diagnostics, so its result still
needs to be measured rather than inferred from the two-layer model.

A separate matched 5e-4 probe could be considered later if reducing acquisition
time remains the priority. Current fluctuations make it a lower priority than
establishing the requested architecture. If the goal instead becomes a steadier
final checkpoint, a separately tested reduction toward 1e-4 after learning
would be more attractive than another increase. Neither change has been run.

The next architecture is context only: no new training is queued or launched.
One matched seed and reused development pools support a directional conclusion;
final confirmation remains unevaluated.

## Evidence and retention

The existing [comparison report](reports/rt-nextlat-fuzzy-a5/d128-b2560-lr3e4-5000/report.md)
and its figures contain the complete recorded curves. Its generic sentence
about a disappointing 5k result is a prospective qualification, not a finding
that this completed run disappointed. The results here supersede that generic
wording as the completed assessment.

Runtime: `.runtime/rt-nextlat-fuzzy-a5/20260920T184640Z-d128-b2560-lr3e4-5000/`.
`completed-assessment.json` records finite-value checks, gradient statistics,
late observations and input hashes. An independent read-only review agreed
with the stability assessment and recommendation.

Final checkpoint SHA256:
`fbe70360dcb2e4429a673244bfd7343ad176263ef53bb9499229f74d533b562f`.

Verified archive:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260920T184640Z-d128-b2560-lr3e4-5000/final-evidence.tar.gz`.
Archive SHA256:
`8bd3e8c1ca9da7e0053cebca4336c290757937d3d0c404a5d2f3aadad2c34e3f`.
The endpoint checkpoint's current bytes match the retention receipt. This
post-run assessment is stored separately in the persistent project directory.
