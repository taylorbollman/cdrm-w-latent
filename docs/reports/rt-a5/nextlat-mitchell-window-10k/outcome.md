# RT + NextLat: Mitchell initialization, positions and layer-2 window

Both requested pilots completed 10,000 updates. **The original full-attention
Mitchell + ALiBi model remains the strongest of these three runs at that
endpoint.** Fixed sinusoids did not improve length generalization; restricting
the second ALiBi recurrent layer to a two-record attention window also did not
improve it. This is a single-seed development result.

All three models start with exactly the same 25 learned tensors, including
the original Mitchell backbone and unchanged NextLat predictor initialization.
The new runs have no identity-centered overrides. They share training data,
every minibatch's order, optimizer, joint state/latent objective, batch size
1,024, length-12 training and FP32 runtime. Each sees 10.24 million training
words. The comparison uses backbone inference, without autonomous NextLat
MLP rollout.

The first pilot changed only ALiBi to fixed unit-amplitude sinusoidal positions
with base 10,000, added once to the original unscaled token embeddings. There
are no learned position parameters or RoPE. The original raw token embedding
still conditions the NextLat predictor.

Before examining that pilot's results, we fixed the positional selection score
as mean exact-prefix accuracy E(t) over t = 13 through 36 at update 10,000.
All 102,400 OOD development words contribute, including those already failing
within the first 12 operations. ALiBi scored **7.2195%**, versus **2.4739%** for
sinusoids, so the second pilot used ALiBi and started again from the original
initialization.

| Model at 10,000 updates | L12 dev whole-word accuracy | E(13) | E(14) | E(16) | M(36) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mitchell + ALiBi, full RT (original baseline) | 98.3047% | 96.8652% | 62.2373% | 1.3350% | 39.1194% |
| Mitchell + sinusoids, full RT | 97.8350% | 54.2559% | 4.9277% | 0.0029% | 36.0196% |
| Mitchell + ALiBi, layer-2 window 2 | 93.0303% | 79.6963% | 26.0879% | 0.3457% | 37.1962% |

Here E(t) requires **every state through position t** to be correct. A(t)
measures only the state at t, while M(t) averages token accuracy through t.
The length curves use prefixes of the same length-36 outputs. All three runs
had zero observed E(36) on these 102,400 words; this does not establish zero
population probability. Length-12 token accuracies on the separate short
development set were 99.7533%, 99.6887% and 98.9552%, respectively.

The window pilot's mean E(13–36) was **4.5815%**, below its full-attention
ALiBi control. It also had more errors within the training length, so this
pilot does not isolate an effect on extrapolation alone. Learning curves were
not monotonic: for example, its smaller routine development check reached
97.44% whole-word accuracy at 7,500 updates. We retained the prospectively
chosen 10,000-update endpoint instead of selecting a favorable checkpoint.
The 1,000/5,000 checkpoints are descriptive in the full report.

Layer 1 remains standard full-prefix RT. For layer-2 input x_t and output z_t,
layer 2 directly attends only to temporary self K/V from x_t and permanent
K/V from z_(t-1). At the first position it attends only to self. The previous
output contains the entire previous block computation, and gradients remain
attached through earlier recurrent outputs. Thus the restriction concerns
direct attention reads; it does not impose two-token memory. The existing
tiled computation is retained, with no claim of sparse-kernel acceleration.

Validation passed: 79 CPU tests across initialization, model semantics,
training/resume, independent attention reference, selection and reporting;
both GPU preflights; and finite 25-update smokes at the actual training shape.
For the windowed model, combined-loss tiled gradients agreed with an
independent scan that physically constructs only the two permitted records
at lengths 2, 12 and 36 (largest absolute parameter-gradient difference
5.07e-7). Saved initial tensors matched the original baseline bitwise.
Both final model/Adam state audits passed, including all 10,000 optimizer
counters. The independent report audit verified all 432 metric rows,
18 evaluations, 12 checkpoint hashes and every paired minibatch-order hash.
Full and boundary plots use exactly the same endpoint rows.

These checks support the implemented semantics. They do not establish that a
windowed model cannot improve with another seed or training recipe. Final
confirmation data remain unused, and no run was extended beyond its pilot
budget. The current evidence supports retaining full-attention Mitchell +
ALiBi as the working reference while reviewing the next experiment.

[Full report and plots](report.md) · [Boundary PDF](length-boundary.pdf) ·
[Full-length PDF](length-full.pdf) · [Metrics CSV](metrics.csv) ·
[W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/u991ouce)

Training runs: [original ALiBi](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/j1pce7q9),
[Mitchell + sinusoids](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/oe69tg2n),
[ALiBi + layer-2 window 2](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/6c59kpwp).

All eight new checkpoints (0/1k/5k/10k for each arm) are retained with verified
checksums under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260914T175404Z-nextlat-mitchell-window/`.
The lineage's `checkpoint-storage.json`, `evidence-storage.json` and evidence
readback receipt record checkpoint and final archive verification. The full
implementation and continuation notes are in
[the handoff](../../../rt-nextlat-mitchell-window-handoff.md).
