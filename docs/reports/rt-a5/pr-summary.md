# A5: paired two-layer Transformer and RT

This change adds a native A5 prefix-state experiment around our existing
Transformer and tiled RT. Both primary models have two blocks, width 512,
eight heads, GELU FFN width 2,048, learned LayerNorm/QK normalization, causal
ALiBi and 6,357,504 parameters. Both RT blocks are recurrent at rho 1.
Corresponding weights are mapped exactly at initialization. No RT/vendor
implementation, precision kernel or CDRM architecture was changed.

The task predicts the accumulated permutation after every operation with a
60-class native head and unshifted CE. The evaluator reports both independent
state accuracy and cumulative prefix exactness; the latter counts a word only
while all predictions through that position remain correct. The first pilot
trains at length 12 and evaluates through length 36.

The experiment uses **FP32 throughout, TF32 and autocast off, no torch.compile
or CUDA graphs**. This follows the approved simplification and avoids a new
mixed-precision study. Short-sequence FP32 timing proved practical, so no
execution optimization was needed for the pilot.

The paired pilot completed 10,000 updates per arm. Each endpoint was evaluated
on 102,400 words per development role:

| Length | Model | Token accuracy | Whole-word exact match | Final-state accuracy |
| --- | --- | ---: | ---: | ---: |
| 12 | SEQ | 39.72% | 0% | 1.69% |
| 12 | RT | 99.88% | 99.18% | 99.37% |
| 36 | SEQ | 14.35% | 0% | 1.67% |
| 36 | RT | 37.36% | 0% | 1.70% |

RT learned the training-length task much more successfully in this bounded
comparison. Neither model produced a fully correct length-36 word among the
102,400 evaluated examples. Higher mean token accuracy on the longer words
includes RT's strong early-prefix performance; it does not establish successful
long-word state tracking. Length-36 CE was 3.5511 for SEQ and 4.6701 for RT,
so RT's advantage in token accuracy does not extend to every metric.

Total measured run time, including initialization of tracking, evaluations and
checkpointing, was 176.76 seconds for SEQ and 571.57 seconds for RT. The actual
training-loop portions were 167.09 and 550.81 seconds. Final checkpoint
parameters and Adam states are finite FP32, and every paired training batch's
order hash matches. See [the full pilot report and plots](pilot/report.md).

Implemented components:

- Native deterministic A5 generation, independent data streams, prefix/full-word
  overlap checks, frozen uint8 arrays, SHA256 manifests and read-only loading.
- Shared two-block model construction, exact canonical weight pairing, explicit
  width variants, same-position loss and separate accuracy metrics.
- An ordinary training loop with matched shuffled epoch order, constant-rate
  AdamW, clipping, online W&B, per-update history and complete checkpoints.
- Checked fresh-process resume and development-only checkpoint evaluation.
- Bounded validation, profiling and paired-result reporting with exported plots.

The initial corpus contains 800,000 training words and 200,000 development
words of length 12, a separate long pool, and 102,400 independent confirmation
words. The length-36 development interface exposes 102,400 words. Complete
12-prefix intersections and complete-word intersections across roles are zero.
The native NumPy sampler preserves the task but does not claim to reproduce
the upstream Python generator's bytes. Confirmation has not received model
evaluation.

Verified implementation evidence:

- All 44 initial focused CPU tests passed: data, paired models, loss/metrics,
  checkpoint/order logic and validation-report arithmetic.
- A further 12 reporter tests passed, including rejection of mismatched or
  incomplete comparisons and checks of the exported plot data.
- Tiled-versus-naive FP32 backward checks passed at B2/D128 and lengths 12/36,
  covering all 21 recurrent parameter tensors. Global gradient relative L2
  errors were approximately 6.43e-7 and 5.84e-7 respectively.
- Both architectures passed causality, batch-order and truncated-prefix checks.
- Both reached 100% token and whole-word accuracy after 10 updates on the
  separate 32-word, length-3 memorization smoke (D128, LR1e-3).
- A fresh-process RT continuation from update 1 to 3 matched uninterrupted
  execution exactly in parameters, Adam state, RNG, data order and progress.

At physical B1024/T12/D512 on the H100, mean synchronized update time after
25 warmup updates was 15.55 ms for SEQ and 53.81 ms for RT over 50 timed
updates. No extra warmup was indicated by the measured timing drift. These
are fixed-fixture forward/backward/clip/Adam timings, excluding logging/data
transfer. The actual training loop and evaluation overhead are reported
separately with the pilot.

The current lineage is `.runtime/rt-a5/20260911T154748Z`. The paired pilot
uses seed 1234, batch 1,024 and 10,000 updates per architecture, with retained
checkpoints at 0/1,000/5,000/10,000. See [usage](../../rt-a5-usage.md),
[the approved plan](../../rt-a5-experiment-plan.md) and the
[W&B project](https://wandb.ai/taylorbollman/rt-a5-state-tracking).

The paper's GPT uses RoPE/RMSNorm/SwiGLU, whereas both of our arms retain
ALiBi/LayerNorm/GELU. This is a matched comparison of our components, not an
exact reproduction of that GPT or NextLat's separately rolled-out latent RNN.
The 10,000-update, single-seed development pilot is 2.5% of the reference
400,000-update budget. It does not establish final convergence or seed
robustness. The longer comparison and capacity ablations remain subsequent
milestones.
