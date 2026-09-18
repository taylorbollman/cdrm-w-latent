# RT + NextLat: combined identity-centered initialization and sinusoidal pilot

This is a new development experiment authorized on 2026-09-14. The user
wanted rapid prototyping with both changes together, followed by review.
Do not start single-change ablations, longer training, or a confirmation-set
evaluation automatically.

## Reproduction and scope

Lineage: `.runtime/rt-a5/20260914T173148Z-nextlat-identity-sinusoidal/`.
The prospective `protocol.json` and `source-freeze.json` identify the run.
The primary comparator is the existing 10k RT+NextLat run at
`.runtime/rt-a5/20260911T191702Z-nextlat/train-rt-nextlat/`.

Both arms have two tiled recurrent blocks, rho 1, D512, H8, GELU FFN2048,
LayerNorm and full-width learned Q/K normalization. Parameter count remains
6,357,504 in the backbone and 1,049,600 in the NextLat predictor: 7,407,104
total. No CDRM or autonomous NextLat predictor rollout is used.

The new arm changes:

1. Both blocks' value projections and attention output projections start as
   `I + Normal(0, 1/D)` independently per entry. The variance is `1/D`, and
   the standard deviation is `1/sqrt(D)`; noise uses isolated CPU seed 1236.
   Within fused K/V weights, only the lower D rows change. Every other
   backbone and predictor parameter retains the original initialization.
2. ALiBi is disabled in the model and both block configurations. Standard
   fixed sine/cosine positions, base 10000, position origin zero and amplitude
   one, are added once to unscaled token embeddings. RoPE remains off. The
   `wpe` module is parameter-free; its deterministic frequency buffer is
   nonpersistent. There are no learned positional weights or added Adam slots.

The [source note](rt-nextlat-identity-sinusoidal-source-note.md) explains why
this is an RT analogue of the IDS4 initialization law, rather than an IDS4
transition reproduction. At the selected variance, matrix perturbations have
order-one RMS size. Neither a near-identity temporal map nor reliable previous
state propagation is guaranteed. LayerNorm, attention mixing, self contribution,
residual paths and MLP remain active.

NextLat continues to predict a detached next post-final-LayerNorm latent from
the current attached latent and the **raw** next-operation embedding. Its
residual three-linear GELU predictor, RMSNorm, SmoothL1 weight 1 and
same-position state CE are unchanged. Ordinary inference calls only the RT
backbone. Predictor seed is 1235; backbone/data-order seed is 1234.

The run uses the frozen A5 corpus, 800k training words, training length 12,
physical batch 1024, constant AdamW LR 1e-4, betas (.9,.95), epsilon 1e-8,
matrix decay .01, vector decay zero, and one global norm clip at 1. All
arithmetic/state is FP32, with autocast, TF32, compile and CUDA graphs off.
The fixed endpoint is 10,000 updates / 10,240,000 word presentations.
Checkpoints are 0, 1k, 5k and 10k, with 102,400 words per development role at
trained checkpoints. Short checks use 4,096 rows every 500 updates. Final
confirmation remains unevaluated.

## Implementation and validation

New files are `scripts/rt_a5_nextlat_variant.py`,
`scripts/rt_a5_nextlat_variant_train.py`, and
`configs/rt_a5_nextlat_variant/base.json`. The trainer imports the **same
function objects** as the baseline for joint loss/backward/clip/Adam,
teacher-conditioned diagnostics, backbone evaluation and word order. It has
a separate checkpoint schema and adds the variant configuration to the strict
resume contract. Do not load this arm through the historical plain factory:
fixed positional buffers are reconstructed from the variant source/config.

The frozen NextLat source identity remains
`1e6d0c63289f01525bc0c19bba6b2646d61df10ddb815bc74f5c111b46f9f961`
(46 sources). The new training identity is
`6a53fe8acd9291417ac9e8708095fa7dab945694fb7dc4f2d1dcd6bd11af8a69`
(49 sources). Historical files were preserved. New validation/reporting
programs have their own recorded source coverage.

Before training: 12 model and 15 trainer CPU tests passed, including exact
unaffected initialization, position formula/input addition, raw predictor
embedding route, causality and exact model/Adam/RNG continuation. The GPU
preflight passed combined-loss naive/tiled comparisons at B2/D128/T12 and
T36, with maximum gradient absolute errors about 4.47e-8 and 2.61e-8.
All 25 discarded B1024/T12/D512 updates had finite FP32 state. This is bounded
correctness coverage; the earlier mixed-precision study was not reopened.

All GPU commands must use `scripts/docker_shell.sh`, with container path and
`nvidia-smi` checks. Use `CDRM_DOCKER_GPUS=none` for CPU artifact inspection.
The exact training command is saved in the lineage's `train.sh`.

## Tracking, retention and interpretation

Training: [W&B run](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/euz1ntfd).
Preflight: [W&B run](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/dl91wmsa).
Retain checkpoints and evidence under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260914T173148Z-nextlat-identity-sinusoidal/`.
The lineage's `retain.py` supports checkpoint upload and final evidence
archiving with checksum receipts. Archive only after reports and audits are
complete; put archive command logs outside the lineage being archived.

Compare E(t), A(t) and M(t) separately on the same OOD words. E(t) requires
all states through position t to be correct; A(t) tests the current state;
M(t) averages token correctness through t. Full and boundary plots must use
the same rows and checkpoint. The primary comparison is the fixed 10k
endpoint; 1k and 5k are diagnostic. A single-seed combined result cannot
attribute an effect to initialization or positions separately.

## Completed endpoint and next-session state

The run stopped at exactly 10,000 updates and W&B synchronized successfully.
No training remains running. Read the
[outcome](reports/rt-a5/nextlat-identity-sinusoidal-10k/outcome.md) and
[detailed report/plots](reports/rt-a5/nextlat-identity-sinusoidal-10k/report.md).
The [W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/gpj4nf96)
contains paired figures and metric tables.

The variant's length-12 whole-word development accuracy is 99.6436%, versus
98.3047% for the baseline. Its OOD E(13)/E(14) are 43.7207%/2.2676%, versus
96.8652%/62.2373%. E(16) and E(36) have zero observed successes in the variant;
the baseline E(16) is 1.3350%. The combined change improves the training-length
fit but worsens extrapolation at this fixed endpoint. The original RT+NextLat
remains the reference; individual initialization/position effects are unresolved.

The actual initial checkpoints passed 37 saved-state checks: exactly the
four intended tensors differ and all unaffected slices/parameters match.
Measured perturbation Frobenius ratios are 0.99915–1.00177; the composed
W_O W_V linear surrogate differs from identity by ratios 1.72495 and 1.73413.
These are not recurrent Jacobian measurements. The final checkpoint passed
30 checks, including all 25 model tensors and Adam states finite FP32,
Adam counters exactly 10k, unchanged source identities, and matched complete
minibatch histories. See `init-state-validation.json` and
`final-state-validation.json`. Five focused reporter tests passed separately.

Final storage receipts are `checkpoint-storage.json`, `evidence-storage.json`
and `evidence-readback.json` in the lineage. The final evidence audit verifies
metric rows and generated artifacts independently. These files identify the
retained checkpoint generations and archive membership. Do not alter a closed
archive's members; future ablations need a fresh lineage and explicit scope.

Suggested follow-up, if authorized: sinusoidal-only with original initialization,
then identity-centered V/O with ALiBi. No follow-up arm, longer continuation,
confirmation evaluation or extra noise sweep has been started.
