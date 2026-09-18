# Outcome: identity-centered V/O initialization plus fixed sinusoidal RT + NextLat

The combined variant completed the fixed **10,000-update** pilot successfully.
It fits the training-length task better at this endpoint, but **generalizes
worse beyond length 12** than the existing RT + NextLat pilot. Retain the
original model as the current length-generalization reference.

| Development metric at 10k | Existing RT + NextLat | Combined variant |
| --- | ---: | ---: |
| Length-12 token accuracy | 99.7533% | 99.9541% |
| Length-12 whole-word accuracy | 98.3047% | 99.6436% |
| OOD exact prefix E(12) | 98.3770% | 99.6523% |
| OOD exact prefix E(13) | 96.8652% | 43.7207% |
| OOD exact prefix E(14) | 62.2373% | 2.2676% |
| OOD exact prefix E(16) | 1.3350% | 0 observed |
| Length-36 mean token accuracy | 39.1194% | 35.6854% |
| Length-36 whole-word accuracy | 0 observed | 0 observed |
| Length-36 CE per token | 3.3475 | 10.8193 |

Each evaluation uses the same 102,400 frozen words per development role.
Whole-word accuracy requires every cumulative state to be correct. E(t)
uses the first t predictions from length-36 words; it is distinct from the
separately generated length-12 development set. Final-state accuracy at 36
is near chance in both arms (1.6426% and 1.5645%; chance is 1/60).

The learning curves differ substantially. The variant starts slowly, then
improves rapidly around updates 4,000–4,500. At 5k it already has better
length-12 whole-word accuracy, 91.5195% versus 68.2754%. Its 5k E(13) is also
higher, 40.4004% versus 34.6738%, while E(14) is lower, 2.0605% versus 6.6807%.
These intermediate observations do not replace the fixed 10k comparison.
The full run, including evaluation/checkpointing, took about 9.69 minutes on
the H100, similar to 9.77 minutes for the original pilot; this is descriptive
timing from separate runs, not a controlled speed benchmark.

## Exactly what changed

Both tiled recurrent blocks initialize W_V and W_O as
`I + independent Normal(0, 1/D)` per entry, with D512 and independent CPU
noise seed 1236. This is an RT analogue of the released IDS4 matrix
initialization law. RT does not have the same input-dependent transition
matrix. The perturbation is substantial: measured `||W-I||_F/||I||_F` is
0.999–1.002, so this is identity-centered rather than near-identity. It
does not guarantee a state-preserving temporal update. The
[source note](../../../rt-nextlat-identity-sinusoidal-source-note.md) gives
the equations and pinned implementation references.

ALiBi is replaced by fixed, unit-amplitude sine/cosine positions, base 10000,
starting at position zero and added once to unscaled backbone token
embeddings. RoPE and learned positional parameters are absent. All other
initial weights, including the NextLat predictor, match the original pilot
bitwise. Its conditioning input remains the raw next-operation embedding.

Data, all 10,000 minibatch orders, the FP32 runtime, loss, optimizer and
evaluation are matched. The joint training update and evaluator are the
original function objects. There is no autonomous NextLat MLP rollout.

## Checks and interpretation

All 27 focused model/trainer tests and five reporter tests passed. Small
tiled/naive FP32 joint-gradient comparisons passed at lengths 12 and 36;
25 discarded actual-batch joint updates were finite. The saved initialization
audit passed 37 checks, and the endpoint audit passed 30: all 25 parameter
tensors and Adam states are finite FP32, with every Adam counter at 10,000.
The observed result does not require a new precision investigation based on
these checks alone. These are bounded checks, not a proof that every possible
input or gradient path has been exhaustively validated.

The two changes were intentionally combined and only one training seed was
run. We cannot attribute the improvement at length 12 or the extrapolation
regression to either change individually. In particular, this outcome does
not rule out a smaller-noise initialization, a different mapping of the IDS4
idea, or either change alone.

If continuing this branch, the informative next experiments are the two
single-change 10k arms: fixed sinusoids with original initialization, and
identity-centered V/O with ALiBi retained. Start with the sinusoidal-only
arm to test the positional substitution without changing learned initial
weights. Those runs have **not** been started. Confirmation remains untouched.

[Detailed report and plots](report.md) · [Full-length PDF](length-full.pdf) ·
[Boundary PDF](length-boundary.pdf) · [Training losses PDF](training-losses.pdf) ·
[W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/gpj4nf96) ·
[W&B training](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/euz1ntfd)

This interpretation is a separate post-run note. The generated report,
CSV, figures and their recorded hashes remain unchanged.
