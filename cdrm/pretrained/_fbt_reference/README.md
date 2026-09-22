# Pinned FBT source evidence

The files here are byte-for-byte evidence from
`xidulu/Full-bandwidth-transformer` at
`7037c60924870aca6e30fac95212b0c7caee052d`. `manifest.json` records their paths,
URLs, sizes and SHA256 hashes; `LICENSE` is the upstream MIT license. The OLMo
implementation does not import or execute this source. The repository is the
author's Nanochat implementation; preserving its feedback operation does not
make an OLMo port a reproduction of all its model/training choices.

## Feedback semantics in nanochat/gpt.py

- `LatentFeedback.forward`, lines 101–114, defines the selected `gate_product`
  fusion as `RMSNorm((W_state h_previous) * sigmoid(W_gate RMSNorm(e_current)))`.
  The state branch is the value; the embedding enters through the gate. The
  fusion is not an embedding-plus-residual addition. Neither RMSNorm has a
  learned scale; the helper supplies no explicit epsilon, so PyTorch chooses
  its input-dtype default. There are two active bias-free D-by-D matrices.
- `init_weights`, lines 387–389 and 434–437, initializes feedback matrices from
  `Uniform(-sqrt(3/D), sqrt(3/D))`, giving standard deviation `1/sqrt(D)`. The
  upstream module owns dormant concatenation/addition alternatives for stable
  checkpoints; these are not necessary to implement the selected fusion.
- `_run_trunk` returns the state after its final RMS normalization. In the
  multi-pass loop, lines 849–861, pass p uses pass p-1's state at position t-1
  and the original embedding at position t. Gradients remain attached across
  passes. The first token stays ordinary. Every pass reruns the shared stack
  with fresh attention state; no cache is transferred between Jacobi passes.
- Masks have shape `[K-1,B,T]`; false positions take ordinary token inputs,
  not the preceding pass's inputs. The mask helper resets at each BOS and
  optionally samples a nonempty ordinary prefix uniformly from lengths 1..L,
  including the all-ordinary case. This protects the feedback edge only; it
  is not a packed-document attention-isolation mask.
- The training-only jitter default is 0.02: independent uniform noise in
  `[-0.02,0.02]` is added to the shifted state before fusion. There is no jitter
  in evaluation. Jitter and sampled prefix masks require explicit RNG policy.
- K counts all full passes, including ordinary pass 0. K=1 is the ordinary
  objective. The loss is `L0 + mean(L1,...,L[K-1])`, with extra-pass coefficient
  fixed to one in this source. Every pass uses all unignored targets, including
  plain-prefix positions; feedback masks do not select loss positions.

## OLMo port boundaries

OLMo retains its native RoPE, non-affine LayerNorm, tied full 50,304-row readout,
unscaled token embeddings and native CE. We do not import Nanochat's smear,
backout, token-value tables, residual scalars, 15-logit softcap or padded-vocab
cropping. Scale calibration of the new RMS-normalized fusion to native OLMo
embedding RMS and the beta interpolation are explicit adaptation choices.
Beta zero must preserve native embeddings exactly. Packed-document rows remain
unsupported; explicit valid/document masks supersede BOS-token inference.

O5 additionally applies the existing horizon-one NextLat objective independently
to each pass's post-finalnorm states, sharing one predictor. This combination
is our experiment, not a claim made by the FBT source. For each CE/latent/KL
term the aggregate raw sum is `S0 + gamma*mean(S_extra)` and its denominator is
the same valid-position count used for a single pass. K1 has no extra term;
gamma defaults to one. Existing NextLat stopped targets, detached auxiliary
readout use, attached source/embedding paths and independent masks remain intact.
