# Pinned NextLat language-model source evidence

This directory retains byte-for-byte files from `JaydenTeoh/NextLat`, revision
`b37d3411ab9b17be8638abbddb9529f0f3a0a5f9`. `manifest.json` records upstream paths,
URLs, sizes and SHA256 checksums. The accompanying upstream MIT license applies
to the retained source. These files are audit/reference material; the production
OLMo implementation does not import this training framework.

The language-model predictor concatenates **next embedding, then current
post-finalnorm state**, applies learned-scale RMSNorm (the upstream class is
named `LayerNorm`, but selects RMSNorm when `bias=False`), and uses three linear
layers with two GELUs and a residual addition of the source state. Its norm
epsilon is `1e-5`; linear initialization is normal with standard deviation
`0.02`. There is no second application of the backbone's final norm.

The pinned 1B language-model configuration uses projection factor **1.6**,
latent SmoothL1 weight **1**, KL weight **1**, auxiliary token CE weight **0**,
and prediction horizon **1**. Predictor width is
`128 * round(proj_factor * 2 * model_dim / 128)`: **6528** for OLMo's width
2048. The pinned 100M configuration uses factor 1.3. The historical A5
configuration instead uses factor 0.5 and KL weight 0; neither its aligned
state labels nor its loss configuration should silently become LM defaults.

For language modeling, the predictor maps `(h[t], e[t+1])` to the target
`stopgrad(h[t+1])`. SmoothL1 uses beta 1 and averages over valid positions and
latent coordinates. KL is `KL(teacher || student)` over the whole vocabulary;
the teacher state is `h[t+1]`, and both distributions predict token `t+2`.
The auxiliary readout use is detached, and so are the target state and teacher
distribution. Source states and conditioning embeddings retain their gradients.
Upstream detaches intermediate leaves as an implementation device, then
explicitly sends their accumulated gradients back through the trunk and token
embedding; permanently detaching these paths would change the method.

OLMo's input embedding and readout are tied, unlike the pinned NextLat LM's
separate matrices. The OLMo adaptation detaches only the auxiliary readout
*use*, retaining ordinary CE and input/conditioning-embedding gradients.
Its explicit same-document pair/triple masks replace upstream EOS-derived
mask construction. Prompt/response loss selection is independent for CE,
latent regression and KL. O3 accepts one document per padded batch row and
rejects packed multiple-document rows, because loss masks alone do not isolate
backbone attention. Packing, autonomous latent rollout, multi-horizon training
and the upstream speculative decoding framework are outside this milestone.

The independent CPU tests verify these hashes and execute only unchanged AST
class definitions for the predictor/configuration/norm to check implementation
parity. The dense objective oracle is written separately in the tests; it does
not reuse the production chunked loss implementation.
