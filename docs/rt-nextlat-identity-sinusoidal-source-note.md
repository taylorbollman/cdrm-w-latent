# RT + NextLat identity-centered initialization and sinusoidal positions: source note

Written 2026-09-14 before the combined development pilot. This note records
the source audit and the intended mathematical changes; it does not claim
that the pilot has completed or that either change improves accuracy.

The user requested a rapid combined pilot inspired by the initialization in
*The Illusion of State in State-Space Models*, together with standard fixed
sinusoidal positional embeddings. The comparison is the existing two-layer
RT + NextLat 10,000-update A5 pilot. Both changes are intentional; this first
comparison cannot attribute an outcome to either change separately.

## What the source specifies and implements

The [paper, Sections 5.2 and 6](https://arxiv.org/html/2404.08819v3#S5.SS2)
defines an input-dependent SSM transition of the form

\[
h_t=\alpha(x_t)h_{t-1}+\bar Bx_t,
\qquad \alpha(x_t)\in\mathbb R^{d_s\times d_s}.
\]

Its initialization discussion motivates matrices centered on the identity
so that transitions can propagate previous state. This is a statement about
an SSM transition matrix, not an attention weight or an RT parameter.

The linked [word-problem repository](https://github.com/jopetty/word-problem/tree/8f910f92e1c70455dcd9376f56032dfc55126188)
imports `IDS4TokenClassifier` from its `sfirah` dependency. The inspected
source revisions are:

| Repository | Inspected commit | Relevant source |
| --- | --- | --- |
| `jopetty/word-problem` | `8f910f92e1c70455dcd9376f56032dfc55126188` | [`src/main.py`, `train_ids4`](https://github.com/jopetty/word-problem/blob/8f910f92e1c70455dcd9376f56032dfc55126188/src/main.py#L1260); [`pyproject.toml`](https://github.com/jopetty/word-problem/blob/8f910f92e1c70455dcd9376f56032dfc55126188/pyproject.toml) |
| `jopetty/sfirah` | `30b5a87342882f42dc3ed4136ee1ade3633c266e` | [`sfirah/ids4.py`](https://github.com/jopetty/sfirah/blob/30b5a87342882f42dc3ed4136ee1ade3633c266e/sfirah/ids4.py#L30); [`sfirah/transformers.py`](https://github.com/jopetty/sfirah/blob/30b5a87342882f42dc3ed4136ee1ade3633c266e/sfirah/transformers.py#L24) |

These are the revisions inspected for this audit, not verified identities
of the authors' historical training environment. The word-problem dependency
declaration does not pin a `sfirah` commit. The inspected IDS4 file's latest
path-specific commit is `7d050262c889d72829b6a7923598ae0c74a10f0f`, dated
2024-03-29.

The released IDS4 implementation initializes a tensor containing one matrix
per input feature:

\[
A_r=I_{d_s}+\frac{G_r}{\sqrt{d_s}},
\qquad (G_r)_{ij}\overset{\mathrm{iid}}\sim\mathcal N(0,1).
\]

Thus the per-entry noise standard deviation is \(1/\sqrt{d_s}\). The
word-problem trainer defaults to model width 255 and state width 110. The
forward code first computes

\[
T_t=\operatorname{GELU}\!\left(\sum_r x_{t,r}A_r\right),
\]

then forms prefix matrix products, projects those products, and adds an
input-dependent output branch. Consequently, for arbitrary embedded inputs,
the actual resulting transition is not literally \(I+\mathcal N(0,\sigma^2)\):
the identity terms are mixed by the input and followed by GELU. The new RT
pilot borrows the explicit **matrix initialization law**, not this IDS4
architecture, its matrix-product computation, or its training protocol.

## Mapping the idea to our recurrent Transformer

Our RT has no learned matrix directly corresponding to \(\alpha(x_t)\).
Within one recurrent block, let \(x_t\) be its input and \(z_t\) its output.
At query position \(t\), the records used for values are

\[
r_{tj}=\begin{cases}
z_j,&j<t,\\
x_t,&j=t.
\end{cases}
\qquad v_{tj}=W_VN(r_{tj}),
\]

where \(N\) is the block's attention LayerNorm. Earlier positions use the
permanent output-derived record, while self-attention uses the provisional
input-derived record. The head-specific attention mixture and block output
are

\[
a_t=\operatorname{concat}_h
  \left(\sum_{j\le t}p_{tjh}\,P_hv_{tj}\right),
\qquad u_t=x_t+W_Oa_t,
\qquad z_t=u_t+\operatorname{MLP}(N_f(u_t)).
\]

Here \(P_h\) selects head \(h\)'s value coordinates and \(p_{tjh}\) is its
softmax attention probability. Query/key projection and normalization still
determine those probabilities. After computing \(z_t\), the block stores its
normalized key/value projections for future positions.

The implementation is in the local
[`recurrent-transformer/olmo/model.py`](../recurrent-transformer/olmo/model.py):
`PreAttentionBlock._eager_forward` projects normalized inputs;
`OLMoRecurrentAutogradBlock._real_forward` exposes the direct scan;
`OLMoRecurrentBlockTiledFunction.forward` performs the same recurrence through
tiled attention updates; and `PostAttentionBlock.forward` applies the
attention residual and MLP. In the inspected source these start near lines
997, 1155, 1329 and 1025 respectively. The new implementation must leave this
historical source file unchanged.

The chosen analogue initializes the existing value and output projections
in **both** recurrent blocks as

\[
W_V=I_D+E_V,\qquad W_O=I_D+E_O,
\qquad (E_V)_{ij},(E_O)_{ij}
  \overset{\mathrm{iid}}\sim\mathcal N(0,1/D).
\]

This uses the released dimension-scaled variance with our square projection
width \(D=512\), giving per-entry standard deviation
\(\sigma=1/\sqrt{512}\approx0.04419417\). The two matrices receive independent
noise. Concretely, \(W_V\) occupies the lower 512 rows of `kv_proj.weight`,
and \(W_O\) is `attn_out.weight`. Their shapes and trainability are unchanged.
Using both maps avoids leaving a randomly initialized output projection to
remix the coordinates immediately after an identity-centered value map.

This is **identity-centered**, not a claim of a small matrix perturbation:
for a fixed vector \(v\),

\[
\mathbb E\|E_Vv\|_2^2=\|v\|_2^2,
\qquad
\frac{\|E_V\|_F}{\|I_D\|_F}\approx1.
\]

The same holds for \(E_O\). Their composition includes both perturbations
and their product, so it also has no small-noise guarantee. The bounded
initialization checks should record the actual Frobenius ratios. Matching
the released variance scaling was selected instead of introducing an
additional small-noise coefficient.

LayerNorm, head-specific attention mixtures, the provisional self record,
the residual input, and the MLP all remain present. Thus this initialization
does **not** make the complete time-step map or its Jacobian an identity,
guarantee preservation of the previous state, or force attention to select
the immediately previous position. It is a practical hypothesis about the
value read/write coordinates. Setting only Q/K to identity would address a
different hypothesis about retrieval, and suppressing residual branches
would address identity through depth rather than temporal memory.

Query/key parameters, learned normalization, MLP parameters, token embedding,
output head, and the NextLat predictor retain their baseline initialization.
The predictor already computes a residual latent prediction; changing its
initialization is outside this combined pilot. No autonomous predictor
rollout is introduced.

## Fixed sinusoidal positions and embedding scope

The inspected `sfirah` Transformer uses additive sine/cosine positions with
base 10000 and starts at position zero. Its positional module adds the
encoding directly to token embeddings; it does not multiply token embeddings
by \(\sqrt D\). The intended RT input is therefore

\[
x^{(0)}_t=e(g_t)+s_t,
\qquad
s_{t,2i}=\sin\!\left(t/10000^{2i/D}\right),
\qquad
s_{t,2i+1}=\cos\!\left(t/10000^{2i/D}\right).
\]

Use the same formula at training length 12 and evaluation length 36, with
correct sequence-axis indexing for our `[batch, sequence, width]` tensors.
The sinusoidal table is fixed, carries no trainable parameter, and is added
once at the backbone input. Embedding dropout remains zero. ALiBi and RoPE
are both disabled. No learned positional embedding is retained.

This last condition requires an explicit implementation: OLMo constructs
`transformer.wpe` as a learned `nn.Embedding` whenever both `alibi` and `rope`
are false (near line 1998), and adds its output in `OLMo.forward` (near line
2250). Merely disabling ALiBi is insufficient. A new model factory must
replace this positional behavior with the fixed encoding while preserving
the historical files and checkpoint contracts.

The baseline learned token embeddings stay unscaled and retain their
initialization. Fixed unit-amplitude sinusoidal coordinates can consequently
be large relative to those embeddings at initialization; this is part of the
chosen positional intervention, not a hidden embedding-rescaling change.
The NextLat auxiliary remains conditioned on the **raw learned embedding of
the next operation**, as in the existing pilot. It does not receive an
extra positional embedding in that conditioning branch. Its target latents
come from the modified backbone, so they naturally reflect the backbone's
positional inputs.

## Interpretation boundary

Training and evaluation should retain the existing pilot's FP32 execution,
NextLat objective, data, order, optimizer and fixed 10,000-update endpoint.
Finite bounded forward/backward checks exercise the new inputs and
initialization without reopening the historical mixed-precision study.
Any observed result is a single-seed combined-development result. A hint of
improvement can motivate later removal of one change at a time; this pilot
does not independently establish an initialization effect, a positional
effect, or equivalence to either source paper's model.
