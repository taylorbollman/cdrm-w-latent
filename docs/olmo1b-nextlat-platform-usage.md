# OLMo O3: NextLat language-model objectives and training platform

O3 adds auxiliary NextLat training to the existing native OLMo backbone and
checks complete optimizer updates and boundary resume. It does **not** start
the O4 comparative learning experiment. FBT is not implemented. GPU acceptance
and measurements belong in the [O3 report](reports/olmo1b-o3/results.md), with
the scope defined by the [O3 protocol](reports/olmo1b-o3/protocol.md).

The backbone remains original OLMo-1B at step60000, approximately 252B tokens:
16 blocks, D2048, full MHA with 16 heads of dimension128, SwiGLU intermediate
8192, non-affine LayerNorm, native FP32 split-half RoPE, and no Q/K normalization.
Its tied embedding/readout has all **50,304** native rows. The tokenizer uses
50,280 IDs; the extra output rows still participate in CE and KL normalization.
See the [native loading guide](olmo1b-native-rt-usage.md) and
[tiled RT execution contract](olmo1b-tiled-rt-usage.md).

The native backbone has **1,176,764,416** parameters. The default NextLat
predictor adds **82,726,912**, for **1,259,491,328** total. Predictor construction
does not reinitialize the backbone. Its seed uses an isolated CPU RNG scope,
so it does not consume the caller's CPU or CUDA random stream. Move the complete
wrapper to the intended device before constructing its optimizer.

## Source and predictor

[`nextlat.py`](../cdrm/pretrained/nextlat.py) follows the released **1B
language-model, horizon-one** configuration at NextLat revision
`b37d3411ab9b17be8638abbddb9529f0f3a0a5f9`. The byte-pinned source,
configuration files, hashes and license are retained in
[`_nextlat_reference/`](../cdrm/pretrained/_nextlat_reference/README.md).

Let `h_t` be the backbone's already final-normalized state at token t, and
`e_t = E[token_t]` its original token embedding. The auxiliary prediction is

\[
\widehat h_{t+1} = h_t + W_3\,\mathrm{GELU}\!\left(
W_2\,\mathrm{GELU}\!\left(W_1\,\mathrm{RMSNorm}([e_{t+1};h_t])\right)
\right).
\]

The learned-scale predictor RMSNorm has epsilon `1e-5`. There are three
bias-free linear layers, two GELUs, and no predictor dropout by default.
Linear weights use normal initialization with standard deviation `0.02`.
The source's class named `LayerNorm` selects RMSNorm when `bias=False`; the
native backbone still uses its own non-affine LayerNorm. The backbone's final
norm is not applied to its states a second time.

`NextLatConfig(model_dim=2048)` selects projection factor **1.6** and hidden
width `128 * round(1.6 * 2 * 2048 / 128) = 6528`. Its initial latent and KL
coefficients are both **1**. The old A5 recipe used factor 0.5 and KL coefficient
0, with aligned state labels; those are not the LM defaults. Multi-horizon
prediction, auxiliary next-next-token CE, and autonomous predictor rollout are
outside O3.

## LM alignment, masks and gradient ownership

CE uses `h_t` to predict token `t+1`. The latent objective compares
`predicted_h[t+1]` with `stopgrad(h[t+1])`. KL compares the token distributions
of these two states, both predicting token **t+2**:

\[
\begin{aligned}
q_{t+2} &= \operatorname{softmax}(\operatorname{sg}(E)
                                     \operatorname{sg}(h_{t+1})),\\
p_{t+2} &= \operatorname{softmax}(\operatorname{sg}(E)\widehat h_{t+1}),\\
L &= L_{\mathrm{CE}} + \lambda_{\mathrm{latent}} L_{\mathrm{latent}}
                         + \lambda_{\mathrm{KL}} L_{\mathrm{KL}},\\
L_{\mathrm{latent}} &= \frac{\sum_t m^{L}_t\sum_d
  \operatorname{SmoothL1}_{\beta=1}(\widehat h_{t+1,d},
                                    \operatorname{sg}(h_{t+1,d}))}
 {D\,\max(1,\sum_t m^{L}_t)},\\
L_{\mathrm{KL}} &= \frac{\sum_t m^{K}_t
  \sum_v q_{t+2,v}(\log q_{t+2,v}-\log p_{t+2,v})}
 {\max(1,\sum_t m^{K}_t)}.
\end{aligned}
\]

Sums include batch rows as well as positions. CE likewise divides by its own
valid target count. Source states and conditioning embeddings remain attached.
The target state and teacher distribution are detached. For auxiliary logits,
only the readout **use** is detached with `F.linear(predicted, E.detach())`.
The single tied parameter E still receives ordinary CE and input/conditioning
embedding gradients. There is no copied or independently optimized readout.

`NextLatBatch` contains these aligned `[B,T]` tensors:

| Field | Meaning |
| --- | --- |
| `input_ids` | int64 native token IDs, including valid lookup IDs at padded positions |
| `valid_mask` | bool token validity; never inferred from the token value |
| `document_ids` | int64 document identity; nonnegative at valid positions |
| `ce_mask` | optional bool selection at the **target token** position t+1 |
| `latent_mask` | optional bool selection at the **target state** position t+1 |
| `kl_mask` | optional bool selection at the **target token** position t+2 |

CE and latent regression require a valid same-document pair `(t,t+1)`.
KL requires a valid same-document triple `(t,t+1,t+2)`. The three optional
selection masks are independent. For example, response-only CE need not
exclude prompt positions from latent supervision. `None` selects every valid
pair/triple for that objective; it does not infer a response boundary.
`build_nextlat_masks(batch)` returns masks of widths `T-1`, `T-1`, and `T-2`.

O3 integration permits **one document per row**, with left or right padding.
It rejects rows containing multiple document IDs before executing the backbone.
Loss masks alone would not prevent attention from crossing document boundaries.
The loss primitive can mask independently computed states using document IDs,
but this is not packed-document attention support. The caller supplies document
boundaries and prompt/response policies explicitly; token IDs do not trigger an
implicit EOS split. Diagnostic fixtures append one explicit native EOS token.

All-empty terms return differentiable zero. Disabled objectives have zero
counts. An optimizer update with no valid positively weighted term is rejected.
Disabling NextLat, or setting both auxiliary coefficients to zero, omits its
predictor and bypasses auxiliary computation.

## APIs and independent execution switches

`NextLatLM` owns a native backbone plus an optional predictor. Its `forward`
is a **training-objective** interface returning `NextLatLosses`; it does not
return logits. `loss_sums` is the equivalent explicit method.

```python
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, NextLatLM
from cdrm.pretrained.recurrent import RTMode

# `backbone` has already been strictly loaded from the verified native artifact.
model = NextLatLM(backbone, NextLatConfig(model_dim=2048), enabled=True).to("cuda")
batch = NextLatBatch(ids, valid_mask, document_ids,
                    ce_mask=response_mask, latent_mask=None, kl_mask=response_mask)
mode = RTMode((0,), 1.0)  # Layer 0 recurrent; the other 15 remain ordinary.
losses = model.loss_sums(batch, backbone_kwargs={"mode": mode})
losses.total.backward()

# Inference uses native states and never invokes the auxiliary predictor.
model.eval()
with torch.no_grad():
    output = model.backbone(ids, attention_mask=valid_mask, mode=mode)
```

The example assumes `torch` is imported and tensors reside on the model device.
For an `OLMoTiledRTForCausalLM` backbone, `RTMode((), 0.0)` is ordinary execution;
`RTMode((0,), 0.0)` checks the ordinary limit through the selected tiled path.
The selected layers/alpha and `NextLatLM(..., enabled=...)` are independent
switches. An ordinary `OLMoForCausalLM` backbone is also supported; omit `mode`
for that class. No FBT passes or FBT loss aggregation are implemented here.

`NextLatLosses.sums`, `.counts`, `.means`, and `.weights` have `ce`, `latent`,
and `kl` keys; `.total` applies the coefficients to the means. A latent sum is
already averaged over coordinates, so its count is valid **pairs**, not scalar
coordinates. `model.counts(batch)` and `model.objective_weights()` expose the
denominators and coefficients before forward for gradient accumulation.

The lower-level `compute_nextlat_loss_sums(hidden_states, token_embeddings,
readout_weight, batch, predictor, config, enabled=True)` accepts attached,
post-finalnorm states and embedding tensors. This is the useful entry point
for independently testing a future backbone; calling it does not establish
that the supplied states respect document boundaries.

`NextLatConfig.vocab_chunk_size` means **valid positions per full-vocabulary
projection chunk**. It does not split the vocabulary or normalize partitions
separately. Each chunk includes all 50,304 rows. Non-reentrant activation
checkpointing recomputes its logits/log probabilities during backward, avoiding
retention of all position-by-vocabulary buffers at once. The default is 32;
O3 GPU validation uses 8 and profiling uses 128 positions per chunk. Ordinary projections
follow caller autocast; CE/KL and latent reduction arithmetic are FP32.

The objective wrapper owns input lookup and validity masks, and disables
backbone logits and caches. Passing `past_key_values`, `use_cache`, external
input tensors, or an overriding attention mask in `backbone_kwargs` is rejected.
Document chunks with attached cache credit need a separate objective alignment
design. The backbone's existing inference cache API remains available directly.

## Optimizer updates and exact boundary resume

[`lm_training.py`](../cdrm/pretrained/lm_training.py) supplies `build_adamw`,
`build_warmup_scheduler`, `optimizer_step`, `TrainingCounters`,
`save_training_checkpoint`, and `load_training_checkpoint`.

Materialize/load the full model, add the predictor, and complete device/dtype
conversion **before** constructing AdamW. The helper owns each trainable
parameter once, including the tied matrix. It applies weight decay to matrices
and excludes vector/scalar parameters. The optional scheduler is a linear
warmup: with N warmup updates, update1 uses `base_lr/N`, updateN uses `base_lr`,
then the learning rate stays constant. It is not a complete O4 training recipe.

`optimizer_step(model, optimizer, microbatches, config=LMTrainingConfig(...),
backbone_kwargs={"mode": mode}, scheduler=scheduler, counters=counters)` computes
separate CE/pair/triple counts over the **whole update** before accumulating.
Each microbatch contributes its raw sum divided by that global term's count.
It does not average microbatch means equally when valid counts differ. The
helper clips the accumulated gradient, performs AdamW, advances the scheduler,
clears gradients, and increments update/microbatch/document/token and objective
position counters. Returned metrics include loss sums/means, counts, objective
weights, gradient norm before clipping, used/next LR, and initialized optimizer
state bytes. `bf16_mixed` requires CUDA; no CPU fallback is provided.

Save only at a completed optimizer boundary with cleared gradients. Checkpoints
contain model and predictor tensors, AdamW moments, scheduler state, counters,
data cursor, resolved configuration, source identity, module training flags,
Python/NumPy/Torch CPU and CUDA RNG, and any explicitly named data generators.
Publication is atomic and refuses to overwrite an existing checkpoint.

Supply a canonical `configuration` dictionary covering model/NextLat config,
NextLat enabled state, RT mode, precision/backend policy, optimizer/scheduler,
and data/masking policy. Supply a `source_fingerprint` containing the native
`checkpoint_sha256` plus relevant code/source identities. These are explicit
caller-owned records; the helper does not infer omitted experiment settings.

On resume, rebuild the same materialized model, optimizer and scheduler, then
call `load_training_checkpoint` with the matching configuration/fingerprint,
named generators, and retained checkpoint SHA256. The loader checks ownership,
aliases, tensor layouts, configuration, source and RNG topology before loading.
It uses `load_state_dict(assign=False)` so optimizer parameter references remain
valid, then restores moments, scheduler, RNG, counters and the data cursor.
Continue using the **returned** counters and cursor. Mid-update resume, changed
GPU topology, distributed accumulation and cross-precision resume are outside
this contract. A failed step does not roll back consumed RNG or data; recover
from a completed boundary when those streams must be reproduced.

## Bounded validation and profiling

Run GPU commands inside the required project container; first confirm the
container working directory and GPU. Reuse the existing verified native
artifact directory. These commands perform no model download:

```bash
bash scripts/docker_shell.sh bash -lc 'test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi'

bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_lm_validate.py --artifacts .runtime/olmo1b-step60000/artifacts --output-dir .runtime/olmo1b-step60000/lm-validation-NEW'

bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_lm_profile.py --artifacts .runtime/olmo1b-step60000/artifacts --output-dir .runtime/olmo1b-step60000/lm-profile-NEW'
```

Use a fresh output directory for every attempt. Validation compares short
actual-checkpoint objectives and gradients to an independent dense objective,
and checks disposable optimizer recovery. Its newly updated states are
diagnostic, not adapted-model candidates. Profiling includes real initialized
AdamW moments and complete steps at zero learning rate: weights stay fixed,
while moments/counters still update. It reports wall/device timing, token and
objective counts, and memory; it is not a maximum-batch search.

Drivers log online under `taylorbollman/pretrained-fbt-rt-nextlat` and record
fixture tokens, masks, model/source hashes and configuration. Retain reports,
code and checkpoint receipts under `gs://fast-chunks`; use persistent project
or home storage for checkpoints needed beyond the session. SSD-only scratch
is disposable. The original native checkpoint remains unchanged.

This milestone supports single-process, one-GPU execution. It does not clear
DDP/FSDP, compile, CUDA graphs, double backward, packed documents, cached LM
training, or comparative learning quality. BF16 finite-gradient observations
are bounded diagnostics, not evidence of long-run training equivalence. O4
still needs an approved dataset/split, masking policy, adaptation budget and
matched ordinary/RT/NextLat learning comparisons.
