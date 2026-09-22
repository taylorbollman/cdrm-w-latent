# Native OLMo feedback reference

O5a provides a bounded implementation, not an optimized generation or training
driver. Read [protocol](reports/olmo1b-o5a/protocol.md), the source snapshot
[audit](../cdrm/pretrained/_fbt_reference/README.md), and the current handoff.

Wrap a strictly loaded native `OLMoTiledRTForCausalLM` in `OLMoFBT`. Construct
the fusion only after the backbone has real weights: its fixed output scale is
the RMS of the native embedding matrix, including all 50,304 rows. New matrices
use an isolated generator, seed20260922, uniform bounds `sqrt(3/D)`. The branch
has two bias-free D-by-D matrices, FP32 RMS reductions, explicit epsilon1e-5,
and no learned normalization or scale parameters. Explicit epsilon, precision
and native embedding-scale calibration are OLMo adaptations, not bitwise
Nanochat reproduction. The native backbone is unchanged.

```python
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTConfig, FBTMode, FBTOnlineMode
from cdrm.pretrained.recurrent import RTMode

core = OLMoFBT(loaded_backbone, FBTConfig()).to("cuda")
mode = FBTMode(num_passes=3, beta=0.37, rt_mode=RTMode((0,), 0.37))
output = core(ids, attention_mask=valid, document_ids=documents,
              mode=mode, return_logits=False)
final = output.last_hidden_state
all_passes = output.pass_hidden_states
```

FBTMode defaults to FBT-only with two passes. **`RTMode(())` means ordinary;
the older `RTMode()` default selects layer0.** Pass0 always uses the explicit
ordinary mode. K1 is ordinary even when a recurrent layer is configured for
extra passes. `enabled=False` runs one ordinary/RT pass using `rt_mode`, ignoring
the feedback pass count. Beta0 bypasses the new fusion exactly; it does not
turn off RT in extra passes. Independent documents occupy separate rows.
Optional padding and explicit positions follow native OLMo conventions.
Packed rows and finite-pass caches are rejected.

Exact online feedback is a different API, without K:

```python
online_mode = FBTOnlineMode(beta=0.37, rt_mode=RTMode((0,), 0.37))
first = core.forward_online(ids[:, :3], mode=online_mode, use_cache=True)
rest = core.forward_online(ids[:, 3:], mode=online_mode,
                           past_key_values=first.past_key_values)
```

It requires an RT-capable native backbone, using an empty selected-layer set
for FBT-only execution. It consumes the freshly completed previous-token post-finalnorm state and
stores current execution's KVs. The cache and its feedback edge remain attached
under autograd. In cached calls, supplied attention masks and document IDs cover
the **entire cached prefix plus current tokens**; explicit positions cover only
current tokens. Changed modes, weights, fusion scale, model conversion or
autocast/gradient context reject a cache. An ordinary-prefix cache is not
accepted. Cache tensors must not be mutated; `.data` writes are unsupported.
No prefix switching, sampled prefix mixin, training jitter, beam search or
production generation integration is implemented here.

For objectives:

```python
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatConfig

model = FBTNextLatLM(core, NextLatConfig(model_dim=2048), enabled=True, gamma=1).to("cuda")
losses = model.loss_sums(batch, backbone_kwargs={"mode": mode})
losses.total.backward()
```

Move the entire wrapper to the chosen device after construction; the NextLat
predictor is initialized on CPU. `enabled` here controls NextLat independently
of FBT. The predictor is never used for inference. The result retains per-pass
`pass_losses` and `pass_coefficients`. Aggregate each objective as pass0 plus
gamma times the mean of extra passes. Consequently aggregate `means['ce']` is
an objective, **not final-pass NLL**. Counts are valid data positions once, so
existing `optimizer_step` normalizes uneven microbatches correctly. K1 has no
extra-pass term; K>1/gamma1 has total pass weight2.

Save configuration explicitly including the backbone config, fusion config,
FBT/RT mode, NextLat config and enabled flag, gamma, precision and optimizer
settings. Existing `save_training_checkpoint`/`load_training_checkpoint` retain
model/buffer, optimizer, scheduler, counters, RNG and cursor state; use an
explicit source fingerprint and immutable artifact paths. Tiny full-state
resume is tested; distributed and changed-topology resume are not.

The actual-checkpoint validation command, always inside the GPU container:

```bash
bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_fbt_validate.py \
  --artifacts .runtime/olmo1b-step60000/artifacts \
  --output-dir .runtime/olmo1b-step60000/o5a-validation-01'
```

The output directory must be new. This validates short finite/online mechanics
and gradients, logs online to W&B, and leaves all model weights unchanged. It
does not train the random fusion, demonstrate quality, profile T512 capacity,
or make BF16 training claims. Retain the selected result/source evidence in
GCS and update the compaction handoff before ending the milestone.
