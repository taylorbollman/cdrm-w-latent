# Native OpenELM RT reference

This is Stage B of the [pretrained research plan](fbt-rt-nextlat-research-plan-v2.md).
It adds a differentiable sequential reference to the validated native OpenELM
adapter. It is deliberately small and slow enough to inspect. It is the
correctness target for a later tiled implementation, not that implementation.

## Exact computation

For a selected layer, let \(x_t\) be its incoming residual stream, \(A\) its
native attention-input RMSNorm, and \(z_t\) its complete output after both
attention and MLP residual additions. The current query and temporary self K/V
are projected from \(A(x_t)\). For every earlier position \(s<t\), persistent
K/V are projected from

\[
m_s=(1-\alpha)x_s+\alpha z_s,\qquad p_s=A(m_s).
\]

Writing \(N_Q,N_K\) for the native headwise learned RMSNorms and \(R_t\) for
RoPE at position \(t\), the attention inputs are

\[
q_t=R_tN_Q(W_Q A(x_t)),\quad
k_s=R_sN_K(W_Kp_s),\quad v_s=W_Vp_s\quad(s<t),
\]

\[
\widetilde{k}_t=R_tN_K(W_KA(x_t)),\qquad
\widetilde{v}_t=W_VA(x_t).
\]

Each query head attends to the corresponding grouped KV head over
\((k_{<t},\widetilde{k}_t)\) and \((v_{<t},\widetilde{v}_t)\), with ordinary
scaled softmax attention. The native block then computes

\[
r_t=x_t+W_O a_t,\qquad
z_t=r_t+\operatorname{SwiGLU}(N_{\mathrm{FFN}}(r_t)).
\]

Only after completing \(z_t\) do we publish the persistent write from
\(m_t\). Thus current self-attention always uses temporary input-derived K/V;
it cannot read its own not-yet-computed persistent output. The final model
normalization is **outside** the layer and outside this memory write.

At \(\alpha=0\), the scan is mathematically ordinary causal attention, including
its gradients. The code actually performs this scan rather than bypassing it.
At \(\alpha=1\), historical memory comes from completed layer outputs. For
fractional alpha, both the input and completed-output branches remain attached
to autograd. Alpha is a fixed per-call configuration scalar, not a learned gate.

The initial actual-checkpoint validation selects **layer index 0 only**. All
other 27 layers remain ordinary native OpenELM. The tiny tests also exercise
top-layer and multiple-layer selections. This is not the restricted-first
architecture used in the historical synthetic experiments.

## API and checkpoint ownership

```python
from cdrm.pretrained.recurrent import OpenELMRecurrentModel, RTMode

# Load exactly the native state through the validated artifact loader as in
# openelm-import-usage.md. There are no new parameters or renamed state keys.
model = OpenELMRecurrentModel(config, device="cuda")
model.load_state_dict(native_state, strict=True)

ordinary = model(ids, mode=RTMode(selected_layers=(), alpha=0))
bridge = model(ids, mode=RTMode(selected_layers=(0,), alpha=0.37))
recurrent = model(ids, mode=RTMode(selected_layers=(0,), alpha=1))

prefix = model(ids[:, :3], mode=RTMode((0,), 1), use_cache=True)
suffix = model(ids[:, 3:], mode=RTMode((0,), 1),
               past_key_values=prefix.past_key_values, use_cache=True)
```

There is one embedding/readout parameter, still counted once by the optimizer.
The native 226 tensor keys and 1,080,153,600 parameters are unchanged. Build any
optimizer after strict loading/device movement. No optimizer or training loop
is part of this reference milestone.

Modes are immutable and passed explicitly to each call. Several forwards can
share weights before one backward without overwriting an earlier call's mode.
`inputs_embeds`, `return_logits=False`, masks and absolute position IDs work as
in the ordinary adapter. Hidden states are returned after final model RMSNorm.

## Cache boundaries

Caches retain attached native-head K/V and normalized **unrotated** keys. RoPE
is applied when reading attention, using stored absolute positions. This avoids
double rotation and preserves OpenELM's 4:1 grouped attention without expanding
the persistent cache.

A cache belongs to its producing model, weights, selected layers, alpha and
execution context. Recreate it when changing weights, device/dtype, autocast,
gradient/inference mode or configured attention backend. Ordinary and recurrent
cache types cannot be interchanged. Grad-enabled cached continuation remains
attached to the prefix graph; normal PyTorch graph-lifetime rules apply.
Unsupported writes through `parameter.data` bypass normal mutation tracking
and must not occur while a cache is live.

Masks cover the entire cached-plus-current key sequence, and cached prefix
validity cannot change. Default positions count valid tokens; explicit cached
coordinates continue after the maximum valid coordinate. Padding queries must
be excluded from losses. These are causal decoder caches, not serializable
optimizer checkpoints or distributed cache objects.

## Reproduce the bounded checks

Use the project container; do not execute CUDA in the host shell:

```bash
bash scripts/docker_shell.sh bash -lc 'test -f /.dockerenv && nvidia-smi'
bash scripts/docker_shell.sh bash -lc 'python scripts/openelm_rt_validate.py \
  --artifacts .runtime/openelm-import/artifacts \
  --output-dir .runtime/openelm-rt-reference/validation-new'
```

The script requires a fresh output directory. It uses physical B1 and a fixed
16-token text prefix from the native tokenizer, the real 300k checkpoint, FP32
parameters, TF32 disabled, and selected layer 0. It logs online to
`taylorbollman/pretrained-fbt-rt-nextlat`.

FP32 checks compare the scan with ordinary native execution at alpha 0 and
with the independent oracle at alpha 0, 0.37 and 1. The oracle uses pinned
CoreNet norm/projection/MLP code, rebuilds historical memory from completed
outputs for each query, and uses explicit matmul/softmax instead of SDPA.
All parameter gradients and the input-embedding activation gradient are
included. Native-sized cache chunks, causality and temporary self-attention
are checked separately.

BF16 math/default checks record finite computation, gradient ownership and
descriptive differences, including alpha 1 versus FP32. These are **not a
BF16 training clearance**: the scan and oracle use different GEMM shapes and
attention reductions, and the checkpoint has not been adapted to recurrence.
The script labels the requested default backend without asserting which fused
kernel was dispatched.

Intentional CPU tests use:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'python -m pytest -q tests/test_openelm_*.py'
```

Small tests include gradients through both fractional memory branches,
historical-input directional finite differences, chunked training gradients,
shared-parameter multiple forwards, nonuniform Q/K normalization gains,
padding, absolute positions and cache rejection. Actual checkpoint scope and
results are in the [Stage B report](reports/openelm-rt-reference/results.md).

## Retain new evidence

`scripts/openelm_rt_retain.py` verifies source hashes and the completed report,
then reuses the Stage A checkpoint's existing GCS object by URI, generation and
checksums. It uploads only the new evidence archive, manifest and receipt.

```bash
bash scripts/docker_shell.sh bash -lc 'env -u GOOGLE_APPLICATION_CREDENTIALS \
  python scripts/openelm_rt_retain.py \
  --artifacts .runtime/openelm-import/artifacts \
  --validation .runtime/openelm-rt-reference/validation-new \
  --checkpoint-receipt docs/reports/openelm-import/storage-receipt.json \
  --report-dir docs/reports/openelm-rt-reference \
  --output-dir .runtime/openelm-rt-reference/retention-new \
  --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-rt-reference/YYYYMMDDTHHMMSSZ'
```

Replace the prefix timestamp with the run's real UTC timestamp. The environment
override is specific to this VM's stale credential-path variable; it selects
the already-mounted valid ADC without changing credentials or `.env`.

Next milestone: native tiled execution/backward against this reference. Long
contexts, large batches, optimizer adaptation, FBT, NextLat and training quality
remain outside this milestone.
