# Native OLMo resource accounting

The estimator in [`resource_estimates.py`](../cdrm/pretrained/resource_estimates.py)
counts the materialized-backward implementation's matrix arithmetic. The F3d
opt-in recompute path needs the explicit correction below; its new option is not
yet an argument of the estimator. The estimator deliberately does not
translate parameter count into `6NT`, or treat estimated FLOPs as measured device
utilization. CUDA graphs reduce scheduling overhead without removing the matrix
work counted here.

The scope is full-backbone training, attached FBT passes, positive pass-loss
coefficients, one unpadded document per row, and no cached prefix. Ordinary
activation checkpointing is optional; RT uses its current custom backward and
the vocabulary losses use their existing checkpointed chunks. Frozen-backbone
adaptation, inference, packed/padded attention, a future RT kernel and sharded
training need separate estimates. Precision affects speed and bytes but does
not change this conventional multiply-add count.

## Parameter ownership

With model width `D`, MLP intermediate width `M`, depth `L`, vocabulary `V`, and
NextLat hidden width `P`, the exact component counts are:

| Component | Unique parameters |
| --- | ---: |
| Native tied embedding/readout and blocks | `VD + L(4D² + 3DM)` |
| FBT fusion | `2D²` |
| NextLat, default bias-free predictor | `P² + 3DP + 2D` |
| Optional NextLat biases | additional `2P + 3D` |

The original OLMo-1B configuration is `D=2048`, `M=8192`, `L=16`, `V=50304`.
Native nonaffine LayerNorm adds no parameters. The NextLat configuration rounds
`P=128 × round(2D × 1.6 / 128)=6528`, matching the source implementation.

| Architecture | Training parameters | Deployable inference parameters |
| --- | ---: | ---: |
| Ordinary or RT | 1,176,764,416 | 1,176,764,416 |
| + FBT | 1,185,153,024 | 1,185,153,024 |
| + NextLat | 1,259,491,328 | 1,176,764,416 |
| + FBT + NextLat | 1,267,879,936 | 1,185,153,024 |

RT adds no weights. Additional FBT passes share existing weights. NextLat adds
82,726,912 training-only parameters; fusion adds 8,388,608. These architecture
counts assume the listed components are actually included. A runtime wrapper
can retain inactive fusion/predictor modules. The separate `parameter_inventory`
function therefore reports unique registered parameters, resident parameter
bytes, trainable parameters, existing gradient participants, optimizer ownership,
and caller-declared executed/inference sets. It deduplicates tied aliases by
parameter identity. Zero gradients still count as participation; absence of a
gradient is not used to infer whether a forward executed.

## Arithmetic conventions and ordinary blocks

One multiply-add is two FLOPs. Let `N=BT`, where `B` is physical batch size and
`T` is padded execution length (this initial estimator requires no padding).
The forward dense work of one ordinary block is

`F_dense = 2N(4D² + 3DM)`.

Fully trainable backward adds `2F_dense`: one input and one weight VJP per
matrix. Ordinary checkpointing adds between `F_dense - 2NDM` and `F_dense`.
Non-reentrant checkpointing may stop after saving the final FF-down inputs,
before executing that last matmul. This is a small implementation-dependent
range, not an assertion that the entire block always recomputes.

Ordinary attention forward is `4BD × pairs`, for QK and PV. The useful causal
pair count is `T(T+1)/2`; full-square execution uses `T²`. Its backward has four
gradient matmuls (`2 × forward`) and can additionally reconstruct QK inside a
Flash backward (`0.5 × forward`). Checkpointing adds another attention forward.
The estimate combines causal work with no internal reconstruction for its low
endpoint, and full-square work plus Flash score reconstruction for its high
endpoint. These endpoints bracket accounting assumptions; they are not rigorous
bounds on issued hardware instructions, tile padding or an arbitrary backend.
Recorded backend traces remain necessary.

## Current RT work

The current RT computes full fused QKV for permanent writes, even though it
discards the Q result. Counting an ideal KV-only projection would understate
actual work. The following terms apply to one selected layer invocation:

| Dense component | Matrix FLOPs |
| --- | ---: |
| Forward: temporary QKV, permanent QKV, finish | `14ND² + 6NDM` |
| Batched temporary/permanent projection reconstruction | `12ND²` |
| Sequential writer forward and input VJP | `12ND²` |
| Sequential finish forward and input VJP | `4ND² + 12NDM` |
| Batched finish reconstruction | `2ND² + 6NDM` |
| Batched projection input and parameter VJPs | `24ND²` |
| Batched finish VJP | `2ND² + 12NDM` |
| Total | **`70ND² + 36NDM`** |

The local VJPs freeze weights. The final batched finish treats attended values
as detached, so its output projection has a weight VJP but no attended-input VJP.
These distinctions matter; multiplying RT forward by three is incorrect.
Completed outputs are retained and reconstructed in parallel, so no sequential
forward recurrence is rerun in backward. Sequential local VJPs still execute.

Let `E=T(T-1)/2` be strict-causal historical pairs. The RT attention matmuls are:

| Attention component | Matrix FLOPs |
| --- | ---: |
| Dyadic forward historical tiles | `4BDE` |
| Complete attention reconstruction in backward | `4BDT²` |
| Reverse historical dV/dP/dK tiles | `6BDE` |
| Final query-gradient dP/dQ | `4BDT²` |

Every strict-causal pair appears once in the forward and reverse tile sums,
including non-power-of-two lengths. The temporary diagonal uses pointwise
operations and is outside the matrix-only estimate. The materialized reference also
materializes full-square FP32 score/probability/error arrays. A single
`[B,H,T,T]` FP32 tensor is `4BHT²` bytes: at `H=16,T=512`, 1 GiB for B64 or 2 GiB
for B128. Several arrays can coexist; neither this per-array number nor a sum of
all named tensors is a measured peak. It identifies a quadratic storage term
in the retained materialized reference.

F3d's opt-in `backward_memory="recompute"` removes those complete backward
attention arrays, retaining row statistics and bounded query/key workspace.
It adds QK reconstruction in the reverse historical tiles (`2BDE`) and in the
final query-gradient stage (`2BDT²`). For the same no-prefix matrix accounting,
add **`2BD(E+T²)` per selected RT invocation** to the existing estimate. At
B64/T512/D2048 this is about0.103 TFLOPs/update for one invocation; multiply
by the actual number of RT layers/feedback calls. Dense, loss and parameter
counts are unchanged. Softmax/pointwise work and kernel tile padding remain
excluded, so this small matrix-work addition does not predict wall time.
The complete model is not claimed to use linear memory: ordinary/padded masks,
long-context eager forward fallback and other model allocations are separate.

## Passes, fusion, predictor and vocabulary losses

With FBT enabled, K includes one ordinary bootstrap. For `R` selected RT layers,
there are `(K-1)R` RT block calls and `KL-(K-1)R` ordinary calls. With FBT disabled,
there is one stack with `R` RT calls. K1 FBT is ordinary regardless of the RT
selection. Beta zero omits fusion computation but does not omit extra stacks.

Each feedback pass projects both previous hidden state and next-token embedding
on the `B(T-1)` suffix positions. Fully trainable forward/backward costs
`12B(T-1)D²` per fusion call. Fusion norms, gating and blends are excluded
pointwise work.

Let `C` be selected CE targets, `J` selected KL triples, and `U` the union of
latent-pair and KL-source-pair positions. The shared predictor is evaluated on
the union, not once for each loss. With all three predictor matrices trainable,
its forward/backward costs `6U(3DP+P²)` per pass. SmoothL1 targets are detached;
regression contributes no separate dense readout.

A vocabulary projection for one position costs `R_vocab=2DV`. The canonical
loss functions checkpoint chunks of selected positions, each using the full
vocabulary:

- CE costs `4C R_vocab`: projection forward, recompute, state VJP and tied-weight
  VJP. The readout is the trainable native embedding matrix.
- KL costs `5J R_vocab`: teacher/student forward, teacher/student recompute,
  and student-state VJP. Teacher states and the auxiliary readout are detached.
  The no-grad teacher is still evaluated again during checkpoint replay.

Chunk size changes peak activation storage and launch overhead, not these
matrix totals. Every FBT pass has its own CE/NextLat computation. The objective
`L0 + gamma × mean(extra losses)` normalizes contributions but does not remove
pass work. Input exposure and objective counts remain per data example; the
ledger reports pass-repeated arithmetic separately. Accumulation scales tokens
and matrix work, not parameter counts.

## Example matching the F3 B64/T512 fixture

The fixture has 32,768 input tokens, 16,384 CE targets, 32,704 latent pairs and
16,384 KL triples per physical update. Latent pairs cover the predictor union.
These are fixture masks, not a recommendation for later learning datasets.
Use counts from the actual prepared layout when masks differ.

```python
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.resource_estimates import LossWork, estimate_training_resources

config = OLMoConfig.native_1b()
estimate = estimate_training_resources(
    config, batch_size=64, sequence_length=512,
    mode=FBTMode(num_passes=2, rt_mode=RTMode((0,))),
    nextlat=NextLatConfig(config.model_dim),
    loss_work=LossWork(ce_targets=16384, latent_pairs=32704,
                       kl_triples=16384, predictor_positions=32704),
    ordinary_checkpointing=True,
)
record = estimate.to_dict()
```

The resulting **matrix-arithmetic estimate** is 644.5–689.3 TFLOPs per update.
All eight architectural combinations, using those same shape/mask assumptions
and ordinary checkpointing, are:

| Configuration | Estimated matrix TFLOPs/update |
| --- | ---: |
| Ordinary | 281.8–304.9 |
| Ordinary + NextLat | 314.9–338.0 |
| RT at layer0 | 294.9–316.5 |
| RT at layer0 + NextLat | 328.0–349.6 |
| FBT K2 | 565.2–611.4 |
| FBT K2 + NextLat | 631.5–677.6 |
| FBT K2 + RT at layer0 | 578.3–623.0 |
| FBT K2 + RT at layer0 + NextLat | 644.5–689.3 |

These rows are analytic estimates, not eight measured throughput tests. They
also explain why arithmetic alone cannot identify the RT bottleneck: replacing
one of sixteen layers adds modest total matrix work while changing dependency
structure, kernel sizes and launch count substantially.

## Coverage and verification

Excluded arithmetic includes normalization, RoPE, nonlinearities, softmax,
elementwise CE/KL/SmoothL1, gather/scatter, embedding-gradient accumulation,
casts, clipping, optimizer/scheduler operations and communication. Allocation,
memory traffic and launch latency are not FLOPs. Frozen-parameter ownership and
zero pass weights require a new work audit. The estimator must be updated if
the RT implementation changes projection shapes or backward reconstruction.

The 20 focused CPU tests compare parameter formulas with real tiny modules,
deduplicate tied aliases, check exposure/pass accounting and reject invalid
counts. Dispatch traces count actual eager `mm`/`bmm` operations for tiny RT
forward/backward at lengths 3, 5 and 8: dense and attention totals agree exactly.
Separate traces verify the canonical CE/KL checkpoint and detached-readout
counts and bracket ordinary checkpoint early stopping. This validates the
accounting against executed code without another GPU numerical campaign; it
does not validate any new GPU kernel or convert an estimate into hardware FLOPs.
