# Native OLMo resource accounting

The estimator in [`resource_estimates.py`](../cdrm/pretrained/resource_estimates.py)
counts native training matrix arithmetic for either RT backward implementation.
Pass `backward_memory="recompute"` for F3d's bounded workspace implementation;
the default remains `"materialized"`. The selected option is included in the
serialized resource card. The estimator deliberately does not
translate parameter count into `6NT`, or treat estimated FLOPs as measured device
utilization. CUDA graphs reduce scheduling overhead without removing the matrix
work counted here.

Pass `kv_only_writes=True` when the RT execution omits the unused Q component
of permanent writes. Its default is `False`, preserving historical matrix
counts; the selected option is included in the resource card. RoPE-table reuse
changes pointwise work and storage, which this matrix-only ledger excludes.

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

The default RT computes full fused QKV for permanent writes, even though it
discards the Q result. The opt-in `kv_only_writes=True` execution reads the K/V
row view of that same packed weight. Temporary input-derived projections still
produce QKV. The following terms apply to one selected layer invocation:

| Dense component | Default full QKV writer | Opt-in KV-only writer |
| --- | ---: | ---: |
| Forward: temporary QKV, permanent write, finish | `14ND² + 6NDM` | `12ND² + 6NDM` |
| Batched temporary/permanent projection reconstruction | `12ND²` | `10ND²` |
| Sequential writer forward and input VJP | `12ND²` | `8ND²` |
| Sequential finish forward and input VJP | `4ND² + 12NDM` | `4ND² + 12NDM` |
| Batched finish reconstruction | `2ND² + 6NDM` | `2ND² + 6NDM` |
| Batched projection input and parameter VJPs | `24ND²` | `20ND²` |
| Batched finish VJP | `2ND² + 12NDM` | `2ND² + 12NDM` |
| Total | **`70ND² + 36NDM`** | **`58ND² + 36NDM`** |

KV-only writes save **`12ND²` matrix FLOPs per RT invocation**, comprising
`2ND²` in forward, `2ND²` in batched reconstruction, `4ND²` in the local writer
forward/input VJP, and `4ND²` in final batched input/weight VJPs. The old full-QKV
projection backward multiplies the zero Q cotangent through the full projection;
the smaller projection removes that matrix work too. Slice backward still
returns the packed parameter's full gradient, with zero Q-row contribution
from the memory branch. The temporary branch supplies its usual Q gradient.

At B64/T512/D2048 the saving is approximately **1.649 TFLOPs per RT block
invocation**. Multiply by actual RT layer/feedback calls and accumulation
microbatches. This does not change parameter counts, attention matrix work,
cache dimensions, MLP work, loss work or supervised-token counts. It is one
third of the permanent projection's matrix work, not one third of RT or model
runtime. Kernel selection, padding, memory traffic and dependency structure
still determine the measured speedup.

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
the estimator adds **`2BD(E+T²)` per selected RT invocation** as the explicit
`rt_attention_probability_recompute` component. No manual correction is needed. At
B64/T512/D2048 this is about0.103 TFLOPs/update for one invocation; multiply
by the actual number of RT layers/feedback calls and accumulation microbatches.
Dense, loss and parameter
counts are unchanged. Softmax/pointwise work and kernel tile padding remain
excluded, so this small matrix-work addition does not predict wall time.
The complete model is not claimed to use linear memory: ordinary/padded masks,
long-context eager forward fallback and other model allocations are separate.

The independent `reuse_rope=True` execution precomputes native FP32 cosine/sine
tables for the actual positions. Dynamic forwarding reuses them within a stack;
prepared forwarding owns them across layers, FBT passes and backward
recomputation. This removes repeated frequency/phase/trigonometric work without
removing the rotations themselves. These operations are pointwise and outside
the matrix ledger, so the matrix estimate is identical for `control` and `rope`.
One prepared table pair occupies `8BTHd` bytes, where `Hd` is head dimension:
32 MiB at B64/T512/Hd128. It is shared rather than replicated per layer/pass.
It adds no model parameters or persistent checkpoint buffers. Include its
allocation in measured setup/steady memory; do not infer a net memory benefit
from this isolated table size.

## Passes, fusion, predictor and vocabulary losses

With FBT enabled, K includes one ordinary bootstrap. For `R` selected RT layers,
there are `(K-1)R` RT block calls and `KL-(K-1)R` ordinary calls. With FBT disabled,
there is one stack with `R` RT calls. K1 FBT is ordinary regardless of the RT
selection. Beta zero omits fusion computation but does not omit extra stacks.
Alpha zero still executes the selected RT code and therefore retains its RT
arithmetic accounting; the ordinary mathematical limit is not a runtime bypass.

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
    backward_memory="recompute",
)
record = estimate.to_dict()
```

The resulting **matrix-arithmetic estimate** is 644.6–689.4 TFLOPs per update;
the materialized reference is 644.5–689.3. All eight architectural combinations,
using those same shape/mask assumptions, ordinary checkpointing and the
default materialized backward, are:

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

For a prepared `StaticFBTTraining` execution, obtain `LossWork` from
`plan.counts` and `plan.loss_layout.needed_source_indices.numel()`. The latter is
the predictor-source union, not the sum of latent and KL counts. Use the actual
model's `backward_memory`, selected RT layers, FBT mode and checkpointing setting.
Also pass the actual `kv_only_writes` setting; omitting it deliberately retains
the historical full-QKV estimate. The example and table above use that default.
An architecture formula should be accompanied by `parameter_inventory` for the
live model and optimizer. Declare the executed and inference name sets from the
configuration; gradient presence alone does not identify those sets. Inactive
registered wrappers can otherwise make resident parameters look like active
architecture parameters.

## Coverage and verification

Excluded arithmetic includes normalization, RoPE, nonlinearities, softmax,
elementwise CE/KL/SmoothL1, gather/scatter, embedding-gradient accumulation,
casts, clipping, optimizer/scheduler operations and communication. Allocation,
memory traffic and launch latency are not FLOPs. Frozen-parameter ownership and
zero pass weights require a new work audit. The estimator must be updated if
the RT implementation changes projection shapes or backward reconstruction.

The focused CPU tests compare parameter formulas with real tiny modules,
deduplicate tied aliases, check exposure/pass accounting and reject invalid
counts. Dispatch traces count actual eager `mm`/`bmm` operations for tiny RT
forward/backward at lengths 3, 5, 8, 33 and 65 with both permanent-writer options
and each backward implementation:
dense and attention totals agree exactly. Lengths above the 32-position workspace
chunk exercise the bounded query/key reconstruction paths while counting their
complete matrix reductions. Separate tests check K1, multiple selected layers,
K2/K3, alpha zero and accumulation scaling; recomputation changes no parameters.
KV-only accounting also checks the `12ND²` saving across selected layers,
feedback passes and accumulation while preserving parameter/objective counts.
Separate traces verify the canonical CE/KL checkpoint and detached-readout
counts and bracket ordinary checkpoint early stopping. This validates the
accounting against executed code without another GPU numerical campaign; it
does not validate any new GPU kernel or convert an estimate into hardware FLOPs.
