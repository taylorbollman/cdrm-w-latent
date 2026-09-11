# Activation checkpointing in the tiled Recurrent Transformer

The model already uses the activation recomputation described in the paper.
It is built into each tiled layer's custom backward and remains active with
CUDA graphs. The separate `activation_checkpointing=None` setting means that
we have not added another checkpoint wrapper around complete layers.

The paper describes retaining each layer's input and output, rebuilding
persistent keys and values from the saved output, and reconstructing attention
without repeating the sequential forward recurrence. Adjacent layers share
boundary activations: one layer's output is the next layer's input.
See [Section 6.1, Activation Checkpointing](https://arxiv.org/html/2604.21215v1#S6.SS1).

The current implementation follows that arrangement:

| Operation | Code |
| --- | --- |
| Save layer input and output through `save_for_backward` | [model.py:1435](../../../recurrent-transformer/olmo/model.py#L1435) |
| Reconstruct queries and temporary KV from the input | [model.py:1472](../../../recurrent-transformer/olmo/model.py#L1472) |
| Reconstruct persistent KV directly from saved outputs | [model.py:1483](../../../recurrent-transformer/olmo/model.py#L1483) |
| Recompute attention probabilities and outputs in batched operations | [model.py:1511](../../../recurrent-transformer/olmo/model.py#L1511) |
| Reconstruct MLP intermediates for the temporal backward pass in chunks | [model.py:1529](../../../recurrent-transformer/olmo/model.py#L1529) |
| Recompute the MLP in a final batched pass for parameter/input gradients | [model.py:1635](../../../recurrent-transformer/olmo/model.py#L1635) |

The context also retains a reference to the attention bias. Persistent KV
reconstruction currently uses a loop over positions, but it does not need to
repeat the recurrence that produced those outputs.

We retain the released 150M setting `bwd_mlp_chunks=4`; at sequence length 512,
the temporal MLP reconstruction covers 128 positions per chunk. The released
[model overlay](../../../recurrent-transformer/configs/kempner/models/150m.yaml)
sets four chunks, while the
[training sweep](../../../recurrent-transformer/sweep_recurrent_150m_512.yaml)
leaves whole-layer checkpointing commented out.

Four chunks do not divide all backward memory by four. Attention probabilities
are reconstructed as a full `[batch, heads, length, length]` tensor, and the
final MLP parameter-gradient pass covers the full sequence. Under our FP32
recurrent-state policy, that attention tensor alone occupies 8 GiB at
batch 512, 16 heads and length 512. This is tensor-size arithmetic from
[recompute_alphas](../../../recurrent-transformer/olmo/model.py#L1287), not a
claim that the complete backward needs only 8 GiB.

An additional checkpoint around every individual layer would retain much of
the same boundary storage and could require rerunning the expensive temporal
forward. I therefore expect limited additional savings for this placement;
that expectation has not been benchmarked. Checkpointing groups of layers
could remove more boundaries, with a different recomputation tradeoff.

**No additional whole-layer checkpointing experiment was run or cleared.**
The [model setter](../../../recurrent-transformer/olmo/model.py#L2041) and
[block-group setter](../../../recurrent-transformer/olmo/model.py#L1920)
reject external checkpointing with tiled recurrence. Existing tests explicitly
expect that rejection. These guards remain intact. Nested checkpointing would
need its own validation because this custom backward performs internal
autograd calls and directly accumulates parameter gradients.

CUDA capture preserves the existing recomputation procedure and records its
GPU operations for replay. The capture validation compares captured and
uncaptured execution at the same precision; it is not an ablation of
checkpointing or a new BF16-versus-FP32 accuracy claim. Capacity and throughput
results should retain both settings explicitly: internal tiled recomputation
enabled, additional whole-layer checkpointing disabled.
