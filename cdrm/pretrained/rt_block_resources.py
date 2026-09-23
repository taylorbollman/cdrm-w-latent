"""Matrix-operation and parameter ledger for isolated, all-recurrent stacks.

These counts describe the executed training schedules, including reconstruction
and local VJPs. They are not a hardware FLOP measurement or an end-to-end LM
estimate; embeddings, readout, objective and optimizer are outside this scope.
"""
from __future__ import annotations

from .olmo import OLMoConfig


def estimate_rt_block_resources(config: OLMoConfig, *, batch_size: int,
        sequence_length: int, backend: str, native_arm: str = "both",
        native_backward: str = "recompute") -> dict:
    """Count one forward/backward through ``config.num_layers`` distinct blocks.

    ``backend`` is ``native`` or ``author`` (the tiled implementations). Native
    control/rope use full permanent QKV; both uses permanent K/V only. Author
    tiling always writes only K/V and materializes backward probabilities.
    Changing MLP chunk count, helper compilation or RoPE reuse does not alter
    this matrix ledger. All weights and the stack input require gradients.
    """
    if not isinstance(config, OLMoConfig):
        raise TypeError("config must be an OLMoConfig")
    for name, value in (("batch_size", batch_size), ("sequence_length", sequence_length)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if backend not in ("native", "author"):
        raise ValueError("backend must be native or author tiled recurrence")
    if native_arm not in ("control", "rope", "both"):
        raise ValueError("native_arm must be control, rope or both")
    if native_backward not in ("materialized", "recompute"):
        raise ValueError("native_backward must be materialized or recompute")
    b, t, d, m, calls = (batch_size, sequence_length, config.model_dim,
                          config.mlp_intermediate_size, config.num_layers)
    n, history = b * t, t * (t - 1) // 2
    per_layer_parameters = 4 * d * d + 3 * d * m
    if backend == "author":
        forward_dense = 12 * n * d * d + 6 * n * d * m
        total_dense = 50 * n * d * d + 36 * n * d * m
        # T=1 reverse tiles use three pointwise multiply/reduce operations,
        # while the native backend includes those pairs in matrix products.
        total_attention = 10 * b * d * history + 6 * b * d * t * t - 6 * b * d * (t // 2)
        backward_policy, kv_only = "materialized", True
    else:
        kv_only = native_arm == "both"
        forward_dense = (12 if kv_only else 14) * n * d * d + 6 * n * d * m
        total_dense = (58 if kv_only else 70) * n * d * d + 36 * n * d * m
        total_attention = 10 * b * d * history + 8 * b * d * t * t
        if native_backward == "recompute":
            total_attention += 2 * b * d * (history + t * t)
        backward_policy = native_backward
    forward_attention = 4 * b * d * history
    breakdown = {}
    for phase, dense, attention in (
        ("forward", forward_dense, forward_attention),
        ("backward", total_dense - forward_dense, total_attention - forward_attention),
        ("total", total_dense, total_attention),
    ):
        breakdown[phase] = {"dense": calls * dense, "attention": calls * attention,
                            "total": calls * (dense + attention)}
    return {
        "schema": "olmo-rt-block-resources-v1", "backend": backend,
        "native_arm": native_arm if backend == "native" else None,
        "backward_memory": backward_policy, "kv_only_writes": kv_only,
        "batch_size": b, "sequence_length": t, "model_dim": d,
        "mlp_intermediate_size": m, "num_heads": config.num_heads, "head_dim": config.head_dim,
        "block_calls": calls, "input_tokens": n, "block_token_work": calls * n,
        "parameters_per_layer": per_layer_parameters,
        "unique_parameters": calls * per_layer_parameters,
        "forward_matrix_flops": breakdown["forward"]["total"],
        "backward_matrix_flops": breakdown["backward"]["total"],
        "total_matrix_flops": breakdown["total"]["total"],
        "matrix_breakdown": breakdown,
        "assumptions": [
            "Multiply-add counts as two FLOPs; counts describe matrix arithmetic, not measured hardware instructions or time.",
            "Every configured layer is a distinct RT block called once; no parameter sharing or extra passes.",
            "All block weights and the stack input require gradients; full-strength recurrence, no prefix cache, no padding.",
            "Native packed QKV and value/gate SwiGLU parameters; nonaffine LayerNorm contributes no parameters.",
            "Backward includes the implementation's reconstruction, local VJPs and batched parameter VJPs.",
            "Author reverse singleton tiles are pointwise and excluded; native singleton pairs remain in matrix products.",
            "Excluded: embeddings, final normalization, vocabulary readout, CE/NextLat/FBT/objectives, optimizer/clipping, communication.",
            "Excluded: normalization, RoPE, nonlinearities, softmax and all other pointwise arithmetic, casts, gather/scatter, launch overhead and hardware padding.",
            "Compiled fusion or chunking can change runtime and workspace without changing these logical matrix counts.",
        ],
    }
