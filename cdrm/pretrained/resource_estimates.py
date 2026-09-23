"""Auditable matrix-arithmetic estimates for native OLMo training.

This is an accounting tool, not a hardware-performance model. It describes the
current dyadic RT/custom VJP and canonical checkpointed vocabulary losses,
including either materialized or recomputed backward probabilities. See
``docs/olmo-resource-accounting.md`` for derivations and excluded work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .nextlat import NextLatConfig
from .olmo import OLMoConfig
from .olmo_fbt import FBTMode


def _integer(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class LossWork:
    """Selected positions in one physical microbatch and one pass.

    Predictor positions are the UNION of latent pairs and KL-source pairs; their
    cardinality cannot generally be inferred by adding the two objective counts.
    Counts must already exclude zero-weight objectives and document boundaries.
    """
    ce_targets: int
    latent_pairs: int = 0
    kl_triples: int = 0
    predictor_positions: int = 0

    def __post_init__(self):
        for name, value in asdict(self).items():
            _integer(name, value)
        if not max(self.latent_pairs, self.kl_triples) <= self.predictor_positions <= self.latent_pairs + self.kl_triples:
            raise ValueError("predictor_positions must count the auxiliary-source union")

    @classmethod
    def full_document(cls, batch_size: int, sequence_length: int,
                      nextlat: NextLatConfig | None = None):
        _integer("batch_size", batch_size, minimum=1)
        _integer("sequence_length", sequence_length, minimum=1)
        pairs = batch_size * max(sequence_length - 1, 0)
        triples = batch_size * max(sequence_length - 2, 0)
        latent = pairs if nextlat is not None and nextlat.lambda_latent else 0
        kl = triples if nextlat is not None and nextlat.lambda_kl else 0
        return cls(pairs, latent, kl, max(latent, kl))


@dataclass(frozen=True)
class FlopComponent:
    name: str
    minimum: int
    maximum: int
    description: str


@dataclass(frozen=True)
class TrainingResourceEstimate:
    components: tuple[FlopComponent, ...]
    input_tokens_per_update: int
    pass_token_work_per_update: int
    objective_positions_per_update: dict[str, int]
    objective_positions_across_passes: dict[str, int]
    ordinary_block_calls_per_microbatch: int
    rt_block_calls_per_microbatch: int
    parameter_counts: dict[str, int]
    assumptions: tuple[str, ...]
    backward_memory: str = "materialized"

    @property
    def matrix_flops_minimum(self) -> int:
        return sum(c.minimum for c in self.components)

    @property
    def matrix_flops_maximum(self) -> int:
        return sum(c.maximum for c in self.components)

    def to_dict(self):
        return {**asdict(self), "matrix_flops_minimum": self.matrix_flops_minimum,
                "matrix_flops_maximum": self.matrix_flops_maximum,
                "schema": "olmo-training-resource-estimate-v1"}


def architecture_parameter_counts(config: OLMoConfig, *, fbt: bool = False,
                                  nextlat: NextLatConfig | None = None) -> dict[str, int]:
    """Unique architecture weights, counting tied readout/lookup exactly once.

    These are component totals, not claims about a particular module's resident,
    trainable or optimizer-owned weights. Use ``parameter_inventory`` for those.
    """
    if type(fbt) is not bool:
        raise ValueError("fbt must be boolean")
    if nextlat is not None and nextlat.model_dim != config.model_dim:
        raise ValueError("NextLat and backbone widths must agree")
    d, m = config.model_dim, config.mlp_intermediate_size
    backbone = config.vocab_size * d + config.num_layers * (4*d*d + 3*d*m)
    fusion = 2*d*d if fbt else 0
    predictor = 0
    if nextlat is not None:
        p = nextlat.hidden_dim
        predictor = 3*d*p + p*p + 2*d
        if nextlat.bias:
            predictor += 2*p + 3*d
    return {"backbone": backbone, "fusion": fusion, "nextlat_training_only": predictor,
            "training_architecture": backbone + fusion + predictor,
            "deployable_inference": backbone + fusion}


def parameter_inventory(model, *, optimizer=None, executed_names=None,
                        inference_names=None) -> dict[str, int | None]:
    """Observe unique tensor ownership without allocating or running the model.

    Optional names are caller-declared sets, not inferred from nonzero gradients.
    ``gradient_participating`` counts existing gradient tensors, including zeros.
    Resident bytes here cover parameters only, not buffers, casts or graph pools.
    """
    named = dict(model.named_parameters(remove_duplicate=False))
    unique = {id(p): p for p in named.values()}

    def selected(names):
        if names is None:
            return None
        names = set(names)
        if names - named.keys():
            raise ValueError(f"Unknown parameter names: {sorted(names - named.keys())}")
        return sum(p.numel() for p in {id(named[n]): named[n] for n in names}.values())

    owned = None
    if optimizer is not None:
        parameters = {id(p): p for group in optimizer.param_groups for p in group["params"]}
        if parameters.keys() - unique.keys():
            raise ValueError("Optimizer owns parameters outside this model")
        owned = sum(p.numel() for p in parameters.values())
    return {"registered_unique": sum(p.numel() for p in unique.values()),
            "resident_parameter_bytes": sum(p.numel()*p.element_size() for p in unique.values()),
            "trainable": sum(p.numel() for p in unique.values() if p.requires_grad),
            "gradient_participating": sum(p.numel() for p in unique.values() if p.grad is not None),
            "optimizer_owned": owned, "executed_declared": selected(executed_names),
            "deployable_inference_declared": selected(inference_names)}


def estimate_training_resources(config: OLMoConfig, *, batch_size: int,
                                sequence_length: int, mode: FBTMode,
                                nextlat: NextLatConfig | None = None,
                                loss_work: LossWork | None = None,
                                ordinary_checkpointing: bool = False,
                                accumulation_steps: int = 1,
                                backward_memory: str = "materialized") -> TrainingResourceEstimate:
    """Estimate dense training matrix arithmetic, with explicit recomputation.

    Scope: all backbone/active branch weights trainable, attached cross-pass
    gradients, positive pass coefficients, no cached prefix, one unpadded document
    per row. Accumulation assumes identical shapes/counts in every microbatch.
    Ordinary attention bounds span ideal causal work through full-square work,
    including optional Flash score reconstruction. They are not rigorous bounds
    on hardware FLOPs: implementation padding and instruction details are omitted.
    ``backward_memory`` selects RT probability storage/recomputation accounting;
    it does not change parameter counts or ordinary-layer execution.
    """
    for name, value in (("batch_size", batch_size), ("sequence_length", sequence_length),
                        ("accumulation_steps", accumulation_steps)):
        _integer(name, value, minimum=1)
    if type(ordinary_checkpointing) is not bool:
        raise ValueError("ordinary_checkpointing must be boolean")
    if backward_memory not in ("materialized", "recompute"):
        raise ValueError("backward_memory must be materialized or recompute")
    if not isinstance(mode, FBTMode):
        raise TypeError("mode must be FBTMode")
    if any(i >= config.num_layers for i in mode.rt_mode.selected_layers):
        raise ValueError("RT selected layer exceeds model depth")
    if sequence_length > config.max_context_length:
        raise ValueError("sequence_length exceeds the native context contract")
    parameters = architecture_parameter_counts(config,
        fbt=mode.enabled and mode.num_passes > 1 and mode.beta > 0, nextlat=nextlat)
    work = loss_work or LossWork.full_document(batch_size, sequence_length, nextlat)
    if not isinstance(work, LossWork):
        raise TypeError("loss_work must be LossWork")
    pairs, triples = batch_size*max(sequence_length-1, 0), batch_size*max(sequence_length-2, 0)
    if max(work.ce_targets, work.latent_pairs, work.predictor_positions) > pairs or work.kl_triples > triples:
        raise ValueError("Selected loss counts exceed available pairs/triples")
    if nextlat is None and work.predictor_positions:
        raise ValueError("Auxiliary selections require NextLat")
    if nextlat is not None and ((not nextlat.lambda_latent and work.latent_pairs) or
                               (not nextlat.lambda_kl and work.kl_triples)):
        raise ValueError("Zero-weight objectives must have zero active counts")
    if not (work.ce_targets or work.predictor_positions):
        raise ValueError("Training estimate requires a positively weighted objective")
    b, t, d, m = batch_size, sequence_length, config.model_dim, config.mlp_intermediate_size
    n = b*t
    passes = mode.num_passes if mode.enabled else 1
    rt_calls = len(mode.rt_mode.selected_layers) * (passes-1 if mode.enabled else 1)
    ordinary_calls = passes*config.num_layers - rt_calls
    fusion_calls = passes-1 if mode.enabled and mode.beta > 0 else 0
    components = []

    def add(name, lower, upper=None, description=""):
        components.append(FlopComponent(name, lower*accumulation_steps,
            (lower if upper is None else upper)*accumulation_steps, description))

    ordinary_forward = 2*n*(4*d*d + 3*d*m)*ordinary_calls
    add("ordinary_dense_forward", ordinary_forward)
    add("ordinary_dense_backward", 2*ordinary_forward,
        description="Input and weight VJPs for each ordinary dense projection.")
    if ordinary_checkpointing:
        add("ordinary_dense_checkpoint_recompute",
            ordinary_forward - 2*n*m*d*ordinary_calls, ordinary_forward,
            "Non-reentrant early stop can skip the final FF-down matmul; full replay is the upper estimate.")
    causal_pairs, square_pairs = t*(t+1)//2, t*t
    ordinary_attention = 4*b*d*ordinary_calls
    add("ordinary_attention_forward", ordinary_attention*causal_pairs,
        ordinary_attention*square_pairs, "QK and PV, causal useful to full-square arithmetic.")
    add("ordinary_attention_backward", 2*ordinary_attention*causal_pairs,
        5*ordinary_attention*square_pairs//2,
        "Four gradient matmuls; upper estimate also includes one Flash QK reconstruction.")
    if ordinary_checkpointing:
        add("ordinary_attention_checkpoint_recompute", ordinary_attention*causal_pairs,
            ordinary_attention*square_pairs, "Replayed attention forward; separate from kernel-internal backward reconstruction.")

    # Current RT calls full fused QKV even where the Q result is discarded.
    add("rt_dense_forward", rt_calls*(14*n*d*d + 6*n*d*m),
        description="Temporary QKV + permanent full QKV + output projection/SwiGLU.")
    add("rt_batched_projection_reconstruction", rt_calls*12*n*d*d)
    add("rt_local_writer_forward_and_input_vjp", rt_calls*12*n*d*d)
    add("rt_local_finish_forward_and_input_vjp", rt_calls*(4*n*d*d + 12*n*d*m))
    add("rt_finish_reconstruction", rt_calls*(2*n*d*d + 6*n*d*m))
    add("rt_batched_projection_vjps", rt_calls*24*n*d*d)
    add("rt_batched_finish_vjp", rt_calls*(2*n*d*d + 12*n*d*m),
        description="Final attended tensor is detached: WO has weight VJP only here.")
    history = t*(t-1)//2
    add("rt_attention_dyadic_forward", rt_calls*4*b*d*history)
    add("rt_attention_completed_reconstruction", rt_calls*4*b*d*t*t,
        description="Complete QK and PV arithmetic; recompute mode bounds workspace using query chunks.")
    add("rt_attention_reverse_history", rt_calls*6*b*d*history,
        description="dV, dP and dK over each strict-causal dyadic pair exactly once.")
    add("rt_attention_final_query_vjp", rt_calls*4*b*d*t*t,
        description="Full-square dP and dQ; temporary diagonal work is pointwise and excluded.")
    if backward_memory == "recompute":
        add("rt_attention_probability_recompute", rt_calls*2*b*d*(history+t*t),
            description="Additional QK in reverse historical tiles and final query VJP; no cached prefix.")
    add("fusion_dense_forward_backward", fusion_calls*12*b*max(t-1, 0)*d*d,
        description="Two D-by-D projections on every suffix position; attached input and weight VJPs.")
    readout_forward = 2*d*config.vocab_size
    add("ce_readout_forward_recompute_backward", passes*work.ce_targets*4*readout_forward,
        description="Forward + checkpoint replay + state VJP + tied-weight VJP.")
    if nextlat is not None:
        p = nextlat.hidden_dim
        add("nextlat_predictor_forward_backward", passes*work.predictor_positions*6*(3*d*p+p*p),
            description="Shared predictor once per union source position per pass; all three dense layers trainable.")
        add("kl_readout_forward_recompute_backward", passes*work.kl_triples*5*readout_forward,
            description="Teacher+student forward, both checkpoint replays, student-state VJP only; readout detached.")
    counts = {"ce": work.ce_targets, "latent": work.latent_pairs,
              "kl": work.kl_triples, "predictor": work.predictor_positions}
    return TrainingResourceEstimate(tuple(components), n*accumulation_steps,
        n*passes*accumulation_steps,
        {k: v*accumulation_steps for k, v in counts.items()},
        {k: v*passes*accumulation_steps for k, v in counts.items()},
        ordinary_calls, rt_calls, parameters,
        ("Multiply-add counts as two FLOPs; estimates count matrix arithmetic, not elapsed time.",
         "Current native full-QKV dyadic RT/custom VJP; no cached prefix or padding.",
         "All active weights trainable; attached pass gradients and positive pass coefficients.",
         "Loss masks change selected readout/predictor work, not dense backbone execution.",
         "Excluded: norms, RoPE, activations, softmax/CE/KL elementwise arithmetic, gather/scatter, casts, optimizer/clipping, communication, launch overhead and hardware padding.",
         "Ordinary attention and checkpoint early-stop ranges are accounting assumptions, not rigorous hardware bounds; CUDA graphs do not remove arithmetic."),
        backward_memory=backward_memory)
