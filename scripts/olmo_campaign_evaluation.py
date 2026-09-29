"""Local, jitter-free FP32 campaign evaluation with live-model preservation.

No distributed reduction, optimizer, cursor, checkpoint or CUDA graph is owned
here. The engine separately guards its graph-owned inputs/tables and boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, replace
import math
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from cdrm.pretrained.distributed_checkpoint import _local_rng, _restore_local_rng
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_lm_common import tree_digests

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'olmo-campaign-local-evaluation-v1'
TERMS = ('ce', 'latent', 'kl')
RUNTIME_FLAGS = ('attention_backend', 'ordinary_attention_backend', 'attention_precision',
    'tile_backend', 'backward_tile_backend', 'backward_memory', 'rt_implementation',
    'ordinary_pointwise_backend', 'ordinary_rope_backend', 'ordinary_activation_checkpointing',
    'ordinary_checkpoint_layers', 'cast_weights_once', 'reuse_rope', 'kv_only_writes',
    'author_precision', 'author_compiled_helpers', 'author_bwd_mlp_chunks', 'author_autocast_cache')
FP32_FLAGS = {'attention_backend': 'sdpa', 'ordinary_attention_backend': 'sdpa',
    'attention_precision': 'fp32', 'tile_backend': 'eager', 'backward_tile_backend': 'eager',
    'ordinary_pointwise_backend': 'eager', 'ordinary_rope_backend': 'native'}
_ACTIVE = ContextVar('campaign_evaluation_models', default=())


def source_hashes():
    names = ('scripts/olmo_campaign_evaluation.py', 'tests/test_campaign_evaluation.py',
        'docs/reports/olmo-campaign-evaluation/evaluator-contract.md',
        'cdrm/pretrained/fbt_training.py', 'cdrm/pretrained/nextlat.py',
        'cdrm/pretrained/campaign_recipe.py', 'cdrm/pretrained/distributed_checkpoint.py',
        'cdrm/pretrained/olmo_tiled.py', 'cdrm/pretrained/olmo_rope.py',
        'scripts/olmo_lm_common.py')
    return {name: sha256_file(ROOT/name) for name in names}


def tensor_metadata(value):
    return (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()),
            value.storage_offset(), str(value.dtype), str(value.device), value._version,
            value.requires_grad)


def _tensors(model):
    return {'parameter/'+name: tensor_metadata(value) for name, value in model.named_parameters()} | {
        'buffer/'+name: tensor_metadata(value) for name, value in model.named_buffers()}


def _generations(model):
    return {(name, attr): getattr(module, attr) for name, module in model.named_modules()
            for attr in ('_cache_generation', '_rt_cache_generation') if hasattr(module, attr)}


@contextmanager
def evaluation_runtime(model, *, device=None, generators=None):
    """Temporarily evaluate this unwrapped model without replacing live weights.

    Entry gradients may be absent or persistent zero buffers. The latter are
    never set to None on the normal path. Unexpected gradient replacement/value
    changes are repaired to the saved zero boundary and still fail integrity.
    Parameter/buffer corruption is detected by ownership/storage/version checks;
    no model-sized copies or unsafe weight rollback are attempted.
    """
    if not isinstance(model, FBTNextLatLM) or not isinstance(model.backbone.backbone, OLMoTiledRTForCausalLM):
        raise TypeError('Require the unwrapped native campaign model')
    parameters = tuple(model.named_parameters())
    actual_device = parameters[0][1].device
    device = actual_device if device is None else torch.device(device)
    if device != actual_device or device.type not in ('cpu', 'cuda') or any(
            p.dtype != torch.float32 or p.device != device for _, p in parameters):
        raise ValueError('Evaluation needs unchanged FP32 masters on the local device')
    if id(model) in _ACTIVE.get():
        raise ValueError('Nested evaluation of the same live model is unsupported')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        # Caller configures precision before graph construction. Do not change
        # process-global training settings inside a model-local evaluation.
        if device.type == 'cuda':
            raise ValueError('Common FP32 evaluation requires TF32 already disabled')
    gradients = [(name, p, p.grad, None if p.grad is None else tensor_metadata(p.grad)) for name,p in parameters]
    if any(g is not None and bool(g.any()) for _,_,g,_ in gradients):
        raise ValueError('Evaluation requires a completed boundary with zero or absent gradients')
    base = model.backbone.backbone
    if base.rt_implementation != 'native':
        raise ValueError('Only the selected native RT implementation is supported')
    flags = {name: getattr(base, name) for name in RUNTIME_FLAGS}
    modes = tuple((module, module.training) for module in model.modules())
    modules = tuple((name, id(module)) for name,module in model.named_modules())
    before, generations = _tensors(model), _generations(model)
    rng = _local_rng(device, generators)
    rng_pin = tree_digests(rng)
    cache = torch.is_autocast_cache_enabled()
    started = time.perf_counter()
    evidence = {'precision': 'fp32_math_eager', 'feedback_jitter': 0.,
        'rope_scope': 'Invocation-owned tables; prepared graph/table ownership is guarded by the caller',
        'integrity_scope': 'Tensor ownership/storage/version, gradient zeros/ownership, runtime/modes/RNG; full byte hashes are caller acceptance evidence',
        'restored': False, 'integrity_passed': False}
    token = _ACTIVE.set((*_ACTIVE.get(), id(model)))
    try:
        for name, value in FP32_FLAGS.items():
            setattr(base, name, value)
        model.eval()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device.type, enabled=False, cache_enabled=False):
            yield evidence
    finally:
        # Restoration must run on forward, scalar-validation and caller errors.
        for name, value in flags.items():
            setattr(base, name, value)
        for module, training in modes:
            module.training = training
        _restore_local_rng(rng, device, generators)
        _ACTIVE.reset(token)
        gradient_identity = all(p.grad is g and (g is None or tensor_metadata(g) == metadata)
                                for _,p,g,metadata in gradients)
        gradient_zero = all(p.grad is None or not bool(p.grad.any()) for _,p,_,_ in gradients)
        # Preserve graph storage if an erroneous callback changed .grad, but
        # expose that error instead of pretending evaluation was harmless.
        with torch.no_grad():
            for _, p, gradient, _ in gradients:
                if gradient is not None and bool(gradient.any()):
                    gradient.zero_()
                if p.grad is not gradient:
                    p.grad = gradient
        checks = {'tensor_metadata_unchanged': _tensors(model) == before,
            'module_ownership_unchanged': tuple((name,id(module)) for name,module in model.named_modules()) == modules,
            'cache_generations_unchanged': _generations(model) == generations,
            'gradient_identity_unchanged': gradient_identity, 'gradient_values_remained_zero': gradient_zero,
            'runtime_restored': all(getattr(base,name) == value for name,value in flags.items()),
            'modes_restored': all(module.training == training for module,training in modes),
            'rng_restored': tree_digests(_local_rng(device,generators)) == rng_pin,
            'autocast_cache_restored': torch.is_autocast_cache_enabled() == cache}
        evidence.update(checks=checks, restored=all(checks[key] for key in
            ('runtime_restored','modes_restored','rng_restored','autocast_cache_restored')),
            integrity_passed=all(checks.values()), elapsed_seconds=time.perf_counter()-started,
            timing_scope='Host wall time includes eager evaluation, scalar synchronization and integrity checks; no throughput claim')
        if not evidence['integrity_passed']:
            raise RuntimeError('Held-out evaluation changed live-state integrity: '+', '.join(k for k,v in checks.items() if not v))


def per_pass_sums(model, batch, recipe):
    """One canonical no-jitter forward; JSON numerators/counts, no reductions.

    Zero-target and entirely dummy batches are valid local contributions. Do
    not average these local means: the caller reduces sums/counts first.
    """
    if id(model) not in _ACTIVE.get() or torch.is_grad_enabled() or any(m.training for m in model.modules()):
        raise ValueError('Use evaluation_runtime with no gradients and every module in eval mode')
    if not isinstance(recipe, CampaignRecipe) or (model.enabled != recipe.nextlat
            or model.config.document_policy != recipe.document_policy
            or model.pass_loss_policy != 'campaign_v1' or model.gamma != 1.):
        raise ValueError('Evaluation model and declared campaign recipe differ')
    base = model.backbone.backbone
    if any(getattr(base,name) != value for name,value in FP32_FLAGS.items()):
        raise ValueError('Evaluation runtime no longer uses the declared FP32 path')
    device = next(model.parameters()).device
    if batch.input_ids.device != device or batch.input_ids.shape[1] != recipe.sequence_length:
        raise ValueError('Evaluation batch must already be on the local device at declared length')
    if torch.is_autocast_enabled(device.type):
        raise ValueError('Evaluation autocast must be disabled')
    before = {name: None if value is None else tensor_metadata(value) for name,value in vars(batch).items()}
    mode = replace(recipe.mode(), feedback_jitter=0.)
    losses = model.loss_sums(batch, backbone_kwargs={'mode':mode, 'feedback_noise':None,
                                                   'right_padded_causal':True})
    counts = model.counts(batch)
    weights = model.objective_weights()
    if len(losses.pass_losses) != mode.num_passes or losses.counts != counts or losses.weights != weights:
        raise ValueError('Canonical per-pass structure/counts/weights differ')
    def scalar_sums(values):
        if set(values) != set(TERMS):
            raise ValueError('Evaluation loss terms differ')
        result = {}
        for term in TERMS:
            value = values[term]
            if value.dtype != torch.float32 or value.numel() != 1 or value.requires_grad:
                raise ValueError('Evaluation loss must be a detached FP32 scalar')
            result[term] = float(value)
            if not math.isfinite(result[term]) or (not weights[term] and (result[term] != 0. or counts[term] != 0)):
                raise ValueError('Evaluation loss is nonfinite or a disabled auxiliary is nonzero')
        return result
    passes = []
    for index, loss in enumerate(losses.pass_losses):
        if loss.counts != counts or loss.weights != weights:
            raise ValueError('Per-pass target eligibility differs')
        passes.append({'index':index,'sums':scalar_sums(loss.sums),'counts':dict(counts)})
    if before != {name:None if value is None else tensor_metadata(value) for name,value in vars(batch).items()}:
        raise ValueError('Evaluation modified its input tensors')
    return {'schema':SCHEMA, 'passes':passes, 'aggregate_sums':scalar_sums(losses.sums),
        'counts':dict(counts),'weights':dict(weights),
        'term_pass_coefficients':{term:list(losses.term_pass_coefficients[term]) for term in TERMS},
        'input_tokens':int(batch.valid_mask.sum()), 'enabled':{term:bool(weights[term]) for term in TERMS},
        'mode':asdict(mode), 'policy':'common_fp32_no_jitter_v1'}
