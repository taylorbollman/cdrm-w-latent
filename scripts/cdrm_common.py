"""Shared, explicit FP32 CDRM NUM/OPS/pilot utilities."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from stage_a_common import (require_cuda_container, seed_all, unique_parameters,
                           configure_compiled_helpers, provenance, rng_state, restore_rng)
from stage_b_train import atomic_json, append_jsonl, file_digest, json_digest, compiler_audit
from r3_mixed_operational import cpu_tree, state_digest, effective_precision, execution_contract


def setup(seed=0):
    hardware = require_cuda_container()
    torch.set_num_threads(1)
    seed_all(seed, deterministic=True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    configure_compiled_helpers(True)
    if torch.is_autocast_enabled('cuda'):
        raise RuntimeError('CDRM reference requires autocast off')
    return {'provenance': provenance(hardware), 'execution_contract': execution_contract()}


def sources():
    paths = set(Path('recurrent-transformer/olmo').rglob('*.py'))
    paths.update(Path(p) for p in ['scripts/cdrm_common.py', 'scripts/cdrm_train.py',
        'scripts/stage_a_common.py', 'scripts/stage_b_train.py', 'scripts/r3_mixed_operational.py',
        'scripts/r3_validation_metrics.py', 'cdrm/mad_data.py',
        'vendors/mad-lab/mad/data/instances.py', 'configs/cdrm/base_d256.json',
        'configs/cdrm/base_d128.json', 'configs/cdrm/base_d128_6.json'])
    return {str(p): file_digest(p) for p in sorted(paths) if p.is_file()}


def config(preset, topology, vocab, length, *, diagnostics=False, gates=None):
    from olmo.config import ModelConfig
    raw = json.loads(Path(f'configs/cdrm/base_{preset}.json').read_text())
    raw.update(vocab_size=vocab, embedding_size=vocab, max_sequence_length=max(512, length),
               recurrent_layers=[raw.get('cdrm_early_layer', 3)] if topology == 'r3' else [],
               recurrent_backend='tiled' if topology == 'r3' else 'naive',
               recurrent_precision_policy='legacy', reference_eager=topology != 'r3',
               precision=None, init_device='cpu', cdrm_enabled=topology in ['cdrm', 'same_depth'],
               cdrm_source='same_depth' if topology == 'same_depth' else 'deep',
               cdrm_output_states=diagnostics)
    raw.update(gates or {})
    return ModelConfig(**raw)


def build_pair_member(preset, topology, vocab, length, seed, *, gates=None):
    """Fresh common SEQ backbone, then explicit weights-only topology construction."""
    from olmo.model import OLMo
    from olmo.checkpoint_conversion import convert_model
    seed_all(seed, deterministic=True)
    base = OLMo(config(preset, 'seq', vocab, length))
    base_digest = state_digest(base.state_dict())
    if topology == 'seq':
        model, conversion = base, {'policy': 'fresh_standard_backbone'}
    else:
        # Isolate adapter initialization from the common backbone RNG history.
        seed_all(seed + 100003, deterministic=True)
        model = OLMo(config(preset, topology, vocab, length, gates=gates))
        if topology == 'r3':
            conversion = convert_model(base, model).to_dict()
        else:
            target = model.state_dict()
            common = base.state_dict()
            if not set(common).issubset(target):
                raise AssertionError('CDRM changed common backbone checkpoint names')
            for name, value in common.items():
                if target[name].shape != value.shape:
                    raise AssertionError(name)
                target[name] = value
            extras = sorted(set(target) - set(common))
            if extras != ['cdrm.bridge_adapter.weight', 'cdrm.deep_adapter.weight']:
                raise AssertionError(f'Unexpected CDRM checkpoint owners: {extras}')
            model.load_state_dict(target, strict=True)
            for name, value in common.items():
                if not torch.equal(value, model.state_dict()[name]):
                    raise AssertionError(f'Backbone copy changed {name}')
            conversion = {'policy': 'strict_common_backbone_copy_with_new_nonzero_adapters',
                          'copied': list(common), 'new': extras}
        del base
    model = model.to(device='cuda', dtype=torch.float32)
    # OLMo's inherited constructor enables flash globally. Establish the actual
    # reference policy after every constructor, then scope forwards explicitly.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    owned = unique_parameters(model)
    seed_all(seed + 200003, deterministic=True)
    return model, {'backbone_initialization_sha256': base_digest, 'conversion': conversion,
                   'parameter_count': sum(p.numel() for p in owned),
                   'parameter_bytes': sum(p.numel() * p.element_size() for p in owned),
                   'model_config': dataclasses.asdict(model.config)}


def optimizer_for(model, lr=5e-4):
    params = unique_parameters(model)
    opt = torch.optim.AdamW(params, lr=lr, betas=(.9,.98), eps=1e-8,
                           weight_decay=0., foreach=False, fused=False)
    flat = [p for g in opt.param_groups for p in g['params']]
    if len(flat) != len({id(p) for p in flat}) or {id(p) for p in flat} != {id(p) for p in params}:
        raise AssertionError('Optimizer parameter ownership mismatch')
    return opt


def loss_sum(logits, labels):
    if logits.dtype != torch.float32:
        raise AssertionError('Expected FP32 logits')
    mask = labels != -100
    count = int(mask.sum().item())
    if count == 0 or not mask.any(-1).all().item():
        raise ValueError('Every independent example needs scored targets')
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction='sum'), count


def metric_counts(logits, labels):
    summed, count = loss_sum(logits, labels)
    valid = labels != -100
    correct = (logits.argmax(-1) == labels) & valid
    return {'ce_sum': float(summed.detach().item()), 'targets': count,
            'correct': int(correct.sum().item()),
            'exact': int((correct | ~valid).all(-1).sum().item()), 'examples': labels.shape[0]}


def finish_metrics(counts):
    return {**counts, 'ce': counts['ce_sum'] / counts['targets'],
            'token_accuracy': counts['correct'] / counts['targets'],
            'sequence_exact_match': counts['exact'] / counts['examples']}


def rms(x):
    return float(x.detach().float().square().mean().sqrt().item())


def state_summaries(states):
    if states is None:
        return {}
    out = {name + '_rms': rms(states[name]) for name in
           ['p3', 'p8', 'candidate', 'hat_m', 'm', 'deep_correction', 'bridge_correction']}
    out['candidate_correction_over_p3_rms'] = out['deep_correction_rms'] / max(out['p3_rms'], 1e-12)
    out['bridge_correction_over_p8_rms'] = out['bridge_correction_rms'] / max(out['p8_rms'], 1e-12)
    return out


def update(model, opt, ids, labels, answer_labels=None, *, monitor=False, clip=1.):
    model.train()
    before = [p.detach().clone() for p in model.parameters()] if monitor else None
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with sdpa_kernel(SDPBackend.MATH):
        output = model(ids, output_cdrm_states=monitor and model.config.cdrm_enabled)
        summed, count = loss_sum(output.logits, labels)
    loss = summed / count
    loss.backward()
    params = list(model.named_parameters())
    if any(p.dtype != torch.float32 or p.grad is None or p.grad.dtype != torch.float32 for _,p in params):
        raise AssertionError('Active model must have every intended FP32 parameter gradient')
    norms = {}
    if monitor:
        for name, p in params:
            if name.startswith('cdrm.') or name.startswith(f'transformer.blocks.{model.config.cdrm_early_layer}.'):
                norms[name] = {'l2': float(p.grad.detach().norm().item()),
                               'nonzero': bool(torch.count_nonzero(p.grad).item())}
    total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True).item())
    opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    row = {'native_loss': float(loss.item()), 'native_targets': count, 'seconds': elapsed,
           'gradient_norm': total_norm, 'clipped': total_norm > clip,
           'clip_coefficient': min(1., clip/(total_norm+1e-6)), 'input_tokens': ids.numel(),
           'learning_rate': opt.param_groups[0]['lr']}
    if not math.isfinite(row['native_loss']):
        raise FloatingPointError('Nonfinite loss')
    if monitor:
        row.update(precision=effective_precision(model,opt,require_gradients=True),
                   state_summaries=state_summaries(output.cdrm_states), parameter_gradient_signals=norms,
                   update_l2=float(torch.stack([(p.detach().double()-old.double()).square().sum()
                       for p,old in zip(model.parameters(),before)]).sum().sqrt().item()),
                   answer=finish_metrics(metric_counts(output.logits.detach(), answer_labels if answer_labels is not None else labels)))
    return row


@torch.no_grad()
def evaluate(model, dataset, batch_size, *, limit=None):
    model.eval()
    totals = {kind:{'ce_sum':0.,'targets':0,'correct':0,'exact':0,'examples':0} for kind in ['native','answer']}
    n = len(dataset.input_ids) if limit is None else min(limit,len(dataset.input_ids))
    started = time.perf_counter()
    for begin in range(0,n,batch_size):
        ids = torch.as_tensor(dataset.input_ids[begin:min(begin+batch_size,n)],device='cuda')
        with sdpa_kernel(SDPBackend.MATH):
            logits = model(ids).logits
        for name, values in [('native',dataset.labels),('answer',dataset.answer_labels)]:
            labels = torch.as_tensor(values[begin:min(begin+batch_size,n)],device='cuda')
            for key,value in metric_counts(logits,labels).items(): totals[name][key]+=value
    torch.cuda.synchronize()
    return {**{k:finish_metrics(v) for k,v in totals.items()},'seconds':time.perf_counter()-started}


def save_checkpoint(path, payload):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.partial')
    torch.save(cpu_tree(payload),tmp)
    tmp.rename(path)
    return {'path':str(path),'sha256':file_digest(path),'bytes':path.stat().st_size}
