"""Tiny pinned-source NUM reproduction; not a training or performance benchmark.

Run via the project Docker launcher, from /workspace/cdrm-w-latent:
python docs/reports/stage-a/upstream_baseline.py
The isolated source is created with git archive at the recorded upstream pin.
"""
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path('/workspace/cdrm-w-latent')
assert Path.cwd() == ROOT and Path('/.dockerenv').exists(), 'Use the project container launcher'
subprocess.run(['nvidia-smi', '-L'], check=True)
SOURCE = ROOT / '.runtime/upstream-a21b42d'
assert (SOURCE / 'olmo/model.py').exists(), 'Create pinned source with git archive first'
sys.path.insert(0, str(SOURCE))
import torch
import olmo.model
from olmo.model import OLMo
from olmo.config import ModelConfig, BlockType, ActivationType, InitFnType
from debug_utils import initialize_recurrent_from_sequential
assert Path(olmo.model.__file__).resolve() == SOURCE / 'olmo/model.py'
assert torch.cuda.is_available(), 'No CPU fallback'
torch.manual_seed(20260906)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
ATOL, RTOL = 1e-5, 1e-4
DTYPE = torch.float64 if '--float64' in sys.argv else torch.float32
REPORT = ROOT / ('docs/reports/stage-a/upstream_baseline_fp64.json' if DTYPE == torch.float64 else 'docs/reports/stage-a/upstream_baseline.json')
report = {
    'evidence_category': 'NUM', 'status': 'running',
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'upstream_revision': 'a21b42d2bc292edb86ed1b62cee4bcab809a9d21',
    'source_path': str(SOURCE), 'model_import': olmo.model.__file__,
    'source_model_sha256': hashlib.sha256((SOURCE / 'olmo/model.py').read_bytes()).hexdigest(),
    'python': platform.python_version(), 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
    'settings': {'seed': 20260906, 'batch': 2, 'sequence_length': 8, 'width': 32,
                 'mlp_width': 128, 'heads': 4, 'layers': 2, 'vocab': 64,
                 'dtype': str(DTYPE), 'matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
                 'float32_matmul_precision': 'highest', 'cuda_graphs': False, 'model_compile': False,
                 'upstream_compiled_helpers': 'retained unchanged',
                 'dropout': 0, 'alibi': True, 'rope': False, 'flash_attention_option': False,
                 'qk_layer_norm': True, 'bwd_mlp_chunks': 1,
                 'atol': ATOL, 'rtol': RTOL},
    'scope': 'Tiny whole-model naive/tiled recurrence at rho=1; unchanged pinned code. '
             'Mean-logit surrogate and random output cotangent. Not a CE, training, or timing benchmark. '
             'Default learned norms only; upstream conversion is intentionally unmodified.',
    'cases': [],
}
def save():
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
def stats(a, b):
    delta = (a.detach() - b.detach()).double()
    ref = a.detach().double()
    norm2 = ref.square().sum().item()
    diff2 = delta.square().sum().item()
    return {'max_abs': delta.abs().max().item(), 'mean_abs': delta.abs().mean().item(),
            'relative_squared_error': diff2 / norm2 if norm2 else (0.0 if diff2 == 0 else None),
            'allclose': bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)),
            'num_elements': a.numel(),
            'num_exceeding_tolerance': int((delta.abs() > (ATOL + RTOL * b.detach().double().abs())).sum().item()),
            'worst_tolerance_ratio': (delta.abs() / (ATOL + RTOL * b.detach().double().abs())).max().item(),
            'violations': [
                {'flat_index': int(i), 'reference': ref.flatten()[i].item(),
                 'tiled': b.detach().flatten()[i].item(), 'abs_diff': delta.abs().flatten()[i].item(),
                 'allowed': (ATOL + RTOL * b.detach().double().abs()).flatten()[i].item()}
                for i in (delta.abs() > (ATOL + RTOL * b.detach().double().abs())).flatten().nonzero().flatten()[:20]
            ]}

save()
try:
    for norm_after in (True, False):
        config = ModelConfig(d_model=32, n_heads=4, n_kv_heads=4, n_layers=2,
            mlp_hidden_size=128, max_sequence_length=8, vocab_size=64, embedding_size=64,
            block_type=BlockType.sequential, init_device='cuda', precision=DTYPE,
            norm_after=norm_after, rope=False, alibi=True, flash_attention=False,
            attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
            activation_type=ActivationType.gelu, init_fn=InitFnType.mitchell,
            attention_layer_norm=True, attention_layer_norm_with_affine=True,
            layer_norm_with_affine=True, bias_for_layer_norm=False, include_bias=False,
            weight_tying=False, bwd_mlp_chunks=1)
        torch.manual_seed(20260906)
        seq = OLMo(config).to(device='cuda', dtype=DTYPE).train()
        naive_cfg, tiled_cfg = copy.deepcopy(config), copy.deepcopy(config)
        naive_cfg.block_type = BlockType.recurrent_autograd
        tiled_cfg.block_type = BlockType.recurrent
        naive, tiled = OLMo(naive_cfg).to(device='cuda', dtype=DTYPE).train(), OLMo(tiled_cfg).to(device='cuda', dtype=DTYPE).train()
        initialize_recurrent_from_sequential(seq, naive)
        initialize_recurrent_from_sequential(seq, tiled)
        del seq
        ids = torch.randint(0, 64, (2, 8), device='cuda')
        for objective in ('upstream_mean_logits_surrogate', 'random_output_cotangent'):
            naive.zero_grad(set_to_none=True)
            tiled.zero_grad(set_to_none=True)
            a, b = naive(ids).logits, tiled(ids).logits
            if objective == 'upstream_mean_logits_surrogate':
                la, lb = a.mean(), b.mean()
            else:
                cotangent = torch.randn_like(a)
                la, lb = (a * cotangent).sum(), (b * cotangent).sum()
            with torch.autograd.set_multithreading_enabled(False):
                la.backward()
                lb.backward()
            params_a, params_b = dict(naive.named_parameters()), dict(tiled.named_parameters())
            gradients, missing = {}, []
            for name in sorted(set(params_a) | set(params_b)):
                pa, pb = params_a.get(name), params_b.get(name)
                if pa is None or pb is None or pa.grad is None or pb.grad is None:
                    missing.append(name)
                else:
                    gradients[name] = stats(pa.grad, pb.grad)
            case = {'norm_after': norm_after, 'objective': objective,
                    'logits': stats(a, b), 'objective_value': {'naive': la.item(), 'tiled': lb.item(),
                    'abs_diff': abs(la.item() - lb.item())}, 'parameters_compared': len(gradients),
                    'missing_gradients': missing, 'gradients': gradients,
                    'max_gradient_abs': max(v['max_abs'] for v in gradients.values()),
                    'gradient_failures': [k for k, v in gradients.items() if not v['allclose']]}
            case['passed'] = case['logits']['allclose'] and not missing and not case['gradient_failures']
            report['cases'].append(case)
            print(json.dumps({k: v for k, v in case.items() if k != 'gradients'}), flush=True)
            save()
        del naive, tiled
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    report['status'] = 'passed' if all(c['passed'] for c in report['cases']) else 'failed'
except Exception:
    report['status'] = 'error'
    report['traceback'] = traceback.format_exc()
    print(report['traceback'], flush=True)
finally:
    save()
print('Report:', REPORT, 'status:', report['status'], flush=True)
sys.exit(0 if report['status'] == 'passed' else 1)
