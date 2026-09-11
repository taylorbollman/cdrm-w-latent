#!/usr/bin/env python3
"""Bounded all-recurrent C4 training with fixed-shape CUDA capture and resume."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_batch_profile import memory, precision_check
from rt_cuda_graph import CapturedRTBackward
from rt_cuda_graph_validate import digest_tensors, validation_config
from stage_a_common import configure_compiled_helpers, require_cuda_container, restore_rng, rng_state, seed_all
from stage_b_train import atomic_json, compiler_audit, file_digest

SCHEMA = 'rt-precision-state-v1'
BATCH = LENGTH = 512
SEED = 20260910
SCHEDULE = {'name': 'CosWithWarmup', 'peak_lr': 1e-3, 'alpha_0': .1,
            'alpha_f': .1, 'warmup_steps': 5000, 'horizon': 12500,
            'indexing': 'update numbers start at 1'}
OPTIMIZER = {'lr': 1e-3, 'betas': (.9, .95), 'eps': 1e-8, 'weight_decay': 0.,
             'foreach': False, 'fused': False}


def learning_rate(update):
    """Released CosWithWarmup, including its nonzero initial warmup LR."""
    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise ValueError('Learning-rate update must be a positive integer')
    peak, warmup, horizon = SCHEDULE['peak_lr'], SCHEDULE['warmup_steps'], SCHEDULE['horizon']
    if update < warmup:
        return peak * (SCHEDULE['alpha_0'] + (1 - SCHEDULE['alpha_0']) * update / warmup)
    if update >= horizon:
        return peak * SCHEDULE['alpha_f']
    floor = peak * SCHEDULE['alpha_f']
    return floor + (peak - floor) * (1 + math.cos(math.pi * (update - warmup) / (horizon - warmup))) / 2


def data_window(completed_updates, rows, batch=BATCH):
    """No shuffle, wrapping, repetition or skipped rows at a resume boundary."""
    if (isinstance(completed_updates, bool) or not isinstance(completed_updates, int)
            or completed_updates < 0 or batch < 1):
        raise ValueError('Invalid completed-update count or batch')
    begin = completed_updates * batch
    end = begin + batch
    if end > rows:
        raise ValueError('Frozen training corpus is exhausted; repetition is not allowed')
    return begin, end


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def save_checkpoint(path, packet):
    if path.exists():
        raise FileExistsError(f'Refusing to replace checkpoint: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('xb') as handle:
        torch.save(packet, handle)
    temporary.rename(path)
    return {'path': str(path), 'sha256': file_digest(path), 'bytes': path.stat().st_size,
            'completed_updates': packet['completed_updates'], 'next_data_row': packet['next_data_row']}


def validate_resume_metadata(packet, contract, endpoint):
    if packet.get('schema') != SCHEMA or packet.get('resume_contract') != contract:
        raise ValueError('Checkpoint schema or source/data/model/runtime resume contract differs')
    completed = packet.get('completed_updates')
    if type(completed) is not int or completed not in (0, 100, 500) or not completed < endpoint:
        raise ValueError('Resume requires an earlier retained endpoint (0 or 100)')
    if packet.get('next_data_row') != completed * BATCH:
        raise ValueError('Checkpoint data offset disagrees with completed updates')
    if (packet.get('model_config') != contract['model_config']
            or packet.get('optimizer_parameter_names') != contract['optimizer_parameter_names']
            or packet.get('seed') != contract['seed']
            or packet.get('data_manifest_sha256') != contract['data']['manifest_sha256']
            or packet.get('source_sha256') != contract['source_sha256']
            or packet.get('runtime') != contract['runtime']
            or packet.get('protocol_sha256') != contract['protocol_sha256']
            or packet.get('heldout_manifest_sha256') != contract['data']['heldout_exclusion_manifest_sha256']
            or packet.get('policy') != contract['model_config']['recurrent_precision_policy']):
        raise ValueError('Checkpoint top-level metadata disagrees with its resume contract')
    if not {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= set(packet.get('rng', {})):
        raise ValueError('Checkpoint lacks complete RNG state')
    if not isinstance(packet.get('initial_state_sha256'), str) or len(packet['initial_state_sha256']) != 64:
        raise ValueError('Checkpoint lacks its original initialization digest')
    data_window(completed, contract['data']['shape'][0])
    return completed


def validate_model_checkpoint(packet, state):
    weights = packet['model']
    if set(weights) != set(state):
        raise ValueError('Checkpoint model state names differ')
    for name, reference in state.items():
        value = weights[name]
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid FP32 checkpoint model tensor: {name}')


def validate_optimizer_checkpoint(packet, named_parameters):
    """Check Adam identity and every moment before loading a retained state."""
    completed = packet['completed_updates']
    state = packet['optimizer']
    groups = state.get('param_groups', [])
    names = packet['optimizer_parameter_names']
    if len(groups) != 1 or names != list(named_parameters):
        raise ValueError('Require one Adam group in canonical named-parameter order')
    group = groups[0]
    indices = group.get('params', [])
    if len(indices) != len(names) or len(set(indices)) != len(indices):
        raise ValueError('Optimizer parameter coverage is invalid')
    for key, expected in OPTIMIZER.items():
        if key == 'lr':
            expected = learning_rate(completed) if completed else OPTIMIZER['lr']
        actual = group.get(key)
        if key == 'betas':
            actual = tuple(actual) if actual is not None else None
        if actual != expected:
            raise ValueError(f'Optimizer option differs: {key}')
    for key in ('amsgrad', 'maximize', 'capturable', 'differentiable'):
        if group.get(key, False):
            raise ValueError(f'Unsupported optimizer option: {key}')
    if not completed:
        if state['state']:
            raise ValueError('Initialization checkpoint must have empty Adam state')
        return
    if set(state['state']) != set(indices):
        raise ValueError('Every trained parameter requires retained Adam moments')
    for name, index in zip(names, indices):
        values = state['state'][index]
        if set(values) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError(f'Unexpected Adam state fields: {name}')
        step = values['step']
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() != completed:
            raise ValueError(f'Adam step disagrees with completed updates: {name}')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = values[key]
            if (not isinstance(value, torch.Tensor) or value.shape != named_parameters[name].shape
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError(f'Invalid FP32 Adam moment: {name}/{key}')
            if key == 'exp_avg_sq' and torch.any(value < 0):
                raise ValueError(f'Negative Adam second moment: {name}')


def load_training_data(manifest_path, endpoint):
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'rt-precision-c4-data-v1'
            or manifest.get('status') != 'complete' or manifest.get('mode') != 'train'):
        raise ValueError('Require a complete frozen RT C4 training manifest')
    role = manifest['roles']['train']
    packing, tokenizer = manifest['packing'], manifest['tokenizer']
    if (packing['sequence_length'] != LENGTH or not packing['append_eos_per_document']
            or not packing['cross_document_attention'] or packing['extra_attention_masks']
            or tokenizer['vocab_size'] != 32100 or tokenizer['padded_model_vocab'] != 32128
            or tokenizer['eos_id'] != 1 or tokenizer['pad_id'] != 0 or not manifest.get('exclusion')):
        raise ValueError('Data packing/tokenizer/held-out-exclusion contract differs')
    root = manifest_path.parent.resolve()
    ids_path, boundaries_path = root / role['ids_path'], root / role['boundaries_path']
    for path, expected in ((ids_path, role['ids_sha256']), (boundaries_path, role['boundaries_sha256'])):
        if path.resolve().parent != root or file_digest(path) != expected:
            raise ValueError('Training IDs or document-boundary file differs from its manifest')
    ids = np.load(ids_path, mmap_mode='r', allow_pickle=False)
    if (ids.dtype != np.uint16 or role['dtype'] != 'uint16' or ids.ndim != 2
            or list(ids.shape) != role['shape'] or ids.shape[1] != LENGTH
            or ids.shape[0] < endpoint * BATCH):
        raise ValueError('Require enough frozen uint16 [rows,512] IDs for the absolute endpoint')
    for begin in range(0, len(ids), 4096):
        if np.max(ids[begin:begin + 4096]) >= tokenizer['vocab_size']:
            raise ValueError('Training corpus contains IDs outside the valid tokenizer vocabulary')
    data = {'manifest_sha256': file_digest(manifest_path), 'ids_sha256': role['ids_sha256'],
            'boundaries_sha256': role['boundaries_sha256'], 'shape': list(ids.shape), 'dtype': str(ids.dtype),
            'batch': BATCH, 'order': 'consecutive source rows; no shuffle, wrapping or repetition',
            'heldout_exclusion_manifest_sha256': manifest['exclusion']['manifest_sha256']}
    return ids, data, manifest


def validate_protocol(protocol, seed):
    if protocol.get('schema') != 'rt-precision-alignment-protocol-v1':
        raise ValueError('Unexpected precision-alignment protocol schema')
    expected = {'layers': 12, 'all_recurrent': True, 'd_model': 1024, 'heads': 16,
                'ffn': 4096, 'length': LENGTH, 'physical_batch': BATCH,
                'backbone_parameters': 151045120, 'total_parameters': 216843264}
    if protocol.get('architecture') != expected or seed not in protocol['training']['seeds']:
        raise ValueError('Protocol architecture or training seed differs')
    schedule = protocol['schedule']
    for name, value in {'warmup_updates': 5000, 'horizon_updates': 12500, 'alpha_0': .1,
                        'alpha_f': .1, 'update_index_first': 1}.items():
        if schedule.get(name) != value:
            raise ValueError(f'Protocol schedule differs: {name}')
    optimizer = protocol['optimizer']
    for name, value in {'name': 'AdamW', 'peak_lr': 1e-3, 'betas': [.9, .95], 'eps': 1e-8,
                        'weight_decay': 0., 'clip_norm': 1., 'foreach': False, 'fused': False}.items():
        if optimizer.get(name) != value:
            raise ValueError(f'Protocol optimizer differs: {name}')
    training = protocol['training']
    for name, value in {'screen_updates': 100, 'continuation_updates': 500, 'cuda_graphs': True,
                        'head_microbatch': 2, 'internal_recomputation': True,
                        'outer_checkpoint': False, 'same_token_order_between_arms': True,
                        'loss': 'sum shifted CE / (B*512)'}.items():
        if training.get(name) != value:
            raise ValueError(f'Protocol training contract differs: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=('bf16_fp32_state', 'legacy'), required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--updates', type=int, choices=(100, 500), required=True,
                        help='Absolute completed-update endpoint, preserving the 12500-step schedule')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh training output directory, including for resume')
    if not args.wandb_project:
        parser.error('Online W&B is required')
    args.output_dir.mkdir(parents=True)
    report = {'schema': 'rt-precision-training-v1', 'status': 'running', 'policy': args.policy,
              'precision': 'bf16', 'seed': args.seed, 'shape': [BATCH, LENGTH],
              'endpoint': args.updates, 'head_chunk_size': 2, 'cuda_graphs': True,
              'backbone_accumulation_steps': 1, 'optimizer_captured': False,
              'schedule': SCHEDULE, 'optimizer': {'name': 'AdamW', **OPTIMIZER, 'clip_norm': 1.},
              'loss': 'Native shifted CE sum divided by B*512; 511 supervised targets per sequence',
              'qualification': 'A 100/500-step prefix of 5000-step warmup does not clear peak-LR behavior or convergence',
              'timing_scope': 'Update time includes data copy, replay, clip, Adam and full finite/FP32 checks; excludes W&B and checkpoint IO',
              'updates': [], 'checkpoints': []}
    tracker = None
    started = time.monotonic()
    try:
        report['hardware'] = require_cuda_container()
        torch.set_num_threads(1)
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.utils.counters.clear()
        cache = os.environ.get('TORCHINDUCTOR_CACHE_DIR')
        if not cache or not Path(cache).is_absolute():
            raise ValueError('Set an explicit absolute TORCHINDUCTOR_CACHE_DIR for retained resume identity')
        source_paths = [Path(__file__), Path('scripts/rt_batch_profile.py'), Path('scripts/rt_cuda_graph.py'),
                        Path('scripts/rt_cuda_graph_validate.py'), Path('scripts/stage_a_common.py'),
                        Path('scripts/stage_b_train.py'), Path('scripts/experiment_tracking.py'),
                        Path('configs/stage_a/full.json')]
        source_paths += sorted(Path('recurrent-transformer/olmo').rglob('*.py'))
        report['source_sha256'] = {}
        for source in source_paths:
            relative = source.relative_to(Path.cwd()) if source.is_absolute() else source
            value = source.read_bytes()
            report['source_sha256'][str(relative)] = hashlib.sha256(value).hexdigest()
            target = args.output_dir / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
        ids, data, manifest = load_training_data(args.data_manifest, args.updates)
        atomic_json(args.output_dir / 'data-manifest.json', manifest)
        report['data'] = data
        protocol = json.loads(args.protocol.read_text())
        validate_protocol(protocol, args.seed)
        report['protocol_sha256'] = file_digest(args.protocol)
        report['heldout_manifest_sha256'] = data['heldout_exclusion_manifest_sha256']
        atomic_json(args.output_dir / 'protocol.json', protocol)
        config = validation_config('full', args.policy)
        report['model_config'] = dataclasses.asdict(config)
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({key: value for key, value in report.items() if key not in ('source_sha256', 'wandb', 'updates')})
        seed_all(args.seed, deterministic=True)
        runtime = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'cache': cache,
                   'gpu_name': report['hardware']['name'], 'capability': report['hardware']['capability'],
                   'driver_version': report['hardware']['nvidia_smi'].splitlines()[0].rsplit(',', 1)[-1].strip(),
                   'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                   'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                   'autocast_cache_enabled': torch.is_autocast_cache_enabled(),
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
        report['runtime'] = runtime
        from olmo.model import OLMo, OLMoRecurrentBlockTiled
        model = OLMo(config).to(device='cuda', dtype=torch.float32).train()
        if (len(model.transformer.blocks) != 12 or not all(
                isinstance(block, OLMoRecurrentBlockTiled) for block in model.transformer.blocks)
                or model.activation_checkpointing_strategy is not None or config.bwd_mlp_chunks != 4):
            raise AssertionError('Require all twelve recurrent blocks and four internal MLP chunks')
        names = list(dict(model.named_parameters()))
        if sum(p.numel() for p in model.parameters()) != 216843264:
            raise AssertionError('Standard RT parameter count differs')
        optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
        contract = {'model_config': report['model_config'], 'optimizer_parameter_names': names,
                    'seed': args.seed, 'data': data, 'source_sha256': report['source_sha256'],
                    'runtime': runtime, 'schedule': SCHEDULE, 'optimizer': OPTIMIZER,
                    'protocol_sha256': report['protocol_sha256'],
                    'batch': BATCH, 'length': LENGTH, 'head_chunk_size': 2, 'clip_norm': 1.}
        report['resume_contract'] = contract
        completed = 0
        initial_digest = digest_tensors(model.state_dict())
        saved_rng = rng_state()
        if args.resume is not None:
            packet = torch.load(args.resume, map_location='cpu', weights_only=False)
            completed = validate_resume_metadata(packet, contract, args.updates)
            validate_model_checkpoint(packet, model.state_dict())
            validate_optimizer_checkpoint(packet, dict(model.named_parameters()))
            model.load_state_dict(packet['model'], strict=True)
            optimizer.load_state_dict(packet['optimizer'])
            initial_digest, saved_rng = packet['initial_state_sha256'], packet['rng']
            report['resume'] = {'path': str(args.resume), 'sha256': file_digest(args.resume),
                                'completed_updates': completed, 'next_data_row': completed * BATCH}
            del packet
        report['initial_state_sha256'] = initial_digest
        report['starting_updates'] = completed
        report['starting_model_sha256'] = digest_tensors(model.state_dict())

        def assert_sources_unchanged():
            for path, expected in report['source_sha256'].items():
                if file_digest(Path(path)) != expected:
                    raise AssertionError(f'Training source changed: {path}')
            if file_digest(args.data_manifest) != data['manifest_sha256']:
                raise AssertionError('Training manifest changed')
            if file_digest(args.protocol) != report['protocol_sha256']:
                raise AssertionError('Frozen training protocol changed')

        def checkpoint(step):
            assert_sources_unchanged()
            packet = {'schema': SCHEMA, 'model_config': report['model_config'],
                      'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                      'optimizer_parameter_names': names, 'completed_updates': step,
                      'next_data_row': step * BATCH, 'seed': args.seed, 'rng': cpu_tree(rng_state()),
                      'initial_state_sha256': initial_digest, 'data_manifest_sha256': data['manifest_sha256'],
                      'source_sha256': report['source_sha256'], 'runtime': runtime,
                      'policy': args.policy, 'protocol_sha256': report['protocol_sha256'],
                      'heldout_manifest_sha256': report['heldout_manifest_sha256'],
                      'resume_contract': contract}
            report['checkpoints'].append(save_checkpoint(args.output_dir / 'checkpoints' / f'step-{step:06d}.pt', packet))
            atomic_json(args.output_dir / 'progress.json', report, replace=True)

        if args.resume is None:
            checkpoint(0)
        begin, end = data_window(completed, len(ids))
        example = torch.from_numpy(np.array(ids[begin:end], dtype=np.int64)).to('cuda')
        # Existing parameters/moments are loaded before capture; warmup performs
        # no optimizer updates. Each graph owns fresh, stable gradient buffers.
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError('Capture construction requires cleared gradients')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        captured = CapturedRTBackward(model, example, head_chunk_size=2, bf16=True,
                                      precision_policy=args.policy)
        if digest_tensors(model.state_dict()) != report['starting_model_sha256']:
            raise AssertionError('Capture warmup changed model weights')
        restore_rng(saved_rng)
        report['capture_setup_memory'] = memory()
        report['capture'] = captured.metadata()
        report['compiler_after_capture'] = compiler_audit(True)
        atomic_json(args.output_dir / 'progress.json', report, replace=True)
        print(json.dumps({'status': 'captured', 'policy': args.policy,
                          'starting_updates': completed, 'memory': report['capture_setup_memory']}), flush=True)
        torch.cuda.reset_peak_memory_stats()
        with (args.output_dir / 'updates.jsonl').open('x') as log:
            for previous in range(completed, args.updates):
                update = previous + 1
                begin, end = data_window(previous, len(ids))
                step_started = time.perf_counter()
                batch = torch.from_numpy(np.array(ids[begin:end], dtype=np.int64)).to('cuda')
                lr = learning_rate(update)
                optimizer.param_groups[0]['lr'] = lr
                loss = captured.replay(batch)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
                optimizer.step()
                check = precision_check(model, optimizer)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - step_started
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite training CE')
                row = {'update': update, 'row_begin': begin, 'row_end': end,
                       'train_ce_b_times_t': value, 'train_ce_supervised_token': value * LENGTH / (LENGTH - 1),
                       'gradient_norm_before_clip': norm.item(), 'learning_rate': lr,
                       'update_seconds': elapsed, 'input_tokens_per_second': BATCH * LENGTH / elapsed,
                       'all_finite_fp32_parameters_gradients_moments': check['all_finite'], 'memory': memory()}
                report['updates'].append(row)
                log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                tracker.log({'update': update, 'train/ce_b_times_t': value,
                             'train/ce_supervised_token': row['train_ce_supervised_token'],
                             'train/gradient_norm_before_clip': row['gradient_norm_before_clip'],
                             'train/learning_rate': lr, 'train/update_seconds': elapsed,
                             'train/input_tokens_per_second': row['input_tokens_per_second'],
                             'train/all_finite_fp32_state': True})
                atomic_json(args.output_dir / 'progress.json', report, replace=True)
                if update in (100, 500):
                    checkpoint(update)
                if update % 10 == 0 or update == args.updates:
                    print(json.dumps({'update': update, 'ce': value, 'lr': lr,
                                      'update_seconds': elapsed}), flush=True)
        report['final_precision_check'] = precision_check(model, optimizer)
        report['compiler'] = compiler_audit(True)
        if report['compiler']['counters'] != report['compiler_after_capture']['counters']:
            raise AssertionError('Compiler activity changed during graph replay')
        assert_sources_unchanged()
        if file_digest(args.data_manifest.parent / manifest['roles']['train']['ids_path']) != data['ids_sha256']:
            raise AssertionError('Frozen training IDs changed during training')
        report['final_model_sha256'] = digest_tensors(model.state_dict())
        report['completed_updates'] = args.updates
        report['next_data_row'] = args.updates * BATCH
        report['status'] = 'complete'
        tracker.summary({'train/completed_updates': args.updates, 'train/passed': True,
                         'train/final_ce_b_times_t': report['updates'][-1]['train_ce_b_times_t'],
                         'train/final_learning_rate': learning_rate(args.updates)})
    except BaseException as error:
        report.update(status='execution_failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        try:
            if tracker is not None:
                tracker.finish(succeeded=report['status'] == 'complete')
        except BaseException as error:
            report.update(status='execution_failed', sync_error_type=type(error).__name__)
            raise
        finally:
            atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'output_dir': str(args.output_dir),
                      'completed_updates': args.updates}), flush=True)


if __name__ == '__main__':
    main()
