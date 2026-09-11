#!/usr/bin/env python3
"""Fresh-process 101->102 CUDA-graph continuation proof from a real RT step100.

The reference phase recaptures at step100, runs 101/102 and retains only the
step101 proof state plus exact step102 digests. The resumed phase is a separate
process that recaptures from the proof state and must reproduce update102.
This does not compare against the original training capture created at step0.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_batch_profile import memory, precision_check
from rt_cuda_graph import CapturedRTBackward
from rt_cuda_graph_validate import digest_tensors, validation_config
from rt_precision_compare import cpu_tree, save_packet
from rt_precision_credit import snapshot_sources, verify_sources
from rt_precision_eval_contract import validate_evaluation_contract
from rt_precision_train import (BATCH, LENGTH, OPTIMIZER, SCHEDULE, data_window,
                                learning_rate, load_training_data, validate_model_checkpoint,
                                validate_optimizer_checkpoint, validate_protocol,
                                validate_resume_metadata)
from stage_a_common import (configure_compiled_helpers, require_cuda_container,
                            restore_rng, rng_state, seed_all)
from stage_b_train import atomic_json, compiler_audit, file_digest

STATE_SCHEMA = 'rt-precision-resume-proof-state-v1'
REPORT_SCHEMA = 'rt-precision-resume-proof-v1'
START, SAVED, TARGET = 100, 101, 102


def validate_phase_arguments(phase, proof_checkpoint, reference_report):
    if phase not in ('reference', 'resumed'):
        raise ValueError('Unknown resume-proof phase')
    supplied = proof_checkpoint is not None, reference_report is not None
    if supplied != ((False, False) if phase == 'reference' else (True, True)):
        raise ValueError('Only resumed phase requires both proof checkpoint and reference report')


def process_identity():
    # PIDs alone can repeat inside new Docker containers.
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': os.getpid(), 'start_ticks': fields[19], 'hostname': socket.gethostname()}


def rng_digest(state):
    digest = hashlib.sha256()
    numpy_state = state['numpy']
    digest.update(repr((state['python'], numpy_state[0], numpy_state[2:])).encode())
    digest.update(np.ascontiguousarray(numpy_state[1]).tobytes())
    tensors = {'torch_cpu': state['torch_cpu']}
    tensors.update({f'torch_cuda_{index}': value for index, value in enumerate(state['torch_cuda'])})
    digest.update(digest_tensors(tensors).encode())
    return digest.hexdigest()


def runtime_identity(hardware):
    cache = os.environ.get('TORCHINDUCTOR_CACHE_DIR')
    if not cache or not Path(cache).is_absolute():
        raise ValueError('Explicit retained absolute TORCHINDUCTOR_CACHE_DIR required')
    return {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'cache': cache,
            'gpu_name': hardware['name'], 'capability': hardware['capability'],
            'driver_version': hardware['nvidia_smi'].splitlines()[0].rsplit(',', 1)[-1].strip(),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'autocast_cache_enabled': torch.is_autocast_cache_enabled(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def training_resume_contract(config, names, *, seed, data, sources, runtime, protocol_sha256):
    """Reconstruct the training authority, without proof-only metadata keys."""
    return {'model_config': dataclasses.asdict(config), 'optimizer_parameter_names': list(names),
            'seed': seed, 'data': data, 'source_sha256': sources,
            'runtime': runtime, 'schedule': SCHEDULE, 'optimizer': OPTIMIZER,
            'protocol_sha256': protocol_sha256,
            'batch': BATCH, 'length': LENGTH, 'head_chunk_size': 2, 'clip_norm': 1.}


def validate_proof_state(packet, contract):
    if packet.get('schema') != STATE_SCHEMA or packet.get('proof_contract') != contract:
        raise ValueError('Proof schema or source/data/protocol/runtime/authority contract differs')
    training = contract['training_resume_contract']
    if (packet.get('completed_updates') != SAVED or type(packet.get('completed_updates')) is not int
            or packet.get('next_data_row') != SAVED * BATCH):
        raise ValueError('Resume proof must start exactly at saved update101 and its next row')
    for key in ('model_config', 'optimizer_parameter_names', 'seed'):
        if packet.get(key) != training[key]:
            raise ValueError(f'Proof top-level identity differs: {key}')
    if packet.get('rng_sha256') != rng_digest(packet['rng']):
        raise ValueError('Proof RNG state differs from its saved digest')
    data_window(SAVED, training['data']['shape'][0])


def validate_reference_report(reference, *, proof_sha256, current_contract, current_process):
    if (reference.get('schema') != REPORT_SCHEMA or reference.get('status') != 'complete'
            or reference.get('phase') != 'reference'):
        raise ValueError('Require a completed reference phase report')
    # Reports have the JSON tuple/list representation; checkpoint contracts are
    # native torch packets. Normalize metadata for this comparison only.
    canonical = json.loads(json.dumps(current_contract, allow_nan=False))
    if reference.get('proof_contract') != canonical:
        raise ValueError('Reference report source/data/protocol/runtime authority differs')
    if reference.get('proof_checkpoint', {}).get('sha256') != proof_sha256:
        raise ValueError('Proof checkpoint is not the one retained by the reference run')
    if reference.get('process') == current_process:
        raise ValueError('Resume proof requires a fresh process')
    expected = reference.get('expected_update102')
    if (not isinstance(expected, dict) or expected.get('update') != TARGET
            or expected.get('row_begin') != SAVED * BATCH
            or expected.get('row_end') != TARGET * BATCH):
        raise ValueError('Reference expected-update identity differs')
    return expected


def exact_update_comparison(reference, actual):
    identity_fields = ('update', 'row_begin', 'row_end', 'learning_rate', 'loss', 'loss_sha256',
                       'gradient_norm_before_clip', 'gradient_norm_sha256', 'raw_gradients_sha256',
                       'post_model_sha256', 'post_optimizer_sha256', 'post_rng_sha256')
    if any(field not in reference or field not in actual for field in identity_fields):
        raise ValueError('Update packet is missing an exact comparison field')
    for side in (reference, actual):
        if not all(math.isfinite(side[key]) for key in ('learning_rate', 'loss', 'gradient_norm_before_clip')):
            raise FloatingPointError('Nonfinite expected or actual update scalar')
    mismatches = [field for field in identity_fields if reference[field] != actual[field]]
    return {'pass': not mismatches, 'exact_fields': list(identity_fields), 'mismatched_fields': mismatches,
            'tolerance': 'Exact native scalar equality and SHA256 of every tensor byte; no numerical tolerance'}


def optimizer_tensors(model, optimizer):
    return {f'{name}/{key}': optimizer.state[parameter][key]
            for name, parameter in model.named_parameters() for key in ('step', 'exp_avg', 'exp_avg_sq')}


def execute_update(model, optimizer, captured, ids, update):
    begin, end = data_window(update - 1, len(ids))
    tokens = torch.from_numpy(np.array(ids[begin:end], dtype=np.int64)).to('cuda')
    lr = learning_rate(update)
    optimizer.param_groups[0]['lr'] = lr
    loss = captured.replay(tokens)
    value = float(loss)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite proof loss')
    raw = {name: parameter.grad for name, parameter in model.named_parameters()}
    if any(gradient is None or gradient.dtype != torch.float32 or not bool(torch.isfinite(gradient).all())
           for gradient in raw.values()):
        raise FloatingPointError('Proof requires every finite FP32 raw parameter gradient')
    raw_digest = digest_tensors(raw)
    # Keep bounded CPU copies only for a possible mismatch packet after stepping.
    raw_cpu = cpu_tree(raw)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False, error_if_nonfinite=True)
    optimizer.step()
    precision = precision_check(model, optimizer)
    if any(float(state['step']) != update for state in optimizer.state.values()):
        raise AssertionError('Every native Adam counter must advance exactly once')
    torch.cuda.synchronize()
    record = {'update': update, 'row_begin': begin, 'row_end': end, 'learning_rate': lr,
              'loss': value, 'loss_sha256': digest_tensors({'loss': loss}),
              'gradient_norm_before_clip': float(norm), 'gradient_norm_sha256': digest_tensors({'norm': norm}),
              'raw_gradients_sha256': raw_digest, 'post_model_sha256': digest_tensors(model.state_dict()),
              'post_optimizer_sha256': digest_tensors(optimizer_tensors(model, optimizer)),
              'post_rng_sha256': rng_digest(rng_state()), 'precision': precision, 'memory': memory()}
    return record, raw_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('reference', 'resumed'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Original completed step100 training checkpoint')
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--heldout-manifest', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--proof-checkpoint', type=Path)
    parser.add_argument('--reference-report', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args = parser.parse_args()
    validate_phase_arguments(args.phase, args.proof_checkpoint, args.reference_report)
    if not args.wandb_project:
        parser.error('Online W&B required')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh proof output directory')
    args.output_dir.mkdir(parents=True)
    tracker, started = None, time.monotonic()
    report = {'schema': REPORT_SCHEMA, 'status': 'running', 'phase': args.phase,
              'process': process_identity(), 'updates': [],
              'scope': 'Physical B512/T512, selected native BF16 policy, same retained source/runtime/cache, clip and Adam outside capture',
              'qualification': 'Proves fresh recapture of saved update101 versus fresh step100 capture continued to102; not the original capture0 after100 updates'}
    try:
        report['hardware'] = require_cuda_container()
        torch.set_num_threads(1)
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.utils.counters.clear()
        sources = snapshot_sources(args.output_dir, extra=(Path(__file__), Path('scripts/rt_precision_train.py'),
                                                           Path('scripts/rt_precision_eval_contract.py')))
        report['source_sha256'] = sources
        training_report = json.loads(args.training_report.read_text())
        seed, policy = training_report['seed'], training_report['policy']
        seed_all(seed, deterministic=True)
        runtime = runtime_identity(report['hardware'])
        report.update(seed=seed, policy=policy, runtime=runtime)
        ids, data, _ = load_training_data(args.data_manifest, TARGET)
        protocol = json.loads(args.protocol.read_text())
        validate_protocol(protocol, seed)
        authority = {'checkpoint_sha256': file_digest(args.checkpoint),
                     'training_report_sha256': file_digest(args.training_report),
                     'protocol_sha256': file_digest(args.protocol),
                     'data_manifest_sha256': file_digest(args.data_manifest),
                     'heldout_manifest_sha256': file_digest(args.heldout_manifest)}
        report['authority'] = authority
        config = validation_config('full', policy)
        from olmo.model import OLMo
        model = OLMo(config).to(device='cuda', dtype=torch.float32).train()
        names = list(dict(model.named_parameters()))
        original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        report['training_identity'] = validate_evaluation_contract(
            training_report, original, checkpoint_sha256=authority['checkpoint_sha256'],
            heldout_manifest_sha256=authority['heldout_manifest_sha256'],
            current_source_sha256=sources, protocol_sha256=authority['protocol_sha256'],
            expected_parameters=dict(model.named_parameters()))
        if original['completed_updates'] != START:
            raise ValueError('Resume proof is anchored only to original update100')
        training_sources = {name: sources[name] for name in training_report['source_sha256']}
        contract = training_resume_contract(config, names, seed=seed, data=data, sources=training_sources,
                                            runtime=runtime, protocol_sha256=authority['protocol_sha256'])
        validate_resume_metadata(original, contract, 500)
        validate_model_checkpoint(original, model.state_dict())
        validate_optimizer_checkpoint(original, dict(model.named_parameters()))
        proof_contract = {'training_resume_contract': contract, 'source_sha256': sources,
                          'authority': authority, 'reference_start': START, 'saved_update': SAVED,
                          'target_update': TARGET}
        report['proof_contract'] = proof_contract
        atomic_json(args.output_dir / 'criteria.json',
                    {'proof_contract': proof_contract, 'acceptance': 'Exact loss/norm/raw-gradient/model/Adam/RNG identity at update102'})
        report['criteria_sha256'] = file_digest(args.output_dir / 'criteria.json')
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({key: value for key, value in report.items() if key not in ('source_sha256', 'proof_contract', 'wandb', 'updates')})
        expected = None
        if args.phase == 'reference':
            state, completed = original, START
        else:
            state = torch.load(args.proof_checkpoint, map_location='cpu', weights_only=False)
            validate_proof_state(state, proof_contract)
            validate_model_checkpoint(state, model.state_dict())
            validate_optimizer_checkpoint(state, dict(model.named_parameters()))
            completed = SAVED
            reference = json.loads(args.reference_report.read_text())
            expected = validate_reference_report(reference, proof_sha256=file_digest(args.proof_checkpoint),
                                                   current_contract=proof_contract, current_process=report['process'])
            report['reference_report'] = {'path': str(args.reference_report), 'sha256': file_digest(args.reference_report)}
            report['proof_checkpoint'] = {'path': str(args.proof_checkpoint), 'sha256': file_digest(args.proof_checkpoint)}
        model.load_state_dict(state['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
        optimizer.load_state_dict(state['optimizer'])
        saved_rng = cpu_tree(state['rng'])
        report['starting_model_sha256'] = digest_tensors(model.state_dict())
        report['starting_optimizer_sha256'] = digest_tensors(optimizer_tensors(model, optimizer))
        del state, original
        gc.collect()
        begin, end = data_window(completed, len(ids))
        example = torch.from_numpy(np.array(ids[begin:end], dtype=np.int64)).to('cuda')
        captured = CapturedRTBackward(model, example, head_chunk_size=2, bf16=True, precision_policy=policy)
        if (digest_tensors(model.state_dict()) != report['starting_model_sha256']
                or digest_tensors(optimizer_tensors(model, optimizer)) != report['starting_optimizer_sha256']):
            raise AssertionError('Recapture warmup changed model or optimizer state')
        restore_rng(saved_rng)
        if rng_digest(rng_state()) != rng_digest(saved_rng):
            raise AssertionError('RNG restore after recapture was not exact')
        report['capture'] = captured.metadata()
        compiler_after_capture = compiler_audit(True)
        if args.phase == 'reference':
            row, raw = execute_update(model, optimizer, captured, ids, SAVED)
            del raw
            report['updates'].append(row)
            saved_rng = cpu_tree(rng_state())
            proof = {'schema': STATE_SCHEMA, 'proof_contract': proof_contract,
                     'model_config': dataclasses.asdict(config), 'optimizer_parameter_names': names,
                     'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                     'completed_updates': SAVED, 'next_data_row': SAVED * BATCH, 'seed': seed,
                     'rng': saved_rng, 'rng_sha256': rng_digest(saved_rng)}
            report['proof_checkpoint'] = save_packet(args.output_dir / 'step-000101-proof.pt', proof)
            del proof
            tracker.log({'update': SAVED, 'resume_proof/loss': row['loss']})
            atomic_json(args.output_dir / 'progress.json', report, replace=True)
            # Exclude checkpoint/tracker bookkeeping from stochastic state in
            # both phases. Dropout-free model execution itself uses no RNG.
            restore_rng(saved_rng)
        actual, raw = execute_update(model, optimizer, captured, ids, TARGET)
        report['updates'].append(actual)
        if args.phase == 'reference':
            report['expected_update102'] = actual
        else:
            report['comparison'] = exact_update_comparison(expected, actual)
            if not report['comparison']['pass']:
                report['mismatch_packet'] = save_packet(args.output_dir / 'actual-update102-mismatch.pt',
                    {'schema': 'rt-precision-resume-mismatch-v1', 'proof_contract': proof_contract,
                     'record': actual, 'raw_gradients': raw, 'model': cpu_tree(model.state_dict()),
                     'optimizer': cpu_tree(optimizer.state_dict()), 'rng': cpu_tree(rng_state())})
                raise AssertionError('Fresh-process update102 did not exactly match the reference')
        del raw
        report['compiler'] = compiler_audit(True)
        if report['compiler']['counters'] != compiler_after_capture['counters']:
            raise AssertionError('Compiler activity changed during captured proof updates')
        verify_sources(sources)
        for key, path in (('checkpoint_sha256', args.checkpoint), ('training_report_sha256', args.training_report),
                          ('protocol_sha256', args.protocol), ('data_manifest_sha256', args.data_manifest),
                          ('heldout_manifest_sha256', args.heldout_manifest)):
            if file_digest(path) != authority[key]:
                raise AssertionError(f'Proof external authority changed: {key}')
        role = json.loads(args.data_manifest.read_text())['roles']['train']
        if file_digest(args.data_manifest.parent / role['ids_path']) != data['ids_sha256']:
            raise AssertionError('Frozen training token stream changed during proof')
        tracker.log({'update': TARGET, 'resume_proof/loss': actual['loss']})
        tracker.summary({'resume_proof/completed_update': TARGET,
                         'resume_proof/exact_resumption_pass': args.phase == 'resumed'})
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='execution_failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        try:
            if tracker:
                tracker.finish(succeeded=report['status'] == 'complete')
        except BaseException as error:
            report.update(status='execution_failed', sync_error_type=type(error).__name__)
            raise
        finally:
            atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'phase': args.phase,
                      'exact_pass': report.get('comparison', {}).get('pass')}), flush=True)


if __name__ == '__main__':
    main()
