#!/usr/bin/env python3
"""Bounded same-precision RT CUDA-graph gradient and optimizer validation."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_batch_profile import head_chunked_backward, precision_check
from rt_cuda_graph import CapturedRTBackward
from stage_a_common import configure_compiled_helpers, require_cuda_container, seed_all
from stage_b_train import atomic_json, compiler_audit

ATOL = 1e-7
RTOL = 1e-5


def digest_tensors(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        if tensor is None:
            raise AssertionError(f'Missing tensor: {name}')
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compare_trees(reference, actual):
    """CPU FP64 reductions, bounded temporary memory, all coordinates checked."""
    if set(reference) != set(actual):
        raise AssertionError('Tensor names differ')
    records = {}
    totals = dict(reference_squared=0., actual_squared=0., error_squared=0., dot=0.)
    failures = nonfinite = count = 0
    maximum = 0.
    exact = True
    for name in sorted(reference):
        left, right = reference[name], actual[name]
        if left is None or right is None or left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'Missing tensor, shape or dtype mismatch: {name}')
        a = left.detach().cpu().reshape(-1)
        b = right.detach().cpu().reshape(-1)
        row = dict(numel=a.numel(), dtype=str(a.dtype), exact=torch.equal(a, b),
                   failures=0, nonfinite=0, max_abs_error=0., reference_squared=0.,
                   actual_squared=0., error_squared=0., dot=0.)
        for start in range(0, a.numel(), 1_048_576):
            x, y = a[start:start+1_048_576].double(), b[start:start+1_048_576].double()
            finite = torch.isfinite(x) & torch.isfinite(y)
            row['nonfinite'] += int((~finite).sum())
            if not finite.all():
                continue
            difference = y-x
            errors = difference.abs()
            row['failures'] += int((errors > ATOL + RTOL*x.abs()).sum())
            row['max_abs_error'] = max(row['max_abs_error'], float(errors.max()) if len(errors) else 0.)
            row['reference_squared'] += float(torch.dot(x, x))
            row['actual_squared'] += float(torch.dot(y, y))
            row['error_squared'] += float(torch.dot(difference, difference))
            row['dot'] += float(torch.dot(x, y))
        for key in totals:
            totals[key] += row[key]
        row['relative_l2'] = math.sqrt(row['error_squared']) / max(math.sqrt(row['reference_squared']), 1e-30)
        denominator = math.sqrt(row['reference_squared'])*math.sqrt(row['actual_squared'])
        row['cosine'] = row['dot']/denominator if denominator else (1. if row['error_squared']==0 else 0.)
        row['pass'] = row['nonfinite']==0 and row['failures']==0
        failures += row['failures']; nonfinite += row['nonfinite']; count += row['numel']
        maximum = max(maximum, row['max_abs_error']); exact = exact and row['exact']
        records[name] = row
    denominator = math.sqrt(totals['reference_squared'])*math.sqrt(totals['actual_squared'])
    return {'pass': failures==0 and nonfinite==0, 'exact': exact,
            'reduction': 'CPU FP64; every coordinate checked', 'atol': ATOL, 'rtol': RTOL,
            'tensor_count': len(records), 'numel': count, 'failure_coordinates': failures,
            'nonfinite_coordinates': nonfinite, 'max_abs_error': maximum,
            'relative_l2': math.sqrt(totals['error_squared'])/max(math.sqrt(totals['reference_squared']),1e-30),
            'cosine': totals['dot']/denominator if denominator else (1. if totals['error_squared']==0 else 0.),
            'tensors': records}


def parameters(model):
    return dict(model.named_parameters())


def gradients(model):
    return {name: parameter.grad for name, parameter in model.named_parameters()}


def optimizer_tensors(model, optimizer):
    result = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state[parameter]
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            if key not in state:
                raise AssertionError(f'Missing Adam state: {name}/{key}')
            result[f'{name}/{key}'] = state[key]
    return result


def assert_pass(record, label):
    if not record['pass']:
        raise AssertionError(f'{label}: {record["failure_coordinates"]} coordinate failures, '
                             f'{record["nonfinite_coordinates"]} nonfinite coordinates')


def validation_config(tier, policy='bf16_fp32_state'):
    from olmo.config import ModelConfig
    raw = json.loads(Path('configs/stage_a/full.json').read_text())
    raw.update(block_type='recurrent', recurrent_layers=None, recurrent_backend='tiled',
               recurrent_write_rho=1., recurrent_precision_policy=policy,
               ordinary_attention_precision_policy='legacy', reference_eager=False,
               cdrm_enabled=False, init_device='cuda', precision=None)
    if tier == 'tiny':
        raw.update(d_model=64,n_heads=4,n_kv_heads=4,mlp_hidden_size=256,n_layers=2,
                   max_sequence_length=16,vocab_size=128,embedding_size=128)
    return ModelConfig(**raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=('tiny','full'), required=True)
    parser.add_argument('--precision', choices=('fp32','bf16'), required=True)
    parser.add_argument('--policy', choices=('bf16_fp32_state', 'legacy'), default='bf16_fp32_state')
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='recurrent-transformer-cuda-graphs')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh validation directory')
    if not args.wandb_project:
        parser.error('Online W&B is required')
    args.output_dir.mkdir(parents=True)
    report = {'schema':'rt-cuda-graph-validation-v1','status':'running','tier':args.tier,
              'precision':args.precision,'policy':args.policy,'seed':20260910,'atol':ATOL,'rtol':RTOL,
              'scope':'Captured versus uncaptured execution at the same precision; no BF16-versus-FP32 comparison',
              'head_chunk_size':2,'backbone_accumulation_steps':1,'optimizer_captured':False,
              'repeated_input_checks':[],'optimizer_updates':[],'guards':{}}
    tracker = None; started = time.monotonic()
    try:
        report['hardware'] = require_cuda_container()
        torch.set_num_threads(1)
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.utils.counters.clear()
        source_paths = [Path(__file__),Path('scripts/rt_cuda_graph.py'),Path('scripts/rt_batch_profile.py'),
                        Path('scripts/stage_a_common.py'),Path('scripts/stage_b_train.py'),
                        Path('scripts/experiment_tracking.py'),Path('configs/stage_a/full.json')]
        source_paths += sorted(Path('recurrent-transformer/olmo').rglob('*.py'))
        report['source_sha256'] = {}
        for source in source_paths:
            relative = source.relative_to(Path.cwd()) if source.is_absolute() else source
            data = source.read_bytes(); report['source_sha256'][str(relative)] = hashlib.sha256(data).hexdigest()
            target = args.output_dir/'source'/relative; target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        report['runtime'] = {'torch':str(torch.__version__),'cuda':torch.version.cuda,
                             'cache':os.environ.get('TORCHINDUCTOR_CACHE_DIR')}
        config = validation_config(args.tier, args.policy)
        report['model_config'] = dataclasses.asdict(config)
        batch = 3 if args.tier=='tiny' else 2
        report['shape'] = [batch,config.max_sequence_length]
        tracker = OnlineTracker(project=args.wandb_project,entity=args.wandb_entity,group=args.wandb_group,
                                name=args.wandb_run_name,output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({key:value for key,value in report.items() if key not in ('source_sha256','wandb')})
        seed_all(20260910,deterministic=True)
        from olmo.model import OLMo
        reference = OLMo(config).to(device='cuda',dtype=torch.float32).train()
        candidate = OLMo(validation_config(args.tier, args.policy)).to(device='cuda',dtype=torch.float32).train()
        candidate.load_state_dict(reference.state_dict(),strict=True)
        bf16 = args.precision=='bf16'
        generator = torch.Generator(device='cuda').manual_seed(20260911)
        ids = torch.randint(0,config.vocab_size,report['shape'],device='cuda',generator=generator)
        initial_state = digest_tensors(candidate.state_dict())
        names = list(candidate.state_dict())
        captured = CapturedRTBackward(candidate,ids,head_chunk_size=2,bf16=bf16,warmup=3,
                                      precision_policy=args.policy)
        report['capture_preserves_state_dict'] = list(candidate.state_dict())==names and digest_tensors(candidate.state_dict())==initial_state
        if not report['capture_preserves_state_dict']:
            raise AssertionError('Capture changed model weights or state_dict names')
        first_gradient_digest = None
        for repeat in range(2):
            reference.zero_grad(set_to_none=True)
            expected_loss = head_chunked_backward(reference,ids,bf16=bf16)
            actual_loss = captured.replay(ids)
            torch.cuda.synchronize()
            row = {'repeat':repeat+1,'ids_sha256':digest_tensors({'ids':ids}),
                   'loss':compare_trees({'loss':expected_loss},{'loss':actual_loss}),
                   'raw_gradients':compare_trees(gradients(reference),gradients(candidate))}
            digest = digest_tensors(gradients(candidate))
            row['same_as_first_captured_gradients_bitwise'] = first_gradient_digest is None or digest==first_gradient_digest
            first_gradient_digest = digest if first_gradient_digest is None else first_gradient_digest
            report['repeated_input_checks'].append(row)
            assert_pass(row['loss'],'Repeated-input loss');assert_pass(row['raw_gradients'],'Repeated-input gradients')
            if not row['same_as_first_captured_gradients_bitwise']:
                raise AssertionError('Repeated replay accumulated or changed gradients at fixed weights and inputs')
            atomic_json(args.output_dir/'progress.json',report,replace=True)
        options = dict(lr=1e-3,betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
        reference_optimizer = torch.optim.AdamW(reference.parameters(),**options)
        candidate_optimizer = torch.optim.AdamW(candidate.parameters(),**options)
        report['optimizer'] = {'name':'AdamW',**options,'clip_norm':1.}
        for update in range(1,4):
            ids = torch.randint(0,config.vocab_size,report['shape'],device='cuda',generator=generator)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_loss = head_chunked_backward(reference,ids,bf16=bf16)
            actual_loss = captured.replay(ids)
            torch.cuda.synchronize()
            row = {'update':update,'ids_sha256':digest_tensors({'ids':ids}),
                   'reference_loss':expected_loss.item(),'captured_loss':actual_loss.item(),
                   'loss':compare_trees({'loss':expected_loss},{'loss':actual_loss}),
                   'raw_gradients':compare_trees(gradients(reference),gradients(candidate))}
            report['optimizer_updates'].append(row)
            assert_pass(row['loss'],'Changed-input loss');assert_pass(row['raw_gradients'],'Changed-input gradients')
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True,foreach=False)
            actual_norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(),1.,error_if_nonfinite=True,foreach=False)
            row['gradient_norm'] = compare_trees({'norm':expected_norm},{'norm':actual_norm})
            reference_optimizer.step();candidate_optimizer.step()
            row['parameters'] = compare_trees(parameters(reference),parameters(candidate))
            row['adam_state'] = compare_trees(optimizer_tensors(reference,reference_optimizer),optimizer_tensors(candidate,candidate_optimizer))
            for key in ('gradient_norm','parameters','adam_state'):
                assert_pass(row[key],f'Post-update {key}')
            tracker.log({'update':update,'validation/raw_gradient_relative_l2':row['raw_gradients']['relative_l2'],
                         'validation/parameter_relative_l2':row['parameters']['relative_l2'],
                         'validation/adam_relative_l2':row['adam_state']['relative_l2'],
                         'validation/loss_abs_error':row['loss']['max_abs_error']})
            atomic_json(args.output_dir/'progress.json',report,replace=True)
        report['final_precision_check'] = precision_check(candidate,candidate_optimizer)
        report['parameters_changed_after_adam'] = digest_tensors(candidate.state_dict())!=initial_state
        if not report['parameters_changed_after_adam']:
            raise AssertionError('Adam updates left model unchanged')
        report['compiler_after_replays'] = compiler_audit(True)
        prefix = config.max_sequence_length//2
        changed = ids.clone();changed[:,prefix:] = (changed[:,prefix:]+1)%config.vocab_size
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            left = candidate(ids,return_pre_logits=True,return_logits=False).pre_logits
            right = candidate(changed,return_pre_logits=True,return_logits=False).pre_logits
        report['causality'] = {'scope':'Native forward pre-logit prefix; capture uses the same native model path, but this is not a separately observed captured-prefix test',
                               'unchanged_prefix_length':prefix,'suffix_changed':not torch.equal(ids,changed),
                               'prefix_bitwise_equal':torch.equal(left[:,:prefix],right[:,:prefix])}
        if not report['causality']['prefix_bitwise_equal']:
            raise AssertionError('Native forward violates prefix causality')
        before_guards = digest_tensors(candidate.state_dict())
        try: captured.replay(ids[:-1])
        except ValueError:report['guards']['invalid_shape_rejected']=True
        else:raise AssertionError('Capture accepted a changed batch shape')
        try:
            with torch.no_grad():captured.replay(ids)
        except RuntimeError:report['guards']['no_grad_rejected']=True
        else:raise AssertionError('Capture accepted no_grad replay')
        first_parameter = next(candidate.parameters());saved_gradient = first_parameter.grad
        first_parameter.grad = None
        try:
            try:captured.replay(ids)
            except RuntimeError:report['guards']['replaced_gradient_buffer_rejected']=True
            else:raise AssertionError('Capture accepted missing persistent gradient storage')
        finally:first_parameter.grad=saved_gradient
        report['guards']['state_dict_preserved'] = digest_tensors(candidate.state_dict())==before_guards and list(candidate.state_dict())==names
        if not report['guards']['state_dict_preserved']:
            raise AssertionError('Guard checks changed model state')
        report['capture'] = captured.metadata()
        report['compiler'] = compiler_audit(True)
        for path,digest in report['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise AssertionError(f'Validation source changed: {path}')
        report['status'] = 'complete'
        tracker.summary({'validation/passed':True,'validation/repeated_replays':2,'validation/adam_updates':3,
                         'validation/capture_preserves_state_dict':report['capture_preserves_state_dict']})
    except BaseException as error:
        report.update(status='execution_failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        try:
            if tracker:tracker.finish(succeeded=report['status']=='complete')
        except BaseException as error:
            report.update(status='execution_failed',sync_error_type=type(error).__name__)
            raise
        finally:atomic_json(args.output_dir/'report.json',report)
    print(json.dumps({'status':report['status'],'output_dir':str(args.output_dir)}),flush=True)


if __name__=='__main__':
    main()
