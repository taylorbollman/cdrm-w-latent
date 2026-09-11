#!/usr/bin/env python3
"""Retained actual-loss triangulation for R3 CUDA BF16 autocast. NUM only."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager, nullcontext
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from r3_backward_validate import cpu, conversion, tensors, step_packet, adam_state
from r3_validation_metrics import compare_tensors
from stage_a_common import require_cuda_container, seed_all, configure_compiled_helpers, provenance
from stage_a_common import rng_state, restore_rng
from stage_b_train import aligned_ce_sum, compiler_audit, file_digest
from experiment_tracking import OnlineTracker, add_wandb_arguments


@contextmanager
def preserve_tracking_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def comparison(reference, actual):
    row = compare_tensors(reference, actual)
    if row["finite"]:
        a, b = reference.double().reshape(-1), actual.double().reshape(-1)
        denominator = a.norm() * b.norm()
        row["cosine"] = float(torch.dot(a, b) / denominator) if denominator > 1e-24 else None
        row["historical_bf16_gradient_screen"] = row["rel_l2"] <= .015625 and row["maxerr_over_ref_rms"] <= .0625
    else:
        row.update(cosine=None, historical_bf16_gradient_screen=False)
    return row


def compare_map(reference, actual):
    if list(reference) != list(actual):
        raise AssertionError("Canonical tensor coverage/order differs")
    rows = {n: comparison(reference[n], actual[n]) for n in reference}
    return {"rows": rows, "count": len(rows),
            "all_present_finite": all(r["finite"] for r in rows.values()),
            "historical_bf16_failed_tensors": [n for n,r in rows.items() if not r["historical_bf16_gradient_screen"]],
            "fp32_elementwise_failed_tensors": [n for n,r in rows.items() if not r["elementwise_pass"]]}


class Observation:
    """Observe module/helper boundaries without replacing any tensors."""
    def __init__(self, model):
        self.model, self.hooks, self.rows = model, [], {}
        self.phase = "forward"
        self.samples = {}

    def record(self, name, values):
        if torch.compiler.is_compiling():
            return
        if isinstance(values, torch.Tensor):
            key = f"{self.phase}/{name}/{values.dtype}"
            row = self.rows.setdefault(key, {"count": 0, "dtype": str(values.dtype),
                "first_shape": list(values.shape), "requires_grad": values.requires_grad})
            row["count"] += 1
        elif isinstance(values, (tuple, list)):
            for i,v in enumerate(values):
                self.record(f"{name}/{i}", v)
        elif isinstance(values, dict):
            for key, value in values.items():
                self.record(f"{name}/{key}", value)

    def internal(self, phase, values, token_index=None):
        self.record(f"internal/{phase}", values)
        if token_index in (0, 1, 64, 127):
            for name, value in values.items():
                if isinstance(value, torch.Tensor) and value.numel() <= 65536:
                    self.samples[f"{self.phase}/{phase}/{token_index}/{name}"] = cpu(value)

    def __enter__(self):
        import olmo.model as module
        self.original_helper = module.recurrent_helper
        def helper(block, fn, *args):
            self.record(f"helper/{fn.__name__}/inputs", args)
            result = self.original_helper(block, fn, *args)
            self.record(f"helper/{fn.__name__}/outputs", result)
            return result
        module.recurrent_helper = helper
        self.block = self.model.transformer.blocks[3]
        self.prior_observer = getattr(self.block, '_recurrent_precision_observer', None)
        self.block._recurrent_precision_observer = self.internal
        for name, layer in self.model.named_modules():
            if (isinstance(layer, torch.nn.Linear) or 'norm' in name
                    or name in ['transformer.wte', 'transformer.blocks.3',
                                'transformer.blocks.3.pre_attention_block']):
                def hook(m, inputs, output, label=name):
                    self.record(label + "/input", inputs)
                    self.record(label + "/output", output)
                self.hooks.append(layer.register_forward_hook(hook))
        return self

    def __exit__(self, *error):
        import olmo.model as module
        module.recurrent_helper = self.original_helper
        if self.prior_observer is None:
            del self.block._recurrent_precision_observer
        else:
            self.block._recurrent_precision_observer = self.prior_observer
        for hook in self.hooks:
            hook.remove()


def graph(model, tokens, bf16, boundaries=None):
    captured = {}
    def capture_embed(module, inputs, output):
        output.retain_grad()
        captured['embedding_output'] = output
    def capture_block(module, inputs):
        inputs[0].retain_grad()
        captured['block3_input'] = inputs[0]
    hooks = [model.transformer.wte.register_forward_hook(capture_embed),
             model.transformer.blocks[3].register_forward_pre_hook(capture_block)]
    if boundaries is not None:
        def capture_output(module, inputs, output):
            value = output[0]
            value.retain_grad()
            boundaries['block3_output'] = value
        hooks.append(model.transformer.blocks[3].register_forward_hook(capture_output))
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            logits = model(tokens).logits
        logits.retain_grad()
    finally:
        for h in hooks:
            h.remove()
    return logits, captured


def backward(model, retained, labels=None, cotangent=None, retain=False):
    logits, captured = retained
    model.zero_grad(set_to_none=True)
    for x in [logits, *captured.values()]:
        x.grad = None
    loss = None
    # Intentionally outside the outer autocast context. CE upcasts BF16 logits.
    if labels is not None:
        summed, count = aligned_ce_sum(logits, labels)
        loss = summed / count
        loss.backward(retain_graph=retain)
    else:
        logits.backward(cotangent, retain_graph=retain)
    parameters = {n: None if p.grad is None else cpu(p.grad) for n,p in model.named_parameters()}
    return {"parameters": parameters,
            "inputs": {n: None if x.grad is None else cpu(x.grad) for n,x in captured.items()},
            "logits": cpu(logits), "cotangent": cpu(logits.grad),
            "loss": loss.item() if loss is not None else None}


def build(payload, backend, policy, tiny=False):
    from olmo.config import ModelConfig
    from olmo.model import OLMo
    cfg = copy.deepcopy(payload['model_config'])
    cfg.update(init_device='cpu', precision=None, recurrent_backend=backend,
               reference_eager=backend == 'naive')
    if policy != 'legacy' or 'recurrent_precision_policy' in ModelConfig.__dataclass_fields__:
        cfg['recurrent_precision_policy'] = policy
    model = OLMo(ModelConfig(**cfg)).cuda().train()
    model.load_state_dict(payload['model'], strict=True)
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=Path('configs/stage_b/pilot.json'))
    parser.add_argument('--policy',choices=['legacy','bf16_fp32_state'],default='legacy')
    parser.add_argument('--baseline-source',type=Path,default=Path('.runtime/r3-bf16/20260906T225438Z/source/pre-bf16-source.json'))
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--batch-offset',type=int,default=0)
    parser.add_argument('--data-offset',type=int,default=0)
    parser.add_argument('--scale-check',action='store_true')
    parser.add_argument('--fp32-reference',type=Path)
    parser.add_argument('--include-tiled-fp32',action='store_true')
    parser.add_argument('--only-tiled',action='store_true',
                        help='Run tiled FP32/BF16 only; requires --include-tiled-fp32.')
    parser.add_argument('--no-observers',action='store_true',
                        help='Disable module/helper dtype instrumentation for confirmation.')
    parser.add_argument('--capture-block-boundaries',action='store_true',
                        help='Retain recurrent input/output values and incoming output gradient for localization.')
    parser.add_argument('--autocast-cache',choices=['on','off'],default='on')
    parser.add_argument('--bf16-reduced-reduction',choices=['on','off'],default='on')
    parser.add_argument('--output-dir',type=Path,required=True)
    add_wandb_arguments(parser)
    args=parser.parse_args()
    if args.only_tiled and (not args.include_tiled_fp32 or args.fp32_reference):
        parser.error('--only-tiled requires --include-tiled-fp32 and no reused naive reference')
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory')
    hardware=require_cuda_container()
    args.output_dir.mkdir(parents=True)
    seed_all(937,deterministic=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    # Set globally so the outer forward AND every nested replay context inherit
    # the same choice. An override on graph() alone would end before backward.
    torch.set_autocast_cache_enabled(args.autocast_cache == 'on')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16_reduced_reduction == 'on'
    torch._dynamo.reset();torch._dynamo.utils.counters.clear()
    configure_compiled_helpers(True)
    report={'schema':'r3-mixed-validation-v1','evidence':'NUM','status':'running',
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'provenance':provenance(hardware),'script_sha256':file_digest(Path(__file__)),
            'settings':{'parameters':'FP32','optimizer_state':'FP32','autocast':'CUDA BF16 for forward only',
                        'grad_scaler':False,'tf32':False,'sdpa':'math','deterministic':True,
                        'fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        'math_sdpa_reduced_precision_reduction':torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                        'whole_model_compile':False,'cuda_graphs':False,'policy':args.policy}}
    report['settings'].update(autocast_cache=torch.is_autocast_cache_enabled(),
                              observers=not args.no_observers,
                              block_boundary_capture=args.capture_block_boundaries)
    started=time.monotonic()
    tracker = None
    if args.wandb_project:
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name,
                                output_dir=args.output_dir, preserve_state=preserve_tracking_rng)
        report['wandb'] = tracker.record
    try:
        from cdrm.synthetic.experiment import generate_training_batch
        plan=json.loads(args.plan.read_text())
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        source_paths = set(payload['identity']['source_sha256']) | set(report['provenance']['source_sha256']) | {
            'scripts/r3_mixed_validate.py', 'scripts/r3_backward_validate.py',
            'scripts/r3_validation_metrics.py', 'scripts/experiment_tracking.py'}
        report['validation_source_sha256']={name:file_digest(Path(name)) for name in sorted(source_paths)}
        for name in sorted(source_paths):
            destination=args.output_dir/'source'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
        baseline=json.loads(args.baseline_source.read_text())['source_sha256']
        changes={}
        for name,expected in payload['identity']['source_sha256'].items():
            if baseline.get(name) != expected:
                raise AssertionError(f'Original training identity not in baseline snapshot: {name}')
            actual=file_digest(Path(name))
            if actual!=expected:
                if name not in ['recurrent-transformer/olmo/model.py','recurrent-transformer/olmo/config.py']:
                    raise AssertionError(f'Unapproved frozen source change: {name}')
                changes[name]={'baseline':expected,'candidate':actual}
        if file_digest(args.plan)!=payload['identity']['plan_sha256']:
            raise AssertionError('Resolved Stage B plan changed')
        if payload['identity']['task']!='mqar' or payload['model_config']['recurrent_layers']!=[3] or payload['model_config']['recurrent_write_rho']!=1.:
            raise AssertionError('Expected authoritative MQAR R3/rho1 checkpoint')
        completed=payload['completed_updates']
        base_batch=generate_training_batch(plan,'mqar',completed)
        if base_batch.sha256!=payload['next_data_sha256']:
            raise AssertionError('Checkpoint next-batch identity changed')
        full=generate_training_batch(plan,'mqar',completed+args.data_offset)
        if not 0<=args.batch_offset<64 or not 1<=args.batch<=64-args.batch_offset:
            raise ValueError('Invalid physical diagnostic batch')
        batch=full.take(slice(args.batch_offset,args.batch_offset+args.batch))
        tokens=torch.as_tensor(batch.input_ids,device='cuda');labels=torch.as_tensor(batch.labels,device='cuda')
        report.update(checkpoint={'path':str(args.checkpoint),'sha256':file_digest(args.checkpoint),'completed_updates':completed},
                      source_changes=changes,baseline_source_sha256=file_digest(args.baseline_source),
                      fixture={'sha256':batch.sha256,'full_batch_sha256':full.sha256,'batch':args.batch,'shape':list(tokens.shape),
                               'data_counter':completed+args.data_offset,'batch_offset':args.batch_offset,'answers':int((labels!=-100).sum()),
                               'alignment':'Stage B aligned answer-only CE / total scored answers, no shift','accumulation':False})
        if tracker:
            tracker.start({'evidence_class':'NUM','fixture':report['fixture'],
                           'checkpoint':report['checkpoint'],'settings':report['settings'],
                           'source':report['provenance']['source_sha256']})
            save_json(args.output_dir/'wandb-run.json',tracker.record)
            print(json.dumps({'wandb_run_url':tracker.record['run_url']}),flush=True)
        settings=plan['training'];lr=settings['learning_rate']/settings['warmup_updates'] if completed==0 else payload['optimizer']['param_groups'][0]['lr']
        packets,steps,samples={},{},{};report['observed_dtypes']={};report['scaling']={}
        modes=[('naive_fp32','naive','legacy',False),('naive_bf16','naive',args.policy,True),('tiled_bf16','tiled',args.policy,True)]
        if args.include_tiled_fp32:modes.insert(1,('tiled_fp32','tiled','legacy',False))
        if args.only_tiled:modes=[m for m in modes if m[0].startswith('tiled_')]
        if args.fp32_reference:
            prior=json.loads((args.fp32_reference/'report.json').read_text())
            if prior['fixture']['used_batch_sha256']!=batch.sha256 or prior['checkpoint']['sha256']!=report['checkpoint']['sha256']:
                raise AssertionError('FP32 packet fixture/checkpoint identity mismatch')
            if prior['status'] != 'diagnostics_complete' or any(prior['settings'][k] != v for k,v in
                    {'parameters':'FP32','computation':'FP32','autocast':False,'tf32':False,'sdpa':'math','deterministic':True}.items()):
                raise AssertionError('Reused reference is not the cleared strict FP32 profile')
            if prior['model_config'] != payload['model_config']:
                raise AssertionError('Reused reference model configuration differs')
            for key, expected in {'lr':lr,'betas':settings['betas'],'eps':settings['eps'],
                                  'weight_decay':settings['weight_decay'],'clip':settings['gradient_clip'],
                                  'initial_state_equal':True,'foreach':False,'fused':False}.items():
                if prior['optimizer'][key] != expected:
                    raise AssertionError(f'Reused reference optimizer differs: {key}')
            inventory=json.loads((args.fp32_reference.parent/'artifact-manifest.json').read_text())
            tensor_path=args.fp32_reference/'ce-and-update-tensors.pt'
            if file_digest(tensor_path)!=inventory['files'][str(tensor_path.relative_to(args.fp32_reference.parent))]['sha256']:
                raise AssertionError('Reused tensor packet differs from its retained inventory')
            old=torch.load(args.fp32_reference/'ce-and-update-tensors.pt',map_location='cpu',weights_only=False)
            packets['naive_fp32']=old['gradients']['naive'];steps['naive_fp32']=old['steps']['naive']
            report['reused_fp32']={'path':str(args.fp32_reference),'report_sha256':file_digest(args.fp32_reference/'report.json'),'tensors_sha256':file_digest(args.fp32_reference/'ce-and-update-tensors.pt')}
            modes=[m for m in modes if m[0]!='naive_fp32']
        for name,backend,policy,bf16 in modes:
            print(f'{name}: policy={policy} B{args.batch} T{tokens.shape[1]} checkpoint={completed}',flush=True)
            model=build(payload,backend,policy)
            if any(p.dtype!=torch.float32 for p in model.parameters()):raise AssertionError('Trainable params must stay FP32')
            if list(dict(model.named_parameters())) != list(payload['model']):
                raise AssertionError('Canonical parameter names/order differ from parent checkpoint')
            opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=tuple(settings['betas']),eps=settings['eps'],weight_decay=settings['weight_decay'],foreach=False,fused=False)
            opt.load_state_dict(copy.deepcopy(payload['optimizer']))
            for group in opt.param_groups:group['lr']=lr
            observation = (nullcontext(SimpleNamespace(rows={}, samples={})) if args.no_observers else Observation(model))
            with sdpa_kernel(SDPBackend.MATH),observation as observer:
                boundaries={} if args.capture_block_boundaries else None
                retained=graph(model,tokens,bf16,boundaries)
                observer.phase='backward/ce'
                packets[name]=backward(model,retained,labels=labels,retain=args.scale_check)
                if any(v is None or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in tensors(packets[name]).values()):
                    raise AssertionError('Every intended gradient must be present finite FP32')
                boundary_packet = None
                if boundaries is not None:
                    value=boundaries['block3_output']
                    boundary_packet={
                        'block3_input':cpu(retained[1]['block3_input']),
                        'block3_output':cpu(value),
                        'block3_output_gradient':cpu(value.grad)}
                if args.scale_check:
                    original=packets[name];report['scaling'][name]={}
                    for scale in [1/32,32.]:
                        observer.phase=f'backward/scaled_{scale}'
                        scaled=backward(model,retained,cotangent=original['cotangent'].cuda()*scale,retain=True)
                        normalized={n:v.double()/scale for n,v in tensors(scaled).items()}
                        report['scaling'][name][str(scale)]=compare_map(tensors(original),normalized)
                    # Restore the actual unscaled CE gradient for the optimizer.
                    observer.phase='backward/restored_ce'
                    packets[name]=backward(model,retained,labels=labels)
                if boundary_packet is not None:
                    packets[name]['boundaries']=boundary_packet
                steps[name]=step_packet(model,opt,settings['gradient_clip'])
                report['observed_dtypes'][name]=observer.rows
                samples[name]=observer.samples
            for state in opt.state.values():
                if any(v.dtype!=torch.float32 for v in state.values() if isinstance(v,torch.Tensor)):
                    raise AssertionError('Adam state must stay FP32')
            report['observed_dtypes'][name]['parameter_and_optimizer_contract']={'parameters':'torch.float32','gradients':'torch.float32','moments':'torch.float32','verified':True}
            if tracker:
                tracker.log({f'ce/{name}':packets[name]['loss'],
                             f'clip_norm/{name}':steps[name]['clip_norm']})
            del model,opt,retained
        report['comparisons']={}
        pairs=[('naive_bf16','naive_fp32'),('tiled_bf16','naive_bf16'),('tiled_bf16','naive_fp32')]
        if args.include_tiled_fp32:pairs.append(('tiled_fp32','naive_fp32'))
        if args.include_tiled_fp32:pairs.append(('tiled_bf16','tiled_fp32'))
        pairs=[pair for pair in pairs if all(arm in packets for arm in pair)]
        for actual,reference in pairs:
            a,b=packets[reference],packets[actual]
            moment=lambda s:{f'{n}/{k}':v for n,st in s['state'].items() for k,v in st.items()}
            report['comparisons'][f'{actual}_vs_{reference}']={
                'logits':compare_tensors(a['logits'],b['logits'],
                    atol=2e-6 if actual.endswith('_fp32') else .002,
                    rtol=2e-5 if actual.endswith('_fp32') else .02),
                'losses':{'reference':a['loss'],'actual':b['loss'],'absolute_error':abs(a['loss']-b['loss'])},
                'gradients':compare_map(tensors(a),tensors(b)),
                'clipping':{reference:{k:steps[reference][k] for k in ['clip_norm','clip_coefficient']},actual:{k:steps[actual][k] for k in ['clip_norm','clip_coefficient']}},
                'moments':compare_map(moment(steps[reference]),moment(steps[actual])),
                'deltas':compare_map(steps[reference]['deltas'],steps[actual]['deltas']),
                'weights':compare_map(steps[reference]['weights'],steps[actual]['weights'])}
        torch.save({'tokens':cpu(tokens),'labels':cpu(labels),'packets':packets,'steps':steps,'observed_intermediates':samples,'model_config':payload['model_config']},args.output_dir/'tensors.pt')
        report['optimizer']={'lr':lr,'betas':settings['betas'],'eps':settings['eps'],'clip':settings['gradient_clip'],'weight_decay':settings['weight_decay'],'initial_steps':[completed],'initial_state':'Identical deep copies of authoritative checkpoint optimizer state; diagnostic LR override equal for every arm.'}
        report['compiler']=compiler_audit(True)
        report['status']='diagnostics_complete'
        if tracker:
            for label, pair in report['comparisons'].items():
                rows=pair['gradients']['rows']
                tracker.summary({f'{label}/worst_gradient_rel_l2':max(r['rel_l2'] for r in rows.values()),
                                 f'{label}/ce_absolute_error':pair['losses']['absolute_error'],
                                 f'{label}/all_gradients_finite':pair['gradients']['all_present_finite']})
                for index,(name,row) in enumerate(rows.items()):
                    tracker.log({'tensor_index':index,
                                 f'{label}/gradient_rel_l2':row['rel_l2'],
                                 f'{label}/gradient_maxerr_over_rms':row['maxerr_over_ref_rms']})
    except BaseException as exc:
        report.update(status='execution_failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        if 'validation_source_sha256' in report:
            report['source_changed_during_run']=[name for name,expected in report['validation_source_sha256'].items()
                                                  if file_digest(Path(name)) != expected]
            if report['source_changed_during_run']:
                report.update(status='execution_failed',error_type='SourceChanged',
                              error='Validation source changed during execution; do not clear this result.')
        save_json(args.output_dir/'report.json',report)
        try:
            if tracker:
                tracker.finish(succeeded=report['status']=='diagnostics_complete')
        except Exception as exc:
            report.update(status='execution_failed',error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            save_json(args.output_dir/'report.json',report)
        if report.get('source_changed_during_run'):
            raise RuntimeError(report['error'])
    print(json.dumps({'status':report['status'],'output':str(args.output_dir)}),flush=True)


if __name__=='__main__':main()
