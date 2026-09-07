#!/usr/bin/env python3
"""Bounded actual-loss FP32 CDRM validation and matched optimizer-step timing."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from cdrm_common import (setup,sources,build_pair_member,optimizer_for,loss_sum,update,
                        state_digest,cpu_tree,atomic_json,save_checkpoint,effective_precision,
                        compiler_audit,state_summaries,file_digest)
from r3_validation_metrics import compare_tensors
from r3_mixed_operational import measured_compiler_audit


def validation_sources():
    result=sources()
    result['scripts/cdrm_validate.py']=file_digest(Path(__file__))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['validate','benchmark'],default='validate')
    parser.add_argument('--preset',choices=['d256','d128','d128_6'],default='d256')
    parser.add_argument('--topology',choices=['seq','r3','cdrm','same_depth'],default='cdrm')
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--task',default='in-context-recall')
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError('Use a fresh output directory')
    args.output_dir.mkdir(parents=True)
    report={'schema':'cdrm-reference-validation-v1','evidence':'NUM/OPS','status':'running',
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    started=time.monotonic()
    try:
        report.update(setup(args.seed));tracked=validation_sources();report['sources']=tracked
        if args.data_root:
            from cdrm.mad_data import load_dataset,task_spec
            data=load_dataset(args.data_root,args.task,'train').take(slice(0,args.batch))
            ids=torch.as_tensor(data.input_ids,device='cuda');labels=torch.as_tensor(data.labels,device='cuda')
            answers=torch.as_tensor(data.answer_labels,device='cuda');vocab=data.manifest['vocab_size']
            fixture={'source':'pinned MAD native training labels','task':args.task,'sha256':data.sha256}
        else:
            from cdrm.synthetic.experiment import generate_training_batch
            plan=json.loads(Path('configs/stage_b/pilot.json').read_text())
            if args.batch>64:raise ValueError('Stage B fixture supports at most B64')
            data=generate_training_batch(plan,'mqar',0).take(slice(0,args.batch))
            ids=torch.as_tensor(data.input_ids,device='cuda');labels=torch.as_tensor(data.labels,device='cuda')
            answers=labels;vocab=1024
            fixture={'source':'authoritative Stage B MQAR aligned answer-only labels','task':'mqar','sha256':data.sha256}
        length=ids.shape[1]
        report['fixture']={**fixture,'batch':ids.shape[0],'length':length,'vocab':vocab,
                           'native_targets':int((labels!=-100).sum()),'answer_targets':int((answers!=-100).sum()),
                           'extra_lm_shift':False}
        if len(ids)!=args.batch:raise ValueError('Requested batch not available')
        if args.mode=='benchmark':
            model,construction=build_pair_member(args.preset,args.topology,vocab,length,args.seed)
            opt=optimizer_for(model)
            report['construction']=construction
            warmup=[]
            for _ in range(args.warmup):warmup.append(update(model,opt,ids,labels))
            report['warmup']=warmup
            report['precision_after_warmup']=effective_precision(model,opt,require_gradients=True)
            before=compiler_audit(args.topology=='r3')
            report['compiler_after_warmup']=before
            report['startup_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            rows=[update(model,opt,ids,labels) for _ in range(args.steps)]
            report['measured']=rows
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            after=compiler_audit(args.topology=='r3');report['compiler_after_measurement']=after
            report['measurement_compilation']=measured_compiler_audit(before,after)
            if not report['measurement_compilation']['steady_compilation_free']:
                raise AssertionError('Compilation occurred during measurement')
            times=[r['seconds'] for r in rows]
            report['timing']={'mean_seconds':statistics.mean(times),'median_seconds':statistics.median(times),
                              'tokens_per_second':args.steps*ids.numel()/sum(times),
                              'scope':'Synchronized actual native loss forward/backward/clip/Adam; prepared GPU data; no diagnostic state retention'}
            report['final_precision']=effective_precision(model,opt,require_gradients=True)
        else:
            seq,seq_info=build_pair_member(args.preset,'seq',vocab,length,args.seed)
            bypass,bypass_info=build_pair_member(args.preset,'cdrm',vocab,length,args.seed,gates={'cdrm_lambda':0.})
            packets={}
            for name,model in [('seq',seq),('lambda_zero',bypass)]:
                with sdpa_kernel(SDPBackend.MATH):
                    output=model(ids,output_hidden_states=True,output_cdrm_states=name=='lambda_zero')
                    summed,n=loss_sum(output.logits,labels);loss=summed/n
                loss.backward()
                packets[name]={'logits':cpu_tree(output.logits),'loss':loss.item(),
                               'gradients':{k:None if p.grad is None else cpu_tree(p.grad) for k,p in model.named_parameters()},
                               'hidden_states':cpu_tree(output.hidden_states)}
                if name=='lambda_zero':
                    if len(output.hidden_states)!=model.config.n_layers+1:raise AssertionError('Hidden convention changed')
                    if not torch.equal(output.hidden_states[model.config.cdrm_late_layer+1],output.cdrm_states['v8']):
                        raise AssertionError('The first suffix hidden input must be v8')
            comparisons={name:compare_tensors(value,packets['lambda_zero']['gradients'][name])
                         for name,value in packets['seq']['gradients'].items()}
            logits=compare_tensors(packets['seq']['logits'],packets['lambda_zero']['logits'])
            extra={k:None if v is None else float(v.abs().max()) for k,v in packets['lambda_zero']['gradients'].items() if k.startswith('cdrm.')}
            report['bypass']={'logits':logits,'gradients':comparisons,'extra_branch_gradient_max':extra,
                              'losses':{k:v['loss'] for k,v in packets.items()},
                              'same_backbone':seq_info['backbone_initialization_sha256']==bypass_info['backbone_initialization_sha256'],
                              'hidden_entries':len(packets['seq']['hidden_states'])}
            if not logits['elementwise_pass'] or not all(r['elementwise_pass'] for r in comparisons.values()) or any(v not in [None,0.] for v in extra.values()):
                raise AssertionError('Lambda-zero equivalence failed')
            save_checkpoint(args.output_dir/'bypass-packets.pt',packets)
            del seq,bypass,packets,output
            torch.cuda.empty_cache()
            model,construction=build_pair_member(args.preset,args.topology,vocab,length,args.seed)
            opt=optimizer_for(model);report['construction']=construction
            with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                original=model(ids,output_cdrm_states=model.config.cdrm_enabled)
                report['initial_corrections']=state_summaries(original.cdrm_states)
                altered=ids.clone();split=length//2;altered[:,split:]=(altered[:,split:]+1)%vocab
                changed=model(altered).logits
                causal=compare_tensors(original.logits[:,:split],changed[:,:split])
                report['causality']=causal
                if not causal['elementwise_pass']:raise AssertionError('Future perturbation changed earlier outputs')
                del original,changed
            row=update(model,opt,ids,labels,answers,monitor=True)
            report['actual_step']=row
            if args.topology in ['cdrm','same_depth'] and any(not row['parameter_gradient_signals'][name]['nonzero'] for name in ['cdrm.deep_adapter.weight','cdrm.bridge_adapter.weight']):
                raise AssertionError('Active adapters lack loss gradients')
            snapshot={'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(opt.state_dict()),
                      'config':construction['model_config'],'completed_updates':1,'fixture':report['fixture']}
            report['checkpoint']=save_checkpoint(args.output_dir/'one-update.pt',snapshot)
            restored,restored_info=build_pair_member(args.preset,args.topology,vocab,length,args.seed)
            restored_opt=optimizer_for(restored)
            loaded=torch.load(args.output_dir/'one-update.pt',map_location='cpu',weights_only=False)
            restored.load_state_dict(loaded['model'],strict=True);restored_opt.load_state_dict(loaded['optimizer'])
            report['round_trip']={'model_exact':state_digest(restored.state_dict())==state_digest(model.state_dict()),
                                  'optimizer_exact':state_digest(restored_opt.state_dict())==state_digest(opt.state_dict())}
            with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                report['round_trip']['logits_exact']=torch.equal(model(ids).logits,restored(ids).logits)
            if not all(report['round_trip'].values()):raise AssertionError('Checkpoint round trip failed')
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            report['memory_scope']='NUM includes concurrent comparison/round-trip models; steady timing/memory is measured separately'
        if validation_sources()!=tracked:raise RuntimeError('Source changed during GPU validation')
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        atomic_json(args.output_dir/'report.json',report)
    print(json.dumps({'status':report['status'],'output':str(args.output_dir)}),flush=True)


if __name__=='__main__':main()
