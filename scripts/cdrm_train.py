#!/usr/bin/env python3
"""Fixed-data FP32 CDRM/SEQ/R3 training with a preserved 200-epoch horizon."""
from __future__ import annotations
import argparse
import copy
import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from cdrm_common import (setup, sources, build_pair_member, optimizer_for, update, evaluate,
                        save_checkpoint, state_digest, cpu_tree, rng_state, restore_rng,
                        atomic_json, append_jsonl, file_digest, json_digest,
                        effective_precision, compiler_audit)

FORMAT='cdrm-fp32-training-v1'
MILESTONES={0,1,5,10,25,50,100,200}


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['train','evaluate'],default='train')
    p.add_argument('--topology',choices=['seq','r3','cdrm','same_depth'],required=True)
    p.add_argument('--preset',choices=['d128','d256','d128_6'],default='d128')
    p.add_argument('--task',default='in-context-recall')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--stop-epochs',type=int,default=25)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--shuffle-seed',type=int,default=45678)
    p.add_argument('--monitor-every',type=int,default=100)
    p.add_argument('--run-purpose',choices=['pilot','screening','comparison'],default='pilot')
    p.add_argument('--screen-early-ace',action='store_true')
    p.add_argument('--ace-token-accuracy',type=float,default=.999)
    p.add_argument('--ace-exact-match',type=float,default=.99)
    p.add_argument('--ace-consecutive-epochs',type=int,default=3)
    p.add_argument('--ace-by-epoch',type=int,default=10)
    p.add_argument('--train-examples',type=int)
    p.add_argument('--ops-development-from-train',action='store_true',
                   help='Use the next32 training examples as an OPS-only monitor; no research dev access')
    p.add_argument('--resume',type=Path)
    p.add_argument('--reference-final',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['dev','final'],default='final')
    p.add_argument('--max-wall-seconds',type=float,default=12000)
    p.add_argument('--deadline-utc',default='2026-09-07T16:18:30+00:00')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.batch<1 or not 1<=args.stop_epochs<=200 or args.monitor_every<1:
        p.error('Require positive batch/monitor interval and endpoint1..200')
    if args.mode=='evaluate' and args.checkpoint is None:
        p.error('Evaluation requires --checkpoint')
    if args.mode=='evaluate' and args.run_purpose=='screening' and args.split=='final':
        p.error('Screening uses development only; reserve final for the selected comparison')
    if args.screen_early_ace and (args.topology!='seq' or args.run_purpose!='screening'):
        p.error('Early-ace stopping is only for explicitly labeled SEQ screening')
    if not (0<=args.ace_token_accuracy<=1 and 0<=args.ace_exact_match<=1
            and args.ace_consecutive_epochs>0 and args.ace_by_epoch>0):
        p.error('Invalid screening thresholds')
    return args


def dataset_identity(dataset):
    return {'input_ids':state_digest(dataset.input_ids),'labels':state_digest(dataset.labels),
            'answer_labels':state_digest(dataset.answer_labels),
            'examples':len(dataset.input_ids),'length':dataset.input_ids.shape[1]}


def compare_resumed(reference, actual):
    from r3_validation_metrics import compare_tensors
    differences={}
    def compare(a,b,name):
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            if state_digest(a)!=state_digest(b):differences[name]=compare_tensors(a,b)
        elif isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):differences[name]={'keys_differ':True};return
            for k in a:compare(a[k],b[k],f'{name}/{k}')
        elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
            if len(a)!=len(b):differences[name]={'lengths_differ':True};return
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{name}/{i}')
        elif isinstance(a,np.ndarray) and isinstance(b,np.ndarray):
            if state_digest(a)!=state_digest(b):differences[name]={'rng_array_mismatch':True}
        elif a!=b:differences[name]={'reference':a,'actual':b}
    for key in ['model','optimizer','scheduler','rng','completed_updates','completed_epochs',
                'batch_in_epoch','next_batch_sha256','identity']:
        compare(reference[key],actual[key],key)
    def numerical_history(payload):
        return [{k:v for k,v in row.items() if k!='seconds'} for row in payload['history']]
    compare(numerical_history(reference),numerical_history(actual),'history')
    for epoch,row in reference['development'].items():
        if epoch not in actual['development']:differences[f'development/{epoch}']={'missing':True};continue
        for key in ['native','answer']:compare(row[key],actual['development'][epoch][key],f'development/{epoch}/{key}')
    return {'bitwise_state_and_nontiming_metrics_equal':not differences,'differences':differences,
            'excluded':['wall times','paths','serialized file bytes','compiler history','checkpoint role ledger']}


def main():
    args=args_parser()
    if args.output_dir.exists():raise FileExistsError('Use a new output directory')
    args.output_dir.mkdir(parents=True)
    report={'format':FORMAT,'evidence':'SYN' if args.mode=='train' else 'SYN-evaluation','status':'running',
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    started=time.monotonic()
    try:
        report.update(setup(args.seed))
        from cdrm.mad_data import load_dataset,epoch_indices
        train=load_dataset(args.data_root,args.task,'train')
        if args.ops_development_from_train:
            if args.train_examples is None or args.train_examples+32>len(train.input_ids):
                raise ValueError('OPS requires an explicit training subset with32 further monitor examples')
            dev=train.take(slice(args.train_examples,args.train_examples+32))
            report['evidence']='OPS'
        else:
            dev=load_dataset(args.data_root,args.task,'dev')
        if args.train_examples is not None:
            if args.train_examples<args.batch or args.train_examples>len(train.input_ids):
                raise ValueError('Invalid explicit training subset')
            train=train.take(slice(0,args.train_examples))
        # Reserved symbols and absent IDs in small samples must not shrink the
        # model vocabulary. The audited native task manifest is authoritative.
        vocab=int(train.manifest['vocab_size'])
        if dev.manifest['vocab_size']!=vocab:raise ValueError('Train/dev vocabulary mismatch')
        length=train.input_ids.shape[1]
        model, construction=build_pair_member(args.preset,args.topology,vocab,length,args.seed)
        opt=optimizer_for(model)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200,eta_min=1e-6)
        tracked_sources=sources()
        identity={'format':FORMAT,'topology':args.topology,'preset':args.preset,'task':args.task,
                  'seed':args.seed,'shuffle_seed':args.shuffle_seed,'physical_batch':args.batch,
                  'model_config':construction['model_config'],'source_sha256':tracked_sources,
                  'train':dataset_identity(train),'dev':dataset_identity(dev),
                  'data_manifest_sha256':json_digest(train.manifest),
                  'backbone_initialization_sha256':construction['backbone_initialization_sha256'],
                  'precision':report['execution_contract'],
                  'optimizer':{'type':'AdamW','lr':5e-4,'betas':[.9,.98],'eps':1e-8,'weight_decay':0.,
                               'foreach':False,'fused':False,'gradient_clip':1.},
                  'schedule':{'type':'CosineAnnealingLR','T_max_epochs':200,'eta_min':1e-6,
                              'step':'after each completed epoch','warmup':False},
                  'monitor_every':args.monitor_every,'accumulation':False,
                  'training_subset':args.train_examples,
                  'development_source':'next32_training_examples_OPS' if args.ops_development_from_train else 'independent_dev'}
        report.update(identity=identity,identity_sha256=json_digest(identity),construction=construction,
                      run_purpose=args.run_purpose,
                      screening={'enabled':args.screen_early_ace,'token_accuracy':args.ace_token_accuracy,
                                 'sequence_exact_match':args.ace_exact_match,
                                 'consecutive_epochs':args.ace_consecutive_epochs,'by_epoch':args.ace_by_epoch},
                      supervision='Native MAD training labels; separate answer labels for retrieval metrics; no additional shift',
                      schedule_interpretation='Explicit epoch stepping implements intended MAD cosine horizon; no dependency on upstream Lightning scheduler registration')
        atomic_json(args.output_dir/'resolved-config.json',identity)
        n=len(train.input_ids);updates_per_epoch=math.ceil(n/args.batch)
        report['updates_per_epoch']=updates_per_epoch
        history=[];development={};completed_epochs=0;completed_updates=0;batch_in_epoch=0
        best_score=float('inf');roles={'milestones':{},'latest':None,'best_dev':None,'pruned':[]}
        permutation_cache={}
        def indices(epoch,batch_index):
            if epoch not in permutation_cache:
                permutation_cache.clear()
                permutation_cache[epoch]=epoch_indices(n,epoch,args.shuffle_seed)
            begin=batch_index*args.batch
            return permutation_cache[epoch][begin:min(begin+args.batch,n)]
        def next_hash():return state_digest(train.input_ids[indices(completed_epochs,batch_in_epoch)])
        def payload():
            return {'format':FORMAT,'identity':identity,'identity_sha256':json_digest(identity),
                    'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                    'rng':rng_state(),'completed_epochs':completed_epochs,'completed_updates':completed_updates,
                    'batch_in_epoch':batch_in_epoch,'next_batch_sha256':next_hash(),
                    'history':history,'development':development,'best_score':best_score,
                    'roles':roles,'initial_weights_only':completed_updates==0,
                    'memory_state':'Rebuilt from each independent input; no cross-minibatch side state'}
        if args.mode=='evaluate':
            checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            if checkpoint['identity']!=identity:raise ValueError('Evaluation identity mismatch')
            model.load_state_dict(checkpoint['model'],strict=True)
            data=dev if args.split=='dev' else load_dataset(args.data_root,args.task,'final')
            report.update(checkpoint={'path':str(args.checkpoint),'sha256':file_digest(args.checkpoint),
                                     'completed_epochs':checkpoint['completed_epochs'],'completed_updates':checkpoint['completed_updates']},
                          evaluation_split=args.split,evaluation_data=dataset_identity(data),
                          metrics=evaluate(model,data,args.batch),status='complete')
            return
        if args.resume:
            old=torch.load(args.resume,map_location='cpu',weights_only=False)
            if old['format']!=FORMAT or old['identity']!=identity or old['identity_sha256']!=json_digest(identity):
                raise ValueError('Resume config/source/runtime/data identity mismatch')
            if old['completed_updates']==0:raise ValueError('Initialization is weights-only; update-zero exact resume is unvalidated')
            model.load_state_dict(old['model'],strict=True);opt.load_state_dict(old['optimizer'])
            scheduler.load_state_dict(old['scheduler']);restore_rng(old['rng'])
            for key,value in [('model',model.state_dict()),('optimizer',opt.state_dict()),('scheduler',scheduler.state_dict()),('rng',rng_state())]:
                if state_digest(value)!=state_digest(old[key]):raise AssertionError(f'{key} restore differs')
            completed_epochs=old['completed_epochs'];completed_updates=old['completed_updates'];batch_in_epoch=old['batch_in_epoch']
            history=copy.deepcopy(old['history']);development=copy.deepcopy(old['development']);best_score=old['best_score']
            roles=copy.deepcopy(old['roles'])
            for value in [roles.get('latest'),roles.get('best_dev'),*roles['milestones'].values()]:
                if value and value.get('sha256') is None:
                    if Path(value['path']).resolve()!=args.resume.resolve():
                        raise AssertionError('Unresolved role refers to a different checkpoint')
                    value.update(sha256=file_digest(args.resume),bytes=args.resume.stat().st_size)
            if next_hash()!=old['next_batch_sha256']:raise AssertionError('Sampler/data position mismatch')
            report['resume']={'path':str(args.resume),'sha256':file_digest(args.resume),'loaded_exact':True,
                              'completed_epochs':completed_epochs,'completed_updates':completed_updates,
                              'batch_in_epoch':batch_in_epoch,'repeated_evaluation':False,'warmup_updates':0}
        else:
            record=save_checkpoint(args.output_dir/'initial-weights.pt',payload())
            roles['milestones']['0']=record
            report['initial_weights']=record
            development['0']=evaluate(model,dev,args.batch)
            best_score=development['0']['answer']['ce']
            roles['best_dev']=record
            atomic_json(args.output_dir/'checkpoint-ledger.json',roles,replace=True)
        if completed_epochs>=args.stop_epochs:raise ValueError('Requested endpoint already completed')
        def retain_checkpoint(tag,*,milestone=None,best=False):
            if sources()!=tracked_sources:raise RuntimeError('Source changed during execution')
            path=args.output_dir/f'{tag}.pt'
            packet=payload()
            # A checkpoint cannot contain its own content hash. Store its role
            # with an explicit unresolved hash and resolve that one role on load.
            saved_roles=copy.deepcopy(roles)
            current={'path':str(path),'sha256':None,'bytes':None}
            saved_roles['latest']=current
            if milestone is not None:saved_roles['milestones'][str(milestone)]=current
            if best:saved_roles['best_dev']=current
            packet['roles']=saved_roles
            record=save_checkpoint(path,packet)
            old_latest=roles['latest'];old_best=roles['best_dev']
            roles['latest']=record
            if milestone is not None:roles['milestones'][str(milestone)]=record
            if best:roles['best_dev']=record
            atomic_json(args.output_dir/'checkpoint-ledger.json',roles,replace=True)
            protected={v['path'] for v in roles['milestones'].values()}
            protected.update(v['path'] for v in [roles['latest'],roles['best_dev']] if v)
            # Only prune artifacts created in this run; parent lineage is immutable.
            for old_record in [old_latest,old_best]:
                if not old_record:continue
                path=Path(old_record['path'])
                if path.parent==args.output_dir and str(path) not in protected and path.exists():
                    path.unlink();roles['pruned'].append(old_record)
            atomic_json(args.output_dir/'checkpoint-ledger.json',roles,replace=True)
            return record
        deadline=datetime.datetime.fromisoformat(args.deadline_utc)
        starting_updates=completed_updates
        paused=False
        early_ace=False
        while completed_epochs<args.stop_epochs:
            if time.monotonic()-started>args.max_wall_seconds-30 or datetime.datetime.now(datetime.timezone.utc)>deadline-datetime.timedelta(seconds=30):
                report['pause_checkpoint']=retain_checkpoint(f'paused-u{completed_updates:07d}')
                paused=True;break
            idx=indices(completed_epochs,batch_in_epoch)
            ids=torch.as_tensor(train.input_ids[idx],device='cuda')
            labels=torch.as_tensor(train.labels[idx],device='cuda')
            answers=torch.as_tensor(train.answer_labels[idx],device='cuda')
            monitor=completed_updates==0 or (completed_updates+1)%args.monitor_every==0
            row=update(model,opt,ids,labels,answers,monitor=monitor)
            row.update(update=completed_updates+1,epoch=completed_epochs+1,batch_in_epoch=batch_in_epoch,
                       indices_sha256=state_digest(idx))
            history.append(row);append_jsonl(args.output_dir/'learning-curve.jsonl',row)
            completed_updates+=1;batch_in_epoch+=1
            if completed_updates%100==0:
                print(json.dumps({'update':completed_updates,'epoch':completed_epochs+1,
                                  'topology':args.topology,'native_loss':row['native_loss']}),flush=True)
            if batch_in_epoch==updates_per_epoch:
                completed_epochs+=1;batch_in_epoch=0;scheduler.step()
                metrics=evaluate(model,dev,args.batch);development[str(completed_epochs)]=metrics
                append_jsonl(args.output_dir/'development.jsonl',{'epoch':completed_epochs,**metrics})
                is_best=metrics['answer']['ce']<best_score
                if is_best:best_score=metrics['answer']['ce']
                early_ace=(args.screen_early_ace
                    and args.ace_consecutive_epochs<=completed_epochs<=args.ace_by_epoch
                    and all(development[str(e)]['answer']['token_accuracy']>=args.ace_token_accuracy
                            and development[str(e)]['answer']['sequence_exact_match']>=args.ace_exact_match
                            for e in range(completed_epochs-args.ace_consecutive_epochs+1,completed_epochs+1)))
                retain_checkpoint(f'epoch-{completed_epochs:04d}',
                                  milestone=completed_epochs if completed_epochs in MILESTONES or completed_epochs==args.stop_epochs or early_ace else None,
                                  best=is_best)
                print(json.dumps({'completed_epoch':completed_epochs,'topology':args.topology,
                                  'dev_answer':metrics['answer'],'best_dev':is_best}),flush=True)
                if early_ace:break
        report.update(status='paused' if paused else 'complete',completed_epochs=completed_epochs,
                      termination_reason='early_ace' if early_ace else ('wall_time' if paused else 'requested_endpoint'),
                      completed_updates=completed_updates,batch_in_epoch=batch_in_epoch,
                      development=development,roles=roles,history=history,
                      updates_this_process=completed_updates-starting_updates,
                      final_precision=effective_precision(model,opt,require_gradients=completed_updates>starting_updates),
                      compiler=compiler_audit(args.topology=='r3'))
        if args.reference_final:
            ref=torch.load(args.reference_final,map_location='cpu',weights_only=False)
            report['recovery_comparison']=compare_resumed(ref,cpu_tree(payload()))
            atomic_json(args.output_dir/'recovery-comparison.json',report['recovery_comparison'])
            if not report['recovery_comparison']['bitwise_state_and_nontiming_metrics_equal']:
                raise AssertionError('Exact recovery comparison failed; numerical differences are retained')
        if sources()!=tracked_sources:raise RuntimeError('Source changed during execution')
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        atomic_json(args.output_dir/'report.json',report)


if __name__=='__main__':main()
