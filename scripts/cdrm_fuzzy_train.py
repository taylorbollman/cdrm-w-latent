#!/usr/bin/env python3
"""Resumable native fuzzy-recall calibration for the paired150M architecture family."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shlex
import sys
import time

import torch

import cdrm_fuzzy_common as common
from cdrm_train import compare_resumed
from experiment_tracking import OnlineTracker,add_wandb_arguments,scalar_metrics

SCHEMA='cdrm-fuzzy-calibration-v1'
MAX_EPOCHS=10
TRAIN_EXAMPLES=12800
DEV_EXAMPLES=1280
CALIBRATION_LENGTH=256
SHUFFLE_SEED=961003


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare-init','train'),default='train')
    parser.add_argument('--arm',choices=common.ARMS)
    parser.add_argument('--precision',choices=tuple(common.PRECISIONS),default='bf16')
    parser.add_argument('--seed',type=int,default=86100)
    parser.add_argument('--initial-checkpoint',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--reference-final',type=Path)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--data-manifest',type=Path)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--lr',type=float,choices=common.LEARNING_RATES,default=5e-4)
    parser.add_argument('--batch-size',type=int,choices=(32,64,128),default=128)
    parser.add_argument('--epochs',type=int,default=MAX_EPOCHS)
    parser.add_argument('--stop-updates',type=int)
    parser.add_argument('--save-updates',type=int,nargs='*',default=[])
    parser.add_argument('--checkpoint-every-epochs',type=int,default=1)
    parser.add_argument('--monitor-every',type=int,default=100)
    add_wandb_arguments(parser);parser.set_defaults(wandb_project='cdrm-150m-fuzzy-recall')
    args=parser.parse_args(argv)
    if args.mode=='train':
        if not all((args.arm,args.initial_checkpoint,args.data_root,args.data_manifest)):
            parser.error('Training requires arm, paired initialization, data root and preparation manifest')
        if not 1<=args.epochs<=MAX_EPOCHS:parser.error('This calibration entry point is bounded to1..10epochs')
        maximum=args.epochs*(TRAIN_EXAMPLES//args.batch_size)
        if args.stop_updates is None:args.stop_updates=maximum
        if not 1<=args.stop_updates<=maximum:parser.error('Stop update lies outside the declared epoch budget')
        if any(not 1<=value<=MAX_EPOCHS*(TRAIN_EXAMPLES//args.batch_size) for value in args.save_updates):parser.error('Invalid checkpoint update')
        if not 1<=args.checkpoint_every_epochs<=MAX_EPOCHS or args.monitor_every<1:parser.error('Invalid checkpoint/monitor cadence')
        if args.reference_final and not args.checkpoint:parser.error('Recovery comparison requires a starting checkpoint')
        if not args.wandb_project:parser.error('Calibration requires online W&B logging')
    elif args.checkpoint or args.initial_checkpoint:parser.error('Initialization preparation does not consume trained/initial checkpoints')
    return args


def cursor_at(completed,train_size,batch_size):
    if train_size%batch_size or completed<0:raise ValueError('Full physical batches and a nonnegative cursor are required')
    return divmod(completed,train_size//batch_size)


def expected_rate(update,base_lr,updates_per_epoch):
    if update<1:raise ValueError('Learning-rate update index starts at1')
    epoch=(update-1)//updates_per_epoch
    return common.SCHEDULE['eta_min']+(base_lr-common.SCHEDULE['eta_min'])*(1+math.cos(math.pi*epoch/50))/2


def batch_indices(completed,train_size,batch_size,shuffle_seed):
    epoch,position=cursor_at(completed,train_size,batch_size)
    return common.epoch_indices(train_size,epoch,shuffle_seed)[position*batch_size:(position+1)*batch_size]


def advance_cursor(epoch,position,updates_per_epoch,scheduler):
    position+=1
    if position==updates_per_epoch:epoch+=1;position=0;scheduler.step()
    return epoch,position


def validate_position(payload,train):
    identity=payload['identity'];batch=identity['physical_batch'];shuffle=identity['shuffle_seed']
    completed=payload['completed_updates'];updates_per_epoch=len(train)//batch
    if cursor_at(completed,len(train),batch)!=(payload['completed_epochs'],payload['batch_in_epoch']):raise ValueError('Checkpoint cursor differs')
    if len(payload['history'])!=completed:raise ValueError('Checkpoint history is incomplete')
    permutation=None;previous_epoch=None
    for update,row in enumerate(payload['history'],1):
        epoch,position=cursor_at(update-1,len(train),batch)
        if (row['update'],row['epoch'],row['batch_in_epoch'])!=(update,epoch+1,position):raise ValueError('History cursor/order differs')
        if not math.isclose(row['learning_rate'],expected_rate(update,identity['base_lr'],updates_per_epoch),rel_tol=1e-12,abs_tol=1e-16):
            raise ValueError('History learning rate differs from the50-epoch cosine schedule')
        if epoch!=previous_epoch:permutation=common.epoch_indices(len(train),epoch,shuffle);previous_epoch=epoch
        indices=permutation[position*batch:(position+1)*batch]
        if row['indices_sha256']!=common.state_digest(indices) or row['batch_sha256']!=train.take(indices).sha256:raise ValueError('History data order differs')
    if payload['next_batch_sha256']!=train.take(batch_indices(completed,len(train),batch,shuffle)).sha256:raise ValueError('Restored next batch differs')
    if payload['scheduler']['last_epoch']!=payload['completed_epochs']:raise ValueError('Scheduler epoch differs')
    rate=expected_rate(completed+1,identity['base_lr'],updates_per_epoch)
    if any(not math.isclose(group['lr'],rate,rel_tol=1e-12,abs_tol=1e-16) for group in payload['optimizer']['param_groups']):
        raise ValueError('Saved optimizer learning rate differs')


def validate_checkpoint(payload,identity,target,*,recovery=False):
    if (payload.get('format')!=common.FORMAT or payload.get('identity_sha256')!=common.json_digest(payload['identity'])
            or payload['identity']!=identity or payload['model_config']!=identity['model_config']
            or payload['arm']!=identity['arm']):raise ValueError('Checkpoint source/config/runtime/data/LR/initialization identity differs')
    common.verify_sources(identity['source_sha256'])
    if target<=payload['completed_updates']:raise ValueError('Continuation must execute at least one update')
    if target<payload['run_target_updates'] and not recovery:raise ValueError('Only an explicit recovery control can shorten an invocation target')


def compare_recovery(reference,actual):
    result=compare_resumed(reference,actual)
    for key in ('arm','precision','model_config','initialization','initial_checkpoint'):
        if common.state_digest(reference[key])!=common.state_digest(actual[key]):result['differences'][key]={'immutable_metadata_differs':True}
    if set(reference['development'])!=set(actual['development']):result['differences']['development/keys']={'scope_differs':True}
    result['bitwise_state_and_nontiming_metrics_equal']=not result['differences']
    result['excluded'].extend(['Invocation stop target','Immediate resume checkpoint reference'])
    return result


def data_contract(args):
    train,dev=(common.load_dataset(args.data_root,common.TASK,split,verify=True) for split in ('train','dev'))
    for split,dataset in (('train',train),('dev',dev)):common.validate_data(dataset,split,length=CALIBRATION_LENGTH)
    if (len(train),len(dev))!=(TRAIN_EXAMPLES,DEV_EXAMPLES):raise ValueError('Calibration requires12800train/1280development examples')
    manifest=json.loads(args.data_manifest.read_text())
    if manifest.get('status')!='complete' or manifest.get('schema')!='cdrm-fuzzy-preparation-v1' or manifest.get('role')!='calibration':
        raise ValueError('Data preparation must be complete frozen calibration data')
    data={split:common.dataset_identity(dataset) for split,dataset in (('train',train),('dev',dev))}
    # Final preparer schema supplies explicit corpus records; do not accept an
    # unrelated preparation document solely because its bytes were retained.
    if manifest.get('task')!=common.TASK or manifest.get('length')!=CALIBRATION_LENGTH or manifest.get('shuffle_seed')!=SHUFFLE_SEED:
        raise ValueError('Preparation task/length/shuffle declaration differs')
    if manifest['protocol']['sha256']!=common.file_digest(args.protocol):raise ValueError('Data and training protocols differ')
    if Path(manifest['data_root']).resolve()!=args.data_root.resolve():raise ValueError('Data root differs from preparation manifest')
    for split in ('train','dev'):
        record=manifest['splits'][split]
        if record['dataset_sha256']!=data[split]['array_sha256'] or record['manifest_sha256']!=data[split]['manifest_sha256']:
            raise ValueError('Preparation manifest and loaded native corpus differ')
    order_record=manifest['epoch_indices'];order_path=Path(order_record['path'])
    if common.file_digest(order_path)!=order_record['sha256']:raise ValueError('Frozen epoch permutations changed')
    import numpy as np
    order=np.load(order_path,allow_pickle=False)
    if order.shape!=(50,len(train)) or order_record['seed']!=SHUFFLE_SEED:raise ValueError('Epoch permutation dimensions/seed differ')
    if any(not np.array_equal(order[epoch],common.epoch_indices(len(train),epoch,SHUFFLE_SEED)) for epoch in range(50)):
        raise ValueError('Saved permutations differ from the declared shared data order')
    return train,dev,data


def run(args,report,tracker):
    common.validate_protocol(args.protocol)
    initial=common.load_initial(args.initial_checkpoint)
    if initial['arm']!=args.arm or initial['initialization']['seed']!=86100:raise ValueError('Calibration requires the corresponding seed86100 paired initialization arm')
    initial_ref=common.reference(args.initial_checkpoint)
    if initial['identity'].get('data_identity') is not None and initial['identity']['data_identity']!=common.reference(args.data_manifest):
        raise ValueError('Initialization is bound to a different calibration manifest')
    if initial['identity']['protocol_sha256']!=common.file_digest(args.protocol):raise ValueError('Initialization and training protocols differ')
    train,dev,data=data_contract(args)
    tracked=common.sources([args.protocol])
    if tracked!=initial['identity']['source_sha256']:raise ValueError('Training dependencies differ from frozen initialization sources')
    report.update(common.setup(initial['initialization']['training_seed']))
    runtime=report['execution_contract']
    if not runtime.get('inductor_cache_directory'):raise ValueError('Exact continuation requires a declared retained compiler cache')
    common.snapshot_sources(args.output_dir,tracked)
    model,construction=common.build_model(args.arm,args.precision,checkpoint=initial,length=CALIBRATION_LENGTH)
    optimizer=common.optimizer_for(model,args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=1e-6)
    common.seed_all(initial['initialization']['training_seed'],deterministic=True)
    identity={'schema':SCHEMA,'arm':args.arm,'precision':args.precision,'model_config':construction['model_config'],
        'source_sha256':tracked,'initial_checkpoint_sha256':initial_ref['sha256'],'initial_identity_sha256':initial['identity_sha256'],
        'initialization':initial['initialization'],'shared_initialization_sha256':initial['initialization']['shared_initialization_sha256'],
        'data':data,'data_manifest':common.reference(args.data_manifest),'protocol_sha256':common.file_digest(args.protocol),
        'execution_contract':runtime,'physical_batch':args.batch_size,'shuffle_seed':SHUFFLE_SEED,
        'base_lr':args.lr,'optimizer':common.OPTIMIZER,'schedule':common.SCHEDULE,'accumulation':False,
        'updates_per_epoch':len(train)//args.batch_size,'maximum_epochs':MAX_EPOCHS,'length':CALIBRATION_LENGTH,
        'development_calendar':'initial, every completed epoch, invocation endpoint',
        'checkpoint_every_epochs':args.checkpoint_every_epochs,'save_updates':sorted(set(args.save_updates)),
        'monitor_every':args.monitor_every,'native_training':'Dense aligned next-token CE including native padding; no new attention padding mask'}
    completed=epoch=position=0;history=[];development={}
    restored=torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    reference=torch.load(args.reference_final,map_location='cpu',weights_only=False) if args.reference_final else None
    if reference is not None and (reference['identity']!=identity or reference['completed_updates']!=args.stop_updates):
        raise ValueError('Recovery reference must have identical experiment identity and endpoint')
    if restored is not None:
        validate_checkpoint(restored,identity,args.stop_updates,recovery=reference is not None)
        validate_position(restored,train)
        model.load_state_dict(restored['model'],strict=True);optimizer.load_state_dict(restored['optimizer'])
        scheduler.load_state_dict(restored['scheduler']);common.restore_rng(restored['rng'])
        for name,value in (('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scheduler',scheduler.state_dict()),('rng',common.rng_state())):
            if common.state_digest(value)!=common.state_digest(restored[name]):raise AssertionError(f'{name} did not restore exactly')
        completed,epoch,position=(restored[key] for key in ('completed_updates','completed_epochs','batch_in_epoch'))
        history=copy.deepcopy(restored['history']);development=copy.deepcopy(restored['development'])
    report.update(identity=identity,identity_sha256=common.json_digest(identity),construction=construction,
        initial_checkpoint=initial_ref,initialization=initial['initialization'],model_config=construction['model_config'],
        history=history,development=development,checkpoints={},
        resume={'checkpoint':common.reference(args.checkpoint) if args.checkpoint else None,'loaded_state_exact':restored is not None,
                'starting_update':completed,'warmup_updates':0,'shared_compiler_cache_required':True})
    common.atomic_json(args.output_dir/'resolved-config.json',identity)
    tracker.start({'evidence_class':'development_calibration','identity':identity,'target_updates':args.stop_updates})
    print(f'W&B: {tracker.record["run_url"]}',flush=True)

    def snapshot():
        return {'format':common.FORMAT,'identity':identity,'identity_sha256':common.json_digest(identity),
            'arm':args.arm,'precision':args.precision,'model_config':construction['model_config'],
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':common.rng_state(),
            'completed_updates':completed,'completed_epochs':epoch,'batch_in_epoch':position,
            'next_batch_sha256':train.take(batch_indices(completed,len(train),args.batch_size,SHUFFLE_SEED)).sha256,
            'history':history,'development':development,'initialization':initial['initialization'],
            'initial_checkpoint':initial_ref,'run_target_updates':args.stop_updates,'memory_state':'Rebuilt for every independent forward'}

    if completed==0:development['0']=common.evaluate(model,dev,args.batch_size,precision=args.precision)
    for update in range(completed+1):
        logged={'update':update}
        if update:logged.update(scalar_metrics(history[update-1],'train'))
        if str(update) in development:logged.update(scalar_metrics(development[str(update)],'dev'))
        if len(logged)>1:tracker.log(logged)
    starting_update=completed
    torch.cuda.reset_peak_memory_stats()
    while completed<args.stop_updates:
        indices=batch_indices(completed,len(train),args.batch_size,SHUFFLE_SEED);batch=train.take(indices)
        ids,labels,answers=(torch.as_tensor(array,device='cuda') for array in (batch.input_ids,batch.labels,batch.answer_labels))
        row=common.update(model,optimizer,ids,labels,answers,precision=args.precision,
                          monitor=completed==0 or (completed+1)%args.monitor_every==0)
        row.update(update=completed+1,epoch=epoch+1,batch_in_epoch=position,batch_sha256=batch.sha256,
                   indices_sha256=common.state_digest(indices))
        completed+=1;epoch,position=advance_cursor(epoch,position,identity['updates_per_epoch'],scheduler)
        history.append(row);common.append_jsonl(args.output_dir/'learning-curve.jsonl',row)
        logged={'update':completed,**scalar_metrics(row,'train')}
        terminal=completed==args.stop_updates and (reference is None or str(completed) in reference['development'])
        if position==0 or terminal:
            development[str(completed)]=common.evaluate(model,dev,args.batch_size,precision=args.precision)
            logged.update(scalar_metrics(development[str(completed)],'dev'))
        if completed in args.save_updates or completed==args.stop_updates or (position==0 and epoch%args.checkpoint_every_epochs==0):
            common.verify_sources(tracked)
            report['checkpoints'][str(completed)]=common.save_checkpoint(args.output_dir/f'u{completed:05d}.pt',snapshot())
        tracker.log(logged)
        if completed%25==0 or completed==args.stop_updates:
            print(f'{args.arm} update{completed}: nativeCE={row["native_loss"]:.6f}, answer_acc={row["answer"]["token_accuracy"]:.6f}',flush=True)
    report.update(completed_updates=completed,completed_epochs=epoch,batch_in_epoch=position,
        training_seconds=sum(row['seconds'] for row in history),invocation_training_seconds=sum(row['seconds'] for row in history[starting_update:]),
        peak_memory_allocated=torch.cuda.max_memory_allocated(),peak_memory_reserved=torch.cuda.max_memory_reserved(),
        final_precision=common.effective_precision(model,optimizer,require_gradients=True),
        compiler=common.compiler_audit(True,require_graphs=args.arm=='cdrm'))
    final=common.cpu_tree(snapshot());validate_position(final,train)
    if reference is not None:
        report['recovery_comparison']=compare_recovery(reference,final)
        common.atomic_json(args.output_dir/'recovery-comparison.json',report['recovery_comparison'])
        if not report['recovery_comparison']['bitwise_state_and_nontiming_metrics_equal']:raise AssertionError('Exact recovery failed; differences retained')
    common.verify_sources(tracked)
    if common.sources([args.protocol])!=tracked:raise RuntimeError('Training dependency inventory changed')
    if common.reference(args.initial_checkpoint)!=initial_ref:raise RuntimeError('Initial checkpoint changed')
    reloaded_train,reloaded_dev,final_data=data_contract(args)
    if final_data!=data:raise RuntimeError('Native corpus changed during training')
    report['status']='complete'


def main(argv=None):
    args=parse_args(argv)
    if args.mode=='prepare-init':
        data_identity=None
        if args.data_manifest:
            manifest=json.loads(args.data_manifest.read_text())
            if (manifest.get('status')!='complete' or manifest.get('role')!='calibration'
                    or manifest['protocol']['sha256']!=common.file_digest(args.protocol)):
                raise ValueError('Initialization data identity requires completed calibration under the same protocol')
            data_identity=common.reference(args.data_manifest)
        return common.save_initial_pair(args.output_dir,args.seed,args.protocol,data_identity)
    if args.output_dir.exists():raise FileExistsError('Use a new output directory')
    args.output_dir.mkdir(parents=True)
    report={'schema':SCHEMA,'status':'running','scope':'Native fuzzy-recall development calibration only; no final data',
        'arm':args.arm,'precision':args.precision,'command':shlex.join([sys.executable,*sys.argv]),
        'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}}
    tracker=OnlineTracker(project=args.wandb_project,entity=args.wandb_entity,group=args.wandb_group,
        name=args.wandb_run_name,output_dir=args.output_dir,preserve_state=common.preserve_tracking_rng)
    report['wandb']=tracker.record;start=time.monotonic()
    try:
        run(args,report,tracker)
        last_development=report['development'][str(max(map(int,report['development'])))]
        tracker.summary({'experiment_status':report['status'],**scalar_metrics(last_development,'last_dev'),
                        'completed_updates':report['completed_updates'],'training_seconds':report['training_seconds']})
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        try:tracker.finish(succeeded=report['status']=='complete')
        except Exception as error:
            report.update(status='failed',final_sync_error_type=type(error).__name__);raise
        finally:
            report['elapsed_seconds']=time.monotonic()-start;common.atomic_json(args.output_dir/'report.json',report)


if __name__=='__main__':main()
