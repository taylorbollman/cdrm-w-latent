#!/usr/bin/env python3
"""Bounded fixed-state diagnosis of F4's retained RT+FBT coordinate-screen failure."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from cdrm.pretrained.artifacts import sha256_file,write_json
from cdrm.pretrained.static_training import StaticFBTTraining
from cdrm.pretrained.lm_training import LMTrainingConfig
from scripts.olmo_f4_resources import (SOURCES as F4_SOURCES, selected_case, build_model,
    changed_batch, configure_determinism, require_container_gpu, validate_prepared_manifest,
    load_native_state_dict, load_native_tokenizer, backend_context, OnlineTracker,
    compare_graph, full_update_parity)
from scripts.olmo_f3e_validate import comparison_metrics
from scripts.olmo_f3d_validate import global_gradient_l2
from scripts.olmo_f3_graph_training import loss_snapshot
from scripts.olmo_lm_common import tensor_digest
SOURCES=tuple(sorted(set(F4_SOURCES)|{"scripts/olmo_f4_roundoff.py"}))
PROTOCOL=ROOT/'docs/reports/olmo1b-f4/roundoff-protocol.md'


def snapshot(plan, result):
    return {'losses':{n:v.detach().cpu().clone() for n,v in loss_snapshot(result).items()},
            'gradients':{n:p.grad.detach().cpu().clone() for n,p in plan.model.named_parameters() if p.grad is not None}}


def compare(candidate,reference):
    if any(candidate[k].keys()!=reference[k].keys() for k in ('losses','gradients')):
        raise AssertionError('Comparison parameter/loss ownership changed')
    result={key:{n:{**comparison_metrics(v,reference[key][n]),'bitwise_equal':torch.equal(v,reference[key][n]),
                        'finite':bool(torch.isfinite(v).all() and torch.isfinite(reference[key][n]).all())}
                 for n,v in candidate[key].items()} for key in ('losses','gradients')}
    result['all_bitwise_equal']=all(r['bitwise_equal'] for group in result.values() for r in group.values())
    result['global_gradient_relative_l2']=global_gradient_l2(result['gradients'].values())
    result['coordinate_screen_failures']=[n for n,r in result['gradients'].items() if r['max_relative']>1/16]
    result['worst_coordinates']={}
    for name in result['coordinate_screen_failures']:
        a,b=candidate['gradients'][name],reference['gradients'][name]
        index=int((a-b).abs().reshape(-1).argmax())
        result['worst_coordinates'][name]={'flat_index':index,'shape':list(a.shape),
            'candidate':float(a.reshape(-1)[index]),'reference':float(b.reshape(-1)[index]),
            'reference_tensor_max':float(b.abs().max())}
    result['finite']=all(r['finite'] for key in ('losses','gradients') for r in result[key].values())
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    args=p.parse_args(argv)
    determinism=configure_determinism(True);runtime=require_container_gpu()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name in SOURCES:
        dst=args.output_dir/'source-snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    shutil.copyfile(PROTOCOL,args.output_dir/'protocol.md')
    case=selected_case('rt-fbt',batch=8,length=512)
    report={'schema':'olmo-f4-roundoff-v1','status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
            'case':asdict(case),'runtime':{k:str(v) for k,v in runtime.items()},'determinism':determinism,
            'source_hashes':{n:sha256_file(ROOT/n) for n in SOURCES},'protocol_sha256':sha256_file(PROTOCOL),
            'arms':{},'comparisons':{},'graph_checks':[],'original_screen_cleared':False}
    tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',group='olmo1b-f4-features',
                          name='olmo-'+args.output_dir.name,output_dir=args.output_dir)
    def save():write_json(args.output_dir/'report.json',report)
    values={}
    try:
        report['checkpoint']=validate_prepared_manifest(args.artifacts)['checkpoint']
        tracker.start({'case':asdict(case),'checkpoint':report['checkpoint'],'scope':'retained-failure diagnosis'})
        print({'wandb':tracker.record['run_url']},flush=True)
        state=load_native_state_dict(args.artifacts);tokenizer=load_native_tokenizer(args.artifacts)
        for precision,memory,backward in (('bf16_mixed','materialized','triton'),('bf16_mixed','recompute','triton'),
                ('bf16_mixed','recompute','eager'),('fp32','materialized','eager'),('fp32','recompute','eager')):
            label=precision+'-'+memory+('-eager' if precision=='bf16_mixed' and backward=='eager' else '');torch.manual_seed(20260922)
            model=build_model(state,case);base=model.backbone.backbone
            base.ordinary_activation_checkpointing=True;base.cast_weights_once=True
            base.tile_backend='triton' if precision=='bf16_mixed' else 'eager';base.backward_tile_backend=backward
            base.backward_memory=memory;base.attention_precision='mixed' if precision=='bf16_mixed' else 'fp32'
            batch=changed_batch(tokenizer,case,0)
            identity={k:None if v is None else tensor_digest(v) for k,v in vars(batch).items()}
            if 'batch_identity' in report and report['batch_identity']!=identity:
                raise AssertionError('Diagnostic arms changed the fixed batch')
            report['batch_identity']=identity
            plan=StaticFBTTraining(model,batch,mode=case.mode(),config=LMTrainingConfig(precision=precision))
            with backend_context('flash' if precision=='bf16_mixed' else 'math'):
                print({'start':label},flush=True)
                result=plan.backward(replay=False);values[label]=snapshot(plan,result);del result
                report['arms'][label]={'losses':{k:float(v) for k,v in values[label]['losses'].items()},
                                      'gradient_tensor_count':len(values[label]['gradients'])}
                if precision=='bf16_mixed':
                    repeated=plan.backward(replay=False);repeat=snapshot(plan,repeated);del repeated
                    comparison=compare(repeat,values[label]);del repeat
                    report['arms'][label]['repeat']=comparison
                    if not comparison['all_bitwise_equal']:raise AssertionError(label+' repeat is not exact')
                if label=='bf16_mixed-recompute':
                    report['comparisons']['bf16_recompute_vs_materialized']=compare(values[label],values['bf16_mixed-materialized'])
                    save();plan.capture(warmup=10)
                    checks=[compare_graph(plan,'candidate_initial_graph')]
                    plan.load_batch(changed_batch(tokenizer,case,1))
                    checks.append(compare_graph(plan,'candidate_changed_tokens_and_overwrite',replays=2))
                    checks.append(full_update_parity(plan,tokenizer,case,updates=3))
                    plan.load_batch(changed_batch(tokenizer,case,2))
                    checks.append(compare_graph(plan,'candidate_changed_weights'))
                    report['graph_checks']=checks
                    if not all(c['passed'] and (not c['name'].startswith('candidate_') or c['all_bitwise_equal']) for c in checks):
                        raise AssertionError('Candidate graph/update path differs')
                if label=='bf16_mixed-recompute-eager':
                    report['comparisons']['bf16_eager_history_vs_materialized']=compare(values[label],values['bf16_mixed-materialized'])
                    report['comparisons']['bf16_fused_vs_eager_history']=compare(values['bf16_mixed-recompute'],values[label])
                if label=='fp32-materialized':
                    for name in ('bf16_mixed-materialized','bf16_mixed-recompute','bf16_mixed-recompute-eager'):
                        report['comparisons'][name+'_vs_fp32']=compare(values[name],values[label])
                        del values[name]
                if label=='fp32-recompute':
                    report['comparisons']['fp32_recompute_vs_materialized']=compare(values[label],values['fp32-materialized'])
                print({'completed':label},flush=True);save()
            del plan,model,base,batch;gc.collect();torch.cuda.empty_cache()
        if not all(c['finite'] for c in report['comparisons'].values()):raise AssertionError('Nonfinite diagnostic')
        assert report['source_hashes']=={n:sha256_file(ROOT/n) for n in SOURCES}
        assert report['protocol_sha256']==sha256_file(PROTOCOL)
        report['status']='completed_diagnostic'
        tracker.log({'diagnosis/original_screen_cleared':0,
                     **{key+'/global_gradient_relative_l2':v['global_gradient_relative_l2'] for key,v in report['comparisons'].items()}},step=1)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error_message=str(error));raise
    finally:
        tracker.finish(succeeded=report['status']=='completed_diagnostic')
        report.update(wandb=tracker.record,finished_utc=datetime.now(timezone.utc).isoformat());save()


if __name__=='__main__':main()
