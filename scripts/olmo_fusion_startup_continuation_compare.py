#!/usr/bin/env python3
"""CPU-only descriptive comparison of completed, SHA-pinned continuations."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from cdrm.pretrained.artifacts import sha256_file, write_json
from scripts.olmo_fusion_startup_update_probe import difference, geometry
from scripts.olmo_lm_common import tree_digests


def read_report(path,digest):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or sha256_file(path)!=digest:
        raise ValueError('Report differs from independent SHA256')
    value=json.loads(path.read_text())
    if (sha256_file(path)!=digest or value.get('schema')!='olmo-fusion-startup-continuation-v1'
            or value.get('status')!='completed_segment' or value.get('passed') is not True
            or not value.get('integrity') or not all(value['integrity'].values())):
        raise ValueError('Require a completed operationally passing continuation')
    return value


def checkpoint(report,report_path,step):
    records=[row for row in report['checkpoints'] if row['optimizer_updates']==step]
    if len(records)!=1:raise ValueError('Missing or ambiguous required checkpoint')
    record=records[0]
    # Recorded paths may use the container mount. Resolve only its basename
    # beside the pinned report, then verify the independently recorded digest.
    path=Path(report_path).parent/Path(record['path']).name
    if path.is_symlink() or sha256_file(path)!=record['sha256']:raise ValueError('Checkpoint pin differs')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if sha256_file(path)!=record['sha256'] or payload['counters']['optimizer_updates']!=step:
        raise ValueError('Checkpoint changed or has the wrong completed step')
    if payload['configuration']!=report['configuration'] or payload['source_fingerprint']!=report['source_fingerprint']:
        raise ValueError('Checkpoint belongs to another continuation')
    return payload


def moments(payload,key):
    groups=payload['optimizer']['param_groups'];ownership=payload['optimizer_ownership']
    if len(groups)!=1 or ownership!=[['state_proj.weight','token_gate.weight']]:
        raise ValueError('Unexpected fusion Adam ownership')
    return {name:payload['optimizer']['state'][identifier][key]
        for identifier,name in zip(groups[0]['params'],ownership[0])}


def compare(fp,bf,fp_path,bf_path,replay=None):
    checks={'paired_precisions':fp['precision']=='fp32' and bf['precision']=='bf16_mixed',
        'same_sources':fp['sources']==bf['sources'],
        'same_origin':fp['configuration']['origin_checkpoint_sha256']==bf['configuration']['origin_checkpoint_sha256'],
        'same_data':fp['configuration']['data_manifest_sha256']==bf['configuration']['data_manifest_sha256'],
        'same_fixture':fp['configuration']['heldout_fixture_sha256']==bf['configuration']['heldout_fixture_sha256'],
        'same_fixed_plan':fp['plan']==bf['plan'],
        'same_initial_boundary':fp['starting_boundary']==bf['starting_boundary'],
        'same_start_end':fp['starting_update']==bf['starting_update']==128 and
            fp['final_counters']==bf['final_counters'] and fp['final_counters']['optimizer_updates']==144,
        'sixteen_updates_each':fp['physical_optimizer_updates']==bf['physical_optimizer_updates']==16 and
            len(fp['updates'])==len(bf['updates'])==16}
    training=[]
    for a,b in zip(fp['updates'],bf['updates']):
        a,b=a['exact'],b['exact'];am,bm=a['metrics'],b['metrics'];step=am['update']
        checks[f'update{step}/matched_data_schedule']=(step==bm['update'] and a['input_pins']==b['input_pins']
            and a['data_metadata']==b['data_metadata'] and am['counters']==bm['counters']
            and am['lr_used']==bm['lr_used'] and am['lr_next']==bm['lr_next'])
        checks[f'update{step}/finite_observations']=all(math.isfinite(float(value)) for m in (am,bm)
            for value in (m['objective'],m['gradient_norm_before_clip'],m['actual_master_delta_geometry']['all']['norm'],*m['pass_ce_sums']))
        training.append({'update':step,'fp32_ce':am['objective'],'bf16_ce':bm['objective'],
            'bf16_minus_fp32_ce':bm['objective']-am['objective'],
            'fp32_raw_gradient_norm':am['gradient_norm_before_clip'],'bf16_raw_gradient_norm':bm['gradient_norm_before_clip'],
            'fp32_master_delta_norm':am['actual_master_delta_geometry']['all']['norm'],
            'bf16_master_delta_norm':bm['actual_master_delta_geometry']['all']['norm'],
            'fp32_clip_scale':am['clip_scale'],'bf16_clip_scale':bm['clip_scale']})
    evaluations=[]
    checks['evaluation_clocks']=[r['update'] for r in fp['evaluations']]==[r['update'] for r in bf['evaluations']]==[128,136,144]
    for a,b in zip(fp['evaluations'],bf['evaluations']):
        checks[f'eval{a["update"]}/scope']=a['counts']==b['counts'] if 'counts' in a else a['ce_targets']==b['ce_targets']
        evaluations.append({'update':a['update'],'fp32_trajectory_common_fp32_ce':a['ce_mean'],
            'bf16_trajectory_common_fp32_ce':b['ce_mean'],'bf16_minus_fp32_ce':b['ce_mean']-a['ce_mean']})
    origin_fp,origin_bf=checkpoint(fp,fp_path,128),checkpoint(bf,bf_path,128)
    checks['saved_origin_exact']=all(tree_digests(origin_fp[key])==tree_digests(origin_bf[key])
        for key in ('model','optimizer','scheduler','rng','counters','data_cursor'))
    parameter_names=('state_proj.weight','token_gate.weight')
    origin={name:origin_fp['model'][name] for name in parameter_names}
    endpoints=[]
    for step in (136,144):
        a,b=checkpoint(fp,fp_path,step),checkpoint(bf,bf_path,step)
        checks[f'checkpoint{step}/matching_clocks']=a['counters']==b['counters'] and a['scheduler']==b['scheduler'] and a['data_cursor']==b['data_cursor']
        checks[f'checkpoint{step}/fixed_scale']=tree_digests(a['model']['output_scale'])==tree_digests(b['model']['output_scale'])==tree_digests(origin_fp['model']['output_scale'])
        ap,bp=({name:payload['model'][name] for name in parameter_names} for payload in (a,b))
        endpoints.append({'update':step,'parameter_separation':geometry(bp,ap),
            'cumulative_delta_from128':geometry(difference(bp,origin),difference(ap,origin)),
            'adam_exp_avg':geometry(moments(b,'exp_avg'),moments(a,'exp_avg')),
            'adam_exp_avg_sq':geometry(moments(b,'exp_avg_sq'),moments(a,'exp_avg_sq'))})
    if replay is not None:
        checks['replay_eight_calls']=replay['starting_update']==136 and replay['physical_optimizer_updates']==8 and len(replay['updates'])==8
        checks['replay_same_contract']=replay['configuration']==bf['configuration'] and replay['sources']==bf['sources']
        golden={row['exact']['metrics']['update']:row['exact'] for row in bf['updates']}
        checks['replay_all_updates_bitwise']=all(row['exact']==golden[row['exact']['metrics']['update']] for row in replay['updates'])
        checks['replay_all_evaluations_bitwise']=replay['evaluations']==bf['evaluations'][1:]
        checks['replay_final_boundary_bitwise']=replay['final_boundary']==bf['final_boundary']
    if not all(checks.values()):raise ValueError('Comparison controls failed: '+','.join(k for k,v in checks.items() if not v))
    return {'checks':checks,'training':training,'common_fp32_evaluations':evaluations,'endpoint_geometry':endpoints,
        'exact_replay_included':replay is not None,
        'qualification':'Descriptive 16-update fusion-only comparison. After the first update, differences include trajectory divergence. No full-backbone, auxiliary, RT, packed or production clearance.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fp32-report','bf16-report','replay-report'):
        p.add_argument('--'+name,type=Path,required=name!='replay-report')
        p.add_argument('--'+name+'-sha256',required=name!='replay-report')
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if (a.replay_report is None)!=(a.replay_report_sha256 is None):p.error('Replay requires both path and SHA256')
    torch.set_num_threads(4)
    fp=read_report(a.fp32_report,a.fp32_report_sha256);bf=read_report(a.bf16_report,a.bf16_report_sha256)
    replay=read_report(a.replay_report,a.replay_report_sha256) if a.replay_report else None
    result=compare(fp,bf,a.fp32_report,a.bf16_report,replay)
    result.update(schema='olmo-fusion-continuation-comparison-v1',passed=True,
        analysis_source_sha256=sha256_file(Path(__file__)),
        report_pins={'fp32':a.fp32_report_sha256,'bf16':a.bf16_report_sha256,'replay':a.replay_report_sha256})
    a.output_dir.mkdir(parents=True,exist_ok=False);write_json(a.output_dir/'report.json',result)
    print(json.dumps({'passed':True,'checks':len(result['checks']),'replay_included':replay is not None,
        'report_sha256':sha256_file(a.output_dir/'report.json')}))


if __name__=='__main__':main()
