#!/usr/bin/env python3
"""CPU-only saved evidence for a six-layer reassigned-head pilot or exact continuation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, _digest_dict, local_path, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, read_input, write_json
from scripts.rt_a5_six_layer_input_report import baseline_evidence, bound, require


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'rt-a5-six-layer-head-report-v1'
TRAIN_SCHEMA = 'rt-a5-six-layer-head-training-v1'
SOURCE_SHA = 'af1fae94e25dfc6982b549157d47ed9492e0e2896a368741a5ec6c39274a434f'
COLUMNS = CSV_COLUMNS
REPORTING_SOURCES = ('scripts/rt_a5_six_layer_head_report.py', 'scripts/rt_a5_six_layer_input_report.py',
    'scripts/rt_a5_nextlat_report.py', 'scripts/rt_a5_length_report.py', 'scripts/rt_a5_report.py',
    'scripts/experiment_tracking.py')


def terminal(report, protocol):
    start, endpoint = protocol['start_update'], protocol['endpoint']
    require((start,endpoint) in ((0,10000),(10000,50000)), 'Unknown head pilot/continuation stage')
    end = report.get('completed_updates')
    require(report.get('schema') == TRAIN_SCHEMA and report.get('start_update') == start
        and report.get('endpoint') == endpoint and type(end) is int and start < end <= endpoint
        and report.get('wandb',{}).get('status') == 'synced', 'Require a finished synced stage')
    require(report.get('confirmation_evaluated') is False and report.get('latent_rollout_evaluated') is False,
        'Confirmation and autonomous rollout must remain unused')
    if report.get('status') == 'complete':
        require(end == endpoint and report.get('requested_endpoint_reached') is True, 'Complete means the actual requested endpoint')
    else:
        stop = report.get('stop_request') or {}
        require(report.get('status') == 'stopped' and end < endpoint and report.get('requested_endpoint_reached') is False
            and stop.get('reason') == 'user_stop_file' and stop.get('observed_after_update') == end
            and local_path(stop['path']).resolve() == local_path(protocol['resolved_args']['stop_file']).resolve(),
            'Require the bound graceful user stop')
        require(hash_file(local_path(stop['path']))['sha256'] == stop['sha256'], 'Stop sentinel changed')
    require(report['embedding_head_enabled'] is True and 'injection_coefficient' not in report and 'schedule' not in report, 'Terminal head route differs')
    return end


def validate_history(history, end, reference):
    require(len(history) == end and len(reference) >= end, 'Exact combined history length differs')
    for update,(row,old) in enumerate(zip(history,reference),1):
        require(row['update'] == old['update'] == update and row['order_chain'] == old['order_chain']
            and row['examples_seen'] == update*1024 and row['embedding_head_enabled'] is True
            and 'injection_coefficient' not in row, 'Global history order/exposure/head route differs')
        require(all(isinstance(row[k],(int,float)) and math.isfinite(row[k])
            for k in ('loss','state_loss','latent_loss','weighted_latent_loss','grad_norm','seconds')), 'Nonfinite history')
        require(row['weighted_latent_loss'] == row['latent_loss']
            and math.isclose(row['loss'],row['state_loss']+row['latent_loss'],rel_tol=2e-6,abs_tol=1e-7), 'Original NextLat objective differs')


def load_evidence(lineage):
    lineage = local_path(lineage).resolve()
    inputs, payloads = {}, {}
    def capture(name,path):
        raw,record = read_input(path); inputs[name],payloads[name] = record,raw
        return raw
    protocol = json.loads(capture('protocol',lineage/'protocol.json'))
    variant = protocol['variant']; start,endpoint = protocol['start_update'],protocol['endpoint']
    require(protocol['schema'] == ('rt-a5-six-layer-head-protocol-v1' if start == 0 else 'rt-a5-six-layer-head-extension-protocol-v1')
        and variant == 'embedding_head' and protocol['n_layers'] == 6
        and protocol['embedding_head_layer'] == 1 and protocol['embedding_head_index'] == 7
        and protocol['embedding_head_enabled'] is True and protocol['additional_parameters'] == 0
        and protocol['window_layer'] == 0 and protocol['window_length'] == 2,
        'Unexpected reassigned-head architecture/protocol')
    require(len(protocol['source_files']) == 66 and _digest_dict(protocol['source_files']) == SOURCE_SHA == protocol['source_sha256'],
        'Frozen 66 source identity differs')
    contract,initial = protocol['strict_contract'],protocol['initialization']
    require(contract['source_sha256'] == SOURCE_SHA and contract['variant'] == variant
        and contract['n_layers'] == 6 and contract['embedding_head_enabled'] is True
        and contract['embedding_head_layer'] == 1 and contract['embedding_head_index'] == 7
        and 'injection_schedule' not in contract and 'injection_coefficient' not in contract,
        'Reassigned-head strict contract differs')
    require(initial['parameter_count'] == 19998208 and initial['parameter_tensors'] == 61
        and initial['baseline_parameter_tensors_changed'] == [] and initial['additional_parameter_count'] == 0
        and initial['embedding_head_layer'] == 1 and initial['embedding_head_index'] == 7
        and initial['model_parameter_sha256'] == initial['shared_six_layer_model_sha256']
        and contract['precision'] == 'fp32' and contract['batch_size'] == 1024
        and contract['objective']['latent_weight'] == 1 and contract['evaluation_route'] == 'backbone_only',
        'Exact61-parameter initialization, objective or evaluation route differs')
    directory = local_path(protocol['training_directory']).resolve()
    require(directory == lineage/'train-head', 'Training directory differs')
    report = json.loads(capture('training_report',directory/'report.json'))
    end = terminal(report,protocol)
    stages = [(lineage,protocol,report)]
    decision = None
    if start:
        parent_protocol_record = bound(protocol['parent_protocol'])
        parent_lineage = local_path(parent_protocol_record['path']).parent
        parent_protocol = json.loads(capture('parent_protocol',parent_lineage/'protocol.json'))
        parent_directory = local_path(parent_protocol['training_directory'])
        bound(protocol['parent_report'],parent_directory/'report.json')
        parent_report = json.loads(capture('parent_report',parent_directory/'report.json'))
        require(terminal(parent_report,parent_protocol) == 10000 and parent_report['status'] == 'complete'
            and parent_protocol['start_update'] == 0 and parent_report['parent_checkpoint'] is None,
            'Require the complete fresh 10k parent')
        require(parent_protocol['strict_contract'] == contract and parent_protocol['initialization'] == initial
            and parent_protocol['source_files'] == protocol['source_files'], 'Continuation source/initialization/contract differs')
        checkpoint = bound(protocol['parent_checkpoint'],parent_directory/'checkpoints/step-010000.pt')
        require(report['parent_checkpoint']['sha256'] == checkpoint['sha256']
            and local_path(report['parent_checkpoint']['path']).resolve() == local_path(checkpoint['path']).resolve(),
            'Reported exact resume checkpoint differs')
        bound(protocol['parent_validation'],parent_lineage/'final-state-validation.json')
        bound(protocol['continuation_decision'])
        decision = json.loads(capture('continuation_decision',local_path(protocol['continuation_decision']['path'])))
        require(decision['decision'] == 'continue_to50000' and decision['E36_count'] > 0
            and decision['words'] == 102400 and decision['checkpoint']['sha256'] == checkpoint['sha256'],
            'Positive full 10k continuation decision differs')
        stages.insert(0,(parent_lineage,parent_protocol,parent_report))
    else:
        require(report['parent_checkpoint'] is None and protocol['resolved_args']['resume'] is None, 'Fresh pilot cannot have a parent')
    history,checkpoints,metrics,curves,rows = [],{},{},{},[]
    stage_records = []
    for index,(stage_lineage,stage_protocol,stage_report) in enumerate(stages):
        label = f'stage{index}'; stage_directory = local_path(stage_protocol['training_directory'])
        stage_end = terminal(stage_report,stage_protocol)
        config = json.loads(capture(label+'_config',stage_directory/'config.json'))
        require(config == stage_protocol['resolved_args'] and stage_report['contract'] == contract
            and stage_report['initialization'] == initial and stage_report['source_files'] == protocol['source_files'],
            'Executed stage differs from its protocol')
        require(capture(label+'_exit',stage_lineage/'training-exit-code.txt').strip() == b'0', 'Successful process exit required')
        if stage_report['status'] == 'stopped':
            capture(label+'_stop',local_path(stage_report['stop_request']['path']))
        for name,digest in protocol['source_files'].items():
            require(hash_file(ROOT/name)['sha256'] == hash_file(stage_directory/'source'/name)['sha256'] == digest,
                'Frozen source or executed snapshot changed')
        expected = sorted(set(step for step in stage_protocol['checkpoint_steps'] if step <= stage_end) | {stage_end})
        require([c['completed_updates'] for c in stage_report['checkpoints']] == expected, 'Stage checkpoint sequence differs')
        for record in stage_report['checkpoints']:
            step = record['completed_updates']
            require(str(step) not in checkpoints and record['embedding_head_enabled'] is True and 'injection_coefficient' not in record
                and record['examples_seen'] == step*1024, 'Duplicate checkpoint or head route/exposure differs')
            checkpoints[str(step)] = bound(record,stage_directory/f'checkpoints/step-{step:06d}.pt')
        observed = set()
        for metric in stage_report['evaluations']:
            step,role = metric['update'],metric['role']
            require((step,role) not in observed and metric['route'] == 'backbone_only'
                and metric['embedding_head_enabled'] is True and 'injection_coefficient' not in metric
                and metric['rows'] == (102400 if step in expected else 4096), 'Evaluation pool/head route differs')
            observed.add((step,role)); converted = metric_rows(metric,'embedding_head')
            rows.extend(converted)
            if step in expected:
                metrics.setdefault(str(step),{})[role] = metric
                curves.setdefault(str(step),{})[role] = converted
        evaluation_steps = set(range(stage_protocol['start_update']+500,stage_end+1,500)) | {s for s in expected if s > 0}
        require(observed == {(s,role) for s in evaluation_steps for role in ROLES}, 'Missing or extra saved evaluations')
        part = [json.loads(line) for line in capture(label+'_history',stage_directory/'history.jsonl').splitlines()]
        require(len(part) == stage_end-stage_protocol['start_update'], 'Stage history length differs')
        history.extend(part)
        state = json.loads(capture(label+'_saved_state',stage_lineage/'final-state-validation.json'))
        require(state['schema'] == 'rt-a5-six-layer-head-final-state-validation-v1' and state['passed'] is True
            and state['completed_updates'] == stage_end and state['source_sha256'] == SOURCE_SHA
            and state['protocol']['sha256'] == hash_file(stage_lineage/'protocol.json')['sha256']
            and state['report']['sha256'] == hash_file(stage_directory/'report.json')['sha256']
            and state['model_parameters'] == 19998208 and state['model_parameter_tensors'] == state['final_active_adam_states'] == 61
            and state['embedding_head_enabled'] is True and state['embedding_head_layer'] == 1 and state['embedding_head_index'] == 7
            and state['all61_baseline_initial_tensors_bitwise_equal'] is True
            and state['shared_six_layer_model_sha256'] == initial['shared_six_layer_model_sha256'], 'Saved-state proof differs')
        require(all(state['checkpoint_inputs'][str(step)]['sha256'] == checkpoints[str(step)]['sha256'] for step in expected),
            'Saved-state proof checked different checkpoints')
        stage_records.append({'protocol':hash_file(stage_lineage/'protocol.json'),'report':hash_file(stage_directory/'report.json'),
            'start_update':stage_protocol['start_update'],'completed_updates':stage_end,'saved_state':state})
    for key in ('preflight','data_manifest','request'):
        bound(protocol[key]); capture(key,local_path(protocol[key]['path']))
    require(json.loads(payloads['preflight'])['passed'] is True
        and inputs['data_manifest']['sha256'] == contract['data_manifest_sha256'], 'Preflight/data proof differs')
    reference = protocol['order_reference']
    for key in ('report','history'): bound(reference[key])
    order_report = json.loads(capture('order_reference_report',local_path(reference['report']['path'])))
    order_history = [json.loads(line) for line in capture('order_reference_history',local_path(reference['history']['path'])).splitlines()]
    require(order_report['status'] == 'complete' and order_report['completed_updates'] >= end, 'Original order reference incomplete')
    for key in ('batch_size','length','train_rows','data_order_seed','data_manifest_sha256'):
        require(order_report['contract'][key] == contract[key], 'Order-reference data differs')
    validate_history(history,end,order_history)
    require(history[-1]['order_chain'] == report['order_chain'], 'Terminal history/report order differs')
    baseline = baseline_evidence(capture,report,curves,order_history[:10000],protocol['reference'])
    # The frozen baseline reader also serves injection reports. Remove its
    # compatibility-only scalar fields: this model has no scalar bypass.
    for row in baseline['metric_rows']:
        row.pop('injection_coefficient', None)
    for point in baseline['comparisons']:
        point['head'] = point.pop('injection')
        point['head'].pop('coefficient', None)
        point['baseline'].pop('coefficient', None)
    rows.extend(baseline['metric_rows'])
    summary={'schema':SCHEMA,'variant':variant,'through_update':end,'requested_endpoint':endpoint,
        'training_status':report['status'],'requested_endpoint_reached':report['requested_endpoint_reached'],
        'embedding_head_enabled':True,'embedding_head_layer':1,'embedding_head_index':7,'additional_parameters':0,'checkpoint_updates':sorted(map(int,checkpoints)),
        'checkpoints':checkpoints,'metrics':metrics,'curves':curves,'metric_rows':rows,'training_curve':training_curve(history,augmented=True),
        'baseline':baseline,'common_checkpoint_updates':baseline['common_checkpoint_updates'],
        'matched_update':max(baseline['common_checkpoint_updates'],default=None),'actual_endpoints':{'head':end,'baseline':10000},
        'protocol':protocol,'stages':stage_records,'inputs':inputs,'training_wandb':report['wandb'],
        'continuation_decision':decision,'matched_minibatch_order_hashes':end,'confirmation_evaluated':False,'latent_rollout_evaluated':False,
        'qualification':'Single-seed reused-development diagnostic selected after earlier results. All 61 baseline initial tensors and the data order match exactly; both models contain 19,998,208 parameters. One existing head (index 7 in block index 1) writes its permanent values from raw token embeddings using its existing value-projection rows. It replaces that head’s contextual permanent values; it does not add an attention head, projection or scalar gate. Permanent keys, other heads and provisional self KV remain contextual and unchanged. The positive full 10k E36 gate permits an exact 50k continuation. Only common retained budgets are matched comparisons; unequal actual endpoints are descriptive. No confirmation or autonomous latent rollout was evaluated.'}

    return summary,payloads


def plot_rows(summary,step,arm='head'):
    view = summary if arm == 'head' else summary['baseline']
    rows = view['curves'][str(step)]['ood_dev']
    require([(r['update'],r['role'],r['length']) for r in rows] == [(step,'ood_dev',t) for t in range(1,37)],
        'Full and boundary require the identical 36-prefix rows')
    return rows


def plots(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    labels={'baseline':'Original six-layer baseline','head':'Six-layer reassigned embedding head'}
    colors={'baseline':'#3574B2','head':'#D65B35'}; figures=[]
    def save(fig,name):
        for suffix in ('png','pdf'): fig.savefig(output/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
        figures.append(name); plt.close(fig)
    def prefix(name,limits,selection,title):
        fig,axes=plt.subplots(1,3,figsize=(14.5,4.8));fig.subplots_adjust(left=.08,right=.99,bottom=.18,top=.79,wspace=.4)
        for axis,key in zip(axes,('E','A','M')):
            for arm,step in selection:
                rows=plot_rows(summary,step,arm)
                axis.plot([r['length'] for r in rows],[r[key] for r in rows],color=colors[arm],label=f'{labels[arm]}: {step:,} updates')
            axis.set(xlim=limits,ylim=(-.025,1.025),xlabel='Prefix t of the same 36-token words',title=f'{key}(t)')
            axis.axvline(12,color='gray',linestyle=':');axis.yaxis.set_major_formatter(PercentFormatter(1));axis.grid(alpha=.2);axis.legend(fontsize=7)
        fig.suptitle(title+' · 102,400 words',y=.97);save(fig,name)
    end,matched=summary['through_update'],summary['matched_update']
    selected=[('baseline',matched),('head',matched)] if matched is not None else [('head',end)]
    title=f'Matched budget: {matched:,} updates' if matched is not None else 'Head endpoint; no shared retained checkpoint yet'
    prefix('length-full',(1,36),selected,title);prefix('length-boundary',(10,18),selected,title)
    if end != 10000:
        prefix('length-unequal-terminals',(1,36),[('baseline',10000),('head',end)],f'Unequal endpoints: baseline 10,000 and head {end:,}; descriptive only')
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.6),layout='constrained')
    for axis,key in zip(axes,('E','A','M')):
        for arm in labels:
            view=summary if arm=='head' else summary['baseline'];steps=view['checkpoint_updates'][1:]
            axis.plot(steps,[plot_rows(summary,s,arm)[-1][key] for s in steps],marker='o',color=colors[arm],label=labels[arm])
        axis.set(xlabel='Completed updates',ylabel=f'{key}(36)',ylim=(-.025,1.025));axis.yaxis.set_major_formatter(PercentFormatter(1));axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.suptitle('Full development pools at actual retained checkpoints');save(fig,'length36-vs-updates')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
    for axis,key,title in zip(axes,('state_ce','latent_loss'),('State CE','NextLat SmoothL1')):
        for arm in labels:
            view=summary if arm=='head' else summary['baseline'];bins=view['training_curve']
            axis.plot([b['update'] for b in bins],[b[key] for b in bins],color=colors[arm],label=labels[arm])
        axis.set(xlabel='Completed updates',ylabel='Loss',title=title+', 100-update means');axis.grid(alpha=.2);axis.legend(fontsize=8)
    save(fig,'training-losses')
    return figures


def markdown(summary):
    end=summary['through_update'];matched=summary['matched_update']
    lines=[f'# Six-layer reassigned embedding head through {end:,} updates','',
        f"Actual status: {summary['training_status']}; requested stage endpoint: {summary['requested_endpoint']:,}.",'',
        'The original six-layer architecture is retained: block index 0 has a two-token attention window and blocks 1–5 use full recurrent attention. In block 1, the existing eighth head (index 7) replaces its permanent values with the existing value-projection rows applied to raw token embeddings. Contextual keys, the other seven heads and temporary self KV remain unchanged. There is no added parameter or scalar gate. Both models have 19,998,208 parameters.','',
        'E(t): every state through t correct. A(t): state t alone correct. M(t): mean token accuracy through t. All full/boundary curves use identical prefixes from the same 102,400 length-36 words.','',
        '| Head update | E(36) | A(36) | M(36) |','|---:|---:|---:|---:|']
    for step in summary['checkpoint_updates'][1:]:
        row=plot_rows(summary,step)[-1];lines.append(f"| {step:,} | {row['E']:.4%} | {row['A']:.4%} | {row['M']:.4%} |")
    lines += ['',f'Latest shared retained budget: {matched:,} updates.' if matched is not None else 'No shared retained trained checkpoint is available.',
        '', '| Shared update | Baseline E(36) | Head E(36) | Baseline M(36) | Head M(36) |','|---:|---:|---:|---:|---:|']
    for point in summary['baseline']['comparisons']:
        a,b=point['baseline'],point['head'];lines.append(f"| {point['update']:,} | {a['E']:.4%} | {b['E']:.4%} | {a['M']:.4%} | {b['M']:.4%} |")
    if end != 10000: lines += ['',f'Actual endpoints differ: head {end:,}, baseline 10,000. [Unequal terminal curves](length-unequal-terminals.pdf) are descriptive.']
    lines += ['',summary['qualification'],'','[Full](length-full.pdf) · [Boundary](length-boundary.pdf) · [E/A/M36 trajectory](length36-vs-updates.pdf) · [Training losses](training-losses.pdf)',
        '',f"[Training W&B]({summary['training_wandb']['run_url']})",'']
    return '\n'.join(lines)


def run(args):
    summary,payloads=load_evidence(args.lineage)
    output=local_path(args.output_dir or summary['protocol']['report_directory']).resolve()
    require(not output.exists(),'Use a fresh report output directory');output.mkdir(parents=True);(output/'inputs').mkdir()
    for name,raw in payloads.items():
        (output/'inputs'/f"{name}{Path(summary['inputs'][name]['path']).suffix}").write_bytes(raw)
    directory=local_path(summary['protocol']['training_directory'])
    for name in summary['protocol']['source_files']:
        destination=output/'training-source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(directory/'source'/name,destination)
    for name in REPORTING_SOURCES:
        destination=output/'reporting-source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    write_json(output/'summary.json',summary)
    with (output/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS);writer.writeheader();writer.writerows(summary['metric_rows'])
    bins=[{**b,'arm':'head'} for b in summary['training_curve']]
    bins += [{**b,'arm':'baseline'} for b in summary['baseline']['training_curve']]
    with (output/'training-curve.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({key for b in bins for key in b}));writer.writeheader();writer.writerows(bins)
    figures=plots(summary,output);(output/'README.md').write_text(markdown(summary))
    tracker=OnlineTracker(project='rt-a5-state-tracking',entity='taylorbollman',output_dir=output,
        group=args.wandb_group or local_path(args.lineage).name,name=f"six-layer-head-report-{summary['through_update']}")
    result={'schema':SCHEMA,'status':'running','variant':summary['variant'],'through_update':summary['through_update'],
        'matched_update':summary['matched_update'],'actual_endpoints':summary['actual_endpoints'],'figures':figures}
    try:
        tracker.start({'schema':SCHEMA,'variant':summary['variant'],'embedding_head_layer':1,'embedding_head_index':7,'additional_parameters':0,'through_update':summary['through_update'],'qualification':summary['qualification']})
        import wandb
        tracker.log({'report/metrics':wandb.Table(columns=list(COLUMNS),data=[[r[k] for k in COLUMNS] for r in summary['metric_rows']]),
            **{f'report/{name}':wandb.Image(str(output/f'{name}.png')) for name in figures}})
        for step in summary['checkpoint_updates'][1:]:
            row=plot_rows(summary,step)[-1];tracker.log({'update':step,**{f'dev/ood_dev/{k}36':row[k] for k in ('E','A','M')}})
        tracker.summary({'through_update':summary['through_update'],'matched_update':summary['matched_update']})
        tracker.finish(succeeded=True);result['status']='complete'
        (output/'README.md').write_text(markdown(summary)+f"\n[Report W&B]({tracker.record['run_url']})\n")
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__)
        try:tracker.finish(succeeded=False)
        except Exception:pass
        raise
    finally:
        result['wandb']=tracker.record
        result['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file() and 'wandb' not in p.relative_to(output).parts and p != output/'report.json'}
        write_json(output/'report.json',result)
    print(json.dumps({'status':result['status'],'output_dir':str(output),'wandb':tracker.record['run_url']}));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--lineage',required=True)
    parser.add_argument('--output-dir');parser.add_argument('--wandb-group');run(parser.parse_args())


if __name__ == '__main__':
    main()
