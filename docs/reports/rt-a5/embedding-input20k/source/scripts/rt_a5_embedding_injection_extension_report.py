#!/usr/bin/env python3
"""Saved-report supplement for the input-injection model's exact 10k-to20k continuation."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import shutil

from scripts import rt_a5_embedding_injection_report as paired
from scripts import rt_a5_l1r_depth_extension_report as continuation
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, _digest_dict, local_path, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, read_input, write_json

ROOT = paired.ROOT
LINEAGE = ROOT / '.runtime/rt-a5/20260915T153500Z-embedding-input20k'
OUTPUT = ROOT / 'docs/reports/rt-a5/embedding-input20k'
SCHEMA = 'rt-a5-embedding-input-extension-comparison-v1'
STEPS = (1000, 5000, 10000, 15000, 20000)
COLORS = {10000: '#236B9E', 20000: '#B95535'}
REPORTING_SOURCES = ('scripts/rt_a5_embedding_injection_extension_report.py',
                     'scripts/rt_a5_l1r_depth_extension_report.py', *paired.REPORTING_SOURCES)


def validate_child(report, config, protocol, parent):
    if (report.get('schema') != 'rt-a5-embedding-injection-training-v1' or report.get('status') != 'complete'
            or report.get('start_update') != 10000 or report.get('endpoint') != 20000 or report.get('completed_updates') != 20000
            or report.get('wandb', {}).get('status') != 'synced' or report.get('confirmation_evaluated') is not False
            or report.get('latent_rollout_evaluated') is not False):
        raise ValueError('Require the complete synced input10k-to20k continuation')
    for key in ('contract', 'source_files', 'initialization'):
        expected = protocol['strict_contract' if key == 'contract' else key]
        if report.get(key) != parent[key] or report[key] != expected:
            raise ValueError(f'Continuation changed parent {key}')
    if (len(report['source_files']) != 62 or _digest_dict(report['source_files']) != protocol['source_sha256']
            or report['contract']['source_sha256'] != protocol['source_sha256']):
        raise ValueError('Continuation source62 identity differs')
    wanted = protocol['parent_checkpoint']; actual = report.get('parent_checkpoint') or {}
    if (actual.get('sha256') != wanted['sha256']
            or local_path(actual.get('path', '')).resolve() != local_path(wanted['path']).resolve()
            or config.get('resume') is None or local_path(config['resume']).resolve() != local_path(wanted['path']).resolve()
            or config != protocol['resolved_args']):
        raise ValueError('Continuation must resume the exact approved input10k checkpoint/configuration')
    if sorted(c['completed_updates'] for c in report['checkpoints']) != [15000, 20000]:
        raise ValueError('Retain only the two new15k/20k checkpoints')


def selected_evaluation(report, step, role):
    if step not in STEPS or role not in ROLES:
        raise ValueError('Use only declared full trajectory checkpoints and development roles')
    selected = [e for e in report['evaluations'] if e['update'] == step and e['role'] == role]
    if len(selected) != 1 or selected[0].get('rows') != 102400 or selected[0].get('route') != 'backbone_only':
        raise ValueError('Every trajectory point requires the same full102400-word evaluation')
    return selected[0], metric_rows(selected[0], 'input')


def make_summary():
    raw, protocol_file = read_input(LINEAGE / 'protocol.json'); protocol = json.loads(raw)
    if (protocol.get('schema') != 'rt-a5-embedding-injection-extension-protocol-v1'
            or protocol.get('start_update') != 10000 or protocol.get('endpoint') != 20000
            or protocol.get('checkpoint_steps') != [15000, 20000] or not protocol.get('qualification')
            or local_path(protocol['training_directory']).resolve() != LINEAGE / 'train-injection'
            or local_path(protocol['report_directory']).resolve() != OUTPUT):
        raise ValueError('Require the authorized separate input20k continuation protocol')
    for key in ('parent_protocol', 'parent_report', 'parent_checkpoint'):
        paired.base.bound_file(protocol[key])
    parent = paired.read_injection('input')
    if (parent['inputs']['protocol']['sha256'] != protocol['parent_protocol']['sha256']
            or parent['inputs']['report']['sha256'] != protocol['parent_report']['sha256']
            or parent['checkpoints']['10000']['sha256'] != protocol['parent_checkpoint']['sha256']):
        raise ValueError('Continuation parent bindings differ from the completed original 10k run')
    directory = LINEAGE / 'train-injection'
    raw, report_file = read_input(directory / 'report.json'); report = json.loads(raw)
    raw, config_file = read_input(directory / 'config.json'); config = json.loads(raw)
    validate_child(report, config, protocol, parent)
    if (LINEAGE / 'training-exit-code.txt').read_text().strip() != '0':
        raise ValueError('Require successful training exit before reporting')
    for name, digest in protocol['source_files'].items():
        if hash_file(ROOT / name)['sha256'] != digest or hash_file(directory / 'source' / name)['sha256'] != digest:
            raise ValueError(f'Frozen current/snapshot source changed: {name}')
    manifest = hash_file(local_path(config['data_dir']) / 'manifest.json')
    if manifest['sha256'] != report['contract']['data_manifest_sha256']:
        raise ValueError('Continuation data manifest changed')
    child_history = continuation.read_history(directory / 'history.jsonl', 10000, 20000, exact_file=True)
    reference = paired.base.read_arm(paired.base.REFERENCE, 'two_layers')
    reference_history = continuation.read_history(reference['inputs']['history']['path'], 0, 20000, exact_file=False)
    combined = continuation.stitch_histories(parent['history'], child_history, reference_history)
    if child_history[-1]['order_chain'] != report['order_chain']:
        raise ValueError('Continuation history/report endpoint order differs')
    checkpoints = copy.deepcopy(parent['checkpoints'])
    for step in (15000, 20000):
        checkpoints[str(step)] = continuation.select_checkpoint(report, directory, step)
    raw, state_file = read_input(LINEAGE / 'final-state-validation.json'); state = json.loads(raw)
    if (state.get('schema') != 'rt-a5-embedding-injection-extension-final-state-validation-v1'
            or state.get('passed') is not True or state.get('start_update') != 10000 or state.get('completed_updates') != 20000
            or state.get('model_parameters') != 13964800 or state.get('model_parameter_tensors') != 44
            or state.get('protocol', {}).get('sha256') != protocol_file['sha256']
            or state.get('report', {}).get('sha256') != report_file['sha256']):
        raise ValueError('Require the completed input20k saved-state check bound to this protocol/report')
    for step in (10000,15000,20000):
        if state['checkpoint_inputs'][str(step)]['sha256'] != checkpoints[str(step)]['sha256']:
            raise ValueError('Saved-state check uses a different parent/continuation checkpoint')
    if state['source_sha256'] != protocol['source_sha256']:
        raise ValueError('Saved-state check uses different training source')
    curves, metrics = copy.deepcopy(parent['curves']), copy.deepcopy(parent['metrics'])
    for step in (15000, 20000):
        curves[str(step)], metrics[str(step)] = {}, {}
        for role in ROLES:
            metrics[str(step)][role], curves[str(step)][role] = selected_evaluation(report, step, role)
    parent.pop('history')
    return {'schema': SCHEMA, 'primary_update': 20000, 'checkpoint_updates': list(STEPS), 'curves': curves, 'metrics': metrics,
            'parent': parent, 'protocol': protocol, 'protocol_input': protocol_file, 'checkpoints': checkpoints,
            'inputs': {'report': report_file, 'config': config_file, 'history': hash_file(directory / 'history.jsonl'),
                       'data_manifest': manifest, 'saved_state': state_file, 'order_reference_history': reference['inputs']['history']},
            'training_curve': training_curve(combined, augmented=True), 'history_segments': [[1,10000],[10001,20000]],
            'matched_minibatch_order_hashes': 20000, 'continuation_wandb': report['wandb'],
            'four_layer_uninjected_control_available': False, 'confirmation_evaluated': False, 'latent_rollout_evaluated': False,
            'scope': 'Input injection only: exact10k-to20k adaptive extension; full1k/5k/10k/15k/20k development trajectory'}


def plot_rows(summary, step):
    if step not in STEPS:
        raise ValueError('Unknown input trajectory checkpoint')
    rows = summary['curves'][str(step)]['ood_dev']
    if [(r['arm'], r['update'], r['role'], r['length']) for r in rows] != [('input',step,'ood_dev',t) for t in range(1,37)]:
        raise ValueError('Full and boundary figures require identical length36 rows at each checkpoint')
    return rows


def plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    names = []
    def save(fig, name):
        for suffix in ('png','pdf'): fig.savefig(output / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
        names.append(name); plt.close(fig)
    for name, limits in (('length-full',(1,36)),('length-boundary',(10,18))):
        fig,axes = plt.subplots(1,3,figsize=(14,4.5)); fig.subplots_adjust(left=.07,right=.99,bottom=.18,top=.80,wspace=.35)
        for axis,key,title in zip(axes,('E','A','M'),('Every state through t correct','Only state t correct','Mean token accuracy through t')):
            for step in (10000,20000):
                rows=plot_rows(summary,step);x=[r['length'] for r in rows]
                axis.plot(x,[r[key] for r in rows],color=COLORS[step],label=f'Input injection at {step:,} updates')
                if key!='M':axis.fill_between(x,[r[f'{key}_low95'] for r in rows],[r[f'{key}_high95'] for r in rows],color=COLORS[step],alpha=.12)
            axis.set(xlim=limits,ylim=(-.025,1.025),xlabel='Prefix t of same length-36 words',title=f'{key}(t): {title}')
            axis.axvline(12,color='gray',linestyle=':');axis.grid(alpha=.2);axis.yaxis.set_major_formatter(PercentFormatter(1));axis.legend(fontsize=8)
        fig.suptitle('Four-layer input injection + NextLat · 10,000 versus 20,000 updates · 102,400 development words',y=.97);save(fig,name)
    fig,axis=plt.subplots(figsize=(9,4.5),layout='constrained')
    axis.plot(STEPS,[plot_rows(summary,s)[-1]['E'] for s in STEPS],marker='o',color=COLORS[20000])
    axis.set(xlabel='Optimizer updates',ylabel='E(36): whole-word accuracy',ylim=(-.025,1.025),xticks=STEPS)
    axis.axvline(10000,color='gray',linestyle=':',label='Continuation starts');axis.yaxis.set_major_formatter(PercentFormatter(1));axis.grid(alpha=.2);axis.legend()
    fig.suptitle('Input-injection trajectory · all points use 102,400 development words');save(fig,'whole-word-vs-updates')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5));fig.subplots_adjust(left=.07,right=.99,bottom=.18,top=.80,wspace=.35)
    rows=summary['training_curve']
    for axis,key,title in zip(axes,('state_ce','latent_loss','loss'),('State CE','Latent SmoothL1','Joint objective')):
        axis.plot([r['update'] for r in rows],[r[key] for r in rows],color=COLORS[20000])
        axis.axvline(10000,color='gray',linestyle=':');axis.set(xlabel='Optimizer updates',ylabel=title,xlim=(0,20000));axis.grid(alpha=.2)
    fig.suptitle('Input injection · stitched first 20k training · 100-update means',y=.97);save(fig,'training-losses')
    return names


def markdown(summary, strength_link):
    old,new=(plot_rows(summary,s)[-1] for s in (10000,20000))
    lines=['# Input injection: continuation from 10k to 20k','',
           f"The same input-injection model's E(36) changes from **{old['E']:.4%} at 10k** to **{new['E']:.4%} at 20k** "
           f"({100*(new['E']-old['E']):+.2f} percentage points).",'',
           'The user extended the budget during the original 10k run after inspecting development results. This supplement compares the model with itself after more training. '
           'The input-versus-value comparison remains a separate matched 10k report; this 20k endpoint has a different training budget.','',
           '| Updates | L12 whole word | E(13) | E(14) | E(36) | A(36) | M(36) |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for step in STEPS:
        rows=plot_rows(summary,step);values=[summary['metrics'][str(step)]['dev']['whole_word_exact_match'],rows[12]['E'],rows[13]['E'],rows[-1]['E'],rows[-1]['A'],rows[-1]['M']]
        lines.append(f'| {step:,} | '+' | '.join(f'{v:.4%}' for v in values)+' |')
    lines+=['','E(t) requires every state through t correct; A(t) checks only state t; M(t) averages correctness through t. '
            'All trajectory points use the same 102,400 development words. Full and boundary plots use identical saved length-36 rows.','',
            'The continuation resumes the exact 10k model/Adam/RNG checkpoint. Both 10k history segments and every minibatch-order hash through 20k are checked. '
            'Architecture, fixed 0.02 coefficient, learned projection, original NextLat objective and 62 training source files remain unchanged. The scalar coefficient is not learned.','',
            'Four RT layers: window of two tokens first, then three full layers; input injection enters the second block. Width 512, batch 1024, full FP32; no TF32, compilation or CUDA graphs. '
            'One seed, reused development data, and no uninjected four-layer control. No final confirmation, autonomous latent rollout or convergence claim.','',
            f'[Learned projection and effective bypass strength]({strength_link})',
            '[Separate matched10k input/value comparison](../embedding-injection10k/report.md)','']
    for name in ('whole-word-vs-updates','length-full','length-boundary','training-losses'):lines += [f'![{name}]({name}.png)','']
    lines += ['[Exact metrics](metrics.csv) · [Provenance and plot data](summary.json). Pointwise Wilson 95% intervals describe variation over words, not training-seed uncertainty.']
    return '\n'.join(lines)+'\n'


def run(args):
    output=local_path(args.output_dir).resolve()
    if output.exists():raise FileExistsError('Use a fresh separate input20k report directory')
    summary=make_summary();summary['projection_strength_report']=args.strength_report_link;output.mkdir(parents=True)
    rows=[r for s in STEPS for role in ROLES for r in summary['curves'][str(s)][role]]
    with (output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_COLUMNS);writer.writeheader();writer.writerows(rows)
    for name in REPORTING_SOURCES:
        path=output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    summary['reporting_sources']={name:hash_file(output/'source'/name) for name in REPORTING_SOURCES};write_json(output/'summary.json',summary)
    figures=plots(summary,output);(output/'report.md').write_text(markdown(summary,args.strength_report_link))
    tracker=OnlineTracker(project='rt-a5-state-tracking',entity='taylorbollman',output_dir=output,group=LINEAGE.name,name='input-injection-nextlat-own10k-to20k')
    result={'schema':SCHEMA,'status':'running','primary_update':20000,'figures':figures}
    try:
        tracker.start({'schema':SCHEMA,'scope':summary['scope'],'qualification':summary['protocol']['qualification']})
        import wandb
        tracker.log({'report/metrics':wandb.Table(columns=list(CSV_COLUMNS),data=[[r[k] for k in CSV_COLUMNS] for r in rows]),
                     **{f'report/{n}':wandb.Image(str(output/f'{n}.png')) for n in figures}})
        for row in summary['training_curve']:
            values={'update':row['update'],**{f'train/{k}':row[k] for k in ('state_ce','latent_loss','loss')}}
            if row['update'] in STEPS:values['dev/E36']=plot_rows(summary,row['update'])[-1]['E']
            tracker.log(values)
        tracker.summary({'input10k/E36':plot_rows(summary,10000)[-1]['E'],'input20k/E36':plot_rows(summary,20000)[-1]['E'],'primary_update':20000})
        tracker.finish(succeeded=True);result['status']='complete'
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__)
        try:tracker.finish(succeeded=False)
        except Exception:pass
        raise
    finally:
        result['wandb']=tracker.record;result['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file() and 'wandb' not in p.relative_to(output).parts and p!=output/'report.json'}
        write_json(output/'report.json',result)
    print(json.dumps({'status':result['status'],'output_dir':str(output),'wandb':tracker.record['run_url']}));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',default=str(OUTPUT))
    parser.add_argument('--strength-report-link',default='../input-bypass-strength/through-020000/README.md')
    run(parser.parse_args())


if __name__=='__main__':main()
