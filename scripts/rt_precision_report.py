#!/usr/bin/env python3
"""Plot closed RT precision evidence; preserve each report build separately."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_tracking import OnlineTracker, add_wandb_arguments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lineage',type=Path,required=True)
    parser.add_argument('--protected-profile',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError('Use a fresh report build directory')
    args.output_dir.mkdir(parents=True)
    report={'schema':'rt-precision-report-v1','status':'running','sources':{},'figures':[],
            'qualification':'Numerical review flags remain separate from execution failure and learning-quality evidence.'}
    tracker=None
    def read(path):
        raw=path.read_bytes(); data=json.loads(raw)
        if data.get('status')!='complete':raise ValueError(f'Refusing unfinished evidence: {path}')
        digest=hashlib.sha256(raw).hexdigest();report['sources'][str(path)]=digest
        target=args.output_dir/'source-reports'/f'{digest}.json';target.parent.mkdir(exist_ok=True)
        if not target.exists():target.write_bytes(raw)
        return data
    def save(fig,name):
        fig.savefig(args.output_dir/f'{name}.png',dpi=170,bbox_inches='tight')
        fig.savefig(args.output_dir/f'{name}.svg',bbox_inches='tight')
        plt.close(fig);report['figures'].append(name)
    try:
        cases={name:read(args.lineage/'numerics'/name/'report.json')
               for name in ('full-b2-seed0','full-b2-seed1','full-b512-seed0')}
        labels=['B2 · seed 0','B2 · seed 1','B512 · seed 0']
        colors={'A':'#2475ad','B':'#d87517'}
        display={'A':'Protected BF16','B':'Released-style BF16'}
        x=np.arange(3); width=.34
        fig,axes=plt.subplots(1,2,figsize=(11,4.1))
        for arm in ('A','B'):
            values=[data['comparisons'][arm+'_vs_C'] for data in cases.values()]
            offset=-width/2 if arm=='A' else width/2
            axes[0].bar(x+offset,[100*v['raw_gradients']['relative_l2'] for v in values],width,
                        color=colors[arm],label=display[arm])
            axes[1].bar(x+offset,[100*v['adam_delta']['relative_l2'] for v in values],width,color=colors[arm])
        axes[0].axhline(1.5625,color='#555555',linestyle='--',linewidth=1,label='Global gradient review threshold')
        axes[0].set_ylabel('Relative L2 error vs FP32 (%)');axes[0].set_title('All parameter gradients')
        axes[1].set_ylabel('Relative L2 distance vs FP32 (%)');axes[1].set_title('First Adam update: distance remains visible')
        for ax in axes:
            ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.17))
        fig.suptitle('12 recurrent layers · D1024 · T512 · identical starting states',fontsize=12)
        fig.tight_layout();save(fig,'initial-gradients-and-adam')

        full=cases['full-b512-seed0'];fig,ax=plt.subplots(figsize=(8.5,4))
        for arm in ('A','B'):
            tensors=full['comparisons'][arm+'_vs_C']['raw_gradients']['tensors']
            values=[100*tensors[f'transformer.blocks.{layer}.q_proj.weight']['relative_l2'] for layer in range(12)]
            ax.plot(range(1,13),values,marker='o',color=colors[arm],label=display[arm])
        ax.axhline(3.125,color='#555555',linestyle='--',linewidth=1,label='Per-tensor review threshold')
        ax.set(xlabel='Recurrent block (1-based)',ylabel='Query-projection gradient L2 error (%)',
               title='Physical B512: localized query-gradient rounding')
        ax.set_xticks(range(1,13));ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.tight_layout();save(fig,'b512-query-gradients')

        previous=read(args.protected_profile)
        legacy=read(args.lineage/'profile/b512-legacy/report.json')
        report['performance_control_qualification']='Protected profile is the preceding same-machine milestone; no randomized timing study.'
        fig,axes=plt.subplots(1,2,figsize=(9,3.8))
        for index,(label,data) in enumerate([('Protected',previous),('Released-style',legacy)]):
            color=list(colors.values())[index]
            axes[0].bar(label,data['timing']['tokens_per_second']/1000,color=color)
            axes[1].bar(label,data['steady_peak_memory']['peak_reserved_bytes']/2**30,color=color)
        axes[0].set(ylabel='Input tokens / second (thousands)',title='Captured physical B512 throughput')
        axes[1].set(ylabel='Peak reserved GPU memory (GiB)',title='Measured steady reservation')
        for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.tight_layout();save(fig,'b512-cost')

        training=[]
        for path in sorted((args.lineage/'train').glob('*/report.json')):
            candidate=json.loads(path.read_text())
            if candidate.get('status')=='complete':training.append(read(path))
        if training:
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            series={}
            for data in training:
                key=(data['policy'],data['seed']); series.setdefault(key,{})
                for row in data['updates']:
                    if row['update'] in series[key] and row!=series[key][row['update']]:
                        raise ValueError('Conflicting training histories for the same arm/seed/update')
                    series[key][row['update']]=row
            for (policy,seed),rows in sorted(series.items()):
                arm='A' if policy=='bf16_fp32_state' else 'B'; ordered=[rows[k] for k in sorted(rows)]
                line='-' if seed==20260910 else '--'
                label=f'{display[arm]} · seed {seed-20260910}'
                axes[0].plot([r['update'] for r in ordered],[r['train_ce_supervised_token'] for r in ordered],
                             color=colors[arm],linestyle=line,label=label,linewidth=1.2)
                axes[1].plot([r['update'] for r in ordered],[r['gradient_norm_before_clip'] for r in ordered],
                             color=colors[arm],linestyle=line,linewidth=1.2)
            axes[0].set(xlabel='Optimizer update',ylabel='CE / supervised token',title='Training warmup prefix')
            axes[1].set(xlabel='Optimizer update',ylabel='Global gradient norm before clipping',title='Gradient trajectory')
            axes[0].legend(fontsize=8);axes[1].set_yscale('log')
            for ax in axes:ax.grid(alpha=.2)
            fig.tight_layout();save(fig,'training-curves')
        script=Path(__file__).read_bytes();(args.output_dir/'report-script.py').write_bytes(script)
        report['script_sha256']=hashlib.sha256(script).hexdigest()
        tracker=OnlineTracker(project=args.wandb_project,entity=args.wandb_entity,group=args.wandb_group,
                              name=args.wandb_run_name,output_dir=args.output_dir)
        report['wandb']=tracker.record
        tracker.start({'scope':'Closed numerical evidence and completed training curves',
                       'lineage':str(args.lineage),'source_sha256':report['sources']})
        import wandb
        tracker.log({name:wandb.Image(str(args.output_dir/f'{name}.png')) for name in report['figures']})
        tracker.summary({'figures':len(report['figures']),'completed_training_reports':len(training)})
        report['status']='complete'
    except BaseException as error:
        report.update(status='execution_failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        try:
            if tracker:tracker.finish(succeeded=report['status']=='complete')
        except BaseException as error:
            report.update(status='execution_failed',sync_error_type=type(error).__name__);raise
        finally:
            (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
