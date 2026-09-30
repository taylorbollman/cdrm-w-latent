"""Aggregate completed fixed-checkpoint probes; no model or training execution."""
from pathlib import Path
import hashlib
import json
import shutil

from scripts.experiment_tracking import OnlineTracker


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'NF origin': 'nf0-forward-02', 'NF endpoint': 'nf32-forward-01',
    'NFR endpoint': 'nfr32-forward-01',
    'NF batch 1': 'nf32-gradient-primary-01',
    'NF batch 2': 'nf32-gradient-conditional-01',
    'NFR batch 1': 'nfr32-gradient-primary-01',
}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb
    directory = ROOT/'.runtime/olmo-feedback-diagnostic'
    output = directory/'summary-01'
    output.mkdir(exist_ok=False)
    reports, pins = {}, {}
    for label, name in CASES.items():
        path = directory/name/'report.json'
        raw = path.read_bytes()
        r = json.loads(raw)
        if (r['status'] != 'complete' or r['tracking']['status'] != 'synced'
                or not all(r[k] for k in ('weights_unchanged','grad_buffers_untouched','rng_unchanged'))):
            raise ValueError('Require completed immutable probes: '+name)
        reports[label] = r
        pins[label] = {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest()}
    forward, gradients = [], []
    for label, r in reports.items():
        for case in r['cases']:
            for p in case['per_pass']:
                forward.append({'model': label, 'beta': case['beta'], 'pass': p['pass'],
                                **p['means'], 'teacher_entropy': p['teacher_entropy_on_kl_positions'],
                                'student_entropy': p['student_entropy_on_kl_positions']})
        if 'gradient' in r:
            if not r['gradient']['passed']:
                raise ValueError('Gradient check failed')
            for component, g in r['gradient']['components'].items():
                gradients.append({'model': label, 'component': component,
                    'norms': g['norms'], 'ce_vs_aux_cosine': g['ce_vs_aux']['cosine'],
                    'ce_first_vs_kl': g['cosines']['ce_first']['kl'],
                    'ce_first_vs_latent': g['cosines']['ce_first']['latent'],
                    'ce_first_vs_ce_later': g['cosines']['ce_first']['ce_later'],
                    'ce_first_dot_joint': g['dot_products']['ce_first']['joint'],
                    'ce_later_dot_joint': g['dot_products']['ce_later']['joint'],
                    'reconstruction_relative_l2': g['reconstruction']['relative_l2_to_joint']})
    figure, axes = plt.subplots(1, 2, figsize=(12,4.6), constrained_layout=True)
    for label, beta in [('NF origin',1),('NF endpoint',1),('NF endpoint',.5),
                         ('NF endpoint',0),('NFR endpoint',1),('NFR endpoint',0)]:
        rows = [r for r in forward if r['model']==label and r['beta']==beta]
        axes[0].plot([r['pass'] for r in rows], [r['ce'] for r in rows], marker='o',
                     label=f'{label}, beta={beta:g}', linestyle='--' if beta==0 else '-')
    axes[0].set(xlabel='Pass', ylabel='CE (nats/target)', xticks=[1,2,3,4],
                title='Same eight dev sequences, T1024')
    axes[0].legend(fontsize=8)
    rows = [r for r in gradients if r['component']=='backbone']
    keys = ['ce_first_vs_kl','ce_first_vs_latent','ce_vs_aux_cosine']
    names = ['First CE vs KL','First CE vs latent','Total CE vs auxiliaries']
    for i,row in enumerate(rows):
        axes[1].bar(np.arange(3)+(i-1)*.24, [row[k] for k in keys], width=.24, label=row['model'])
    axes[1].axhline(0,color='black',linewidth=.7)
    axes[1].set(xticks=np.arange(3), xticklabels=names, ylabel='Gradient cosine on backbone',
                ylim=(-1,1), title='Fixed two-row training probes')
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle('FP32, no jitter, fixed saved weights — no optimizer updates',fontsize=11)
    figure.savefig(output/'feedback-diagnostic.pdf')
    figure.savefig(output/'feedback-diagnostic.png',dpi=160)
    plt.close(figure)
    tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', output_dir=output,
        group='feedback-diagnostic',name='saved-checkpoint-feedback-summary')
    tracker.start({'scope':'fixed-weight diagnosis; six completed probes; no training', 'report_pins':pins})
    tracker.log({'diagnostic':wandb.Image(str(output/'feedback-diagnostic.png')),
        'forward':wandb.Table(columns=list(forward[0]),data=[[r[k] for k in forward[0]] for r in forward]),
        'gradient':wandb.Table(columns=['model','component',*keys,'ce_first_dot_joint','ce_later_dot_joint'],
            data=[[r[k] for k in ['model','component',*keys,'ce_first_dot_joint','ce_later_dot_joint']]
                  for r in gradients])})
    tracker.finish(succeeded=True)
    name = 'scripts/olmo_feedback_summary.py'
    target = output/'source-snapshot'/name
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT/name,target)
    report = {'status':'complete','sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()},
              'report_pins':pins,'tracking':tracker.record,'forward':forward,'gradients':gradients,
              'optimizer_updates':0, 'scope':'Small FP32/no-jitter diagnostics; not BF16 update/quality clearance'}
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(tracker.record['run_url'],flush=True)


if __name__ == '__main__':
    main()
