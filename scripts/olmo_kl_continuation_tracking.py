"""Publish an already pinned completed-pair analysis to the user's W&B project."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--summary-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert digest(args.summary)==args.summary_sha256
    report=json.loads(args.summary.read_text())
    assert report['status']=='complete_pair' and set(report['arms'])=={'KL1','KL0.1'}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'input-snapshot').mkdir()
    shutil.copyfile(args.summary,args.output/'input-snapshot/summary.json')
    import wandb
    tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',entity='taylorbollman',
        output_dir=args.output,group='kl-paired-continuation',name='paired-nf-kl-continuation-summary')
    tracker.start({'scope':'NF32-to64; inherited Adam; KL-only branch; single development prefix; no precision clearance',
        'summary_sha256':args.summary_sha256})
    images={}
    for name in ('development-raw-losses','training-dynamics','timing-memory'):
        source=args.summary.parent/(name+'.png')
        assert digest(source)==report['artifacts'][source.name]['sha256']
        shutil.copyfile(source,args.output/source.name)
        images[name]=wandb.Image(str(source))
    rows=[]
    for label,arm in report['arms'].items():
        for entry in arm['development']:
            for p in entry['passes']:
                rows.append([label,entry['after_update'],p['pass'],*[p['means'][t] for t in ('ce','latent','kl')]])
    tracker.log({**images,'development':wandb.Table(columns=['branch','update','pass','raw_ce','raw_latent','raw_kl'],data=rows)})
    tracker.finish(succeeded=True)
    assert digest(args.summary)==args.summary_sha256
    name='scripts/olmo_kl_continuation_tracking.py'
    target=args.output/'source-snapshot'/name;target.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__),target)
    result={'status':'complete','sources':{name:digest(Path(__file__))},'tracking':tracker.record,
        'summary_sha256':args.summary_sha256,'scope':'Presentation of completed paired analysis; no optimizer updates'}
    (args.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(tracker.record['run_url'])


if __name__=='__main__':main()
