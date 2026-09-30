"""CPU-only declaration and dry resolution; never activate or launch training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from cdrm.pretrained.artifacts import sha256_file,write_json
from scripts import olmo_nfr_128_contract as contract


def reference(path,expected):
    path=Path(path)
    target=path/'manifest.json' if path.is_dir() else path
    if sha256_file(target)!=expected:raise ValueError('Original retained authority changed: '+str(path))
    return {'path':str(path),'sha256':expected}


def declaration():
    base=contract.ROOT/'.runtime/olmo-nfr-kl-continuation'
    report=base/'native-nfr-reduced-32to64-01'
    return {'schema':contract.SCHEMA,'activation':'after_healthy_matched_NFR64_curves',
        'arm':'NFR','origin_update':64,'stop_update':128,'planned_updates':128,'kl_weight':.1,
        'named_checkpoints':[80,96,100,112,128],'extra_evaluation_updates':[100],
        'storage_prefix':'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/nfr-stability-128/20260930-01',
        'original_scope':reference(base/'scope-declaration.json','8485bd04135b4393bff5ec78a011635b539ab292f37fc0a6d7678e7866f40dac'),
        'parent64_report':reference(report/'report.json','9de466ea50ea837db1aaf6f0e88f23675c6a13af1f24a196fab992ea54a96e8c'),
        'parent64_checkpoint':reference('/mnt/localssd/cdrm-checkpoints/nfr-kl-continuation/native-nfr-reduced-32to64-01/update-000064',
            '4f24a233c89e2eb20a0c355d1a3dda530fb906981402297278b2decffde611ea'),
        'parent64_publication':reference(report/'checkpoint-publications/update-000064.json',
            '094934f8391c6323f9a63e35758af296d80df69da2db9c8538492321f3d7ad7c')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--refresh-resolution',action='store_true',help='Update only derived dry resolution before source freeze')
    args=parser.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=args.refresh_resolution)
    scope=declaration();path=args.output_dir/'scope.json'
    if args.refresh_resolution:
        if json.loads(path.read_text())!=scope:raise ValueError('Existing scope differs; never rewrite authority')
    else:write_json(path,scope)
    result=contract.resolve(scope,path)
    write_json(args.output_dir/'resolved.json',result)
    write_json(args.output_dir/'runtime-sources.json',result['sources'])
    print({'status':result['status'],'scope_sha256':sha256_file(path),
        'resolved_sha256':sha256_file(args.output_dir/'resolved.json'),'source_count':len(result['sources']),
        'new_input_tokens':result['new_input_tokens'],'stop_update':128})


if __name__=='__main__':main()
