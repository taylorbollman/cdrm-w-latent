"""Conditional native NFR KL comparison with the accepted KL execution engine.

Only native NFR update32 through64 is added. Historical200/210 and the separate
FBT-only208 runtime remain immutable. This contract never restores tensors.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.distributed_checkpoint import inspect_distributed_checkpoint
from scripts import olmo_kl_continuation as accepted
from scripts import olmo_campaign_manifest as legacy

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='olmo-nfr-kl-scope-v1'
PROTOCOL='docs/reports/olmo-nfr-kl-continuation/protocol.md'
REFS=('declaration','resolved','parent_report','parent_checkpoint')


def source_hashes(scope_path):
    frozen=json.loads((ROOT/'.runtime/olmo-kl-continuation/runtime-sources.json').read_text())
    if len(frozen)!=210 or accepted.source_hashes()!=frozen:
        raise ValueError('Historical210 KL runtime changed')
    names=('scripts/olmo_nfr_kl_contract.py','scripts/olmo_nfr_kl_execute.py',
           'tests/test_nfr_kl_contract.py',PROTOCOL)
    path=Path(scope_path).absolute()
    if not path.is_relative_to(ROOT) or path.is_symlink():raise ValueError('Scope authority must be a real project file')
    # The explicit scope bytes are part of child checkpoint identity, not only
    # a report attachment. They are not relabeled as executable Python.
    return dict(sorted((frozen|{n:sha256_file(ROOT/n) for n in names}
        |{str(path.relative_to(ROOT)):sha256_file(path)}).items()))


def validate_scope(scope):
    legacy.exact_fields(scope,('schema','activation','arm','parent_update','review_stop','planned_updates',
        'kl_weights','declaration','resolved','parent_report','parent_checkpoint','storage_prefix'), 'NFR KL scope')
    if (scope['schema']!=SCHEMA or scope['activation']!='conditional_after_fbt_only_assessment'
            or scope['arm']!='NFR' or scope['parent_update']!=32 or scope['review_stop']!=64
            or scope['planned_updates']!=128 or scope['kl_weights']!=[1.,.1]):
        raise ValueError('Only conditional NFR32-to64 KL1/.1 with unchanged128plan is declared')
    for key in REFS:
        legacy.exact_fields(scope[key],('path','sha256'),key)
        legacy.local_path(scope[key]['path']);legacy.pin(scope[key]['sha256'])
    from scripts.olmo_campaign_loop_run import storage_location
    storage_location(scope['storage_prefix'])
    return scope


def validate_arguments(args,scope):
    validate_scope(scope)
    if (args.arm!='NFR' or args.stop_after!=64 or args.observation!='lean'
            or args.checkpoint_mode!='async' or args.storage_prefix!=scope['storage_prefix']
            or args.branch not in ('control','reduced')
            or args.kl_weight!=(1. if args.branch=='control' else .1)):
        raise ValueError('Command lies outside explicit native NFR comparison')
    for key in REFS:
        actual=getattr(args,key)
        pin=getattr(args,'parent_manifest_sha256' if key=='parent_checkpoint' else key+'_sha256')
        if Path(actual)!=Path(scope[key]['path']) or pin!=scope[key]['sha256']:
            raise ValueError('Command reference differs from NFR scope: '+key)


def authenticated_parent(args,scope):
    """Authenticate the same parent lineage as NF, under an explicit NFR scope."""
    validate_arguments(args,scope)
    parent=inspect_distributed_checkpoint(args.parent_checkpoint,
        expected_manifest_sha256=args.parent_manifest_sha256,verify_state=False)
    reference=legacy.read_json(args.parent_report,args.parent_report_sha256,limit=128*1024**2)
    if (args.spec['kind']!='native' or reference.get('schema')!=accepted.accepted.SCHEMA
            or reference.get('arm')!='NFR' or reference.get('status') not in ('completed_plan','stopped_at_boundary')
            or reference.get('sources')!=accepted.accepted.source_hashes()
            or reference['configuration']!=parent['metadata']['configuration']
            or reference['source_fingerprint']!=parent['metadata']['source_fingerprint']
            or reference['declaration_sha256']!=args.declaration_sha256
            or reference['resolved_sha256']!=args.resolved_sha256):
        raise ValueError('Require pinned completed original NFR parent lineage')
    matches=[r for r in reference['published_checkpoints'] if r['manifest_sha256']==args.parent_manifest_sha256]
    if len(matches)!=1 or matches[0]['state']!=parent['state']:
        raise ValueError('NFR parent has no unique verified publication')
    boundaries=[r['boundary_by_rank'] for r in reference['local_checkpoints']
                if r['receipt']['manifest_sha256']==args.parent_manifest_sha256]
    if len(boundaries)!=1:raise ValueError('NFR parent has no unique independently saved boundary')
    config=parent['metadata']['configuration'];payload=config['execution_identity']['payload']
    if (parent['counters']['optimizer_updates']!=32 or payload['arm']!='NFR'
            or payload['model_contract']['mode']['rt_mode']!={'selected_layers':[0,15],'alpha':1.}
            or payload['model_contract']['mode']['num_passes']!=4
            or payload['model_contract']['weights']!={'ce':1.,'latent':1.,'kl':1.}
            or config['schedule']['planned_updates']!=128
            or len(args.spec['plan']['updates'])!=128
            or args.spec['plan']!=payload['plan']):
        raise ValueError('NFR parent mode, update or unchanged finite plan differs')
    args.parent_boundary_by_rank=boundaries[0];args.parent=parent;args.reference=reference
    return parent


def arguments_from_scope(scope):
    """CPU-only preflight defaults; output paths are assigned by the real launcher."""
    validate_scope(scope)
    fields={key:Path(scope[key]['path']) for key in REFS}
    fields.update({('parent_manifest_sha256' if key=='parent_checkpoint' else key+'_sha256'):scope[key]['sha256'] for key in REFS})
    return SimpleNamespace(**fields,arm='NFR',stop_after=64,observation='lean',checkpoint_mode='async',
        branch='control',kl_weight=1.,storage_prefix=scope['storage_prefix'])


def resolve(scope,scope_path):
    args=arguments_from_scope(scope)
    args.spec=accepted.load_spec(args)
    parent=authenticated_parent(args,scope)
    payload=parent['metadata']['configuration']['execution_identity']['payload']
    return {'schema':'olmo-nfr-kl-dry-resolution-v1','status':'validated_not_launched',
        'scope':scope,'scope_sha256':sha256_file(scope_path),'sources':source_hashes(scope_path),
        'parent_identity_sha256':parent['metadata']['configuration']['execution_identity']['sha256'],
        'plan_sha256':legacy.digest(args.spec['plan']),
        'schedule':parent['metadata']['configuration']['schedule'],
        'origin_update':32,'stop_update':64,'new_input_tokens_per_branch':32*524288,
        'checks':{'parent_original_NFR':True,'two_native_RT_layers':True,'unchanged128plan':True,
            'paired_exact_saved_parent':True,'KL_only_intervention':True,'historical210_sources_unchanged':True},
        'execution':'Unchanged accepted olmo_kl_continuation.run_stage and engine; explicit new native NFR authorizer',
        'qualification':'Metadata verification only; full tensors, Adam and RNG checked on actual strict checkpoint load',
        'estimated_pair_hours_on_two_H100':3.5}
