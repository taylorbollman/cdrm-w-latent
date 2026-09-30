"""Narrow, metadata-only authority for selected reduced-KL NFR64-to128.

Original32 is retained as provenance. The populated reduced64 checkpoint is
strictly loaded under its original configuration before a new execution scope
is recorded. This module never restores model tensors or initializes CUDA.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.distributed_checkpoint import inspect_distributed_checkpoint
from scripts import olmo_nfr_kl_contract as prior
from scripts import olmo_kl_continuation as accepted
from scripts import olmo_campaign_manifest as legacy

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='olmo-nfr-128-scope-v1'
PROTOCOL='docs/reports/olmo-nfr-stability-128/protocol.md'
REFS=('original_scope','parent64_report','parent64_checkpoint','parent64_publication')
NAMES=('scripts/olmo_nfr_128_contract.py','scripts/olmo_nfr_128_execute.py',
       'scripts/olmo_nfr_128_engine.py','scripts/olmo_nfr_128_prepare.py',
       'tests/test_nfr_128_contract.py',PROTOCOL)


def frozen_sources(original_scope_path):
    sources=json.loads((ROOT/'.runtime/olmo-nfr-kl-continuation/runtime-sources.json').read_text())
    if len(sources)!=215 or prior.source_hashes(original_scope_path)!=sources:
        raise ValueError('Historical NFR215 execution sources changed')
    return sources


def source_hashes(scope_path,scope):
    path=Path(scope_path).absolute()
    if not path.is_relative_to(ROOT) or path.is_symlink():
        raise ValueError('Continuation authority must be a real project file')
    sources=frozen_sources(legacy.local_path(scope['original_scope']['path']))
    return dict(sorted((sources|{n:sha256_file(ROOT/n) for n in NAMES}
        |{str(path.relative_to(ROOT)):sha256_file(path)}).items()))


def validate_scope(scope):
    legacy.exact_fields(scope,('schema','activation','arm','origin_update','stop_update',
        'planned_updates','kl_weight','named_checkpoints','extra_evaluation_updates',
        'storage_prefix',*REFS),'NFR128 scope')
    if (scope['schema']!=SCHEMA or scope['activation']!='after_healthy_matched_NFR64_curves'
            or scope['arm']!='NFR' or scope['origin_update']!=64 or scope['stop_update']!=128
            or scope['planned_updates']!=128 or scope['kl_weight']!=.1
            or scope['named_checkpoints']!=[80,96,100,112,128]
            or scope['extra_evaluation_updates']!=[100]):
        raise ValueError('Only unchanged reduced-KL NFR64-to128 is authorized')
    for key in REFS:
        legacy.exact_fields(scope[key],('path','sha256'),key)
        legacy.local_path(scope[key]['path']);legacy.pin(scope[key]['sha256'])
    from scripts.olmo_campaign_loop_run import storage_location
    storage_location(scope['storage_prefix'])
    return scope


def original_arguments(scope):
    validate_scope(scope)
    ref=scope['original_scope']
    old=legacy.read_json(legacy.local_path(ref['path']),ref['sha256'])
    args=prior.arguments_from_scope(old)
    args.branch='reduced';args.kl_weight=.1
    return old,args


def arguments_from_scope(scope):
    _,args=original_arguments(scope)
    args.stop_after=128;args.storage_prefix=scope['storage_prefix']
    return args


def validate_arguments(args,scope):
    expected=arguments_from_scope(scope)
    for name in (*prior.REFS,'parent_manifest_sha256','parent_report_sha256',
                 'declaration_sha256','resolved_sha256','arm','stop_after','observation',
                 'checkpoint_mode','branch','kl_weight','storage_prefix'):
        if getattr(args,name)!=getattr(expected,name):
            raise ValueError('Command differs from NFR128 authority: '+name)
    if (getattr(args,'resume',None) is None)!=(getattr(args,'resume_manifest_sha256',None) is None):
        raise ValueError('Child resume requires path and independent SHA together')


def verify_publication(publication,parent):
    if (publication['manifest_sha256']!=parent['manifest_sha256']
            or publication['state']!=parent['state'] or publication['metadata']!=parent['metadata']
            or publication['counters']!=parent['counters']):
        raise ValueError('Saved64 cloud publication differs from local committed metadata')
    retention=publication.get('retention',{})
    objects=retention.get('objects',[])
    expected={parent['state']['sha256'],parent['manifest_sha256']}
    if (retention.get('download_sha256_verified') is not True or retention.get('create_only') is not True
            or len(objects)!=2 or {x.get('sha256') for x in objects}!=expected
            or any(set(x.get('verification',{}))!={'download_sha256','server_md5','server_size','sha256_metadata'}
                or not all(v is True for v in x['verification'].values()) for x in objects)):
        raise ValueError('Require fully verified immutable cloud64 publication')


def authenticate_parent64(args,scope):
    """Authenticate old32 provenance and old64 strict-load authority, no tensors."""
    validate_arguments(args,scope)
    old,original=original_arguments(scope)
    original.spec=args.spec
    original32=prior.authenticated_parent(original,old)
    sources=frozen_sources(original_scope_path=legacy.local_path(scope['original_scope']['path']))
    read=lambda key:legacy.read_json(legacy.local_path(scope[key]['path']),scope[key]['sha256'],limit=128*1024**2)
    parent=inspect_distributed_checkpoint(legacy.local_path(scope['parent64_checkpoint']['path']),
        expected_manifest_sha256=scope['parent64_checkpoint']['sha256'],verify_state=False)
    reference=read('parent64_report');publication=read('parent64_publication')
    verify_publication(publication,parent)
    config=parent['metadata']['configuration'];identity=config['execution_identity'];payload=identity['payload']
    if (args.spec['kind']!='native' or reference.get('schema')!=accepted.SCHEMA
            or reference.get('status')!='stopped_at_boundary' or reference.get('segment_stop_after')!=64
            or reference.get('arm')!='NFR' or reference.get('wandb',{}).get('status')!='synced'
            or reference.get('last_verified_cloud_update')!=64
            or reference.get('sources')!=sources or reference.get('configuration')!=config
            or reference.get('source_fingerprint')!=parent['metadata']['source_fingerprint']
            or reference.get('final_counters')!=parent['counters']
            or reference.get('original_configuration')!=original32['metadata']['configuration']
            or reference.get('parent_report_sha256')!=original.parent_report_sha256
            or reference.get('parent_manifest_sha256')!=original.parent_manifest_sha256
            or reference.get('declaration_sha256')!=args.declaration_sha256
            or reference.get('resolved_sha256')!=args.resolved_sha256
            or identity['sha256']!=legacy.digest(payload)):
        raise ValueError('Reduced64 must retain the completed original NFR215 lineage')
    if (parent['counters']['optimizer_updates']!=64 or payload['arm']!='NFR'
            or payload['model_contract']['mode']['rt_mode']!={'selected_layers':[0,15],'alpha':1.}
            or payload['model_contract']['mode']['num_passes']!=4
            or payload['model_contract']['weights']!={'ce':1.,'latent':1.,'kl':.1}
            or config['model']['lambda_kl']!=.1
            or config['objective_branch']!=reference['branch']
            or config['objective_branch']['review_stop']!=64
            or config['schedule']!=original32['metadata']['configuration']['schedule']
            or config['schedule']['planned_updates']!=128
            or len(args.spec['plan']['updates'])!=128 or args.spec['plan']!=payload['plan']
            or payload['sources']!=sources or config['parameters']!=payload['model_contract']):
        raise ValueError('Saved64 objective, recurrence, ownership or unchanged128plan differs')
    matches=[p for p in reference['published_checkpoints'] if p['manifest_sha256']==parent['manifest_sha256']]
    boundaries=[r['boundary_by_rank'] for r in reference['local_checkpoints']
                if r['receipt']['manifest_sha256']==parent['manifest_sha256']]
    if (matches!=[publication] or len(boundaries)!=1 or boundaries[0]!=reference['final_boundary_by_rank']
            or reference['final_clocks'].get('scheduler_epoch')!=64
            or reference['final_clocks'].get('adam_parameters',0)<=0):
        raise ValueError('Saved64 must have a unique independently recorded populated-Adam boundary')
    args.original_parent=original32;args.original_reference=original.reference
    args.parent64=parent;args.parent64_reference=reference;args.parent64_boundary_by_rank=boundaries[0]
    args.original_sources=sources
    return parent


def continuation_evaluation(plan,scope):
    result=copy.deepcopy(plan)
    result['scheduled_updates']=sorted(set(plan['scheduled_updates'])|set(scope['extra_evaluation_updates']))
    return result


def continuation_metadata(scope,scope_sha256,parent):
    return {'schema':'olmo-nfr-128-metadata-continuation-v1','scope_sha256':scope_sha256,
        'parent_manifest_sha256':parent['manifest_sha256'],
        'parent_report_sha256':scope['parent64_report']['sha256'],
        'parent_identity_sha256':parent['metadata']['configuration']['execution_identity']['sha256'],
        'parent_update':64,'stop_update':128,'named_checkpoints':scope['named_checkpoints'],
        'extra_evaluation_updates':scope['extra_evaluation_updates'],
        'objective_change':False,'state_transition':'identity_only_after_strict_original64_load'}


def continued_configuration(parent_config,continuation,sources,evaluation):
    result=copy.deepcopy(parent_config)
    result['schema']='olmo-nfr-128-execution-v1'
    result['continuation']=copy.deepcopy(continuation)
    payload=result['execution_identity']['payload']
    payload.update(scope='Exact saved reduced-KL NFR64 continuation; unchanged objective and128 token schedule',
        sources=copy.deepcopy(sources),continuation=copy.deepcopy(continuation),evaluation=copy.deepcopy(evaluation))
    result['execution_identity']['sha256']=legacy.digest(payload)
    return result


def validate_continuation_transition(parent_config,configuration,model,branch,continuation):
    """Whitelist metadata deltas; no tensor writes, RNG draws, or model edits."""
    expected=continued_configuration(parent_config,continuation,
        configuration['execution_identity']['payload']['sources'],
        configuration['execution_identity']['payload']['evaluation'])
    expected_eval=continuation_evaluation(parent_config['execution_identity']['payload']['evaluation'],
        {'extra_evaluation_updates':[100]})
    if (configuration!=expected or branch!=parent_config['objective_branch']
            or configuration['execution_identity']['payload']['evaluation']!=expected_eval
            or continuation['parent_update']!=64 or continuation['stop_update']!=128
            or continuation['objective_change'] is not False
            or model.objective_weights()!={'ce':1.,'latent':1.,'kl':.1}
            or model.predictor.config.lambda_kl!=.1
            or model.config.to_dict()!=parent_config['model']):
        raise ValueError('Continuation changed model, objective or undeclared configuration')
    return {'schema':'olmo-nfr-128-transition-check-v1','metadata_only':True,
        'objective_unchanged':True,'parent_identity_sha256':continuation['parent_identity_sha256'],
        'child_identity_sha256':configuration['execution_identity']['sha256']}


def validate_activation(path,pin,scope,scope_sha256):
    receipt=legacy.read_json(path,pin)
    legacy.exact_fields(receipt,('schema','decision','continuation_sha256','parent64_manifest_sha256',
        'endpoint_reports','rationale'),'NFR128 activation')
    if (receipt['schema']!='olmo-nfr-128-activation-v1' or receipt['decision']!='continue_reduced_to128'
            or receipt['continuation_sha256']!=scope_sha256
            or receipt['parent64_manifest_sha256']!=scope['parent64_checkpoint']['sha256']
            or set(receipt['endpoint_reports'])!={'control','reduced'}
            or not isinstance(receipt['rationale'],str) or not receipt['rationale'].strip()):
        raise ValueError('Matched-endpoint continuation activation differs')
    endpoints={}
    for name,ref in receipt['endpoint_reports'].items():
        legacy.exact_fields(ref,('path','sha256'),name)
        legacy.pin(ref['sha256'])
        report=legacy.read_json(legacy.local_path(ref['path']),ref['sha256'],limit=128*1024**2)
        preservation=report.get('preservation',{})
        if (report.get('schema')!='olmo-nfr-endpoint-curves-v1' or report.get('status')!='completed'
                or report.get('arm')!='NFR' or report.get('after_update')!=64
                or report.get('case')!=name or report.get('kl_weight')!={'control':1.,'reduced':.1}[name]
                or report.get('num_passes')!=32 or report.get('optimizer_updates_performed')!=0
                or any(report.get(k) is not True for k in ('weights_unchanged','rng_unchanged','gradient_buffers_absent'))
                or preservation.get('integrity_passed') is not True or preservation.get('restored') is not True
                or not preservation.get('checks') or not all(v is True for v in preservation['checks'].values())
                or report['result'].get('input_tokens')!=8192 or report['result'].get('ce_targets')!=8184
                or report['result'].get('policy')!='common_fp32_no_jitter_v1' or report['result'].get('beta')!=1.
                or [p['pass'] for p in report['result']['passes']]!=list(range(1,33))):
            raise ValueError('Endpoint observation or preservation differs: '+name)
        endpoints[name]=report
    control,reduced=endpoints['control'],endpoints['reduced']
    if (any(control[k]!=reduced[k] for k in ('endpoint_scope','membership_sha256',
            'index_manifest_sha256','batch_tensor_sha256','probe_policy'))
            or reduced['input_authorities']['checkpoint']!=scope['parent64_checkpoint']
            or any(r['input_authorities']['checkpoint']!=r['endpoint_scope']['declaration']['checkpoints'][name]
                   for name,r in endpoints.items())):
        raise ValueError('Endpoint pair is not matched to the requested saved64 continuation')
    return receipt


def resolve(scope,scope_path):
    args=arguments_from_scope(scope);args.spec=accepted.load_spec(args)
    parent=authenticate_parent64(args,scope)
    evaluation=continuation_evaluation(args.spec['evaluation_plan'],scope)
    metadata=continuation_metadata(scope,sha256_file(scope_path),parent)
    sources=source_hashes(scope_path,scope)
    config=continued_configuration(parent['metadata']['configuration'],metadata,sources,evaluation)
    return {'schema':'olmo-nfr-128-resolution-v1','status':'validated_not_launched',
        'scope':scope,'scope_sha256':sha256_file(scope_path),'sources':sources,
        'continuation':metadata,'configuration':config,
        'parent_identity_sha256':metadata['parent_identity_sha256'],
        'plan_sha256':legacy.digest(args.spec['plan']),'schedule':config['schedule'],
        'evaluation_plan':evaluation,'origin_update':64,'stop_update':128,
        'new_input_tokens':64*524288,'estimated_hours_two_H100':4,
        'checks':{'original32_lineage':True,'reduced64_fully_published':True,
            'populated_Adam_boundary_recorded':True,'unchanged128_data_and_LR_plan':True,
            'historical215_sources_unchanged':True,'no_objective_or_model_change':True},
        'qualification':'Metadata-only preflight; actual tensors/Adam/cursor/RNG verified on strict GPU load'}
