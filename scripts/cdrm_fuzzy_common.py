"""Native MAD fuzzy recall and paired 150M-family model/training utilities.

CPU initialization is explicit. Model execution never silently falls back from
CUDA. This module does not read or extend any frozen five-block checkpoint.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.mad_data import load_dataset, epoch_indices
from cdrm_common import state_summaries, save_checkpoint
import cdrm_tiled_common as old
from stage_a_common import unique_parameters, seed_all, rng_state, restore_rng
from stage_b_train import atomic_json, append_jsonl, file_digest, json_digest, compiler_audit
from r3_mixed_operational import cpu_tree, state_digest, effective_precision

TASK='fuzzy-in-context-recall'
INIT_FORMAT='cdrm-fuzzy-initialization-v1'
FORMAT='cdrm-fuzzy-training-v1'
ARMS=('cdrm','seq')
PRECISIONS={'fp32':'fp32','bf16':'bf16_fp32_state'}
PARAMETERS={'cdrm':153175040,'seq':153437184}
ADAPTERS={'cdrm.deep_adapter.weight','cdrm.bridge_adapter.weight'}
PRESETS={arm:Path(f'configs/cdrm/fuzzy_150m_{arm}.json') for arm in ARMS}
SCHEDULE={'kind':'cosine_per_completed_epoch','T_max':50,'eta_min':1e-6,'warmup':0}
OPTIMIZER={'name':'AdamW','betas':[.9,.98],'eps':1e-8,'weight_decay':0.,'clip':1.,'foreach':False,'fused':False}
LEARNING_RATES=(1e-4,5e-4,1e-3)
preserve_tracking_rng=old.preserve_tracking_rng
snapshot_sources=old.snapshot_sources
verify_sources=old.verify_sources
seed_cpu=old.seed_cpu
native_loss_sum=old.loss_sum
loss_sum=native_loss_sum
metric_counts=old.metric_counts
finish_metrics=old.finish_metrics


def optimizer_for(model,lr=5e-4):
    parameters=unique_parameters(model)
    return torch.optim.AdamW(parameters,lr=lr,betas=(.9,.98),eps=1e-8,weight_decay=0.,foreach=False,fused=False)


def config(arm='cdrm',precision='bf16',backend='tiled',*,length=256,eager=False,overrides=None):
    from olmo.config import ModelConfig
    if arm not in ARMS or precision not in PRECISIONS or backend not in ('tiled','naive'):
        raise ValueError('Unsupported fuzzy architecture, precision or scan backend')
    if not 4<=length<=512:raise ValueError('Require an explicit actual length within the512-token capacity')
    if backend=='naive' and arm=='cdrm' and precision!='fp32':raise ValueError('Naive CDRM remains an FP32 diagnostic oracle')
    raw=json.loads(PRESETS[arm].read_text())
    raw.update(cdrm_enabled=arm=='cdrm',cdrm_backend=backend,
        cdrm_precision_policy=PRECISIONS[precision] if arm=='cdrm' else 'fp32',
        ordinary_attention_precision_policy='fp32',reference_eager=backend=='naive' or eager,
        cdrm_source='deep',cdrm_read_mode='history',cdrm_rho=1.,cdrm_epsilon=.1,cdrm_lambda=.01,
        cdrm_output_states=False,recurrent_layers=[],precision=None,init_device='cpu')
    raw.update(overrides or {})
    return ModelConfig(**raw)


def sources(extra_paths=()):
    paths=set(Path('recurrent-transformer/olmo').rglob('*.py'))
    paths.update(PRESETS.values())
    paths.update(Path(name) for name in ('scripts/cdrm_fuzzy_common.py','scripts/cdrm_fuzzy_train.py',
        'scripts/cdrm_common.py','scripts/cdrm_train.py','scripts/cdrm_tiled_common.py',
        'scripts/stage_a_common.py','scripts/stage_b_train.py','scripts/r3_mixed_operational.py',
        'scripts/r3_validation_metrics.py','scripts/experiment_tracking.py','cdrm/mad_data.py',
        'vendors/mad-lab/mad/data/instances.py','vendors/mad-lab/configs/tasks/fuzzy-in-context-recall.yml'))
    paths.update(Path(name) for name in extra_paths)
    if any(not path.is_file() for path in paths):raise ValueError('A declared training dependency is missing')
    return {str(path):file_digest(path) for path in sorted(paths)}


def setup(seed=0):
    runtime=old.setup(seed)
    runtime['execution_contract'].update(ordinary_attention_precision_policy='fp32',
        supported_mixed_policy='All ordinary attention FP32; ordinary MLP/head and tiled fabric BF16; FP32 parameters/residuals/norms/state/CE/Adam')
    return runtime


def parameter_counts(model):
    named=dict(model.named_parameters())
    if any(p.device.type=='meta' for p in named.values()):
        if len(named)!=len({id(p) for p in named.values()}):raise AssertionError('Duplicate canonical parameter identities')
    else:unique_parameters(model)
    total=sum(p.numel() for p in named.values())
    tables=sum(named[name].numel() for name in ('transformer.wte.weight','transformer.ff_out.weight'))
    adapters=sum(p.numel() for name,p in named.items() if name.startswith('cdrm.'))
    return {'total':total,'adapters':adapters,'input_and_output_tables':tables,
            'ordinary_backbone_excluding_tables_and_adapters':total-tables-adapters,
            'canonical_parameter_tensors':len(named)}


def copy_paired_initialization(cdrm,seq):
    """Copy equal shapes only; resized MLP tensors retain their native draws."""
    source=cdrm.state_dict();target=seq.state_dict()
    if set(source)-set(target)!=ADAPTERS or set(target)-set(source):raise ValueError('Unexpected pairing owners')
    shared,resized=[],[]
    with torch.no_grad():
        for name,value in target.items():
            if source[name].shape==value.shape:
                value.copy_(source[name]);shared.append(name)
            else:
                if not name.endswith(('.ff_proj.weight','.ff_out.weight')) or not name.startswith('transformer.blocks.'):
                    raise ValueError(f'Unexpected non-MLP shape difference: {name}')
                resized.append(name)
    if len(resized)!=2*seq.config.n_layers:raise AssertionError('Exactly two resized MLP tensors per block are required')
    if any(not torch.equal(source[name],target[name]) for name in shared):raise AssertionError('Paired shared weights differ')
    return {'policy':'Copy shape-identical embeddings, output head, attention and norms; retain native resized MLP initialization',
            'copied_names':shared,'native_resized_mlp_names':resized,'additional_cdrm_owners':sorted(ADAPTERS),
            'shared_initialization_sha256':state_digest({name:source[name] for name in shared}),
            'shape_identical_parameters_copied_bitwise':True}


def cpu_initial_pair(seed,*,length=256,overrides=None):
    from olmo.model import OLMo
    seed_cpu(seed)
    cdrm=OLMo(config('cdrm','fp32','naive',length=length,overrides=overrides))
    # Isolate the two nonzero adapter draws from the backbone generator history.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+100003)
        with torch.no_grad():
            for name in ('cdrm.deep_adapter.weight','cdrm.bridge_adapter.weight'):
                torch.nn.init.normal_(cdrm.state_dict()[name],std=cdrm.config.cdrm_adapter_init_scale/math.sqrt(cdrm.config.d_model))
    seed_cpu(seed+100003)
    seq=OLMo(config('seq','fp32','naive',length=length,overrides=overrides))
    pairing=copy_paired_initialization(cdrm,seq)
    models={'cdrm':cdrm,'seq':seq}
    counts={arm:parameter_counts(model) for arm,model in models.items()}
    if overrides is None and any(counts[arm]['total']!=PARAMETERS[arm] for arm in ARMS):
        raise AssertionError('Actual unique parameter counts differ from the declared comparison')
    if any(torch.count_nonzero(cdrm.state_dict()[name]).item()==0 for name in ADAPTERS):raise AssertionError('Adapters must have nonzero initialization')
    pairing.update(seed=seed,adapter_seed=seed+100003,seq_native_seed=seed+100003,training_seed=seed+200003,
        parameter_counts=counts,model_sha256={arm:state_digest(model.state_dict()) for arm,model in models.items()},
        native_initialization='Released Mitchell initialization; no reshaping, padding, partial copying or variance changes to resized MLPs')
    return models,pairing


def validate_protocol(path,seed=None):
    document=json.loads(Path(path).read_text())
    if document.get('schema')!='cdrm-fuzzy-prospective-protocol-v1':raise ValueError('Require the prospective fuzzy protocol')
    if seed is not None and seed not in (document['initialization']['calibration_model_seed'],document['initialization']['additional_numerical_model_seed']):
        raise ValueError('Initialization seed was not declared before generation')
    if document['initialization']['adapter_seed_offset']!=100003:raise ValueError('Adapter seed offset differs')
    for arm in ARMS:
        cfg=config(arm,'fp32','naive');declared=document['models'][arm]
        for key,field in (('width','d_model'),('heads','n_heads'),('layers','n_layers'),('mlp','mlp_hidden_size')):
            if declared[key]!=getattr(cfg,field):raise ValueError('Protocol/model architecture differs')
        if declared['parameters']!=PARAMETERS[arm]:raise ValueError('Protocol parameter count differs')
        if arm=='cdrm':
            for key,field in (('early_layer','cdrm_early_layer'),('late_layer','cdrm_late_layer'),('rho','cdrm_rho'),('epsilon','cdrm_epsilon'),('lambda','cdrm_lambda')):
                if declared[key]!=getattr(cfg,field):raise ValueError('Protocol CDRM topology/gates differ')
    opt=document['optimizer'];schedule=document['schedule']
    for key,value in OPTIMIZER.items():
        if opt.get('clip_norm' if key=='clip' else key)!=value:raise ValueError('Protocol optimizer differs')
    if schedule!={'epochs':50,'eta_min':1e-6,'name':'epoch_cosine','step':'after completed epoch','warmup_epochs':0}:
        raise ValueError('Protocol schedule differs')
    if document['calibration']['learning_rates']!=list(LEARNING_RATES) or document['calibration']['epochs']!=10:
        raise ValueError('Protocol calibration budget differs')
    if document['data']['task']!=TASK or document['precision']['ordinary_attention_precision_policy']!='fp32':
        raise ValueError('Protocol task/precision differs')
    return document


def save_initial_pair(output_dir,seed,protocol,data_identity=None):
    output=Path(output_dir)
    if output.exists():raise FileExistsError('Paired initialization requires a new output directory')
    protocol=Path(protocol);validate_protocol(protocol,seed)
    tracked=sources([protocol]);output.mkdir(parents=True)
    snapshot_sources(output,tracked)
    atomic_json(output/'source-freeze.json',{'source_sha256':tracked,'protocol':reference(protocol),'seed':seed,
                'status':'frozen_before_model_initialization','declared_parameter_counts':PARAMETERS})
    report={'schema':'cdrm-fuzzy-paired-initialization-v1','status':'running',
            'source_sha256':tracked,'protocol':reference(protocol),'data_identity':data_identity,'checkpoints':{}}
    try:
        models,pairing=cpu_initial_pair(seed);report['initialization']=pairing
        for arm,model in models.items():
            cfg=dataclasses.asdict(model.config)
            identity={'format':INIT_FORMAT,'arm':arm,'model_config':cfg,'source_sha256':tracked,
                      'protocol_sha256':file_digest(protocol),'initialization':pairing,'data_identity':data_identity}
            payload={'format':INIT_FORMAT,'identity':identity,'identity_sha256':json_digest(identity),
                'arm':arm,'model_config':cfg,'model':model.state_dict(),'initialization':pairing,
                'completed_updates':0,'completed_epochs':0,'batch_in_epoch':0,
                'optimizer':optimizer_for(model,5e-4).state_dict()}
            report['checkpoints'][arm]=save_checkpoint(output/f'init-{arm}.pt',payload)
        verify_sources(tracked);report['status']='complete'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:atomic_json(output/'report.json',report)
    return report


def reference(path):
    path=Path(path)
    return {'path':str(path),'sha256':file_digest(path),'bytes':path.stat().st_size}


def load_initial(path):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload.get('format')!=INIT_FORMAT or payload.get('completed_updates')!=0
            or payload['optimizer']['state'] or payload['identity_sha256']!=json_digest(payload['identity'])):
        raise ValueError('Require a fresh paired initialization with empty Adam state')
    verify_sources(payload['identity']['source_sha256'])
    if payload['model_config']!=payload['identity']['model_config'] or payload['arm']!=payload['identity']['arm']:
        raise ValueError('Initial arm/config disagrees with its identity')
    if state_digest(payload['model'])!=payload['initialization']['model_sha256'][payload['arm']]:
        raise ValueError('Initial weights changed')
    shared={name:payload['model'][name] for name in payload['initialization']['copied_names']}
    if state_digest(shared)!=payload['initialization']['shared_initialization_sha256']:raise ValueError('Paired shared initialization changed')
    return payload


def build_model(arm,precision='bf16',backend='tiled',*,checkpoint=None,device='cuda',length=256,eager=False):
    from olmo.config import ModelConfig
    from olmo.model import OLMo
    if checkpoint is None:raise ValueError('Prepare an explicit paired initialization before constructing an arm')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False) if isinstance(checkpoint,(str,Path)) else checkpoint
    if payload.get('format') not in (INIT_FORMAT,FORMAT) or payload.get('arm')!=arm:
        raise ValueError('Checkpoint belongs to a different fuzzy architecture or lineage')
    if payload['identity_sha256']!=json_digest(payload['identity']):raise ValueError('Checkpoint identity digest differs')
    if payload['arm']!=payload['identity']['arm'] or payload['model_config']!=payload['identity']['model_config']:
        raise ValueError('Checkpoint arm/config disagrees with its identity')
    verify_sources(payload['identity']['source_sha256'])
    expected=dataclasses.asdict(config(arm,precision,backend,length=length,eager=eager))
    allowed={'cdrm_backend','cdrm_precision_policy','reference_eager'}
    if any(payload['model_config'].get(name)!=value for name,value in expected.items() if name not in allowed):
        raise ValueError('Checkpoint architecture differs beyond explicit diagnostic backend/precision controls')
    model=OLMo(ModelConfig(**expected)).to(device=device,dtype=torch.float32)
    model.load_state_dict(payload['model'],strict=True)
    if state_digest(model.state_dict())!=state_digest(payload['model']):raise AssertionError('Model construction changed checkpoint weights')
    counts=parameter_counts(model)
    if counts['total']!=PARAMETERS[arm]:raise AssertionError('Unexpected parameter count')
    if str(device).startswith('cuda'):
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    return model,{'model_config':expected,'parameter_counts':counts,'parameter_count':counts['total'],
        'starting_weights_sha256':state_digest(payload['model']),'allowed_diagnostic_changes':sorted(allowed),
        'actual_fixture_length':length,'initialization':copy.deepcopy(payload['initialization'])}


def dataset_identity(dataset):
    return {'array_sha256':dataset.sha256,'examples':len(dataset),'length':dataset.input_ids.shape[1],
        'manifest_sha256':dataset.manifest.get('manifest_sha256'),'task':dataset.manifest['task'],
        'vocab_size':dataset.manifest['vocab_size'],'native_training_objective':dataset.manifest['native_training_objective']}


def validate_data(dataset,split,*,length=256):
    if (split not in ('train','dev') or dataset.manifest['task']!=TASK or dataset.manifest['vocab_size']!=16
            or dataset.input_ids.shape[1]!=length or dataset.manifest['split']!=split):
        raise ValueError('Require the declared train/development native fuzzy-recall corpus')
    if split=='train' and np.any(dataset.labels==-100):raise ValueError('Native fuzzy training must retain dense targets including padding')
    if split=='dev' and not np.array_equal(dataset.labels,dataset.answer_labels):raise ValueError('Development uses exactly native masked answer targets')
    if np.any(dataset.answer_labels==15):raise ValueError('Padding must not count as retrieval success')


def update(model,optimizer,ids,labels,answers,*,precision='bf16',monitor=False):
    if not ids.is_cuda:raise RuntimeError('Training execution requires the authorized CUDA container')
    model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
    with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
        output=model(ids,output_cdrm_states=monitor and model.config.cdrm_enabled)
    expected=torch.bfloat16 if precision=='bf16' else torch.float32
    if output.logits.dtype!=expected:raise AssertionError('Observed logits precision differs')
    summed,count=native_loss_sum(output.logits,labels);loss=summed/count;loss.backward()
    if any(p.dtype!=torch.float32 or p.grad is None or p.grad.dtype!=torch.float32 for p in model.parameters()):
        raise AssertionError('Every canonical owner must receive an FP32 gradient')
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True).item())
    optimizer.step()
    row={'native_loss':float(loss.item()),'native_targets':count,'input_tokens':ids.numel(),
        'gradient_norm':norm,'clipped':norm>1.,'clip_coefficient':min(1.,1./(norm+1e-6)),
        'learning_rate':optimizer.param_groups[0]['lr'],'logits_dtype':str(output.logits.dtype),'ce_dtype':str(loss.dtype),
        'native':finish_metrics(metric_counts(output.logits.detach(),labels)),
        'answer':finish_metrics(metric_counts(output.logits.detach(),answers)),
        'all_parameter_and_gradient_dtypes_fp32':True,'all_gradients_finite':True,
        'full_state_monitor_performed':monitor}
    if monitor:row['effective_precision']=effective_precision(model,optimizer,require_gradients=True)
    if monitor and model.config.cdrm_enabled:row['state_summaries']=state_summaries(output.cdrm_states)
    if not math.isfinite(row['native_loss']) or loss.dtype!=torch.float32:raise FloatingPointError('Native CE must be finite FP32')
    torch.cuda.synchronize();row['seconds']=time.perf_counter()-start
    return row


@torch.no_grad()
def evaluate(model,dataset,batch_size,*,precision='bf16'):
    if next(model.parameters()).device.type!='cuda':raise RuntimeError('Evaluation requires the authorized CUDA container')
    modes={module:module.training for module in model.modules()}
    totals={name:{'ce_sum':0.,'targets':0,'correct':0,'exact':0,'examples':0} for name in ('native','answer')}
    start=time.perf_counter()
    try:
        with preserve_tracking_rng():
            model.eval()
            for begin in range(0,len(dataset),batch_size):
                ids=torch.as_tensor(dataset.input_ids[begin:begin+batch_size],device='cuda')
                with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                    logits=model(ids).logits
                for name,array in (('native',dataset.labels),('answer',dataset.answer_labels)):
                    labels=torch.as_tensor(array[begin:begin+batch_size],device='cuda')
                    for key,value in metric_counts(logits,labels).items():totals[name][key]+=value
            torch.cuda.synchronize()
    finally:
        for module,training in modes.items():module.training=training
    return {**{name:finish_metrics(values) for name,values in totals.items()},'seconds':time.perf_counter()-start}
