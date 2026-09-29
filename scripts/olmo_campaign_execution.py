"""Shared all-arm packed execution at recoverable optimizer boundaries.

The versioned launcher supplies an immutable identity. Observation verbosity and
segment limits do not change the objective or checkpoint lineage. Historical
acceptance helpers and checkpoint formats are left unchanged.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import torch

from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.campaign_recipe import ARMS, CampaignRecipe, feedback_noise_for_rows
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
from cdrm.pretrained.distributed_checkpoint import (DistributedCheckpointError,
    inspect_distributed_checkpoint, load_distributed_checkpoint, save_distributed_checkpoint)
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, optimizer_ownership
from cdrm.pretrained.lm_training import parameter_layout
from scripts.experiment_tracking import scalar_metrics
from scripts.olmo_campaign_loop import LifecycleError, LoopPolicy, StopRequest, run_loop
from scripts.olmo_campaign_loop_guarded import retain_without_rng
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_restart import boundary, checkpoint_disk_preflight
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_recovery import seed_local

SCHEMA = 'olmo-campaign-execution-v1'
CURSOR_SCHEMA = 'olmo-campaign-execution-cursor-v1'
WORLD_SIZE = 2
WARMUP = 11


@dataclass(frozen=True)
class SegmentOptions:
    output_dir: Path
    checkpoint_root: Path
    storage_prefix: str
    max_updates: int
    checkpoint_seconds: int = 600
    checkpoint_every_updates: int = 1
    observation: str = 'lean'
    resume: Path | None = None
    resume_manifest_sha256: str | None = None
    stop_file: Path | None = None

    def __post_init__(self):
        if (type(self.max_updates) is not int or self.max_updates < 0
                or type(self.checkpoint_every_updates) is not int or self.checkpoint_every_updates < 1
                or self.observation not in ('lean', 'acceptance')
                or (self.resume is None) != (self.resume_manifest_sha256 is None)):
            raise ValueError('Invalid segment/observation/recovery declaration')
        LoopPolicy(self.max_updates, self.checkpoint_seconds)
        if self.resume_manifest_sha256 is not None:
            from scripts.olmo_campaign_manifest import pin
            pin(self.resume_manifest_sha256)
        from scripts.olmo_campaign_loop_run import storage_location
        storage_location(self.storage_prefix)


def model_contract(model, recipe, optimizer=None):
    """Actual ownership for all component selections, including dormant fusion."""
    if recipe.arm not in ARMS or model.pass_loss_policy != 'campaign_v1':
        raise ValueError('Unsupported campaign model')
    params = dict(model.named_parameters())
    components = {key: set() for key in ('backbone', 'fusion', 'predictor')}
    for name in params:
        if name.startswith('backbone.backbone.'):
            components['backbone'].add(name)
        elif name.startswith('backbone.fusion.'):
            components['fusion'].add(name)
        elif name.startswith('predictor.'):
            components['predictor'].add(name)
        else:
            raise ValueError('Unaccounted model parameter')
    expected = components['backbone'] | (components['fusion'] if recipe.feedback else set()) | components['predictor']
    active = {name for name,p in params.items() if p.requires_grad}
    weights = {'ce':1., 'latent':float(recipe.nextlat), 'kl':float(recipe.nextlat)}
    if (active != expected or not components['backbone'] or len(components['fusion']) != 2
            or bool(components['predictor']) != recipe.nextlat or model.enabled != recipe.nextlat
            or (model.predictor is not None) != recipe.nextlat or model.objective_weights() != weights
            or model.gamma != 1 or model.config.document_policy != recipe.document_policy
            or any(p.dtype != torch.float32 for p in params.values())
            or model.backbone.readout_weight is not model.backbone.token_embeddings.weight):
        raise ValueError('Active/dormant parameter ownership or objective differs')
    ownership = None
    if optimizer is not None:
        ownership = optimizer_ownership(model, optimizer)
        expected_components = {'backbone'} | ({'fusion'} if recipe.feedback else set()) | ({'predictor'} if recipe.nextlat else set())
        if {group.get('component') for group in optimizer.param_groups} != expected_components:
            raise ValueError('Optimizer component telemetry differs')
        for group, names in zip(optimizer.param_groups, ownership):
            if set(names) - components[group['component']]:
                raise ValueError('Optimizer group includes another component')
    return {'arm':recipe.arm, 'mode':asdict(recipe.mode()), 'weights':weights,
        'resident_parameters':sum(p.numel() for p in params.values()),
        'trainable_parameters':sum(params[n].numel() for n in active),
        'component_parameters':{k:sum(params[n].numel() for n in names) for k,names in components.items()},
        'dormant_fusion_parameters':0 if recipe.feedback else sum(params[n].numel() for n in components['fusion']),
        'parameter_layout':parameter_layout(model), 'optimizer_ownership':ownership,
        'tied_readout':True, 'cursor_schema':CURSOR_SCHEMA}


def cursor_record(cursor, *, rank, batch_size):
    return {'schema':CURSOR_SCHEMA, 'cursor':asdict(cursor), 'rank':rank,
            'world_size':WORLD_SIZE, 'physical_batch_per_rank':batch_size}


def expected_counters(plans, completed, recipe, batch_size):
    if type(completed) is not int or not 0 <= completed <= len(plans):
        raise ValueError('Completed update is outside immutable plan')
    result = TrainingCounters()
    for plan in plans[:completed]:
        result.optimizer_updates += 1
        result.input_tokens += plan.counts.valid_tokens
        result.documents += plan.counts.packed_rows
        result.microbatches += WORLD_SIZE*((len(plan.rows)+WORLD_SIZE*batch_size-1)//(WORLD_SIZE*batch_size))
        result.ce_positions += plan.counts.ce_targets
        if recipe.nextlat:
            result.latent_pairs += plan.counts.latent_pairs
            result.kl_triples += plan.counts.kl_triples
    return result


def validate_cursor(data, record, counters, plans, recipe, *, rank, batch_size, restore=False):
    if not plans or asdict(counters) != asdict(expected_counters(plans, counters.optimizer_updates, recipe, batch_size)):
        raise ValueError('Counters differ from enabled losses and committed data plan')
    cursor = plans[0].start_cursor if counters.optimizer_updates == 0 else plans[counters.optimizer_updates-1].next_cursor
    if record != cursor_record(cursor, rank=rank, batch_size=batch_size):
        raise ValueError('Checkpoint cursor/partition differs')
    if restore:
        data.restore_cursor(record['cursor'])
    if data.cursor() != cursor:
        raise ValueError('Reader cursor differs from committed data plan')
    return cursor


def materialize(data, plan, recipe, *, rank, batch_size, width):
    packed = data.rank_batches(plan, rank=rank, world_size=WORLD_SIZE, physical_batch_size=batch_size)
    noises = tuple(feedback_noise_for_rows(recipe, keys, logical_update=plan.start_cursor.next_update,
        sequence_length=recipe.sequence_length, width=width, physical_batch_size=batch_size)
        for keys in packed.keys)
    return packed, noises


def validate_clocks(optimizer, scheduler, counters):
    """Check saved optimizer/schedule clocks at boundaries, including terminal resume."""
    updates=counters.optimizer_updates
    rates=scheduler.get_lr()
    if (scheduler.last_epoch!=updates or scheduler.completed_tokens!=counters.input_tokens
            or scheduler._step_count!=updates+1 or scheduler.get_last_lr()!=rates
            or [group['lr'] for group in optimizer.param_groups]!=rates):
        raise ValueError('Scheduler/learning-rate clocks differ from committed counters')
    owned=[parameter for group in optimizer.param_groups for parameter in group['params']]
    if updates==0:
        if optimizer.state:raise ValueError('Origin Adam must have no inherited moments')
    else:
        if set(optimizer.state)!=set(owned):raise ValueError('Completed Adam state must cover all owned parameters')
        for parameter in owned:
            state=optimizer.state[parameter]
            step=state.get('step')
            if not isinstance(step,torch.Tensor) or step.numel()!=1 or step.item()!=updates:
                raise ValueError('Adam update clock differs from committed counters')
            for name in ('exp_avg','exp_avg_sq'):
                value=state.get(name)
                if not isinstance(value,torch.Tensor) or value.shape!=parameter.shape or value.dtype!=torch.float32:
                    raise ValueError('Completed Adam moments differ from FP32 master geometry')
    return {'optimizer_updates':updates,'input_tokens':counters.input_tokens,
        'scheduler_epoch':scheduler.last_epoch,'adam_parameters':len(optimizer.state)}


def transition_imported_model(model, historical_recipe, target_recipe):
    """State-preserving activation of declared packed/RT execution after strict NF import."""
    if (historical_recipe.arm != 'NF' or historical_recipe.document_policy != 'isolated-v1'
            or target_recipe.arm not in ('NF','NFR') or target_recipe.document_policy != 'continuous-stream-v1'
            or model.config.document_policy != 'isolated-v1'):
        raise ValueError('Unsupported adapted-state transition')
    if (historical_recipe.fusion_seed != target_recipe.fusion_seed
            or historical_recipe.predictor_seed != target_recipe.predictor_seed
            or target_recipe.feedback_jitter != .02):
        raise ValueError('Adapted model seeds or feedback configuration differ')
    before = tree_digests(model.state_dict())
    identities = {n:id(p) for n,p in model.named_parameters()}
    flags = {n:p.requires_grad for n,p in model.named_parameters()}
    modes = {n:m.training for n,m in model.named_modules()}
    if not all(flags.values()):
        raise ValueError('Adapted construction must already have all active parameters trainable')
    old = model.config.to_dict()
    model.config = replace(model.config, document_policy=target_recipe.document_policy)
    checks = {'state_exact':tree_digests(model.state_dict()) == before,
        'parameter_identities_exact':identities == {n:id(p) for n,p in model.named_parameters()},
        'trainability_exact':flags == {n:p.requires_grad for n,p in model.named_parameters()},
        'modes_exact':modes == {n:m.training for n,m in model.named_modules()},
        'only_nextlat_policy':old == {k:v for k,v in model.config.to_dict().items() if k!='document_policy'}}
    if not all(checks.values()):
        raise ValueError('Adapted transition changed unrelated model state')
    model_contract(model, target_recipe)
    return {'checks':checks, 'historical_recipe':historical_recipe.to_dict(),
            'target_recipe':target_recipe.to_dict(), 'loaded_optimizer':False,
            'transition':'NF isolated fusion weights -> declared packed NF/NFR, fresh all-active Adam'}


def run_segment(*, options, coordinator, model, recipe, data, plans, optimizer, scheduler,
                configuration, source_fingerprint, device, tracker, report, batch_size,
                validate_resume_metadata):
    """One shared math path; recover without a reference report or live graphs."""
    from scripts.olmo_campaign_execution_observer import ExecutionObserver
    rank = coordinator.rank
    if coordinator.world_size != WORLD_SIZE or not 0 <= options.max_updates <= len(plans):
        raise ValueError('Require accepted two-rank topology and bounded segment')
    training = LMTrainingConfig(precision=configuration['training']['precision'], max_grad_norm=recipe.max_grad_norm)
    contract = model_contract(model,recipe,optimizer)
    generators = {'data':torch.Generator().manual_seed(recipe.jitter_seed+rank),
                  'local':torch.Generator(device=device).manual_seed(recipe.jitter_seed+WORLD_SIZE+rank)}
    seed_local(recipe.jitter_seed+rank, device)
    counters = TrainingCounters()
    runner = None
    report['timing'] = {'scope':'Wall regions are not optimized steady-state throughput; checkpoint boundary hashing and serialization are separate from transfer.'}
    def persist():
        report['wandb'] = tracker.record if tracker is not None else None
        write_json(options.output_dir/'report.json',report)
    def current_boundary():
        return boundary(model,optimizer,scheduler,counters,
            cursor_record(data.cursor(),rank=rank,batch_size=batch_size),device,generators)
    if options.resume is not None:
        def preflight():
            saved = inspect_distributed_checkpoint(options.resume,
                expected_manifest_sha256=options.resume_manifest_sha256, verify_state=False)
            validate_resume_metadata(saved, configuration['execution_identity'])
            return saved
        started=time.perf_counter()
        saved=coordinator.call('committed resume metadata',preflight)
        restored=load_distributed_checkpoint(options.resume,model,optimizer,scheduler=scheduler,
            configuration=configuration,source_fingerprint=source_fingerprint,generators=generators,
            expected_manifest_sha256=options.resume_manifest_sha256,device=device)
        counters=restored['counters']
        coordinator.call('restore committed cursor',lambda:validate_cursor(data,restored['data_cursor'],counters,
            plans,recipe,rank=rank,batch_size=batch_size,restore=True))
        report['resume']={'manifest_sha256':options.resume_manifest_sha256,
            'completed_update':counters.optimizer_updates,'reference_report_required':False}
        report['timing']['restore_seconds_by_rank']=coordinator.gather(time.perf_counter()-started)
    if counters.optimizer_updates > options.max_updates:
        raise ValueError('Segment limit precedes restored completed boundary')
    report['origin_clocks']=coordinator.call('origin optimizer/schedule clocks',lambda:validate_clocks(optimizer,scheduler,counters))
    report['origin_boundary_by_rank']=coordinator.gather(current_boundary())
    report['adam_resident_before_ddp']=bool(optimizer.state)
    observer=ExecutionObserver(options.observation,model=model,optimizer=optimizer,scheduler=scheduler,
        device=device,generators=generators,max_grad_norm=recipe.max_grad_norm)
    coordinator.call('origin evidence',persist,rank_zero=True)

    def save(number,reason):
        destination=options.checkpoint_root/f'update-{number:06d}'
        coordinator.call('checkpoint optimizer/schedule clocks',lambda:validate_clocks(optimizer,scheduler,counters))
        coordinator.call('checkpoint disk preflight',lambda:checkpoint_disk_preflight(destination,model))
        started=time.perf_counter();before=current_boundary()
        state=coordinator.gather(before['state'])
        if any(x!=state[0] for x in state):raise LifecycleError('Checkpoint replicas differ')
        report['timing'].setdefault('checkpoint_observation_seconds_by_rank',{})[str(number)]=coordinator.gather(time.perf_counter()-started)
        started=time.perf_counter()
        try:
            with runner.checkpoint_boundary() if runner is not None else nullcontext():
                receipt=save_distributed_checkpoint(destination,model,optimizer,scheduler=scheduler,
                    counters=counters,data_cursor=cursor_record(data.cursor(),rank=rank,batch_size=batch_size),
                    configuration=configuration,source_fingerprint=source_fingerprint,generators=generators,device=device)
        except DistributedCheckpointError:
            raise LifecycleError('Coordinated save failed; last retained boundary remains authoritative') from None
        report['timing'].setdefault('checkpoint_write_seconds_by_rank',{})[str(number)]=coordinator.gather(time.perf_counter()-started)
        started=time.perf_counter()
        if not all(coordinator.gather(before==current_boundary())):
            raise LifecycleError('Checkpoint changed live model/Adam/RNG/cursor boundary')
        report['timing'].setdefault('checkpoint_postcheck_seconds_by_rank',{})[str(number)]=coordinator.gather(time.perf_counter()-started)
        report.setdefault('local_checkpoints',[]).append({'optimizer_update':number,'reason':reason,
            'receipt':receipt,'boundary_by_rank':coordinator.gather(before)})
        coordinator.call('local checkpoint evidence',persist,rank_zero=True)
        return receipt
    def retain(receipt):
        started=time.perf_counter()
        result=retain_without_rng(receipt,options.storage_prefix)
        report['timing'].setdefault('checkpoint_retention_seconds',{})[str(receipt['counters']['optimizer_updates'])]=time.perf_counter()-started
        return result
    def publish(receipt):
        report.setdefault('published_checkpoints',[]).append(receipt)
        write_json(options.output_dir/'latest-checkpoint.json',receipt)
        persist()
    stop=StopRequest(options.stop_file)
    policy=LoopPolicy(options.max_updates,options.checkpoint_seconds,
        tuple(range(options.checkpoint_every_updates,len(plans)+1,options.checkpoint_every_updates)),save_initial=True)
    prepared=False
    def prepare():
        nonlocal runner,prepared
        # Lazy setup: initial checkpoint/stop and already-complete resumes do not
        # index a nonexistent next plan or capture an unused graph.
        started=time.perf_counter();before=current_boundary()
        packed,noises=coordinator.call('preparation data',lambda:materialize(data,plans[counters.optimizer_updates],recipe,
            rank=rank,batch_size=batch_size,width=model.config.model_dim))
        counts=sum_objective_counts(coordinator.gather(sum_objective_counts([model.counts(b) for b in packed.batches])))
        adapter=CampaignObjective(model,packed.batches[0],mode=recipe.mode(),global_counts=counts,
            world_size=WORLD_SIZE,feedback_noise=noises[0],config=training)
        runner=CampaignDDPGraphTraining(adapter)
        def phase(name,action):
            report['stage']='prepare/'+name+'/'+action
            coordinator.call('preparation evidence',persist,rank_zero=True)
        runner.prepare(warmup=WARMUP,phase_observer=phase)
        runner.capture(warmup=WARMUP,release_transient_cache=True,phase_observer=phase)
        exact=coordinator.gather(before==current_boundary())
        if not all(exact):raise LifecycleError('Graph preparation changed complete boundary')
        report['preparation_boundary_exact']=exact
        report['timing']['preparation_seconds_by_rank']=coordinator.gather(time.perf_counter()-started)
        report['memory_after_capture']=coordinator.gather(memory())
        prepared=True
    def update():
        if not prepared:prepare()
        index=counters.optimizer_updates
        coordinator.call('validate committed cursor',lambda:validate_cursor(data,
            cursor_record(data.cursor(),rank=rank,batch_size=batch_size),counters,plans,recipe,rank=rank,batch_size=batch_size))
        started=time.perf_counter()
        packed,noises=coordinator.call('update data',lambda:materialize(data,plans[index],recipe,
            rank=rank,batch_size=batch_size,width=model.config.model_dim))
        materialization=time.perf_counter()-started
        started=time.perf_counter()
        inputs=coordinator.call('acceptance input observation',lambda:tree_digests(
            {'batches':[vars(b) for b in packed.batches],'noise':noises,'keys':packed.keys,
             'cursor':asdict(data.cursor())})) if options.observation=='acceptance' else None
        input_observation=time.perf_counter()-started
        torch.cuda.synchronize(device);started=time.perf_counter()
        result=runner.backward(packed.batches,feedback_noises=noises,replay=True)
        torch.cuda.synchronize(device);backward_seconds=time.perf_counter()-started
        gradients=coordinator.call('acceptance gradient observation',observer.after_backward) if options.observation=='acceptance' else observer.after_backward()
        started=time.perf_counter()
        metrics=runner.step(result,optimizer,scheduler=scheduler,counters=counters)
        coordinator.call('commit completed cursor',lambda:data.commit(plans[index].start_cursor,plans[index]))
        torch.cuda.synchronize(device);step_seconds=time.perf_counter()-started
        coordinator.call('post-update clocks',lambda:validate_cursor(data,
            cursor_record(data.cursor(),rank=rank,batch_size=batch_size),counters,plans,recipe,rank=rank,batch_size=batch_size))
        observation=coordinator.call('update observation',lambda:observer.after_update(metrics,counters=counters,
            cursor=cursor_record(data.cursor(),rank=rank,batch_size=batch_size),input_record=inputs,raw_gradients=gradients))
        rows=coordinator.gather(observation)
        # Metrics are globally reduced by the common runner; every rank should
        # report identical completed counters and loss sums under either observer.
        if any(row['metrics']!=rows[0]['metrics'] for row in rows):raise LifecycleError('Replica metrics disagree')
        report.setdefault('updates',{})[str(counters.optimizer_updates)]=rows
        report.setdefault('observations',{})[str(counters.optimizer_updates)]={
            'timing_by_rank':coordinator.gather({'materialization_host':materialization,'input_hash_host':input_observation,
                'backward':backward_seconds,'optimizer_and_cursor':step_seconds}),
            'memory_by_rank':coordinator.gather(memory()),'rank_data':coordinator.gather(packed.accounting)}
        coordinator.call('completed update evidence',persist,rank_zero=True)
        return metrics
    def log(metrics):
        tracker.log({'update':counters.optimizer_updates,**scalar_metrics(metrics,'train')},step=counters.optimizer_updates)
    with stop.installed():
        report['loop']=run_loop(coordinator=coordinator,policy=policy,completed=lambda:counters.optimizer_updates,
            update=update,log=log,save=save,publish_checkpoint=publish,stop=stop,retain=retain,
            restored=options.resume is not None)
    report['final_clocks']=coordinator.call('final optimizer/schedule clocks',lambda:validate_clocks(optimizer,scheduler,counters))
    report['final_boundary_by_rank']=coordinator.gather(current_boundary())
    report['final_counters']=asdict(counters)
    report['runner_by_rank']=coordinator.gather(None if runner is None else runner.metadata)
    report['segment_completed']=counters.optimizer_updates==options.max_updates
    report['plan_completed']=counters.optimizer_updates==len(plans)
    report['graph_prepared']=prepared
    if model_contract(model,recipe,optimizer)!=contract:raise LifecycleError('Final component ownership changed')
    coordinator.call('final segment evidence',persist,rank_zero=True)
