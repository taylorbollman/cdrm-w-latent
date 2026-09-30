"""NFR64-to128 metadata-only continuation of the accepted KL segment.

Mathematical callbacks remain literal copies of the frozen KL engine. Changes
are confined to strict old64 loading, a state-preserving metadata transition,
and declared named checkpoint boundaries. No objective transition is applied.
"""
from contextlib import nullcontext
from dataclasses import asdict
import time
import torch
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
from cdrm.pretrained.distributed_checkpoint import (DistributedCheckpointError,
    inspect_distributed_checkpoint, load_distributed_checkpoint, save_distributed_checkpoint)
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from scripts.experiment_tracking import scalar_metrics
from scripts.olmo_campaign_loop import LifecycleError, LoopPolicy, StopRequest
from scripts.olmo_pilot_async_loop import run_loop
from scripts.olmo_pilot_async_storage import AsyncCheckpointRetention
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_restart import boundary, checkpoint_disk_preflight
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_recovery import seed_local
from scripts.olmo_campaign_execution import (SegmentOptions, cursor_record,
    expected_counters, validate_cursor, materialize, validate_clocks, transition_imported_model,
    CURSOR_SCHEMA, WORLD_SIZE, WARMUP)

from scripts.olmo_kl_branch import branch_model_contract
from scripts.olmo_nfr_128_contract import validate_continuation_transition

SCHEMA = 'olmo-nfr-128-execution-v1'

def run_segment(*, options, coordinator, model, recipe, data, plans, optimizer, scheduler,
                configuration, source_fingerprint, device, tracker, report, batch_size,
                validate_resume_metadata, storage, evaluation=None, checkpoint_mode="async",
                branch=None, parent_configuration=None, parent_fingerprint=None, transition_parent=False,
                parent_boundary_by_rank=None, continuation=None, named_checkpoints=()):
    """Strictly load old64 or a child; leave the existing KL0.1 objective intact."""
    from scripts.olmo_campaign_execution_observer import ExecutionObserver
    if checkpoint_mode not in ("async", "blocking"):
        raise ValueError("Explicit async or blocking checkpoint mode is required")
    rank = coordinator.rank
    if coordinator.world_size != WORLD_SIZE or not 0 <= options.max_updates <= len(plans):
        raise ValueError('Require accepted two-rank topology and bounded segment')
    training = LMTrainingConfig(precision=configuration['training']['precision'], max_grad_norm=recipe.max_grad_norm)
    if branch is None or options.resume is None or continuation is None:
        raise ValueError('Continuation requires an explicit parent or child checkpoint')
    def model_contract(model, recipe, optimizer):
        return branch_model_contract(model,recipe,optimizer,kl_weight=branch['kl_weight'])
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
            if transition_parent:
                if (saved['manifest_sha256'] != continuation['parent_manifest_sha256']
                        or saved['counters']['optimizer_updates'] != continuation['parent_update']
                        or saved['metadata']['configuration'] != parent_configuration
                        or saved['metadata']['source_fingerprint'] != parent_fingerprint):
                    raise ValueError('Parent checkpoint differs from original authenticated configuration')
            else:
                validate_resume_metadata(saved, configuration['execution_identity'])
            return saved
        started=time.perf_counter()
        saved=coordinator.call('committed resume metadata',preflight)
        restored=load_distributed_checkpoint(options.resume,model,optimizer,scheduler=scheduler,
            configuration=parent_configuration if transition_parent else configuration,
            source_fingerprint=parent_fingerprint if transition_parent else source_fingerprint,generators=generators,
            expected_manifest_sha256=options.resume_manifest_sha256,device=device)
        counters=restored['counters']
        coordinator.call('restore committed cursor',lambda:validate_cursor(data,restored['data_cursor'],counters,
            plans,recipe,rank=rank,batch_size=batch_size,restore=True))
        report['resume']={'manifest_sha256':options.resume_manifest_sha256,
            'completed_update':counters.optimizer_updates,'reference_report_required':True,
            'parent_files_required':True,'parent_boundary_compared':bool(transition_parent)}
        report['timing']['restore_seconds_by_rank']=coordinator.gather(time.perf_counter()-started)
        if transition_parent:
            before=coordinator.gather(current_boundary())
            if before != parent_boundary_by_rank:
                raise LifecycleError('Loaded parent differs from its independently recorded saved boundary')
            changes=coordinator.call('declared metadata-only continuation',lambda:validate_continuation_transition(
                parent_configuration, configuration, model, branch, continuation))
            after=coordinator.gather(current_boundary())
            if before != after:
                raise LifecycleError('Metadata continuation changed weights/Adam/schedule/counters/cursor/RNG')
            report['continuation_transition']={'parent_manifest_sha256':continuation['parent_manifest_sha256'],
                'before_boundary_by_rank':before,'after_boundary_by_rank':after,
                'configuration_change_by_rank':coordinator.gather(changes)}
        else:
            report['branch_resume']=True
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
        coordinator.call('owned SSD checkpoint destination',lambda:storage.destination(number),rank_zero=True)
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
    manager=None
    def create_manager():
        nonlocal manager
        manager=AsyncCheckpointRetention(storage, evidence_dir=options.output_dir,
            storage_prefix=options.storage_prefix, worker_timeout_seconds=480,
            source_pins=report['sources'])
        return {'maximum_pending':1,'worker_timeout_seconds':480,'mode':checkpoint_mode}
    report['checkpoint_transport']=coordinator.call('background retention setup',create_manager,rank_zero=True)
    def submit(receipt):
        job=manager.submit(receipt)
        report.setdefault('async_submissions',[]).append(job)
        persist()
        return job
    def poll(*,wait=False):
        return manager.drain() if wait else manager.poll()
    def accept(record):
        receipt=record['receipt'];number=receipt['counters']['optimizer_updates']
        report.setdefault('published_checkpoints',[]).append(receipt)
        report.setdefault('storage_publications',[]).append(record['storage_publication'])
        report.setdefault('async_completions',[]).append(record)
        report['last_verified_cloud_update']=number
        persist()
        return number
    stop=StopRequest(options.stop_file)
    policy=LoopPolicy(options.max_updates,options.checkpoint_seconds,
        tuple(sorted(set(range(options.checkpoint_every_updates,len(plans)+1,options.checkpoint_every_updates))
                     |set(named_checkpoints))),save_initial=True)
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
        update_started=time.perf_counter()
        update_started_unix=time.time()
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
        if evaluation is not None:
            evaluation.run_if_due(counters.optimizer_updates, model=model, runner=runner,
                generators=generators, training_data=data, boundary=current_boundary, persist=persist,log_metrics=False)
        torch.cuda.synchronize(device)
        report.setdefault('update_wall_seconds_by_rank',{})[str(counters.optimizer_updates)]=coordinator.gather(time.perf_counter()-update_started)
        report.setdefault('update_intervals_unix_seconds',{})[str(counters.optimizer_updates)]={'started':update_started_unix,'finished':time.time()}
        coordinator.call('complete update wall evidence',persist,rank_zero=True)
        return metrics
    def log(metrics):
        dev={} if evaluation is None else evaluation.metrics_for(counters.optimizer_updates)
        tracking={'update':counters.optimizer_updates,**scalar_metrics(metrics,'train'),**dev,
            'checkpoint/last_local_update':report.get('local_checkpoints',[{'optimizer_update':0}])[-1]['optimizer_update'],
            'checkpoint/last_verified_cloud_update':report.get('last_verified_cloud_update',
                report.get('resume',{}).get('completed_update',-1)),
            'checkpoint/worker_pending':manager.pending if manager is not None else False}
        tracker.log(tracking,step=counters.optimizer_updates)
    # A restored scheduled boundary is evaluated once in this segment. We do
    # not infer publication from another process's report or mutate its checkpoint.
    if options.resume is not None and evaluation is not None:
        evaluation.run_if_due(counters.optimizer_updates, model=model, runner=None,
            generators=generators, training_data=data, boundary=current_boundary, persist=persist)
    try:
        with stop.installed():
            report['loop']=run_loop(coordinator=coordinator,policy=policy,completed=lambda:counters.optimizer_updates,
                update=update,log=log,save=save,submit_checkpoint=submit,poll_checkpoint=poll,
                accept_checkpoint=accept,stop=stop,restored=options.resume is not None,
                blocking_checkpoints=checkpoint_mode=='blocking')
    except BaseException:
        # No CUDA or distributed calls here; failed-update cleanup must not
        # invent another collective. Pending local files are never deleted.
        if manager is not None:report['async_teardown']=manager.abort()
        raise
    else:
        if manager is not None:manager.close()
    report['final_clocks']=coordinator.call('final optimizer/schedule clocks',lambda:validate_clocks(optimizer,scheduler,counters))
    report['final_boundary_by_rank']=coordinator.gather(current_boundary())
    report['final_counters']=asdict(counters)
    report['runner_by_rank']=coordinator.gather(None if runner is None else runner.metadata)
    report['segment_completed']=counters.optimizer_updates==options.max_updates
    report['plan_completed']=counters.optimizer_updates==len(plans)
    report['graph_prepared']=prepared
    if model_contract(model,recipe,optimizer)!=contract:raise LifecycleError('Final component ownership changed')
    coordinator.call('final segment evidence',persist,rank_zero=True)
