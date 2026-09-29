"""Versioned shared execution with declared read-only evaluation boundaries."""
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
from scripts.olmo_campaign_loop import LifecycleError, LoopPolicy, StopRequest, run_loop
from scripts.olmo_campaign_loop_guarded import retain_without_rng
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_restart import boundary, checkpoint_disk_preflight
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_recovery import seed_local
from scripts.olmo_campaign_execution import (SegmentOptions, model_contract, cursor_record,
    expected_counters, validate_cursor, materialize, validate_clocks, transition_imported_model,
    CURSOR_SCHEMA, WORLD_SIZE, WARMUP)

SCHEMA = 'olmo-campaign-evaluation-execution-v1'

def run_segment(*, options, coordinator, model, recipe, data, plans, optimizer, scheduler,
                configuration, source_fingerprint, device, tracker, report, batch_size,
                validate_resume_metadata, evaluation=None):
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
        if evaluation is not None:
            evaluation.run_if_due(counters.optimizer_updates, model=model, runner=runner,
                generators=generators, training_data=data, boundary=current_boundary, persist=persist,log_metrics=False)
        return metrics
    def log(metrics):
        dev={} if evaluation is None else evaluation.metrics_for(counters.optimizer_updates)
        tracker.log({'update':counters.optimizer_updates,**scalar_metrics(metrics,'train'),**dev},step=counters.optimizer_updates)
    # A restored scheduled boundary is evaluated once in this segment. We do
    # not infer publication from another process's report or mutate its checkpoint.
    if options.resume is not None and evaluation is not None:
        evaluation.run_if_due(counters.optimizer_updates, model=model, runner=None,
            generators=generators, training_data=data, boundary=current_boundary, persist=persist)
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
