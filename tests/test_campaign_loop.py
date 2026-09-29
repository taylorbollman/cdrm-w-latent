"""Boundary lifecycle controls, real Gloo coordination and saved Adam recovery."""
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import signal

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.olmo_campaign_loop import Coordinator, LifecycleError, LoopPolicy, StopRequest, run_loop
from cdrm.pretrained.distributed_checkpoint import save_distributed_checkpoint, load_distributed_checkpoint
from cdrm.pretrained.lm_training import TrainingCounters, build_adamw
from scripts.olmo_lm_common import tree_digests


class LocalCoordinator:
    rank, world_size = 0, 1
    def gather(self, value):
        return [value]
    def same(self, phase, value):
        return value
    def call(self, phase, function, *, rank_zero=False):
        try:
            return function()
        except Exception as exc:
            raise LifecycleError(f"{phase}: {exc}") from exc


def local_run(tmp_path, *, policy=None, stop=None, restored=False, fail=None, retain=None):
    events, state, clock = [], {"update": 0}, [0.]
    def update():
        events.append("update")
        if fail == "update":
            raise RuntimeError("invalid update")
        state["update"] += 1
        clock[0] += 11
        return dict(state)
    def log(metrics):
        events.append("log")
        if fail == "log":
            raise OSError("logging failed")
        if stop is not None:
            stop.request(signal.SIGTERM)
    def save(number, reason):
        events.append(("save",number,reason))
        return {"update":number}
    def publish(receipt):
        events.append(("publish",receipt["update"]))
    values = dict(coordinator=LocalCoordinator(), policy=policy or LoopPolicy(2, 10),
        completed=lambda:state["update"], update=update, log=log, save=save,
        publish_checkpoint=publish, stop=stop or StopRequest(), retain=retain,
        clock=lambda:clock[0], restored=restored)
    return values, events, state


def test_fake_clock_cadence_and_limit_save_each_committed_boundary(tmp_path):
    kwargs, events, state = local_run(tmp_path)
    result = run_loop(**kwargs)
    assert result == {"start_update":0,"completed_update":2,"stop_reason":"update_limit",
                      "checkpoints_published":3,"last_saved_update":2}
    assert [v for v in events if isinstance(v,tuple) and v[0]=="save"] == [
        ("save",0,"initial"),("save",1,"cadence"),("save",2,"update_limit")]
    assert state["update"] == 2


def test_signal_is_only_flag_and_finishes_current_update(tmp_path):
    stop = StopRequest()
    kwargs, events, state = local_run(tmp_path,stop=stop,policy=LoopPolicy(3))
    result = run_loop(**kwargs)
    assert state["update"] == 1 and result["stop_reason"] == f"signal:{signal.SIGTERM}"
    assert events[-2:] == [("save",1,f"signal:{signal.SIGTERM}"),("publish",1)]


def test_existing_stop_file_saves_initial_and_runs_no_update(tmp_path):
    path = tmp_path/"STOP"; path.touch()
    kwargs,events,state = local_run(tmp_path,stop=StopRequest(path))
    assert run_loop(**kwargs)["stop_reason"] == "stop_file"
    assert state["update"] == 0 and "update" not in events
    assert len([v for v in events if isinstance(v,tuple) and v[0]=="save"]) == 1


def test_log_failure_preserves_completed_boundary_and_no_next_update(tmp_path):
    kwargs,events,state = local_run(tmp_path,fail="log")
    with pytest.raises(LifecycleError,match="logging failed"):
        run_loop(**kwargs)
    assert state["update"] == 1
    assert events[-2:] == [("save",1,"logging_failure"),("publish",1)]


def test_failed_update_never_attempts_emergency_save(tmp_path):
    kwargs,events,state = local_run(tmp_path,fail="update")
    with pytest.raises(RuntimeError,match="invalid update"):
        run_loop(**kwargs)
    assert state["update"] == 0 and events[-1] == "update"


def test_retention_failure_does_not_publish_or_advance(tmp_path):
    def reject(receipt):
        raise OSError("retention failed")
    kwargs,events,state = local_run(tmp_path,retain=reject)
    with pytest.raises(LifecycleError,match="retention failed"):
        run_loop(**kwargs)
    assert events == [("save",0,"initial")] and state["update"] == 0


def test_restored_initial_boundary_is_not_overwritten(tmp_path):
    kwargs,events,_ = local_run(tmp_path,restored=True)
    run_loop(**kwargs)
    assert ("save",0,"initial") not in events


def test_signal_handler_restored_even_after_error():
    previous = {number:signal.getsignal(number) for number in (signal.SIGINT,signal.SIGTERM)}
    stop = StopRequest()
    with pytest.raises(ValueError):
        with stop.installed():
            assert signal.getsignal(signal.SIGTERM) == stop.request
            raise ValueError("scope failure")
    assert all(signal.getsignal(number)==handler for number,handler in previous.items())


@pytest.mark.parametrize("kwargs", [{"max_updates":-1}, {"max_updates":True},
    {"max_updates":1,"checkpoint_seconds":601}, {"max_updates":1,"checkpoint_seconds":float('nan')},
    {"max_updates":1,"checkpoint_updates":(1,1)}])
def test_policy_rejects_invalid_boundary_contract(kwargs):
    with pytest.raises(ValueError):
        LoopPolicy(**kwargs)


def _gloo_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",rank=rank,world_size=2,init_method=f"file://{rendezvous}",
                            timeout=timedelta(seconds=45))
    coordinator, directory = Coordinator(), Path(directory)
    configuration, fingerprint = {"loop_test":1}, {"sha256":"a"*64,"source":"fixed-tiny"}
    try:
        def objects():
            torch.manual_seed(777)
            model = torch.nn.Linear(2,2,bias=False)
            optimizer = build_adamw(model,lr=.003,fused=False)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=.9)
            return model,optimizer,scheduler
        model,optimizer,scheduler = objects()
        counters, cursor = TrainingCounters(), {"next_update":0,"rank":rank}
        stop, saved = StopRequest(), []
        def update():
            x = torch.tensor([[1.,2.],[2.,3.]]) + counters.optimizer_updates
            model(x).square().mean().backward()
            for parameter in model.parameters():
                dist.all_reduce(parameter.grad); parameter.grad.div_(2)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            counters.optimizer_updates += 1
            counters.input_tokens += 4
            cursor["next_update"] += 1
            return {"update":counters.optimizer_updates}
        def save(number,reason):
            receipt = save_distributed_checkpoint(directory/f"saved-{number}",model,optimizer,
                scheduler=scheduler,counters=counters,data_cursor=cursor,
                configuration=configuration,source_fingerprint=fingerprint,device=torch.device("cpu"))
            saved.append((number,reason,receipt))
            return receipt
        def log(metrics):
            raise OSError("rank zero injected I/O")
        errors = []
        try:
            run_loop(coordinator=coordinator,policy=LoopPolicy(3),completed=lambda:counters.optimizer_updates,
                update=update,log=log,save=save,publish_checkpoint=lambda receipt:None,stop=stop)
        except LifecycleError as exc:
            errors.append(str(exc))
        else:
            raise AssertionError("Required logging failure did not propagate")
        assert coordinator.gather(errors)[0] == coordinator.gather(errors)[1]
        assert counters.optimizer_updates == cursor["next_update"] == 1
        assert [(n,r) for n,r,_ in saved] == [(0,"initial"),(1,"logging_failure")]
        # Build uninterrupted update2 reference from the safely saved state.
        update()
        expected = tree_digests({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),"counters":asdict(counters),"cursor":cursor})
        model,optimizer,scheduler = objects()
        restored = load_distributed_checkpoint(directory/"saved-1",model,optimizer,scheduler=scheduler,
            configuration=configuration,source_fingerprint=fingerprint,
            expected_manifest_sha256=saved[-1][2]["manifest_sha256"],device=torch.device("cpu"))
        counters,cursor = restored["counters"],restored["data_cursor"]
        stop = StopRequest()
        # A request on rank1 alone must stop both ranks after the same update.
        def after_update():
            result = update()
            if rank == 1:
                stop.request(signal.SIGTERM)
            return result
        result = run_loop(coordinator=coordinator,policy=LoopPolicy(3),completed=lambda:counters.optimizer_updates,
            update=after_update,log=lambda row:None,save=save,publish_checkpoint=lambda receipt:None,
            stop=stop,restored=True)
        assert result["completed_update"] == 2 and result["stop_reason"] == f"signal:{signal.SIGTERM}"
        assert tree_digests({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),"counters":asdict(counters),"cursor":cursor}) == expected
        for phase in ("persist", "finish"):
            try:
                coordinator.call(phase,lambda: (_ for _ in ()).throw(OSError("rank zero only")),rank_zero=True)
            except LifecycleError as exc:
                assert "rank 0" in str(exc)
            else:
                raise AssertionError("Rank-zero failure not coordinated")
        (directory/f"rank-{rank}.json").write_text(json.dumps({"passed":True,"errors":errors,"stop":result}))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason="Gloo unavailable")
def test_real_gloo_logging_failure_checkpoint_restore_and_one_rank_stop(tmp_path):
    mp.spawn(_gloo_worker,args=(str(tmp_path/"init"),str(tmp_path)),nprocs=2,join=True)
    assert all(json.loads((tmp_path/f"rank-{rank}.json").read_text())["passed"] for rank in (0,1))
