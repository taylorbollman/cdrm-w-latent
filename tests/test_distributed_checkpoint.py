"""Checkpoint contract tests plus real two-process Gloo error/recovery checks."""
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cdrm.pretrained import distributed_checkpoint as checkpoint
from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import TrainingCounters, build_adamw


class TinyCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3, 3, bias=False)
        self.readout = nn.Linear(3, 3, bias=False)
        self.readout.weight = self.embed.weight
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        return self.readout(self.embed(x).tanh()) + self.bias


CONFIG = {"precision": "fp32", "runtime": {"optimizer": "scalar", "graphs": False}}
SOURCE = {"checkpoint_sha256": "a" * 64, "revision": "test-source"}
COUNTERS = TrainingCounters(optimizer_updates=2, microbatches=4, documents=8,
                            input_tokens=32, ce_positions=24, latent_pairs=12,
                            kl_triples=8)


def _objects():
    torch.manual_seed(137)
    model = TinyCheckpointModel()
    optimizer = build_adamw(model, lr=0.003, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    for _ in range(2):
        _update(model, optimizer, scheduler)
    model.embed.eval()
    return model, optimizer, scheduler


def _update(model, optimizer, scheduler):
    model(torch.arange(12, dtype=torch.float32).view(4, 3) / 10).square().mean().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _save(path, objects, **kwargs):
    model, optimizer, scheduler = objects
    values = dict(scheduler=scheduler, counters=COUNTERS, data_cursor={"offset": 7},
                  configuration=CONFIG, source_fingerprint=SOURCE)
    values.update(kwargs)
    return checkpoint.save_distributed_checkpoint(path, model, optimizer, **values)


def _load(path, objects, **kwargs):
    model, optimizer, scheduler = objects
    values = dict(scheduler=scheduler, configuration=CONFIG, source_fingerprint=SOURCE)
    values.update(kwargs)
    return checkpoint.load_distributed_checkpoint(path, model, optimizer, **values)


@pytest.fixture
def local_collectives(monkeypatch):
    monkeypatch.setattr(checkpoint, "_context", lambda group: (0, 1))
    monkeypatch.setattr(checkpoint, "_gather", lambda value, group: [value])


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _draw(generator):
    return {"python": random.random(), "numpy": float(np.random.rand()),
            "torch": torch.rand(5), "explicit": torch.rand(5, generator=generator)}


def _rewrite(path, mutate):
    state_path = path / checkpoint.STATE_FILENAME
    payload = torch.load(state_path, weights_only=True)
    mutate(payload)
    torch.save(payload, state_path)
    manifest_path = path / checkpoint.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["state"]["sha256"] = sha256_file(state_path)
    manifest["state"]["size_bytes"] = state_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))


def test_roundtrip_complete_state_rng_aliases_and_continuation(tmp_path, local_collectives):
    objects = _objects()
    generator = torch.Generator().manual_seed(100)
    random.seed(19)
    np.random.seed(27)
    path = tmp_path / "boundary"
    record = _save(path, objects, generators={"data": generator})
    expected_rng = _draw(generator)
    _update(*objects)
    import copy
    expected = [copy.deepcopy(item.state_dict()) for item in objects]
    fresh = _objects()
    fresh[0].train()
    result = _load(path, fresh, expected_manifest_sha256=record["manifest_sha256"], generators={"data": generator})
    assert asdict(result["counters"]) == asdict(COUNTERS)
    assert result["data_cursor"] == {"offset": 7}
    assert fresh[0].embed.weight is fresh[0].readout.weight
    assert not fresh[0].embed.training and fresh[0].training
    _equal(_draw(generator), expected_rng)
    _update(*fresh)
    for item, saved in zip(fresh, expected):
        _equal(item.state_dict(), saved)
    assert sorted(item.name for item in path.iterdir()) == ["manifest.json", "state.pt"]
    assert checkpoint.inspect_distributed_checkpoint(path) == record


def test_requires_initialized_group(tmp_path):
    with pytest.raises(RuntimeError, match="initialized process group"):
        _save(tmp_path / "missing", _objects())


def test_rejects_live_gradients_before_directory_created(tmp_path, local_collectives):
    objects = _objects()
    next(objects[0].parameters()).grad = torch.ones_like(next(objects[0].parameters()))
    with pytest.raises(checkpoint.DistributedCheckpointError, match="cleared gradients"):
        _save(tmp_path / "bad", objects)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("complete", [False, True])
def test_never_overwrites_existing_directory(tmp_path, local_collectives, complete):
    path = tmp_path / "boundary"
    objects = _objects()
    if complete:
        _save(path, objects)
    else:
        path.mkdir()
        (path / "partial").write_bytes(b"partial")
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(checkpoint.DistributedCheckpointError, match="FileExistsError"):
        _save(path, objects)
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}


def test_interrupted_save_has_no_commit_and_keeps_partial_bytes(tmp_path, local_collectives, monkeypatch):
    def fail(payload, stream):
        stream.write(b"partial checkpoint")
        raise OSError("simulated disk full")
    monkeypatch.setattr(torch, "save", fail)
    path = tmp_path / "interrupted"
    with pytest.raises(checkpoint.DistributedCheckpointError, match="disk full"):
        _save(path, _objects())
    assert path.is_dir() and list(path.glob(".state.pt.*.tmp"))
    with pytest.raises(ValueError, match="committed manifest is missing"):
        checkpoint.inspect_distributed_checkpoint(path)


def test_failure_between_state_and_manifest_is_incomplete(tmp_path, local_collectives, monkeypatch):
    original = checkpoint._publish
    def publish(path, writer):
        if path.name == checkpoint.MANIFEST_FILENAME:
            raise OSError("before commit")
        original(path, writer)
    monkeypatch.setattr(checkpoint, "_publish", publish)
    path = tmp_path / "interrupted"
    with pytest.raises(checkpoint.DistributedCheckpointError, match="before commit"):
        _save(path, _objects())
    assert (path / "state.pt").is_file()
    with pytest.raises(ValueError, match="manifest is missing"):
        checkpoint.inspect_distributed_checkpoint(path)


@pytest.mark.parametrize("mutation, message", [
    ("contents", "state SHA256"), ("size", "wrong size"),
    ("missing", "missing or has wrong size"), ("filename", "filename differs"),
    ("schema", "schema differs"), ("world", "world size differs"),
])
def test_file_and_manifest_validation(tmp_path, local_collectives, mutation, message):
    path = tmp_path / "boundary"
    objects = _objects()
    _save(path, objects)
    state = path / "state.pt"
    if mutation == "contents":
        data = bytearray(state.read_bytes())
        data[-1] ^= 1
        state.write_bytes(data)
    elif mutation == "size":
        state.write_bytes(b"truncated")
    elif mutation == "missing":
        state.unlink()
    else:
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if mutation == "filename":
            manifest["state"]["filename"] = "../other.pt"
        elif mutation == "schema":
            manifest["schema"] = "other"
        else:
            manifest["world_size"] = 2
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(checkpoint.DistributedCheckpointError, match=message):
        _load(path, objects)


def test_manifest_expected_hash(tmp_path, local_collectives):
    path = tmp_path / "boundary"
    objects = _objects()
    _save(path, objects)
    with pytest.raises(checkpoint.DistributedCheckpointError, match="manifest SHA256"):
        _load(path, objects, expected_manifest_sha256="b" * 64)


@pytest.mark.parametrize("change", ["configuration", "source", "optimizer", "scheduler", "layout", "generators"])
def test_strict_resume_metadata_before_mutation(tmp_path, local_collectives, change):
    path = tmp_path / "boundary"
    objects = _objects()
    _save(path, objects)
    kwargs = {}
    if change == "configuration":
        kwargs["configuration"] = {**CONFIG, "precision": "bf16_mixed"}
    elif change == "source":
        kwargs["source_fingerprint"] = {"sha256": "b" * 64}
    elif change == "optimizer":
        objects[1].defaults["fused"] = True
    elif change == "scheduler":
        kwargs["scheduler"] = None
    elif change == "layout":
        objects[0].bias.requires_grad_(False)
    else:
        kwargs["generators"] = {"extra": torch.Generator()}
    before = objects[0].embed.weight.detach().clone()
    with pytest.raises(checkpoint.DistributedCheckpointError):
        _load(path, objects, **kwargs)
    assert torch.equal(objects[0].embed.weight, before)


@pytest.mark.parametrize("corruption, message", [
    ("alias", "aliases disagree"), ("moment", "moment shape/dtype"),
    ("foreign", "foreign state"), ("rank", "rank-state mapping"),
    ("counter", "counters disagree"), ("cursor", "cursors disagree"),
    ("mode", "module ownership/training"), ("rng", "RNG device differs"),
    ("scheduler", "scheduler state disagrees"), ("group", "group settings disagree"),
])
def test_payload_semantic_checks(tmp_path, local_collectives, corruption, message):
    path = tmp_path / "boundary"
    objects = _objects()
    _save(path, objects)
    def mutate(payload):
        if corruption == "alias":
            payload["model"]["readout.weight"] = payload["model"]["readout.weight"].clone() + 1
        elif corruption == "moment":
            next(iter(payload["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(9)
        elif corruption == "foreign":
            payload["optimizer"]["state"][999] = {}
        elif corruption == "rank":
            payload["rank_states"][0]["rank"] = 1
        elif corruption == "counter":
            payload["counters"]["optimizer_updates"] = 3
        elif corruption == "cursor":
            payload["rank_states"][0]["data_cursor"]["offset"] = 8
        elif corruption == "mode":
            payload["module_training"][""] = 1
        elif corruption == "rng":
            payload["rank_states"][0]["rng"]["device"] = "cuda:0"
        elif corruption == "scheduler":
            payload["scheduler"]["last_epoch"] += 1
        else:
            payload["optimizer"]["param_groups"][0]["lr"] += 0.1
    _rewrite(path, mutate)
    with pytest.raises(checkpoint.DistributedCheckpointError, match=message):
        _load(path, objects)


def test_local_cuda_rng_never_queries_all_devices(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Peer GPU RNG access is forbidden")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    calls = []
    fake_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: calls.append(("get", str(device))) or fake_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: calls.append(("set", str(device))))
    device = torch.device("cuda:1")
    state = checkpoint._local_rng(device, {})
    checkpoint._restore_local_rng(state, device, {})
    assert calls == [("get", "cuda:1"), ("set", "cuda:1")]


def _gloo_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{rendezvous}", timeout=timedelta(seconds=45))
    try:
        objects = _objects()
        generator = torch.Generator().manual_seed(500 + rank)
        random.seed(900 + rank)
        np.random.seed(1000 + rank)
        torch.manual_seed(1100 + rank)
        path = Path(output) / "valid"
        receipt = _save(path, objects, data_cursor={"rank": rank, "offset": 12 + rank}, generators={"data": generator})
        expected_rng = _draw(generator)
        _update(*objects)
        import copy
        expected_state = [copy.deepcopy(item.state_dict()) for item in objects]
        fresh = _objects()
        loaded = _load(path, fresh, generators={"data": generator}, expected_manifest_sha256=receipt["manifest_sha256"])
        assert loaded["data_cursor"] == {"rank": rank, "offset": 12 + rank}
        _equal(_draw(generator), expected_rng)
        _update(*fresh)
        for item, state in zip(fresh, expected_state):
            _equal(item.state_dict(), state)
        errors = []
        # One participant has dirty gradients: both return before any I/O.
        if rank == 1:
            fresh[0].bias.grad = torch.zeros_like(fresh[0].bias)
        try:
            _save(Path(output) / "dirty", fresh)
        except checkpoint.DistributedCheckpointError as exc:
            errors.append(str(exc))
        else:
            raise AssertionError("Dirty rank should fail both participants")
        fresh[1].zero_grad(set_to_none=True)
        # Rank-zero I/O failure and configuration mismatch cannot strand rank1.
        for name, kwargs in (("valid", {}), ("mismatch", {"configuration": {**CONFIG, "rank": rank}})):
            try:
                _save(Path(output) / name, fresh, **kwargs)
            except checkpoint.DistributedCheckpointError as exc:
                errors.append(str(exc))
            else:
                raise AssertionError("Expected coordinated save rejection")
        for name in ("counter-mismatch", "scheduler-mismatch", "optimizer-mismatch"):
            kwargs = {}
            if name == "counter-mismatch":
                kwargs["counters"] = TrainingCounters(**{**asdict(COUNTERS), "optimizer_updates": 2 + rank})
            elif name == "scheduler-mismatch":
                fresh[2].last_epoch += rank
            else:
                fresh[1].param_groups[0]["lr"] += 0.001 * rank
            try:
                _save(Path(output) / name, fresh, **kwargs)
            except checkpoint.DistributedCheckpointError as exc:
                errors.append(str(exc))
            else:
                raise AssertionError("Expected global boundary-state mismatch rejection")
            finally:
                if name == "scheduler-mismatch":
                    fresh[2].last_epoch -= rank
                elif name == "optimizer-mismatch":
                    fresh[1].param_groups[0]["lr"] -= 0.001 * rank
        # A rank-local file-read error reaches both ranks without tensor mutation.
        original = checkpoint.inspect_distributed_checkpoint
        if rank == 1:
            def fail_read(*args, **kwargs):
                raise OSError("rank-local read failure")
            checkpoint.inspect_distributed_checkpoint = fail_read
        before = fresh[0].embed.weight.detach().clone()
        try:
            _load(path, fresh, generators={"data": generator})
        except checkpoint.DistributedCheckpointError as exc:
            errors.append(str(exc))
        else:
            raise AssertionError("Expected coordinated read rejection")
        assert torch.equal(before, fresh[0].embed.weight)
        checkpoint.inspect_distributed_checkpoint = original
        collected = [None, None]
        dist.all_gather_object(collected, errors)
        assert collected[0] == collected[1]
        assert len(errors) == 7
        (Path(output) / f"rank-{rank}.json").write_text(json.dumps({"passed": True, "errors": errors}))
    finally:
        dist.destroy_process_group()


def test_real_two_rank_gloo_recovery_and_coordinated_errors(tmp_path):
    mp.spawn(_gloo_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    assert all(json.loads((tmp_path / f"rank-{rank}.json").read_text())["passed"] for rank in range(2))
    assert len(list((tmp_path / "valid").glob("*.pt"))) == 1
    assert not (tmp_path / "dirty").exists()
    assert not (tmp_path / "mismatch").exists()
