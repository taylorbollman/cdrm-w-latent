"""Buffer transport lifetime, wire compatibility and native consolidation parity."""
import copy
from datetime import timedelta
import gc
import io
from types import SimpleNamespace
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.optim import ZeroRedundancyOptimizer

from cdrm.pretrained import zero1_training as zero1


def payload():
    base = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    return {'state': {0: {'step': torch.tensor(3.), 'exp_avg': base[:, ::2],
                         'exp_avg_sq': base.clone().square()}},
            'param_groups': [{'params': [0, 1], 'lr': .01, 'betas': (.9, .95),
                              'param_names': ['weight', 'unused']}], 'empty': {}}


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right): assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('obj', [{}, {'state': {}, 'param_groups': []}, payload()])
def test_serialized_tensor_exact_native_bytes_roundtrip_and_owner_lifetime(monkeypatch, obj):
    native_buffer = io.BytesIO()
    torch.save(obj, native_buffer)
    # Keep a weak reference to the actual BytesIO owner: frombuffer must retain
    # it on CPU, not just happen to read bytes before they become invalid.
    buffers = []
    original_factory = io.BytesIO
    def factory(*args, **kwargs):
        buffer = original_factory(*args, **kwargs)
        buffers.append(weakref.ref(buffer))
        return buffer
    monkeypatch.setattr(zero1.io, 'BytesIO', factory)
    def forbidden(*args, **kwargs):
        raise AssertionError('Per-byte tensor construction is forbidden')
    monkeypatch.setattr(zero1.torch, 'ByteTensor', forbidden)
    tensor = zero1._serialized_state_tensor(obj, torch.device('cpu'))
    gc.collect()
    assert buffers[0]() is not None
    assert tensor.dtype == torch.uint8 and tensor.ndim == 1
    assert tensor.numpy().tobytes() == native_buffer.getvalue()
    restored = torch.load(original_factory(tensor.numpy()), weights_only=False)
    assert_tree_equal(obj, restored)
    del tensor
    gc.collect()
    assert buffers[0]() is None


@pytest.mark.parametrize('src', [0, 1])
def test_broadcast_wire_order_sender_identity_and_receiver_mapping(monkeypatch, src):
    obj = payload(); group = object(); wire = []
    monkeypatch.setattr(zero1.dist, 'get_rank', lambda: src)
    def send(tensor, *, src, group, async_op):
        assert group is expected_group and async_op is False
        wire.append((src, tensor.clone()))
    expected_group = group
    monkeypatch.setattr(zero1.dist, 'broadcast', send)
    result = zero1._broadcast_state_object(obj, src, group, torch.device('cpu'))
    assert result is obj
    assert len(wire) == 2 and wire[0][1].dtype == torch.int64
    assert wire[0][1].shape == (1,) and wire[1][1].dtype == torch.uint8
    assert wire[0][1].item() == wire[1][1].numel()
    monkeypatch.setattr(zero1.dist, 'get_rank', lambda: 1-src)
    pending = list(wire)
    def receive(tensor, *, src, group, async_op):
        assert group is expected_group and async_op is False
        expected_src, data = pending.pop(0)
        assert src == expected_src
        tensor.copy_(data)
    monkeypatch.setattr(zero1.dist, 'broadcast', receive)
    restored = zero1._broadcast_state_object(None, src, group, torch.device('cpu'))
    assert not pending
    assert_tree_equal(obj, restored)


@pytest.mark.parametrize('rank,target,expected_sources,expected_shards', [
    (0, 0, [11, 19], [7, 11, 19]), (1, 0, [11, 19], []),
    (0, 1, [7, 19], []), (1, 1, [7, 19], [7, 11, 19]),
])
def test_subgroup_rank_mapping_and_discard_order_match_native(
        monkeypatch, rank, target, expected_sources, expected_shards):
    world_ranks = [7, 11, 19]; group = object(); broadcasts = []; events = []
    def global_rank(actual_group, index):
        assert actual_group is group
        return world_ranks[index]
    monkeypatch.setattr(zero1.dist.distributed_c10d, 'get_global_rank', global_rank)
    def broadcast(obj, *, src_rank, group, device):
        assert group is expected_group and device.type == 'cpu'
        broadcasts.append(src_rank)
        return {'owner': src_rank, 'tensor': torch.tensor(src_rank)}
    expected_group = group
    monkeypatch.setattr(zero1, '_broadcast_state_object', broadcast)
    local = {'owner': world_ranks[rank], 'tensor': torch.tensor(world_ranks[rank])}
    optimizer = SimpleNamespace(rank=rank, global_rank=world_ranks[rank], world_size=3,
        process_group=group, _default_device=torch.device('cpu'), param_groups=[{'lr': .1}],
        optim=SimpleNamespace(state_dict=lambda: local, param_groups=[{'lr': .01}]),
        _check_overlap_initialized=lambda: events.append('overlap_checked'),
        _sync_param_groups=lambda src, dst: events.append(('sync', src, dst)),
        _all_state_dicts=['stale'])
    zero1.FP32Zero1AdamW.consolidate_state_dict(optimizer, to=target)
    assert broadcasts == expected_sources
    assert [state['owner'] for state in optimizer._all_state_dicts] == expected_shards
    assert events[0] == 'overlap_checked' and events[1][0] == 'sync'


def _real_worker(rank, filename):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method=f'file://{filename}', timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(17)
        model = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Linear(3, 2))
        optimizer = zero1.build_zero1_adamw(model, lr=.001, fused=False)
        # Include both empty-state serialization and initialized Adam moments.
        for initialized in (False, True):
            if initialized:
                model(torch.ones(2, 5)).sum().backward()
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = .002
            original_parameters = copy.deepcopy(model.state_dict())
            for target in (0, 1):
                ZeroRedundancyOptimizer.consolidate_state_dict(optimizer, to=target)
                local_after_native = copy.deepcopy(optimizer.optim.state_dict())
                expected = copy.deepcopy(optimizer.state_dict()) if rank == target else None
                optimizer.consolidate_state_dict(to=target)
                if rank == target:
                    assert_tree_equal(expected, optimizer.state_dict())
                    assert len(optimizer._all_state_dicts) == 2
                else:
                    assert not optimizer._all_state_dicts
                assert_tree_equal(local_after_native, optimizer.optim.state_dict())
                assert_tree_equal(original_parameters, model.state_dict())
                assert not optimizer.state
                dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo unavailable')
def test_real_gloo_empty_and_initialized_consolidation_matches_native_both_targets(tmp_path):
    mp.spawn(_real_worker, args=(str(tmp_path/'native-parity.init'),), nprocs=2, join=True)
