from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.distributed_checkpoint import _local_rng, _restore_local_rng
from cdrm.pretrained.lm_training import TrainingCounters
from scripts.olmo_two_gpu_recovery import (
    cursor_for, draw_rng, ensure_persistent_checkpoint_directory, parse_args,
    recovery_batch, seed_local, validate_cursor,
)


def test_checkpoint_directory_rejects_ssd_and_symlink_into_ssd(tmp_path, monkeypatch):
    disposable = tmp_path / "disposable"
    disposable.mkdir()
    monkeypatch.setenv("CDRM_LOCALSSD_MOUNT", str(disposable))
    alias = tmp_path / "alias"
    alias.symlink_to(disposable, target_is_directory=True)
    for path in (disposable, disposable / "checkpoint", alias / "checkpoint"):
        with pytest.raises(ValueError, match="persistent"):
            ensure_persistent_checkpoint_directory(path)
    persistent = tmp_path / "persistent" / "checkpoint"
    assert ensure_persistent_checkpoint_directory(persistent) == persistent


def test_checkpoint_directory_rejects_existing_partial(tmp_path):
    with pytest.raises(FileExistsError):
        ensure_persistent_checkpoint_directory(tmp_path)


@pytest.mark.parametrize("key,value", [("rank", 1), ("world_size", 3), ("next_update", 3),
                                       ("batch_size_per_rank", 2), ("length", 1024)])
def test_cursor_resume_mismatch_rejected(key, value):
    case = SimpleNamespace(batch_size=1, length=512)
    cursor = cursor_for(case, 0, 2)
    counters = TrainingCounters(optimizer_updates=2)
    validate_cursor(cursor, case, 0, counters)
    cursor[key] = value
    with pytest.raises(ValueError, match="cursor"):
        validate_cursor(cursor, case, 0, counters)


def test_probe_reproduces_python_numpy_torch_and_explicit_generators(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No all-device CUDA RNG call is allowed")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    device = torch.device("cpu")
    seed_local(27, device)
    generators = {"data": torch.Generator().manual_seed(12), "local": torch.Generator().manual_seed(13)}
    state = _local_rng(device, generators)
    expected = draw_rng(device, generators)
    assert draw_rng(device, generators) != expected
    _restore_local_rng(state, device, generators)
    assert draw_rng(device, generators) == expected


def test_cli_actual_defaults_and_short_tiny_override():
    args = parse_args(["--case", "combined", "--output-dir", "out", "--checkpoint-dir", "saved"])
    assert not args.tiny and args.batch_size == 1 and args.length == 512
    args = parse_args(["--tiny", "--case", "rt", "--length", "8", "--output-dir", "out", "--checkpoint-dir", "saved"])
    assert args.tiny and args.length == 8


def test_combined_resume_keeps_auxiliary_targets_with_new_tokens():
    case = SimpleNamespace(batch_size=1, length=8, nextlat=True)
    preparation = recovery_batch(case, None, 1, 0, 0, tiny=True)
    resumed = recovery_batch(case, None, 2, 0, 0, tiny=True)
    assert resumed.latent_mask.any() and resumed.kl_mask.any()
    assert not torch.equal(resumed.input_ids, preparation.input_ids)
    assert torch.equal(resumed.latent_mask, preparation.latent_mask)
    assert torch.equal(resumed.kl_mask, preparation.kl_mask)
    for rank, micro in ((0, 1), (1, 0), (1, 1)):
        batch = recovery_batch(case, None, 2, rank, micro, tiny=True)
        assert not batch.latent_mask.any() and not batch.kl_mask.any()


@pytest.mark.parametrize("flags", [("--batch-size", "0"), ("--length", "3")])
def test_invalid_dimensions_fail_before_gpu_setup(flags):
    with pytest.raises(SystemExit):
        parse_args(["--case", "rt", "--output-dir", "out", "--checkpoint-dir", "saved", *flags])
