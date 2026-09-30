"""Changed-rank same-objective acceptance; CUDA/NCCL is a separate CLI check."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cdrm.pretrained.campaign_recipe import CampaignRecipe
from scripts.olmo_allocation_acceptance import (ROWS, fixture_for_rank, literal_counts,
    logical_fixture, parse_args, run_acceptance)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def test_repartition_preserves_real_rows_masks_and_keyed_jitter_with_different_padding():
    recipe = CampaignRecipe("NFR", sequence_length=8, rt_layers=(0, 1), document_policy="continuous-stream-v1")
    for update in range(3):
        snapshots = []
        physical_rows = []
        for world_size in (1, 2, 8):
            by_key, all_batches = {}, []
            fixtures = [fixture_for_rank(recipe, update, rank=r, world_size=world_size) for r in range(world_size)]
            order = [k for slot in range(len(fixtures[0][0])) for f in fixtures for k in f[2][slot]]
            assert order == [r.key for r in logical_fixture(update).rows]
            for batches, noises, keys in fixtures:
                all_batches.extend(batches)
                for batch, noise, batch_keys in zip(batches, noises, keys):
                    for row, key in enumerate(batch_keys):
                        by_key[key] = tuple(getattr(batch, name)[row] for name in vars(batch)) + tuple(n[row] for n in noise)
                    assert not batch.valid_mask[len(batch_keys):].any()
                    assert not any(n[len(batch_keys):].any() for n in noise)
            snapshots.append((by_key, literal_counts(all_batches)))
            physical_rows.append(sum(b.input_ids.shape[0] for b in all_batches))
        assert len(set(physical_rows)) > 1
        for other, counts in snapshots[1:]:
            assert counts == snapshots[0][1]
            assert other.keys() == snapshots[0][0].keys()
            for key, values in other.items():
                assert all(torch.equal(a, b) for a, b in zip(values, snapshots[0][0][key]))
        assert len(snapshots[0][0]) == ROWS[update]
        assert snapshots[0][1]["ce"] > snapshots[0][1]["latent"] > snapshots[0][1]["kl"] > 0


def _acceptance_worker(rank, rendezvous, world_size, arm, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
        init_method=f"file://{rendezvous}", timeout=timedelta(seconds=180))
    try:
        result, state = run_acceptance(arm, device="cpu", case="eager")
        assert result["passed"], result
        if rank == 0:
            torch.save(state, output)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("arm", ["B", "NFR"])
def test_one_and_two_rank_updates_match_independent_global_objective_and_each_other(tmp_path, arm):
    paths = []
    for world_size in (1, 2):
        path = tmp_path / f"{arm}-world{world_size}.pt"
        mp.spawn(_acceptance_worker, args=(str(tmp_path / f"world{world_size}.init"), world_size, arm, str(path)),
                 nprocs=world_size, join=True)
        paths.append(path)
    first, second = (torch.load(p, weights_only=True) for p in paths)
    for name, value in first["model"].items():
        torch.testing.assert_close(value, second["model"][name], atol=3e-6, rtol=3e-5, msg=name)
    for name, state in first["optimizer"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, second["optimizer"][name][key], atol=5e-7, rtol=1e-4, msg=f"{name}/{key}")
    assert first["scheduler"] == second["scheduler"]
    # Physical work deliberately differs, while useful exposure and step clocks do not.
    assert first["counters"]["microbatches"] != second["counters"]["microbatches"]
    assert {k: v for k, v in first["counters"].items() if k != "microbatches"} == {
        k: v for k, v in second["counters"].items() if k != "microbatches"}


def test_cli_explicit_device_no_cpu_graph_fallback(tmp_path):
    common = ["--output-dir", str(tmp_path)]
    assert parse_args(common + ["--device", "cpu", "--case", "eager"]).arms == ("B", "NFR")
    for extra in (["--device", "cpu", "--case", "graph"],
                  ["--device", "cuda", "--case", "graph", "--arms", "B,B"]):
        with pytest.raises(SystemExit):
            parse_args(common + extra)
