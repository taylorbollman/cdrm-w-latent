from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from scripts.olmo_f1_common import IntegrationCase
from scripts.olmo_two_gpu_graph_recovery import (
    graph_configuration, graph_cursor, release_graph_and_hooks, validate_graph_cursor,
)


def test_release_synchronizes_drops_graph_and_explicitly_removes_old_hooks(monkeypatch):
    events = []
    runtime = SimpleNamespace(device=torch.device("cuda:1"), graph=object(), graph_result=object(), stream=object())
    def synchronize(device):
        assert runtime.graph is not None and runtime.graph_result is not None
        events.append(("synchronize", str(device)))
    def remove():
        assert runtime.graph is None and runtime.graph_result is None
        events.append(("remove_hooks", None))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    runtime.ddp = SimpleNamespace(_remove_autograd_hooks=remove)
    release_graph_and_hooks(runtime)
    assert events == [("synchronize", "cuda:1"), ("remove_hooks", None)]
    assert runtime.ddp is None and runtime.stream is None


def test_missing_private_teardown_api_fails_before_touching_graph(monkeypatch):
    runtime = SimpleNamespace(ddp=object(), device=torch.device("cuda:0"), graph=object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: pytest.fail("Should fail before CUDA work"))
    original = runtime.graph
    with pytest.raises(RuntimeError, match="teardown API"):
        release_graph_and_hooks(runtime)
    assert runtime.graph is original


def test_graph_cursor_has_one_physical_microbatch_and_rejects_eager_layout():
    case = IntegrationCase("combined", fbt=True, nextlat=True, rt_layers=(0, 15), batch_size=1, length=512)
    cursor = graph_cursor(case, 1, 4)
    counters = TrainingCounters(optimizer_updates=4)
    validate_graph_cursor(cursor, case, 1, counters)
    assert cursor["microbatches_per_rank"] == 1
    assert cursor["batch_generator"] == "olmo_two_gpu_graph.fixed_batch-v1"
    cursor["microbatches_per_rank"] = 2
    with pytest.raises(ValueError, match="cursor"):
        validate_graph_cursor(cursor, case, 1, counters)


def test_checkpoint_configuration_pins_static_graph_global_counts_and_participation():
    model = SimpleNamespace(backbone=SimpleNamespace(backbone=SimpleNamespace()),
                            config=SimpleNamespace(to_dict=lambda: {"ce_chunk_size": 2048}),
                            enabled=True, gamma=1.0)
    case = IntegrationCase("combined", fbt=True, nextlat=True, rt_layers=(0, 15), batch_size=1, length=512)
    counts = [{"ce": 400, "latent": 300, "kl": 200}, {"ce": 300, "latent": 200, "kl": 100}]
    result = graph_configuration(model, case, LMTrainingConfig(precision="bf16_mixed"),
                                 SimpleNamespace(tiny=False), counts, ["weight", "predictor.weight"])
    assert result["graphs"] and result["ddp"]["static_graph"]
    assert not result["ddp"]["gradient_as_bucket_view"] and not result["ddp"]["find_unused_parameters"]
    assert result["global_counts"] == {"ce": 700, "latent": 500, "kl": 300}
    assert result["counts_by_rank"] == counts
    assert result["expected_active_names"] == ["weight", "predictor.weight"]
    assert result["microbatches_per_rank"] == 1 and result["warmup_backward_calls_per_graph"] == 11
    assert result["recovery_scope"] == "both_continuations_rebuild_actual_ddp_graphs"
