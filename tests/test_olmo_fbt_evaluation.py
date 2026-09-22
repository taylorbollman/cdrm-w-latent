"""Independent held-out per-pass metrics and evaluation state preservation."""

from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.fbt_evaluation import evaluate_fbt_batches
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, FBTOnlineMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(8221)


def _model(wrapped=False):
    core = OLMoFBT(OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math"))
    if wrapped:
        return FBTNextLatLM(core, NextLatConfig(32, dropout=.4), enabled=True)
    return core


def _batches():
    ids = torch.tensor([[2, 3, 5, 8, 13, 21, 34], [1, 1, 4, 7, 10, 17, 22], [6, 9, 1, 1, 1, 1, 1]])
    valid = torch.tensor([[True] * 7, [False] * 2 + [True] * 5, [True] * 2 + [False] * 5])
    docs = torch.tensor([[41] * 7, [88] * 7, [41] * 7])
    ce = torch.ones_like(valid)
    ce[0, 1:3] = False
    one = NextLatBatch(ids, valid, docs, ce_mask=ce)
    two = NextLatBatch(torch.tensor([[12, 9, 6, 3]]), torch.ones(1, 4, dtype=torch.bool),
                      torch.full((1, 4), 41), ce_mask=torch.tensor([[False, False, False, True]]))
    return [one, two]


def _dense_oracle(core, batches, mode):
    """Independent scalar token selection and dense logsumexp/gather loss."""
    records = None
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            kwargs = dict(attention_mask=batch.valid_mask, document_ids=batch.document_ids,
                          mode=mode, return_logits=False)
            if isinstance(mode, FBTOnlineMode):
                states = (core.forward_online(batch.input_ids, use_cache=False, **kwargs).last_hidden_state,)
            else:
                states = core(batch.input_ids, **kwargs).pass_hidden_states
            if records is None:
                records = [[] for _ in states]
            for pass_index, hidden in enumerate(states):
                logits = F.linear(hidden, core.readout_weight).float()
                log_z = logits.logsumexp(-1)
                for row in range(batch.input_ids.shape[0]):
                    if not bool(batch.valid_mask[row].any()):
                        continue
                    losses, hits = [], 0
                    for target in range(1, batch.input_ids.shape[1]):
                        if not (batch.valid_mask[row, target - 1] and batch.valid_mask[row, target]):
                            continue
                        if batch.document_ids[row, target - 1] != batch.document_ids[row, target]:
                            continue
                        if batch.ce_mask is not None and not batch.ce_mask[row, target]:
                            continue
                        label = batch.input_ids[row, target]
                        losses.append(float(log_z[row, target - 1] - logits[row, target - 1, label]))
                        hits += int(logits[row, target - 1].argmax() == label)
                    records[pass_index].append(dict(batch_index=batch_index, row_index=row,
                        document_id=int(batch.document_ids[row][batch.valid_mask[row]][0]),
                        ce_sum=math.fsum(losses), ce_count=len(losses), next_token_correct=hits))
    return records


@pytest.mark.parametrize("mode", [FBTMode(num_passes=1), FBTMode(num_passes=2, beta=.37),
                                 FBTMode(num_passes=3), FBTMode(num_passes=2, beta=0),
                                 FBTMode(enabled=False, num_passes=3), FBTOnlineMode(beta=.37)])
@pytest.mark.parametrize("chunk_size", [1, 128])
def test_each_pass_matches_independent_dense_token_weighted_oracle(mode, chunk_size):
    core = _model()
    batches = _batches()
    expected = _dense_oracle(core, batches, mode)
    actual = evaluate_fbt_batches(core, iter(batches), mode=mode, precision="fp32",
                                  chunk_size=chunk_size, include_document_records=True)
    assert len(actual["passes"]) == len(expected)
    assert actual["documents"] == 4
    assert actual["batches"] == 2
    assert actual["input_tokens"] == 18
    for index, (got, rows) in enumerate(zip(actual["passes"], expected)):
        total, count = math.fsum(row["ce_sum"] for row in rows), sum(row["ce_count"] for row in rows)
        hits = sum(row["next_token_correct"] for row in rows)
        assert got["ce_count"] == count == 10
        assert got["ce_sum"] == pytest.approx(total, abs=4e-6)
        assert got["mean_nll"] == pytest.approx(total / count, abs=1e-6)
        assert got["perplexity"] == pytest.approx(math.exp(total / count), rel=1e-6)
        assert got["next_token_correct"] == hits
        assert got["next_token_accuracy"] == hits / count
        assert got["pass_index"] == (None if isinstance(mode, FBTOnlineMode) else index)
        for got_row, wanted in zip(got["document_records"], rows):
            for key in ("document_id", "batch_index", "row_index", "ce_count", "next_token_correct"):
                assert got_row[key] == wanted[key]
            assert got_row["ce_sum"] == pytest.approx(wanted["ce_sum"], abs=2e-6)
        assert [row["document_id"] for row in got["document_records"]] == [41, 88, 41, 41]
    json.dumps(actual, allow_nan=False)


def test_repeated_ordinary_passes_report_native_nll_not_two_ce_training_objective():
    core = _model()
    single = evaluate_fbt_batches(core, _batches(), mode=FBTMode(num_passes=1), precision="fp32")
    repeated = evaluate_fbt_batches(core, _batches(), mode=FBTMode(num_passes=3, beta=0), precision="fp32")
    for result in repeated["passes"]:
        for key in ("ce_sum", "ce_count", "mean_nll", "perplexity", "next_token_accuracy"):
            assert result[key] == single["passes"][0][key]
    assert "mean_nll" not in repeated


def test_full_native_vocabulary_and_position_chunk_limit(monkeypatch):
    core = _model()
    with torch.no_grad():
        core.readout_weight.zero_()
        core.readout_weight[66, 0] = 10
    batch = NextLatBatch(torch.tensor([[2, 66, 4, 66]]), torch.ones(1, 4, dtype=torch.bool),
                        torch.zeros(1, 4, dtype=torch.long))
    def fixed_states(ids, **kwargs):
        assert kwargs["return_logits"] is False
        hidden = torch.zeros(*ids.shape, 32)
        hidden[..., 0] = 1
        return SimpleNamespace(pass_hidden_states=(hidden, hidden), last_hidden_state=hidden)
    monkeypatch.setattr(core, "forward", fixed_states)
    original = F.linear
    shapes = []
    def checked_projection(hidden, weight, *args, **kwargs):
        if weight is core.readout_weight:
            shapes.append((hidden.shape, weight.shape))
        return original(hidden, weight, *args, **kwargs)
    monkeypatch.setattr(F, "linear", checked_projection)
    result = evaluate_fbt_batches(core, [batch], mode=FBTMode(), precision="fp32", chunk_size=2)
    assert shapes and all(shape[0][0] <= 2 and shape[1][0] == 67 for shape in shapes)
    for metrics in result["passes"]:
        assert metrics["next_token_correct"] == 2
        assert metrics["mean_nll"] == pytest.approx(math.log(math.exp(10) + 66) - 20 / 3, abs=1e-6)


@pytest.mark.parametrize("mode", [FBTMode(), FBTOnlineMode()])
def test_evaluation_bypasses_predictor_preserves_parameters_grads_rng_and_all_train_flags(mode, monkeypatch):
    model = _model(wrapped=True)
    core = model.backbone
    model.train()
    model.predictor.eval()
    core.fusion.token_gate.eval()
    flags = {name: module.training for name, module in model.named_modules()}
    parameters = dict(model.named_parameters())
    for index, parameter in enumerate(parameters.values()):
        parameter.grad = None if index % 2 else torch.full_like(parameter, .125)
    weights = {name: p.detach().clone() for name, p in parameters.items()}
    grads = {name: p.grad for name, p in parameters.items()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation called training or NextLat predictor")
    monkeypatch.setattr(model, "forward", forbidden)
    monkeypatch.setattr(model, "loss_sums", forbidden)
    monkeypatch.setattr(model.predictor, "forward", forbidden)
    method = "forward_online" if isinstance(mode, FBTOnlineMode) else "forward"
    original = getattr(core, method)
    def checked(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        assert kwargs["return_logits"] is False
        if method == "forward_online":
            assert kwargs["use_cache"] is False
            assert "past_key_values" not in kwargs
        return original(*args, **kwargs)
    monkeypatch.setattr(core, method, checked)
    rng = torch.get_rng_state().clone()
    evaluate_fbt_batches(model, _batches(), mode=mode, precision="fp32")
    assert torch.equal(torch.get_rng_state(), rng)
    assert {name: module.training for name, module in model.named_modules()} == flags
    assert torch.is_grad_enabled()
    for name, parameter in parameters.items():
        assert torch.equal(parameter, weights[name])
        assert parameter.grad is grads[name]
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, torch.full_like(parameter, .125))


@pytest.mark.parametrize("failure", ["forward", "iterator", "nonfinite", "overflow", "pass_count"])
def test_errors_restore_every_module_training_flag(failure, monkeypatch):
    model = _model(wrapped=True)
    model.eval()
    model.backbone.fusion.train()
    model.predictor.mlp[0].train()
    flags = [module.training for module in model.modules()]
    batches = _batches()
    error = FloatingPointError
    if failure == "forward":
        def fail(*args, **kwargs):
            raise RuntimeError("synthetic forward failure")
        monkeypatch.setattr(model.backbone, "forward", fail)
        error = RuntimeError
    elif failure == "iterator":
        first_batch = batches[0]
        def fail_iter():
            yield first_batch
            raise RuntimeError("synthetic iterator failure")
        batches = fail_iter()
        error = RuntimeError
    else:
        def extreme(ids, **kwargs):
            hidden = torch.ones(*ids.shape, 32)
            return SimpleNamespace(pass_hidden_states=(hidden,) if failure == "pass_count" else (hidden, hidden))
        monkeypatch.setattr(model.backbone, "forward", extreme)
        if failure == "pass_count":
            error = ValueError
        else:
            with torch.no_grad():
                model.backbone.readout_weight.zero_()
                model.backbone.readout_weight[66, 0] = float("nan") if failure == "nonfinite" else 1000
    with pytest.raises(error):
        evaluate_fbt_batches(model, batches, mode=FBTMode(), precision="fp32")
    assert [module.training for module in model.modules()] == flags
    assert torch.is_grad_enabled()


def test_padding_gap_and_target_ce_masks_with_unscored_rows_do_not_dilute_metrics():
    core = _model()
    batch = _batches()[0]
    valid = batch.valid_mask.clone()
    valid[0, 4] = False
    zero = torch.zeros_like(valid)
    batch = replace(batch, valid_mask=valid, latent_mask=zero, kl_mask=zero)
    empty = replace(batch, valid_mask=zero)
    unscored = replace(batch, ce_mask=zero)
    expected = _dense_oracle(core, [batch], FBTMode())
    result = evaluate_fbt_batches(core, [empty, unscored, batch], mode=FBTMode(), precision="fp32", include_document_records=True)
    assert result["documents"] == 6
    for index, metrics in enumerate(result["passes"]):
        total = math.fsum(row["ce_sum"] for row in expected[index])
        count = sum(row["ce_count"] for row in expected[index])
        assert metrics["ce_count"] == count == 7
        assert metrics["mean_nll"] == pytest.approx(total / count, abs=1e-6)
        assert all(row["mean_nll"] is None and row["ce_count"] == 0 for row in metrics["document_records"][:3])
    json.dumps(result, allow_nan=False)


def test_empty_eval_and_packed_documents_are_rejected():
    core = _model()
    for batches in ([], [replace(_batches()[0], ce_mask=torch.zeros_like(_batches()[0].valid_mask))]):
        with pytest.raises(ValueError, match="no valid CE"):
            evaluate_fbt_batches(core, batches, mode=FBTMode(), precision="fp32")
    batch = _batches()[0]
    docs = batch.document_ids.clone()
    docs[0, 2] = 22
    with pytest.raises(ValueError, match="Packed documents"):
        evaluate_fbt_batches(core, [replace(batch, document_ids=docs)], mode=FBTOnlineMode(), precision="fp32")


@pytest.mark.parametrize("kwargs", [{"mode": FBTMode(rt_mode=RTMode((0,)))},
                                   {"mode": FBTOnlineMode(rt_mode=RTMode((0,)))},
                                   {"mode": None}, {"chunk_size": 0}, {"chunk_size": True},
                                   {"precision": "bf16_mixed"}, {"precision": "fp16"},
                                   {"include_document_records": 1}])
def test_invalid_options_rejected(kwargs):
    options = dict(mode=FBTMode(), precision="fp32")
    options.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        evaluate_fbt_batches(_model(), _batches(), **options)
