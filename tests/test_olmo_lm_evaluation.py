"""Independent held-out NLL/accuracy, state-preservation and isolation checks."""

from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.lm_evaluation import evaluate_batches
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, NextLatLM
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(9404)


def _model(*, recurrent=False, enabled=True):
    config = OLMoConfig.tiny()
    backbone = (OLMoTiledRTForCausalLM(config, attention_backend="math", attention_precision="fp32")
                if recurrent else OLMoForCausalLM(config, attention_backend="math"))
    return NextLatLM(backbone, NextLatConfig(config.model_dim, dropout=0.4), enabled=enabled)


def _batches():
    ids = torch.tensor([[2, 3, 5, 8, 13, 21, 34], [1, 1, 4, 7, 10, 17, 22], [6, 9, 1, 1, 1, 1, 1]])
    valid = torch.tensor([[True] * 7, [False] * 2 + [True] * 5, [True] * 2 + [False] * 5])
    docs = torch.arange(3)[:, None].expand_as(ids).clone()
    ce = torch.ones_like(valid)
    ce[0, 1:3] = False
    one = NextLatBatch(ids, valid, docs, ce_mask=ce)
    two = NextLatBatch(torch.tensor([[12, 9, 6, 3]]), torch.ones(1, 4, dtype=torch.bool),
                      torch.full((1, 4), 14), ce_mask=torch.tensor([[False, False, False, True]]))
    return [one, two]


def _dense_oracle(model, batches, mode):
    """Token loops plus logsumexp/gather; does not call production masks/losses."""
    flags = [(module, module.training) for module in model.modules()]
    records = []
    try:
        model.eval()
        with torch.no_grad():
            kwargs = {} if mode is None else {"mode": mode}
            for batch_index, batch in enumerate(batches):
                hidden = model.backbone(batch.input_ids, attention_mask=batch.valid_mask,
                                        return_logits=False, **kwargs).last_hidden_state
                logits = F.linear(hidden, model.backbone.readout_weight).float()
                normalizers = torch.logsumexp(logits, dim=-1)
                for row in range(batch.input_ids.shape[0]):
                    losses, correct = [], 0
                    for target in range(1, batch.input_ids.shape[1]):
                        valid = bool(batch.valid_mask[row, target - 1] and batch.valid_mask[row, target])
                        same_doc = batch.document_ids[row, target - 1] == batch.document_ids[row, target]
                        selected = batch.ce_mask is None or bool(batch.ce_mask[row, target])
                        if valid and same_doc and selected:
                            token = int(batch.input_ids[row, target])
                            losses.append(float(normalizers[row, target - 1] - logits[row, target - 1, token]))
                            correct += int(logits[row, target - 1].argmax() == token)
                    if bool(batch.valid_mask[row].any()):
                        records.append({"batch_index": batch_index, "row_index": row,
                                        "ce_count": len(losses), "ce_sum": math.fsum(losses),
                                        "next_token_correct": correct})
    finally:
        for module, flag in flags:
            module.training = flag
    total = math.fsum(row["ce_sum"] for row in records)
    count = sum(row["ce_count"] for row in records)
    correct = sum(row["next_token_correct"] for row in records)
    return dict(ce_sum=total, ce_count=count, mean_nll=total / count,
                perplexity=math.exp(total / count), next_token_correct=correct,
                next_token_accuracy=correct / count, document_records=records)


@pytest.mark.parametrize("alpha", [None, 0.0, 0.37, 1.0])
@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_token_weighted_metrics_match_independent_dense_oracle(alpha, chunk_size):
    model = _model(recurrent=alpha is not None)
    mode = None if alpha is None else RTMode((0,), alpha)
    batches = _batches()
    expected = _dense_oracle(model, batches, mode)
    actual = evaluate_batches(model, iter(batches), mode=mode, precision="fp32", chunk_size=chunk_size,
                              include_document_records=True)
    for key in ("ce_sum", "mean_nll", "perplexity"):
        assert actual[key] == pytest.approx(expected[key], rel=8e-7, abs=2e-6), key
    for key in ("ce_count", "next_token_correct", "next_token_accuracy"):
        assert actual[key] == expected[key], key
    assert actual["ce_count"] == 10
    assert actual["documents"] == 4
    assert actual["batches"] == 2
    assert actual["input_tokens"] == 18
    for actual_row, expected_row in zip(actual["document_records"], expected["document_records"]):
        assert actual_row["ce_count"] == expected_row["ce_count"]
        assert actual_row["ce_sum"] == pytest.approx(expected_row["ce_sum"], rel=8e-7, abs=2e-6)
        assert actual_row["next_token_correct"] == expected_row["next_token_correct"]
    # Serialization must not rely on allowing NaN or infinity.
    json.dumps(actual, allow_nan=False)


def test_full_native_vocabulary_is_used_for_nll_and_argmax():
    model = _model()
    with torch.no_grad():
        model.backbone.readout_weight.zero_()
        model.backbone.readout_weight[66, 0] = 10
    batch = NextLatBatch(torch.tensor([[2, 66, 4, 66]]), torch.ones(1, 4, dtype=torch.bool),
                        torch.zeros(1, 4, dtype=torch.long))

    def fixed_states(ids, **kwargs):
        assert kwargs["return_logits"] is False
        hidden = torch.zeros(*ids.shape, 32)
        hidden[..., 0] = 1
        return SimpleNamespace(last_hidden_state=hidden)

    model.backbone.forward = fixed_states
    result = evaluate_batches(model, [batch], precision="fp32", chunk_size=2)
    assert result["next_token_correct"] == 2
    assert result["next_token_accuracy"] == 2 / 3
    expected = math.log(math.exp(10) + 66) - 20 / 3
    assert result["mean_nll"] == pytest.approx(expected, abs=1e-6)


def test_evaluation_never_calls_nextlat_or_changes_weights_grads_rng_or_mixed_train_flags():
    model = _model(recurrent=True)
    model.train()
    model.predictor.eval()
    model.backbone.norm.eval()
    flags = {name: module.training for name, module in model.named_modules()}
    parameters = dict(model.named_parameters())
    for index, parameter in enumerate(parameters.values()):
        parameter.grad = None if index % 2 else torch.full_like(parameter, 0.25)
    weights = {name: p.detach().clone() for name, p in parameters.items()}
    grads = {name: p.grad for name, p in parameters.items()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Held-out inference must not execute a training or predictor forward")

    model.forward = forbidden
    model.loss_sums = forbidden
    model.predictor.forward = forbidden
    native_forward = model.backbone.forward

    def check_native(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        assert kwargs["return_logits"] is False
        return native_forward(*args, **kwargs)

    model.backbone.forward = check_native
    rng = torch.get_rng_state().clone()
    result = evaluate_batches(model, _batches(), mode=RTMode((0,), 1), precision="fp32")
    assert result["ce_count"] == 10
    assert torch.equal(torch.get_rng_state(), rng)
    assert {name: module.training for name, module in model.named_modules()} == flags
    assert torch.is_grad_enabled()
    for name, parameter in parameters.items():
        assert torch.equal(parameter, weights[name])
        assert parameter.grad is grads[name]
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, torch.full_like(parameter, 0.25))


@pytest.mark.parametrize("failure", ["forward", "iterator", "nonfinite", "overflow"])
def test_failure_restores_every_module_training_flag(failure):
    model = _model()
    model.eval()
    model.backbone.layers[0].train()
    model.predictor.mlp[0].train()
    flags = {name: module.training for name, module in model.named_modules()}
    batch = _batches()[0]
    if failure == "forward":
        def fail(*args, **kwargs):
            raise RuntimeError("synthetic forward failure")
        model.backbone.forward = fail
        batches, error = [batch], RuntimeError
    elif failure == "iterator":
        def fail_iter():
            yield batch
            raise RuntimeError("synthetic iterator failure")
        batches, error = fail_iter(), RuntimeError
    else:
        def extreme_states(ids, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(*ids.shape, 32))
        model.backbone.forward = extreme_states
        with torch.no_grad():
            model.backbone.readout_weight.zero_()
            model.backbone.readout_weight[66, 0] = float("nan") if failure == "nonfinite" else 1000
        batches, error = [batch], FloatingPointError
    with pytest.raises(error):
        evaluate_batches(model, batches, precision="fp32")
    assert {name: module.training for name, module in model.named_modules()} == flags
    assert torch.is_grad_enabled()


def test_masks_are_at_target_positions_and_auxiliary_masks_do_not_affect_evaluation():
    model = _model()
    batch = _batches()[0]
    # An invalid interior position breaks both adjacent transitions.
    valid = batch.valid_mask.clone()
    valid[0, 4] = False
    batch = replace(batch, valid_mask=valid)
    expected = _dense_oracle(model, [batch], None)
    zero = torch.zeros_like(valid)
    altered = replace(batch, latent_mask=zero, kl_mask=zero)
    actual = evaluate_batches(model, [altered], precision="fp32")
    assert actual["ce_count"] == expected["ce_count"] == 7
    assert actual["mean_nll"] == pytest.approx(expected["mean_nll"], abs=1e-6)
    assert "document_records" not in actual


def test_empty_rows_or_documents_do_not_dilute_global_token_average():
    model = _model()
    batch = _batches()[1]
    empty = replace(batch, valid_mask=torch.zeros_like(batch.valid_mask))
    unscored = replace(batch, ce_mask=torch.zeros_like(batch.valid_mask))
    result = evaluate_batches(model, [empty, unscored, batch], precision="fp32", include_document_records=True)
    reference = evaluate_batches(model, [batch], precision="fp32")
    assert result["mean_nll"] == reference["mean_nll"]
    assert result["documents"] == 2
    assert result["ce_count"] == 1
    assert len(result["document_records"]) == 2
    assert result["document_records"][0]["mean_nll"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("empty_kind", ["iterator", "mask", "single_token"])
def test_empty_evaluation_is_rejected(empty_kind):
    model = _model()
    batch = _batches()[1]
    if empty_kind == "iterator":
        batches = []
    elif empty_kind == "mask":
        batches = [replace(batch, ce_mask=torch.zeros_like(batch.valid_mask))]
    else:
        batches = [NextLatBatch(batch.input_ids[:, :1], batch.valid_mask[:, :1], batch.document_ids[:, :1])]
    with pytest.raises(ValueError, match="no valid CE"):
        evaluate_batches(model, batches, precision="fp32")


def test_packed_documents_are_rejected_before_backbone_forward():
    model = _model()
    batch = _batches()[1]
    documents = batch.document_ids.clone()
    documents[:, 2:] = 99
    batch = replace(batch, document_ids=documents)

    def forbidden(*args, **kwargs):
        raise AssertionError("Packed data must fail before attention executes")

    model.backbone.forward = forbidden
    with pytest.raises(ValueError, match="Packed documents"):
        evaluate_batches(model, [batch], precision="fp32")


def test_explicit_modes_prevent_implicit_recurrence_and_reject_ordinary_layer_selection():
    batch = _batches()[1]
    with pytest.raises(ValueError, match="explicit RTMode"):
        evaluate_batches(_model(recurrent=True), [batch], precision="fp32")
    with pytest.raises(ValueError, match="ordinary backbone"):
        evaluate_batches(_model(), [batch], mode=RTMode((0,), 1), precision="fp32")
    ordinary = _model(enabled=False)
    implicit = evaluate_batches(ordinary, [batch], precision="fp32")
    explicit = evaluate_batches(ordinary, [batch], mode=RTMode((), 0), precision="fp32")
    assert implicit["ce_sum"] == explicit["ce_sum"]


@pytest.mark.parametrize("kwargs,match", [
    ({"precision": "bf16_mixed"}, "requires CUDA"),
    ({"precision": "fp16"}, "precision"),
    ({"precision": "fp32", "chunk_size": 0}, "chunk_size"),
    ({"precision": "fp32", "chunk_size": True}, "chunk_size"),
])
def test_invalid_runtime_options_are_rejected_without_cpu_fallback(kwargs, match):
    with pytest.raises(ValueError, match=match):
        evaluate_batches(_model(), _batches(), **kwargs)
