"""Independent LM-shift, gradient-ownership and dense NextLat objective checks."""

import ast
import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from cdrm.pretrained.nextlat import (
    NextLatBatch,
    NextLatConfig,
    NextLatLM,
    NextLatPredictor,
    build_nextlat_masks,
    compute_nextlat_loss_sums,
)
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(314159)


def _source_predictor(config):
    """Execute only unchanged upstream class ASTs, never framework imports."""
    directory = Path(__file__).resolve().parents[1] / "cdrm/pretrained/_nextlat_reference"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["revision"] == "b37d3411ab9b17be8638abbddb9529f0f3a0a5f9"
    for record in manifest["files"]:
        data = (directory / record["file"]).read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
    namespace = dict(torch=torch, nn=nn, F=F, dataclass=dataclass)
    for filename, names in (
        ("model_base.py", {"LayerNorm"}),
        ("model_nextlat.py", {"NextLatDynamicsModel"}),
    ):
        tree = ast.parse((directory / filename).read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        assert {node.name for node in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    source_config = SimpleNamespace(
        n_embd=config.model_dim, proj_factor=config.proj_factor,
        bias=config.bias, dropout=config.dropout,
    )
    return namespace["NextLatDynamicsModel"](source_config)


def _batch(*, length=7, batch_size=2):
    tokens = torch.randint(2, 61, (batch_size, length))
    valid = torch.ones_like(tokens, dtype=torch.bool)
    docs = torch.arange(batch_size).unsqueeze(1).expand_as(tokens).clone()
    return NextLatBatch(tokens, valid, docs)


def _oracle_masks(batch):
    """Explicit index loops so production vectorized mask alignment is checked."""
    batch_size, length = batch.input_ids.shape
    masks = {key: torch.zeros(batch_size, max(0, length - distance), dtype=torch.bool)
             for key, distance in (("ce", 1), ("latent", 1), ("kl", 2))}
    for row in range(batch_size):
        for key, distance in (("ce", 1), ("latent", 1), ("kl", 2)):
            selection = getattr(batch, key + "_mask")
            for time in range(length - distance):
                positions = range(time, time + distance + 1)
                valid = all(bool(batch.valid_mask[row, p]) for p in positions)
                same_doc = len({int(batch.document_ids[row, p]) for p in positions}) == 1
                selected = selection is None or bool(selection[row, time + distance])
                masks[key][row, time] = valid and same_doc and selected
    return masks


def _dense_oracle(hidden, embeddings, readout, batch, predictor, config, *, enabled=True):
    """Full-vocabulary reference using separate masking and ordinary autograd."""
    masks = _oracle_masks(batch)
    logits = F.linear(hidden[:, :-1], readout).float()
    losses = F.cross_entropy(logits.flatten(0, 1), batch.input_ids[:, 1:].flatten(), reduction="none")
    ce = (losses.reshape_as(masks["ce"]) * masks["ce"]).sum()
    zero = hidden.sum() * 0
    sums = {"ce": ce, "latent": zero, "kl": zero}
    counts = {"ce": int(masks["ce"].sum()), "latent": 0, "kl": 0}
    if enabled and (config.lambda_latent or config.lambda_kl):
        predicted = predictor(hidden[:, :-1], embeddings[:, 1:])
        if config.lambda_latent:
            elementwise = F.smooth_l1_loss(predicted.float(), hidden[:, 1:].detach().float(), reduction="none")
            sums["latent"] = (elementwise.mean(dim=-1) * masks["latent"]).sum()
            counts["latent"] = int(masks["latent"].sum())
        if config.lambda_kl:
            student_logits = F.linear(predicted[:, :-1], readout.detach()).float()
            teacher_logits = F.linear(hidden[:, 1:-1].detach(), readout.detach()).float()
            log_student = F.log_softmax(student_logits, dim=-1)
            log_teacher = F.log_softmax(teacher_logits, dim=-1)
            kl = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=-1)
            sums["kl"] = (kl * masks["kl"]).sum()
            counts["kl"] = int(masks["kl"].sum())
    means = {key: value / max(counts[key], 1) for key, value in sums.items()}
    total = means["ce"] + config.lambda_latent * means["latent"] + config.lambda_kl * means["kl"]
    return SimpleNamespace(sums=sums, counts=counts, means=means, total=total)


def _assert_gradients(actual, expected, *, atol=2e-6, rtol=3e-5):
    assert len(actual) == len(expected)
    for index, (a, e) in enumerate(zip(actual, expected)):
        if a is None or e is None:
            other = e if a is None else a
            assert other is None or torch.count_nonzero(other) == 0, index
        else:
            torch.testing.assert_close(a, e, atol=atol, rtol=rtol, msg=f"gradient {index}")


def test_predictor_matches_pinned_language_model_classes_outputs_and_all_derivatives():
    config = NextLatConfig(model_dim=32)
    actual = NextLatPredictor(config)
    source = _source_predictor(config)
    source.load_state_dict(actual.state_dict(), strict=True)
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    embedding = torch.randn_like(hidden, requires_grad=True)
    source_hidden = hidden.detach().clone().requires_grad_()
    source_embedding = embedding.detach().clone().requires_grad_()
    result = actual(hidden, embedding)
    expected = source(source_hidden, source_embedding)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    probe = torch.randn_like(result)
    actual_grads = torch.autograd.grad((result * probe).sum(), (hidden, embedding, *actual.parameters()))
    source_grads = torch.autograd.grad((expected * probe).sum(), (source_hidden, source_embedding, *source.parameters()))
    _assert_gradients(actual_grads, source_grads, atol=0, rtol=0)


def test_language_model_predictor_defaults_differ_from_historical_a5():
    config = NextLatConfig(model_dim=2048)
    assert config.proj_factor == 1.6
    assert 128 * round(config.proj_factor * 2 * config.model_dim / 128) == 6528
    assert config.lambda_latent == config.lambda_kl == 1.0
    assert config.bias is False and config.dropout == 0


def test_pair_triple_masks_respect_documents_padding_and_independent_target_policy():
    batch = _batch(length=8)
    batch.valid_mask[0, 0] = False
    batch.valid_mask[1, -1] = False
    batch.document_ids[0, 4:] = 9
    batch.document_ids[1, 3:6] = 8
    ce = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    latent = torch.ones_like(ce)
    latent[:, 2] = False
    kl = torch.zeros_like(ce)
    kl[:, [2, 3, 6, 7]] = True
    batch = replace(batch, ce_mask=ce, latent_mask=latent, kl_mask=kl)
    actual = build_nextlat_masks(batch)
    expected = _oracle_masks(batch)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key
    assert not torch.equal(actual["ce"], actual["latent"])


@pytest.mark.parametrize("chunk_size", [1, 11, 67, 103])
@pytest.mark.parametrize("aux_weights", [(1.0, 1.0), (0.3, 0.0), (0.0, 0.7)])
def test_chunked_objective_matches_independent_dense_values_and_all_gradients(chunk_size, aux_weights):
    config = NextLatConfig(model_dim=32, vocab_chunk_size=chunk_size,
                           lambda_latent=aux_weights[0], lambda_kl=aux_weights[1])
    predictor = NextLatPredictor(config)
    source_predictor = copy.deepcopy(predictor)
    batch = _batch()
    batch.valid_mask[0, 0] = False
    batch.valid_mask[1, -1] = False
    batch.document_ids[0, 4:] = 3
    ce_policy = torch.ones_like(batch.valid_mask)
    ce_policy[:, :3] = False
    batch = replace(batch, ce_mask=ce_policy)
    hidden = torch.randn(2, 7, 32, requires_grad=True)
    embeddings = torch.randn_like(hidden, requires_grad=True)
    # All 67 rows must participate, including those outside the tokenizer's 61.
    readout = (0.2 * torch.randn(67, 32)).requires_grad_()
    expected_inputs = [v.detach().clone().requires_grad_() for v in (hidden, embeddings, readout)]
    actual = compute_nextlat_loss_sums(hidden, embeddings, readout, batch, predictor, config)
    expected = _dense_oracle(*expected_inputs, batch, source_predictor, config)
    for key in expected.sums:
        assert actual.counts[key] == expected.counts[key]
        torch.testing.assert_close(actual.sums[key], expected.sums[key], atol=3e-6, rtol=2e-6)
        torch.testing.assert_close(actual.means[key], expected.means[key], atol=1e-6, rtol=2e-6)
    torch.testing.assert_close(actual.total, expected.total, atol=2e-6, rtol=2e-6)
    actual_grads = torch.autograd.grad(actual.total, (hidden, embeddings, readout, *predictor.parameters()), allow_unused=True)
    expected_grads = torch.autograd.grad(expected.total, (*expected_inputs, *source_predictor.parameters()), allow_unused=True)
    _assert_gradients(actual_grads, expected_grads)


class _ProbePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3))
        self.seen = None

    def forward(self, current_states, next_token_embeds):
        self.seen = (current_states.detach().clone(), next_token_embeds.detach().clone())
        return current_states + self.scale * next_token_embeds


@pytest.mark.parametrize("objective", ["latent", "kl"])
def test_only_source_and_conditioning_routes_receive_auxiliary_credit(objective):
    config = NextLatConfig(model_dim=32, lambda_latent=float(objective == "latent"),
                           lambda_kl=float(objective == "kl"), vocab_chunk_size=11)
    batch = _batch(length=4, batch_size=1)
    off = torch.zeros_like(batch.valid_mask)
    latent = off.clone()
    latent[:, 1] = True
    kl = off.clone()
    kl[:, 2] = True
    batch = replace(batch, ce_mask=off, latent_mask=latent, kl_mask=kl)
    hidden = torch.randn(1, 4, 32, requires_grad=True)
    embeddings = torch.randn_like(hidden, requires_grad=True)
    readout = (0.2 * torch.randn(67, 32)).requires_grad_()
    predictor = _ProbePredictor()
    result = compute_nextlat_loss_sums(hidden, embeddings, readout, batch, predictor, config)
    result.total.backward()
    assert hidden.grad[:, 0].norm() > 0
    assert torch.count_nonzero(hidden.grad[:, 1:]) == 0  # detached target and teacher
    assert embeddings.grad[:, 1].norm() > 0
    assert torch.count_nonzero(embeddings.grad[:, [0, 2, 3]]) == 0
    assert readout.grad is None or torch.count_nonzero(readout.grad) == 0
    assert predictor.scale.grad.abs() > 0
    torch.testing.assert_close(predictor.seen[0], hidden[:, 0])
    torch.testing.assert_close(predictor.seen[1], embeddings[:, 1])


def test_latent_loss_divides_by_valid_pairs_and_latent_coordinates():
    config = NextLatConfig(model_dim=32, lambda_kl=0)
    batch = _batch(length=4, batch_size=1)
    off = torch.zeros_like(batch.valid_mask)
    latent_mask = off.clone()
    latent_mask[:, 1] = True
    batch = replace(batch, ce_mask=off, latent_mask=latent_mask)
    hidden = torch.zeros(1, 4, 32, requires_grad=True)
    embeddings = torch.zeros_like(hidden)
    embeddings[:, 1, 0] = 2
    predictor = _ProbePredictor()
    predictor.scale.data.fill_(1)
    result = compute_nextlat_loss_sums(hidden, embeddings, torch.randn(67, 32), batch, predictor, config)
    assert result.counts["latent"] == 1
    # beta=1 SmoothL1(2, 0)=1.5, exactly one of 32 coordinates in one pair.
    torch.testing.assert_close(result.means["latent"], torch.tensor(1.5 / 32), atol=0, rtol=0)


@pytest.mark.parametrize("enabled,weights", [(False, (1.0, 1.0)), (True, (0.0, 0.0))])
def test_zero_weight_or_disabled_nextlat_bypasses_predictor_exactly(enabled, weights):
    config = NextLatConfig(model_dim=32, lambda_latent=weights[0], lambda_kl=weights[1])
    batch = _batch(length=4)
    predictor = NextLatPredictor(config)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled NextLat must not execute predictor")

    predictor.forward = forbidden
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    readout = torch.randn(67, 32, requires_grad=True)
    embeddings = torch.full_like(hidden, float("nan"), requires_grad=True)
    result = compute_nextlat_loss_sums(hidden, embeddings, readout, batch, predictor, config, enabled=enabled)
    expected = F.cross_entropy(F.linear(hidden[:, :-1], readout).flatten(0, 1), batch.input_ids[:, 1:].flatten())
    torch.testing.assert_close(result.total, expected)
    assert result.counts["latent"] == result.counts["kl"] == 0
    result.total.backward()
    assert embeddings.grad is None
    assert all(parameter.grad is None for parameter in predictor.parameters())


@pytest.mark.parametrize("length", [1, 2, 5])
def test_empty_valid_objectives_return_finite_differentiable_zero(length):
    config = NextLatConfig(model_dim=32)
    batch = _batch(length=length)
    off = torch.zeros_like(batch.valid_mask)
    batch = replace(batch, ce_mask=off, latent_mask=off, kl_mask=off)
    hidden = torch.randn(2, length, 32, requires_grad=True)
    embeddings = torch.randn_like(hidden, requires_grad=True)
    readout = torch.randn(67, 32, requires_grad=True)
    predictor = NextLatPredictor(config)
    result = compute_nextlat_loss_sums(hidden, embeddings, readout, batch, predictor, config)
    assert result.total.item() == 0
    assert all(count == 0 for count in result.counts.values())
    result.total.backward()
    assert hidden.grad is None or torch.count_nonzero(hidden.grad) == 0
    assert embeddings.grad is None or torch.count_nonzero(embeddings.grad) == 0
    assert readout.grad is None or torch.count_nonzero(readout.grad) == 0


def test_tied_embedding_keeps_ce_and_conditioning_paths_without_auxiliary_readout_credit():
    config = NextLatConfig(model_dim=32, vocab_chunk_size=13)
    batch = _batch(length=4, batch_size=1)
    hidden = torch.randn(1, 4, 32, requires_grad=True)
    readout = (0.2 * torch.randn(67, 32)).requires_grad_()
    predictor = NextLatPredictor(config)
    embeddings = F.embedding(batch.input_ids, readout)
    result = compute_nextlat_loss_sums(hidden, embeddings, readout, batch, predictor, config)
    total_grad = torch.autograd.grad(result.total, readout, retain_graph=True)[0]
    ce_grad = torch.autograd.grad(result.means["ce"], readout, retain_graph=True)[0]
    aux_grad = torch.autograd.grad(result.means["latent"] + result.means["kl"], readout)[0]
    torch.testing.assert_close(total_grad, ce_grad + aux_grad, atol=2e-7, rtol=2e-6)
    assert aux_grad.norm() > 0
    conditioned = set(batch.input_ids[:, 1:].flatten().tolist())
    unused = [i for i in range(readout.shape[0]) if i not in conditioned]
    assert torch.count_nonzero(aux_grad[unused]) == 0
    # Extra native output rows still obtain ordinary CE readout credit.
    assert ce_grad[61:].norm() > 0


@pytest.mark.parametrize("alpha", [None, 0.0, 0.37, 1.0])
def test_integrated_ordinary_and_tiled_rt_objectives_match_scan_dense_all_parameter_gradients(alpha):
    config = OLMoConfig.tiny()
    if alpha is None:
        backbone = OLMoForCausalLM(config, attention_backend="math")
        reference = OLMoForCausalLM(config, attention_backend="math")
        kwargs = {}
    else:
        backbone = OLMoTiledRTForCausalLM(config, attention_backend="math", attention_precision="fp32")
        reference = OLMoRTForCausalLM(config, attention_backend="math")
        kwargs = {"mode": RTMode((0,), alpha)}
    reference.load_state_dict(backbone.state_dict(), strict=True)
    nextlat = NextLatConfig(model_dim=config.model_dim, vocab_chunk_size=3)
    model = NextLatLM(backbone, nextlat)
    reference_predictor = copy.deepcopy(model.predictor)
    batch = _batch(length=7)
    batch.valid_mask[0, 0] = False
    batch.valid_mask[1, -2:] = False
    off = torch.ones_like(batch.valid_mask)
    off[:, :3] = False
    batch = replace(batch, ce_mask=off)
    actual = model.loss_sums(batch, backbone_kwargs=kwargs)
    embeddings = reference.token_embeddings(batch.input_ids)
    hidden = reference(inputs_embeds=embeddings, attention_mask=batch.valid_mask, return_logits=False,
                       **kwargs).last_hidden_state
    expected = _dense_oracle(hidden, embeddings, reference.readout_weight, batch, reference_predictor, nextlat)
    assert actual.counts == expected.counts == model.counts(batch)
    torch.testing.assert_close(actual.total, expected.total, atol=2e-6, rtol=2e-6)
    actual_parameters = tuple(model.backbone.parameters()) + tuple(model.predictor.parameters())
    expected_parameters = tuple(reference.parameters()) + tuple(reference_predictor.parameters())
    actual_grads = torch.autograd.grad(actual.total, actual_parameters)
    expected_grads = torch.autograd.grad(expected.total, expected_parameters)
    _assert_gradients(actual_grads, expected_grads, atol=5e-6, rtol=8e-5)
    assert len({id(p) for p in model.parameters()}) == len(actual_parameters)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_training_uses_post_finalnorm_once_and_inference_calls_only_backbone():
    backbone = OLMoForCausalLM(OLMoConfig.tiny(), attention_backend="math")
    model = NextLatLM(backbone, NextLatConfig(model_dim=32))
    batch = _batch(length=5)
    captured = []
    handle = backbone.norm.register_forward_hook(lambda module, args, result: captured.append(result))
    try:
        training = model.loss_sums(batch)
        assert len(captured) == 1
        training.total.backward()
        assert len(captured) == 1
        captured.clear()

        def forbidden(*args, **kwargs):
            raise AssertionError("Native inference must not call the NextLat predictor")

        model.predictor.forward = forbidden
        with torch.no_grad():
            output = model.backbone(batch.input_ids, attention_mask=batch.valid_mask)
        assert len(captured) == 1
        assert output.logits.shape == (2, 5, 67)
        assert torch.equal(output.last_hidden_state, captured[0])
    finally:
        handle.remove()


def test_packed_documents_are_rejected_before_backbone_attention():
    model = NextLatLM(OLMoForCausalLM(OLMoConfig.tiny()), NextLatConfig(model_dim=32))
    batch = _batch(length=6)
    batch.document_ids[0, 3:] = 8

    def forbidden(*args, **kwargs):
        raise AssertionError("Reject packed rows before running attention")

    model.backbone.forward = forbidden
    with pytest.raises(ValueError, match="[Pp]acked"):
        model.loss_sums(batch)
    with pytest.raises(ValueError, match="[Pp]acked"):
        model.counts(batch)
    # The loss-only primitive can explicitly mask independent precomputed states.
    assert not build_nextlat_masks(batch)["ce"][0, 2]


@pytest.mark.parametrize("forbidden", [{"use_cache": True}, {"past_key_values": None}, {"attention_mask": None}])
def test_full_document_training_rejects_cache_or_external_mask_overrides(forbidden):
    model = NextLatLM(OLMoForCausalLM(OLMoConfig.tiny()), NextLatConfig(model_dim=32))
    with pytest.raises(ValueError, match="full-document|cache|owns"):
        model.loss_sums(_batch(), backbone_kwargs=forbidden)


def test_separate_document_rows_and_future_tokens_cannot_contaminate_prior_states():
    backbone = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32")
    model = NextLatLM(backbone, NextLatConfig(model_dim=32))
    batch = _batch(length=6)
    captured = []
    handle = backbone.norm.register_forward_hook(lambda module, args, result: captured.append(result.detach().clone()))
    try:
        with torch.no_grad():
            model.loss_sums(batch, backbone_kwargs={"mode": RTMode((0,), 1)})
            changed = replace(batch, input_ids=batch.input_ids.clone())
            changed.input_ids[1] = (changed.input_ids[1] + 3) % 61
            changed.input_ids[0, -2:] = (changed.input_ids[0, -2:] + 7) % 61
            model.loss_sums(changed, backbone_kwargs={"mode": RTMode((0,), 1)})
        torch.testing.assert_close(captured[0][0, :-2], captured[1][0, :-2], atol=0, rtol=0)
        assert not torch.equal(captured[0][1], captured[1][1])
    finally:
        handle.remove()


def test_shared_tiled_parameters_across_three_losses_return_the_sum_of_owned_gradients():
    backbone = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32")
    model = NextLatLM(backbone, NextLatConfig(model_dim=32, vocab_chunk_size=3))
    batches = [_batch(length=length, batch_size=1) for length in (4, 5, 6)]
    modes = [RTMode((0,), alpha) for alpha in (0.0, 0.37, 1.0)]
    parameters = tuple(model.parameters())
    expected = [torch.zeros_like(parameter) for parameter in parameters]
    for batch, mode in zip(batches, modes):
        loss = model.loss_sums(batch, backbone_kwargs={"mode": mode}).total
        grads = torch.autograd.grad(loss, parameters)
        expected = [total + grad for total, grad in zip(expected, grads)]
    losses = [model.loss_sums(batch, backbone_kwargs={"mode": mode}).total for batch, mode in zip(batches, modes)]
    actual = torch.autograd.grad(sum(losses), parameters)
    _assert_gradients(actual, expected, atol=4e-6, rtol=5e-5)
    assert all(parameter.grad is None for parameter in parameters)
