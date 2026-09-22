"""Fixed-layout loss parity, target ownership and replay-structure guards."""
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from cdrm.pretrained.fbt_training import aggregate_pass_losses
from cdrm.pretrained.nextlat import (
    NextLatBatch, NextLatConfig, NextLatPredictor, compute_nextlat_loss_sums,
)
from cdrm.pretrained.static_nextlat import (
    PreparedNextLatLayout, compute_static_nextlat_loss_sums,
)


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(718)


def batch_fixture(*, masked=True, length=7, rows=3):
    ids = (torch.arange(rows * length).reshape(rows, length) * 3 + 1) % 17
    valid = torch.ones_like(ids, dtype=torch.bool)
    docs = torch.arange(rows).unsqueeze(1).expand_as(ids).clone()
    if not masked:
        return NextLatBatch(ids, valid, docs)
    valid[0, 0] = False  # left padding
    valid[-1, -1] = False  # right padding
    if rows > 1 and length > 4:
        valid[1, 3] = False  # an internal hole must break pairs/triples
    docs.masked_fill_(~valid, -1)
    ce = torch.ones_like(valid)
    ce[0, :3] = False
    latent = (torch.arange(ids.numel()).reshape_as(ids) % 3) != 0
    kl = (torch.arange(ids.numel()).reshape_as(ids) % 2) == 0
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def leaves(config, batch):
    generator = torch.Generator().manual_seed(231)
    hidden = (torch.randn(*batch.input_ids.shape, config.model_dim, generator=generator) * .2).requires_grad_()
    # Extra vocabulary rows remain part of CE/KL, just as for the native model.
    readout = (torch.randn(23, config.model_dim, generator=generator) * .1).requires_grad_()
    return hidden, readout, NextLatPredictor(config)


def compare_records(actual, expected, actual_parameters, expected_parameters):
    assert actual.counts == expected.counts
    assert actual.weights == expected.weights
    for term in ("ce", "latent", "kl"):
        torch.testing.assert_close(actual.sums[term], expected.sums[term], atol=0, rtol=0)
        torch.testing.assert_close(actual.means[term], expected.means[term], atol=0, rtol=0)
    torch.testing.assert_close(actual.total, expected.total, atol=0, rtol=0)
    actual_gradients = torch.autograd.grad(actual.total, actual_parameters, allow_unused=True)
    expected_gradients = torch.autograd.grad(expected.total, expected_parameters, allow_unused=True)
    for index, (actual_gradient, expected_gradient) in enumerate(zip(actual_gradients, expected_gradients)):
        # Preserve absent gradients, not just numerical zeros. AdamW ownership
        # differs when an otherwise unused parameter receives an attached zero.
        assert (actual_gradient is None) == (expected_gradient is None), index
        if actual_gradient is not None:
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-7, rtol=2e-5, msg=f"gradient {index}")


@pytest.mark.parametrize("weights,enabled", [((1., 1.), True), ((.3, 0.), True),
    ((0., .7), True), ((0., 0.), True), ((.3, .7), False)])
@pytest.mark.parametrize("chunk_size", [1, 4, 32])
@pytest.mark.parametrize("masked", [False, True])
def test_static_matches_canonical_values_and_every_gradient_including_tied_embeddings(weights, enabled, chunk_size, masked):
    config = NextLatConfig(model_dim=64, lambda_latent=weights[0], lambda_kl=weights[1],
                           vocab_chunk_size=chunk_size)
    batch = batch_fixture(masked=masked)
    layout = PreparedNextLatLayout.from_batch(batch, config, enabled=enabled)
    h, w, predictor = leaves(config, batch)
    expected_h, expected_w, expected_predictor = leaves(config, batch)
    actual = compute_static_nextlat_loss_sums(h, F.embedding(batch.input_ids, w), w,
        batch.input_ids, predictor, config, layout, enabled=enabled)
    expected = compute_nextlat_loss_sums(expected_h, F.embedding(batch.input_ids, expected_w),
        expected_w, batch, expected_predictor, config, enabled=enabled)
    compare_records(actual, expected, (h, w, *predictor.parameters()),
                    (expected_h, expected_w, *expected_predictor.parameters()))


@pytest.mark.parametrize("length", [1, 2, 3, 7])
def test_zero_objective_counts_keep_attached_hidden_zero_and_no_other_gradients(length):
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture(masked=False, length=length)
    off = torch.zeros_like(batch.valid_mask)
    batch = replace(batch, ce_mask=off, latent_mask=off, kl_mask=off)
    layout = PreparedNextLatLayout.from_batch(batch, config)
    h, w, predictor = leaves(config, batch)
    expected_h, expected_w, expected_predictor = leaves(config, batch)
    actual = compute_static_nextlat_loss_sums(h, F.embedding(batch.input_ids, w), w,
        batch.input_ids, None, config, layout)
    expected = compute_nextlat_loss_sums(expected_h, F.embedding(batch.input_ids, expected_w),
        expected_w, batch, None, config)
    assert actual.counts == {"ce": 0, "latent": 0, "kl": 0}
    compare_records(actual, expected, (h, w, *predictor.parameters()),
                    (expected_h, expected_w, *expected_predictor.parameters()))


@pytest.mark.parametrize("length", [1, 2])
def test_short_sequences_match_without_artificial_pairs_or_triples(length):
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture(masked=False, length=length)
    layout = PreparedNextLatLayout.from_batch(batch, config)
    h, w, predictor = leaves(config, batch)
    expected_h, expected_w, expected_predictor = leaves(config, batch)
    actual = compute_static_nextlat_loss_sums(h, F.embedding(batch.input_ids, w), w,
        batch.input_ids, predictor, config, layout)
    expected = compute_nextlat_loss_sums(expected_h, F.embedding(batch.input_ids, expected_w),
        expected_w, batch, expected_predictor, config)
    compare_records(actual, expected, (h, w, *predictor.parameters()),
                    (expected_h, expected_w, *expected_predictor.parameters()))


class ProbePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3))
        self.seen = None

    def forward(self, hidden, embeddings):
        self.seen = (hidden.detach().clone(), embeddings.detach().clone())
        return hidden + self.scale * embeddings


def test_predictor_union_and_remapping_have_explicit_row_major_target_order():
    config = NextLatConfig(model_dim=64, vocab_chunk_size=2)
    batch = batch_fixture(masked=False, rows=2, length=6)
    off = torch.zeros_like(batch.valid_mask)
    latent, kl = off.clone(), off.clone()
    latent[0, [1, 4]] = True
    latent[1, [2, 5]] = True
    kl[0, [2, 3]] = True
    kl[1, [3, 4]] = True
    batch = replace(batch, ce_mask=off, latent_mask=latent, kl_mask=kl)
    layout = PreparedNextLatLayout.from_batch(batch, config)
    assert layout.needed_source_indices.tolist() == [0, 1, 3, 7, 8, 10]
    assert layout.latent_prediction_indices.tolist() == [0, 2, 3, 5]
    assert layout.latent_target_indices.tolist() == [1, 4, 8, 11]
    assert layout.kl_prediction_indices.tolist() == [0, 1, 3, 4]
    assert layout.kl_teacher_indices.tolist() == [1, 2, 8, 9]
    h, w, _ = leaves(config, batch)
    predictor = ProbePredictor()
    embeddings = F.embedding(batch.input_ids, w)
    compute_static_nextlat_loss_sums(h, embeddings, w, batch.input_ids, predictor, config, layout)
    expected_sources = torch.stack([h[0, 0], h[0, 1], h[0, 3], h[1, 1], h[1, 2], h[1, 4]])
    expected_embeddings = torch.stack([embeddings[0, 1], embeddings[0, 2], embeddings[0, 4],
                                      embeddings[1, 2], embeddings[1, 3], embeddings[1, 5]])
    torch.testing.assert_close(predictor.seen[0], expected_sources, atol=0, rtol=0)
    torch.testing.assert_close(predictor.seen[1], expected_embeddings, atol=0, rtol=0)


@pytest.mark.parametrize("term", ["latent", "kl"])
def test_auxiliary_target_and_readout_are_detached_but_conditioning_is_attached(term):
    config = NextLatConfig(model_dim=64, lambda_latent=float(term == "latent"),
                           lambda_kl=float(term == "kl"))
    batch = batch_fixture(masked=False, rows=1, length=4)
    off = torch.zeros_like(batch.valid_mask)
    latent, kl = off.clone(), off.clone()
    latent[0, 1] = True
    kl[0, 2] = True
    batch = replace(batch, ce_mask=off, latent_mask=latent, kl_mask=kl)
    layout = PreparedNextLatLayout.from_batch(batch, config)
    h, w, _ = leaves(config, batch)
    embeddings = torch.randn_like(h, requires_grad=True)
    predictor = ProbePredictor()
    result = compute_static_nextlat_loss_sums(h, embeddings, w, batch.input_ids, predictor, config, layout)
    h_grad, e_grad, w_grad, p_grad = torch.autograd.grad(result.total,
        (h, embeddings, w, predictor.scale), allow_unused=True)
    assert h_grad[0, 0].abs().sum() > 0
    assert torch.count_nonzero(h_grad[0, 1:]) == 0
    assert e_grad[0, 1].abs().sum() > 0
    assert torch.count_nonzero(e_grad[0, [0, 2, 3]]) == 0
    assert w_grad is None
    assert p_grad.abs() > 0


def test_static_losses_preserve_nonuniform_fbt_pass_weighting():
    config = NextLatConfig(model_dim=64, lambda_latent=.4, lambda_kl=.8, vocab_chunk_size=4)
    batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(batch, config)
    h, w, predictor = leaves(config, batch)
    eh, ew, ep = leaves(config, batch)
    actual_passes, expected_passes = [], []
    for scale in (1., .7, 1.3):
        actual_passes.append(compute_static_nextlat_loss_sums(h * scale, F.embedding(batch.input_ids, w),
            w, batch.input_ids, predictor, config, layout))
        expected_passes.append(compute_nextlat_loss_sums(eh * scale, F.embedding(batch.input_ids, ew),
            ew, batch, ep, config))
    actual = aggregate_pass_losses(actual_passes, gamma=.3)
    expected = aggregate_pass_losses(expected_passes, gamma=.3)
    assert actual.pass_coefficients == (1., .15, .15)
    assert actual.counts == layout.counts  # not multiplied by K
    compare_records(actual, expected, (h, w, *predictor.parameters()), (eh, ew, *ep.parameters()))


class ForbidDynamicSelection(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        forbidden = ("_local_scalar_dense", "nonzero", "masked_select", "_unique", "unique_dim")
        assert not any(name in str(func) for name in forbidden), str(func)
        return func(*args, **(kwargs or {}))


def test_tensor_execution_and_backward_need_no_dynamic_selection_or_host_scalar_reads():
    config = NextLatConfig(model_dim=64, vocab_chunk_size=2)
    batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(batch, config)
    h, w, predictor = leaves(config, batch)
    with ForbidDynamicSelection():
        result = compute_static_nextlat_loss_sums(h, F.embedding(batch.input_ids, w), w,
            batch.input_ids, predictor, config, layout)
        result.total.backward()
    assert torch.isfinite(h.grad).all()


def test_layout_accepts_new_token_values_and_owns_structure_snapshots():
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(batch, config)
    replacement = replace(batch, input_ids=(batch.input_ids + 2) % 17)
    layout.validate_batch(replacement)
    counts = layout.counts
    counts["ce"] = -1
    weights = layout.weights
    weights["kl"] = -1
    assert layout.counts["ce"] >= 0 and layout.weights["kl"] == 1
    batch.ce_mask[0, 0] = ~batch.ce_mask[0, 0]
    with pytest.raises(ValueError, match="ce_mask changed"):
        layout.validate_batch(batch)


def test_reused_layout_reads_changed_ce_targets_and_conditioning_tokens():
    config = NextLatConfig(model_dim=64, vocab_chunk_size=4)
    old_batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(old_batch, config)
    batch = replace(old_batch, input_ids=(old_batch.input_ids + 5) % 17)
    layout.validate_batch(batch)
    h, w, predictor = leaves(config, batch)
    eh, ew, ep = leaves(config, batch)
    actual = compute_static_nextlat_loss_sums(h, F.embedding(batch.input_ids, w), w,
        batch.input_ids, predictor, config, layout)
    expected = compute_nextlat_loss_sums(eh, F.embedding(batch.input_ids, ew), ew, batch, ep, config)
    compare_records(actual, expected, (h, w, *predictor.parameters()), (eh, ew, *ep.parameters()))


def test_layout_preparation_preserves_caller_tensors_and_rng():
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture()
    snapshots = {name: None if value is None else value.clone() for name, value in batch.__dict__.items()}
    rng = torch.random.get_rng_state().clone()
    PreparedNextLatLayout.from_batch(batch, config)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, value in batch.__dict__.items():
        if value is not None:
            assert torch.equal(value, snapshots[name])


def test_static_preserves_canonical_behavior_with_frozen_readout_and_independent_embeddings():
    config = NextLatConfig(model_dim=64, vocab_chunk_size=3)
    batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(batch, config)
    h, w, predictor = leaves(config, batch)
    eh, ew, ep = leaves(config, batch)
    w.requires_grad_(False)
    ew.requires_grad_(False)
    e = F.embedding(batch.input_ids, w).requires_grad_()
    ee = F.embedding(batch.input_ids, ew).requires_grad_()
    actual = compute_static_nextlat_loss_sums(h, e, w, batch.input_ids, predictor, config, layout)
    expected = compute_nextlat_loss_sums(eh, ee, ew, batch, ep, config)
    compare_records(actual, expected, (h, e, *predictor.parameters()), (eh, ee, *ep.parameters()))


@pytest.mark.parametrize("field", ["valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"])
def test_layout_rejects_changed_structure_even_if_loss_counts_would_match(field):
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(batch, config)
    value = getattr(batch, field).clone()
    if field == "document_ids":
        value[0, batch.valid_mask[0]] = 20  # valid one-document change
    elif field == "valid_mask":
        value[0, 1] = False
    else:
        value[0, 0] = ~value[0, 0]  # inactive target still changes static policy
    with pytest.raises(ValueError, match=f"{field} changed"):
        layout.validate_batch(replace(batch, **{field: value}))


def test_none_and_explicit_equal_policy_are_distinct_layouts():
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture(masked=False)
    layout = PreparedNextLatLayout.from_batch(batch, config)
    with pytest.raises(ValueError, match="ce_mask changed"):
        layout.validate_batch(replace(batch, ce_mask=torch.ones_like(batch.valid_mask)))


def test_index_mutation_is_rejected_before_replay():
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(batch, config)
    layout.ce_source_indices.add_(1)
    with pytest.raises(ValueError, match="index tensors were modified"):
        layout.validate_batch(batch)


def test_host_batch_validation_is_allowed_before_transfer_to_execution_device():
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(batch, config)
    # Exercise device-policy metadata on CPU only; this is not GPU execution.
    host_validation_layout = replace(layout, device=torch.device("cuda:0"))
    host_validation_layout.validate_batch(batch)
    with pytest.raises(ValueError, match="shape/device"):
        host_validation_layout.validate_execution(batch.input_ids, config, enabled=True)


def test_internal_structure_snapshot_mutation_is_rejected():
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(batch, config)
    layout._structure[0].logical_not_()
    with pytest.raises(ValueError, match="structure snapshots were modified"):
        layout.validate_integrity()


@pytest.mark.parametrize("which", ["enabled", "weights", "chunk_size", "shape", "dtype", "hidden", "readout"])
def test_metadata_changes_reject_without_tensor_execution(which):
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture()
    layout = PreparedNextLatLayout.from_batch(batch, config)
    h, w, predictor = leaves(config, batch)
    embeddings = F.embedding(batch.input_ids, w)
    ids, enabled = batch.input_ids, True
    if which == "enabled":
        enabled = False
    elif which == "weights":
        config = replace(config, lambda_kl=.5)
    elif which == "chunk_size":
        config = replace(config, vocab_chunk_size=3)
    elif which == "shape":
        ids = ids[:1]
    elif which == "dtype":
        ids = ids.int()
    elif which == "hidden":
        h = h[:, :-1]
    elif which == "readout":
        w = w[:, :-1]
    with pytest.raises(ValueError):
        compute_static_nextlat_loss_sums(h, embeddings, w, ids, predictor, config, layout, enabled=enabled)


@pytest.mark.parametrize("malformed", ["packed", "negative_doc", "integer_mask", "bad_shape", "enabled"])
def test_layout_validation_rejects_invalid_canonical_inputs(malformed):
    config = NextLatConfig(model_dim=64)
    batch = batch_fixture(masked=False)
    enabled = True
    if malformed == "packed":
        batch.document_ids[0, 3:] = 9
    elif malformed == "negative_doc":
        batch.document_ids[0, 0] = -1
    elif malformed == "integer_mask":
        batch = replace(batch, valid_mask=batch.valid_mask.int())
    elif malformed == "bad_shape":
        batch = replace(batch, latent_mask=batch.valid_mask[:, :-1])
    else:
        enabled = 1
    with pytest.raises((ValueError, TypeError)):
        PreparedNextLatLayout.from_batch(batch, config, enabled=enabled)
