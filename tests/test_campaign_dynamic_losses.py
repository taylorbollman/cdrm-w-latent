"""CPU differential checks for fixed-capacity, changing-selection losses.

These qualify loss algebra and graph inputs, not actual CUDA replay or DDP.
"""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from cdrm.pretrained.campaign_losses import DynamicNextLatLayout, compute_dynamic_nextlat_loss_sums
from cdrm.pretrained.nextlat import (NextLatBatch, NextLatConfig, NextLatPredictor,
                                   compute_nextlat_loss_sums)


TERMS = ("ce", "latent", "kl")


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(8283)


def make_batch(kind="mixed", rows=3, length=7):
    ids = (torch.arange(rows*length).reshape(rows, length)*3+1) % 19
    valid = torch.ones_like(ids, dtype=torch.bool)
    docs = torch.arange(rows)[:, None].expand_as(ids).clone()
    ce, latent, kl = (valid.clone() for _ in range(3))
    if kind == "mixed":
        valid[-1, 2:] = False
        valid[0, :1] = False
        ce[:, 2] = False
        latent[0] = False
        kl[-1] = False
        kl[0, ::2] = False
    elif kind == "different":
        valid[0] = False
        valid[-1, -1] = False
        latent[:, ::2] = False
        kl[:, :3] = False
        ce[-1, ::2] = False
    elif kind == "empty":
        valid[:] = False
    elif kind == "targets_empty":
        ce[:] = False
        latent[:] = False
        kl[:] = False
    elif kind == "ce_only":
        latent[:] = False
        kl[:] = False
    elif kind != "full":
        raise ValueError(kind)
    docs.masked_fill_(~valid, -1)
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def leaves(config, batch, *, tied=True):
    generator = torch.Generator().manual_seed(921)
    hidden = (torch.randn(*batch.input_ids.shape, config.model_dim, generator=generator)*.2).requires_grad_()
    readout = (torch.randn(23, config.model_dim, generator=generator)*.1).requires_grad_()
    predictor = NextLatPredictor(config)
    embeddings = (F.embedding(batch.input_ids, readout) if tied else
                  (torch.randn(*hidden.shape, generator=generator)*.1).requires_grad_())
    return hidden, embeddings, readout, predictor


def compare_gradients(actual_loss, expected_loss, actual_parameters, expected_parameters):
    actual = torch.autograd.grad(actual_loss, actual_parameters, allow_unused=True)
    expected = torch.autograd.grad(expected_loss, expected_parameters, allow_unused=True)
    for index, (a, e) in enumerate(zip(actual, expected)):
        if e is None:
            # Stable dense execution intentionally keeps locally inactive
            # parameters attached with zero gradients for DDP participation.
            assert a is None or torch.count_nonzero(a) == 0, index
        else:
            assert a is not None, index
            torch.testing.assert_close(a, e, atol=3e-7, rtol=4e-5, msg=f"gradient {index}")


@pytest.mark.parametrize("enabled,latent,kl", [(True,1.,1.), (True,.3,.7),
    (True,0.,.7), (True,.3,0.), (False,1.,1.)])
@pytest.mark.parametrize("kind", ["full", "mixed", "different", "empty", "targets_empty", "ce_only"])
def test_values_and_raw_tied_parameter_gradients_match_canonical(enabled, latent, kl, kind):
    config = NextLatConfig(64, lambda_latent=latent, lambda_kl=kl, vocab_chunk_size=4,
                           ce_chunk_size=5)
    batch = make_batch(kind)
    layout = DynamicNextLatLayout.from_batch(batch, config, enabled=enabled)
    h, embeds, w, predictor = leaves(config, batch)
    eh, eembeds, ew, epredictor = leaves(config, batch)
    result = compute_dynamic_nextlat_loss_sums(h, embeds, w, batch.input_ids,
        predictor, config, layout, enabled=enabled)
    expected = compute_nextlat_loss_sums(eh, eembeds, ew, batch, epredictor, config, enabled=enabled)
    assert layout.counts == expected.counts
    assert layout.weights == expected.weights
    for term in TERMS:
        torch.testing.assert_close(result[term], expected.sums[term], atol=8e-7, rtol=3e-6)
    objective = sum(result[t]*layout.weights[t]/max(layout.counts[t], 1) for t in TERMS)
    compare_gradients(objective, expected.total, (h,w,*predictor.parameters()),
                      (eh,ew,*epredictor.parameters()))


def test_multiple_changed_rows_masks_and_counts_keep_owned_buffer_addresses():
    config = NextLatConfig(64, vocab_chunk_size=3)
    layout = DynamicNextLatLayout.from_batch(make_batch("full"), config)
    original = {name: value.data_ptr() for name, value in vars(layout).items() if isinstance(value, torch.Tensor)}
    for kind in ("mixed", "empty", "different", "targets_empty", "full"):
        batch = make_batch(kind)
        layout.load_batch(batch)
        layout.validate_integrity()
        h, embeddings, readout, predictor = leaves(config, batch)
        eh, eembeddings, ereadout, epredictor = leaves(config, batch)
        actual = compute_dynamic_nextlat_loss_sums(h, embeddings, readout, batch.input_ids,
                                                  predictor, config, layout)
        expected = compute_nextlat_loss_sums(eh, eembeddings, ereadout, batch, epredictor, config)
        assert layout.counts == expected.counts
        assert {name: getattr(layout, name).data_ptr() for name in original} == original
        for term in TERMS:
            torch.testing.assert_close(actual[term], expected.sums[term], atol=8e-7, rtol=3e-6)
        coefficients = torch.tensor([1/17, .3/11, .7/7])
        compare_gradients(sum(actual[t]*coefficients[i] for i,t in enumerate(TERMS)),
            sum(expected.sums[t]*coefficients[i] for i,t in enumerate(TERMS)),
            (h,readout,*predictor.parameters()), (eh,ereadout,*epredictor.parameters()))


@pytest.mark.parametrize("term", ["latent", "kl"])
def test_auxiliary_target_and_readout_detachments_are_preserved(term):
    config = NextLatConfig(64, vocab_chunk_size=3)
    batch = make_batch("full", rows=1)
    # One target: no overlap between predicting input and detached target state.
    masks = {name: torch.zeros_like(batch.valid_mask) for name in ("ce_mask", "latent_mask", "kl_mask")}
    masks[term+"_mask"][0, 3] = True
    batch = replace(batch, **masks)
    layout = DynamicNextLatLayout.from_batch(batch, config)
    hidden, embeddings, weight, predictor = leaves(config, batch, tied=False)
    result = compute_dynamic_nextlat_loss_sums(hidden, embeddings, weight, batch.input_ids,
                                              predictor, config, layout)
    gh, ge, gw = torch.autograd.grad(result[term], (hidden, embeddings, weight), allow_unused=True)
    assert gw is None  # KL projects through the readout as a detached constant.
    source = 2 if term == "latent" else 1
    assert gh[0, source].abs().sum() > 0
    assert ge[0, source+1].abs().sum() > 0
    assert torch.count_nonzero(gh[0, source+1:]) == 0
    assert torch.count_nonzero(ge[0, source+2:]) == 0


def test_empty_local_slots_have_finite_zero_gradients_for_all_active_parameters():
    config = NextLatConfig(64, vocab_chunk_size=4)
    batch = make_batch("empty")
    layout = DynamicNextLatLayout.from_batch(batch, config)
    h, embeddings, w, predictor = leaves(config, batch)
    result = compute_dynamic_nextlat_loss_sums(h, embeddings, w, batch.input_ids,
                                              predictor, config, layout)
    grads = torch.autograd.grad(sum(result.values()), (h,w,*predictor.parameters()))
    assert layout.counts == {t: 0 for t in TERMS}
    assert all(value.item() == 0 for value in result.values())
    assert all(torch.isfinite(g).all() and not torch.count_nonzero(g) for g in grads)


def test_unequal_microbatch_raw_sums_with_global_device_coefficients_match_whole_batch():
    config = NextLatConfig(64, lambda_latent=.3, lambda_kl=.7, vocab_chunk_size=4)
    batch = make_batch("mixed", rows=4)
    h, embeddings, w, predictor = leaves(config, batch)
    eh, eembeddings, ew, epredictor = leaves(config, batch)
    reference = compute_nextlat_loss_sums(eh, eembeddings, ew, batch, epredictor, config)
    coefficients = torch.tensor([reference.weights[t]/reference.counts[t] for t in TERMS])
    total = 0
    # Rows have unequal selected CE/latent/KL counts. No per-microbatch average.
    for part in (slice(0, 1), slice(1, 4)):
        small = NextLatBatch(**{name: value[part] for name,value in vars(batch).items()})
        layout = DynamicNextLatLayout.from_batch(small, config)
        sums = compute_dynamic_nextlat_loss_sums(h[part], embeddings[part], w, small.input_ids,
                                                predictor, config, layout)
        total = total + sum(sums[t]*coefficients[i] for i,t in enumerate(TERMS))
    torch.testing.assert_close(total, reference.total, atol=8e-7, rtol=3e-6)
    compare_gradients(total, reference.total, (h,w,*predictor.parameters()), (eh,ew,*epredictor.parameters()))


class TensorExecutionTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.matrix_shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("Tensor-only execution read a tensor scalar on the host")
        if func is torch.ops.aten.nonzero.default:
            raise AssertionError("Tensor-only execution made a data-dependent selection")
        if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
            self.matrix_shapes.append((str(func), tuple(tuple(x.shape) for x in args if isinstance(x, torch.Tensor))))
        return func(*args, **(kwargs or {}))


def test_tensor_only_execution_has_constant_dense_work_for_changing_and_empty_masks():
    config = NextLatConfig(64, vocab_chunk_size=4)
    layout = DynamicNextLatLayout.from_batch(make_batch("full"), config)
    observations = []
    for kind in ("full", "mixed", "empty", "ce_only"):
        batch = make_batch(kind)
        layout.load_batch(batch)
        h, embeddings, w, predictor = leaves(config, batch)
        with TensorExecutionTrace() as trace:
            sums = compute_dynamic_nextlat_loss_sums(h, embeddings, w, batch.input_ids,
                                                    predictor, config, layout)
            sum(sums.values()).backward()
        observations.append(trace.matrix_shapes)
    assert observations[0]
    assert all(observation == observations[0] for observation in observations[1:])


@pytest.mark.parametrize("buffer", ["pair_source_indices", "ce_weights", "latent_weights", "kl_weights"])
def test_external_inplace_mutations_reject_before_loading_another_batch(buffer):
    batch = make_batch()
    layout = DynamicNextLatLayout.from_batch(batch, NextLatConfig(64))
    getattr(layout, buffer).zero_()
    with pytest.raises(ValueError, match="modified outside"):
        layout.load_batch(make_batch("full"))


def test_replaced_buffer_and_bad_proposed_batch_do_not_silently_change_layout():
    batch = make_batch()
    layout = DynamicNextLatLayout.from_batch(batch, NextLatConfig(64))
    before = layout.ce_weights.clone(), layout.counts
    bad = replace(batch, document_ids=batch.document_ids.clone())
    bad.document_ids[1, 3] = 111
    with pytest.raises(ValueError, match="Packed"):
        layout.load_batch(bad)
    assert torch.equal(layout.ce_weights, before[0])
    assert layout.counts == before[1]
    object.__setattr__(layout, "ce_weights", layout.ce_weights.clone())
    with pytest.raises(ValueError, match="modified outside"):
        layout.validate_integrity()


@pytest.mark.parametrize("length", [1, 2])
def test_short_context_scope_is_explicit(length):
    ids = torch.zeros((1,length), dtype=torch.long)
    batch = NextLatBatch(ids, torch.ones_like(ids,dtype=torch.bool), torch.zeros_like(ids))
    with pytest.raises(ValueError, match="length >= 3"):
        DynamicNextLatLayout.from_batch(batch, NextLatConfig(64))


def test_dropout_and_changed_objective_reject():
    batch = make_batch()
    with pytest.raises(ValueError, match="zero predictor dropout"):
        DynamicNextLatLayout.from_batch(batch, NextLatConfig(64, dropout=.1))
    config = NextLatConfig(64)
    layout = DynamicNextLatLayout.from_batch(batch, config)
    with pytest.raises(ValueError, match="configuration"):
        layout.validate_execution(batch.input_ids, replace(config, lambda_kl=.2), enabled=True)


def test_zero_weight_objectives_skip_their_dense_auxiliary_branch():
    config = NextLatConfig(64, lambda_latent=0., lambda_kl=0.)
    batch = make_batch()
    layout = DynamicNextLatLayout.from_batch(batch, config)
    hidden, embeddings, readout, _ = leaves(config, batch)
    result = compute_dynamic_nextlat_loss_sums(hidden, embeddings, readout,
        batch.input_ids, None, config, layout)
    assert layout.counts["latent"] == layout.counts["kl"] == 0
    assert result["latent"].item() == result["kl"].item() == 0


def test_unselected_out_of_range_ce_targets_are_not_interpreted_as_labels():
    config = NextLatConfig(64)
    batch = make_batch("empty")
    layout = DynamicNextLatLayout.from_batch(batch, config)
    hidden, embeddings, readout, predictor = leaves(config, batch)
    ids = torch.full_like(batch.input_ids, -999)
    result = compute_dynamic_nextlat_loss_sums(hidden, embeddings, readout, ids,
                                              predictor, config, layout)
    assert result["ce"].item() == 0
