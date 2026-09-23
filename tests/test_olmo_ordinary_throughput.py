"""CPU correctness of throughput fixtures and position-chunked full-vocabulary CE."""
import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.nextlat import NextLatConfig, _ce_chunk, _chunked_sum
from cdrm.pretrained.static_nextlat import PreparedNextLatLayout
from scripts.olmo_ordinary_throughput import make_batch, make_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("supervision,expected", [("half", 256), ("full", 511)])
def test_real_length_fixture_counts_target_positions_without_double_counting(supervision, expected):
    batch = make_batch(3, 512, seed=137, supervision=supervision, vocab_size=67)
    layout = PreparedNextLatLayout.from_batch(batch, NextLatConfig(model_dim=32), enabled=False)
    assert layout.counts == {"ce": 3*expected, "latent": 0, "kl": 0}
    assert layout.needed_source_indices.numel() == 0
    assert batch.input_ids.shape == (3, 512)
    assert batch.input_ids.dtype == torch.long
    assert bool(batch.valid_mask.all())
    assert 0 <= int(batch.input_ids.min()) <= int(batch.input_ids.max()) < 67
    # Attention rows are independent full documents, including random special IDs.
    assert all(torch.unique(batch.document_ids[row]).numel() == 1 for row in range(3))
    assert torch.unique(batch.document_ids[:, 0]).numel() == 3


@pytest.mark.parametrize("length", [2, 8, 32])
def test_full_and_half_fixtures_differ_only_in_supervision_not_data(length):
    half = make_batch(2, length, seed=41, supervision="half", vocab_size=67)
    full = make_batch(2, length, seed=41, supervision="full", vocab_size=67)
    assert torch.equal(half.input_ids, full.input_ids)
    assert torch.equal(half.valid_mask, full.valid_mask)
    assert torch.equal(half.document_ids, full.document_ids)
    counts = [PreparedNextLatLayout.from_batch(batch, NextLatConfig(model_dim=32), enabled=False).counts["ce"]
              for batch in (half, full)]
    assert counts == [2*(length//2), 2*(length-1)]


def test_random_batch_is_reproducible_and_does_not_consume_model_initialization_rng():
    torch.manual_seed(19)
    before = torch.random.get_rng_state().clone()
    first = make_batch(2, 32, seed=7, supervision="full", vocab_size=67)
    same = make_batch(2, 32, seed=7, supervision="full", vocab_size=67)
    changed = make_batch(2, 32, seed=8, supervision="full", vocab_size=67)
    assert torch.equal(first.input_ids, same.input_ids)
    assert not torch.equal(first.input_ids, changed.input_ids)
    assert torch.equal(before, torch.random.get_rng_state())


def test_requested_six_layer_geometry_and_reference_override_preserve_native_width():
    config = make_config()
    assert (config.num_layers, config.model_dim, config.num_heads, config.head_dim,
            config.mlp_intermediate_size, config.vocab_size) == (6, 2048, 32, 64, 8192, 50304)
    reference = make_config(layers=16, heads=16, mlp=8192)
    assert (reference.num_layers, reference.model_dim, reference.num_heads, reference.head_dim) == (16, 2048, 16, 128)
    assert reference.rope_freq_constant == config.rope_freq_constant == 10000.


@pytest.mark.parametrize("chunk", [128, 2048])
def test_checkpointed_position_chunks_match_dense_full_vocabulary_ce_and_both_gradients(chunk):
    generator = torch.Generator().manual_seed(361)
    states = torch.randn(512, 32, generator=generator, dtype=torch.float32).requires_grad_()
    readout = (torch.randn(67, 32, generator=generator, dtype=torch.float32)*.15).requires_grad_()
    # Include the highest vocabulary rows to catch accidental vocabulary truncation.
    targets = torch.arange(512, dtype=torch.long).remainder(67)
    dense_states = states.detach().clone().requires_grad_()
    dense_readout = readout.detach().clone().requires_grad_()
    actual = _chunked_sum(_ce_chunk, states, readout, targets, chunk, weight_second=True)
    expected = F.cross_entropy(dense_states @ dense_readout.T, targets, reduction="sum")
    actual_gradients = torch.autograd.grad(actual/512, (states, readout))
    expected_gradients = torch.autograd.grad(expected/512, (dense_states, dense_readout))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-4)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=2e-7)
        assert bool(torch.isfinite(actual_gradient).all())
        assert int(torch.count_nonzero(actual_gradient)) > 0
