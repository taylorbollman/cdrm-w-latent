"""CPU checks for head-only chunking, shifted supervision and full-backbone credit."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rt_batch_profile import head_chunked_backward


class TinyBackbone(torch.nn.Module):
    def __init__(self, *, scale_logits=False):
        super().__init__()
        self.config = SimpleNamespace(scale_logits=scale_logits, d_model=7)
        self.transformer = torch.nn.ModuleDict({
            'wte': torch.nn.Embedding(13, 7),
            'backbone': torch.nn.Linear(7, 7),
            'ff_out': torch.nn.Linear(7, 13, bias=False),
        })
        self.forward_batches = []
        self.hidden_backward_calls = 0

    def forward(self, ids, *, return_pre_logits, return_logits):
        assert return_pre_logits and not return_logits
        self.forward_batches.append(len(ids))
        features = torch.tanh(self.transformer.backbone(self.transformer.wte(ids)))
        # Credit through a causal computation before the head boundary.
        hidden = features.cumsum(dim=1) / torch.arange(1, ids.shape[1] + 1)[None, :, None]
        hidden.retain_grad()
        hidden.register_hook(self._count_hidden_backward)
        self.hidden = hidden
        return SimpleNamespace(pre_logits=hidden)

    def _count_hidden_backward(self, gradient):
        self.hidden_backward_calls += 1
        return gradient


@pytest.mark.parametrize('batch,chunk,scale_logits', [(1, 2, False), (4, 2, False), (5, 2, True)])
def test_chunked_head_matches_full_shifted_loss_and_every_parameter_gradient(batch, chunk, scale_logits):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(702)
        reference = TinyBackbone(scale_logits=scale_logits)
        candidate = copy.deepcopy(reference)
        ids = torch.randint(0, 13, (batch, 6))

    hidden = reference(ids, return_pre_logits=True, return_logits=False).pre_logits
    logits = reference.transformer.ff_out(hidden)
    if scale_logits:
        logits = logits * reference.config.d_model ** -0.5
    target = ids[:, 1:].reshape(-1)
    summed = F.cross_entropy(logits[:, :-1].reshape(-1, 13), target, reduction='sum')
    expected = summed / ids.numel()
    expected.backward()

    head_batches = []
    handle = candidate.transformer.ff_out.register_forward_pre_hook(
        lambda module, args: head_batches.append(len(args[0])))
    try:
        actual = head_chunked_backward(candidate, ids, chunk_size=chunk, bf16=False)
    finally:
        handle.remove()

    torch.testing.assert_close(actual, expected.detach(), rtol=2e-6, atol=2e-7)
    assert not actual.requires_grad
    assert candidate.forward_batches == [batch]
    assert candidate.hidden_backward_calls == 1
    assert head_batches == [min(chunk, batch - start) for start in range(0, batch, chunk)]
    torch.testing.assert_close(candidate.hidden.grad, reference.hidden.grad, rtol=2e-6, atol=2e-8)
    assert torch.count_nonzero(candidate.hidden.grad[:, -1]).item() == 0
    for name, parameter in candidate.named_parameters():
        expected_gradient = dict(reference.named_parameters())[name].grad
        assert parameter.grad is not None and expected_gradient is not None, name
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad), name
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=3e-6, atol=3e-8, msg=name)


def test_invalid_head_chunk_or_unshiftable_input_fails_before_model_execution():
    model = TinyBackbone()
    for ids, chunk in [(torch.zeros(2, 3, dtype=torch.long), 0),
                       (torch.zeros(2, 1, dtype=torch.long), 2),
                       (torch.zeros(3, dtype=torch.long), 2)]:
        with pytest.raises(ValueError, match='positive head chunks'):
            head_chunked_backward(model, ids, chunk_size=chunk, bf16=False)
    assert model.forward_batches == []
