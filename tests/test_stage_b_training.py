"""CPU checks of answer-loss weighting, schedule boundaries, and resume identity."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cdrm.synthetic.common import SyntheticBatch

_path = Path(__file__).resolve().parents[1] / "scripts/stage_b_train.py"
_spec = importlib.util.spec_from_file_location("stage_b_train", _path)
trainer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer)


def test_answer_alignment_scores_terminal_marker_without_second_shift():
    logits = torch.tensor([[[0.0, 8.0, 0.0], [0.0, 0.0, 9.0]]], requires_grad=True)
    labels = torch.tensor([[-100, 2]])
    loss, count = trainer.aligned_ce_sum(logits, labels)
    assert count == 1 and loss.item() < 0.001
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, 1]) == 3
    single, count = trainer.aligned_ce_sum(logits[:, :1], torch.tensor([[1]]))
    assert count == 1 and single.item() < 0.001


def test_microbatch_sum_weighting_matches_unequal_answer_counts_and_empty_chunk():
    torch.manual_seed(17)
    initial = torch.randn(4, 5, 7, dtype=torch.float64)
    labels = torch.tensor([[-100, -100, -100, -100, 1],
                           [0, 2, 4, -100, 6],
                           [-100, -100, -100, -100, -100],
                           [3, -100, 1, -100, -100]])
    whole, chunks = initial.clone().requires_grad_(), initial.clone().requires_grad_()
    full_loss, total = trainer.aligned_ce_sum(whole, labels)
    (full_loss / total).backward()
    accumulated_loss = 0.0
    for row in range(4):
        summed, count = trainer.aligned_ce_sum(chunks[row:row + 1], labels[row:row + 1])
        (summed / total).backward()
        accumulated_loss += float(summed.detach())
    assert total == 7
    assert accumulated_loss == pytest.approx(float(full_loss.detach()), abs=1e-12)
    torch.testing.assert_close(whole.grad, chunks.grad, atol=1e-14, rtol=1e-14)


def test_schedule_restarts_from_completed_count_and_keeps_full_horizon():
    settings = {"updates": 2000, "warmup_updates": 100,
                "learning_rate": 1e-3, "final_lr_multiplier": 0.1}
    uninterrupted = [trainer.learning_rate_at_update(update, settings) for update in range(2000)]
    interrupted = [trainer.learning_rate_at_update(update, settings) for update in range(137)]
    interrupted += [trainer.learning_rate_at_update(update, settings) for update in range(137, 2000)]
    assert interrupted == uninterrupted
    assert uninterrupted[0] == pytest.approx(1e-5)
    assert uninterrupted[99] == pytest.approx(1e-3)
    assert uninterrupted[-1] == pytest.approx(1e-4)
    assert all(first <= second for first, second in zip(uninterrupted[:99], uninterrupted[1:100]))
    assert all(first >= second for first, second in zip(uninterrupted[99:-1], uninterrupted[100:]))
    assert uninterrupted[199] > 0.0009  # A 200-update stop is not a 200-update cosine run.


def test_resume_checks_plan_sources_model_data_and_offset():
    identity = {"plan": {"updates": 2000}, "source_sha256": {"file.py": "abc"}}
    model = {"recurrent_layers": [3], "vocab_size": 26}
    payload = {"format": trainer.FORMAT, "identity": identity,
               "identity_sha256": trainer.json_digest(identity), "model_config": model,
               "next_data_sha256": "next", "data_offset_examples": 128,
               "training_counters": {"examples": 128}}
    trainer.validate_resume(payload, identity, model, "next")
    for key, replacement in (("identity_sha256", "wrong"), ("next_data_sha256", "other"),
                             ("model_config", {"recurrent_layers": []}),
                             ("data_offset_examples", 0)):
        changed = copy.deepcopy(payload)
        changed[key] = replacement
        with pytest.raises(ValueError):
            trainer.validate_resume(changed, identity, model, "next")
    changed_identity = copy.deepcopy(identity)
    changed_identity["source_sha256"]["file.py"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        trainer.validate_resume(payload, changed_identity, model, "next")


def test_frozen_training_stream_integrity(tmp_path):
    inputs = np.arange(24, dtype=np.int64).reshape(4, 6)
    labels = np.full_like(inputs, -100)
    labels[:, -1] = np.arange(4)
    path = tmp_path / "training.npz"
    np.savez_compressed(path, input_ids=inputs, labels=labels)
    stream = hashlib.sha256()
    for begin in (0, 2):
        stream.update(SyntheticBatch(inputs[begin:begin + 2], labels[begin:begin + 2], [{}, {}]).sha256.encode())
    manifest = {"training": {"path": str(path), "global_batch": 2, "examples": 4,
        "sha256_file": trainer.file_digest(path), "training_stream_sha256": stream.hexdigest()},
        "audit": {"status": "passed", "training_stream_sha256": stream.hexdigest()}}
    actual = trainer.load_training_arrays(manifest, {"global_batch": 2, "updates": 2})
    np.testing.assert_array_equal(actual[0], inputs)
    np.testing.assert_array_equal(actual[1], labels)
    corrupted = copy.deepcopy(manifest)
    corrupted["training"]["sha256_file"] = "invalid"
    with pytest.raises(ValueError, match="SHA-256"):
        trainer.load_training_arrays(corrupted, {"global_batch": 2, "updates": 2})
    corrupted = copy.deepcopy(manifest)
    corrupted["audit"]["training_stream_sha256"] = "invalid"
    with pytest.raises(ValueError, match="split audit"):
        trainer.load_training_arrays(corrupted, {"global_batch": 2, "updates": 2})


def test_retained_json_is_not_overwritten(tmp_path):
    path = tmp_path / "result.json"
    trainer.atomic_json(path, {"value": "original"})
    with pytest.raises(FileExistsError):
        trainer.atomic_json(path, {"value": "changed"})
    assert "original" in path.read_text()


def test_r3_initial_resume_fails_closed_without_blocking_validated_boundaries():
    with pytest.raises(ValueError, match="cold initialization resume path"):
        trainer.validate_resume_boundary("r3", 0)
    trainer.validate_resume_boundary("r3", 2)
    trainer.validate_resume_boundary("seq", 0)
