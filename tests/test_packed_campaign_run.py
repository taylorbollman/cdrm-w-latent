"""Real shard/index cursor contracts used by the two-GPU readiness runner."""
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained import document_shards
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.packed_campaign_data import PackedCampaignData, build_packed_index
from scripts.olmo_packed_campaign_run import (ROOT, cursor_record, parse_args,
                                             restore_training_cursor, token_plan)


@pytest.fixture
def actual_data(tmp_path, monkeypatch):
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return SimpleNamespace(ids=[100+b for b in text.encode()])
    monkeypatch.setattr(document_shards, "_load_tokenizer", lambda _: Tokenizer())
    raw = b"".join((json.dumps({"id": str(i), "text": str(i)+"x"*(i+8)})+"\n").encode()
                   for i in range(7))
    source_file = tmp_path/"source.jsonl"
    source_file.write_bytes(raw)
    source = LocalJSONLSource(SourcePin("fixture", "https://example.invalid/fixture", "pinned",
        hashlib.sha256(raw).hexdigest()), source_file)
    corpus, index = tmp_path/"corpus", tmp_path/"index"
    document_shards.prepare_document_shards([source], corpus, tokenizer_path="unused",
        split_policy=SplitPolicy(19, (("train", 1),)), max_documents_per_shard=2)
    build_packed_index(corpus, index, split="train", length=8)
    return corpus, index


def test_full_schedule_peek_and_validation_do_not_consume_reader(actual_data):
    with PackedCampaignData(*actual_data) as data:
        start = data.cursor()
        plan = token_plan(data, 24)
        assert data.cursor() == start
        assert sum(plan) == data.total_tokens
        assert all(v == 24 for v in plan[:-1])
        assert 0 < plan[-1] <= 24
        counters = TrainingCounters()
        cursor = restore_training_cursor(data, cursor_record(start, rank=0, batch_size=12),
            counters, plan, rank=0, batch_size=12)
        update = data.peek_update(cursor, 24)
        # Prefetch is irrelevant to the committed cursor and clocks.
        data.peek_update(update.next_cursor, 24)
        assert data.cursor() == start
        cursor = data.commit(cursor, update)
        counters.optimizer_updates, counters.input_tokens = 1, plan[0]
        assert restore_training_cursor(data, cursor_record(cursor, rank=0, batch_size=12),
            counters, plan, rank=0, batch_size=12) == cursor
        assert data.cursor() == cursor


def test_fresh_reader_restore_retains_next_batch_and_rejects_clock_or_partition_drift(actual_data):
    with PackedCampaignData(*actual_data) as original:
        plan = token_plan(original, 24)
        update = original.peek_update(original.cursor(), 24)
        cursor = original.commit(original.cursor(), update)
        record = cursor_record(cursor, rank=1, batch_size=12)
        expected = original.peek_update(cursor, 24)
    counters = TrainingCounters(optimizer_updates=1, input_tokens=plan[0])
    with PackedCampaignData(*actual_data) as fresh:
        actual = restore_training_cursor(fresh, record, counters, plan,
                                         rank=1, batch_size=12, install=True)
        assert fresh.peek_update(actual, 24) == expected
        for changed in (dict(record, rank=0), dict(record, physical_batch_per_rank=8),
                        dict(record, cursor=asdict(replace(cursor, next_chunk=cursor.next_chunk+1)))):
            with pytest.raises(ValueError):
                restore_training_cursor(fresh, changed, counters, plan, rank=1, batch_size=12)
        with pytest.raises(ValueError, match="clocks"):
            restore_training_cursor(fresh, record, replace(counters, input_tokens=plan[0]+1),
                                    plan, rank=1, batch_size=12)


def test_cli_requires_restore_pins_and_fixed_training_shape():
    common = ["--corpus", "corpus", "--index", "index", "--index-sha256", "a"*64,
              "--checkpoint-dir", "checkpoint", "--output-dir", str(ROOT/".runtime/packed-cli-test")]
    args = parse_args(common+["--phase", "write"])
    assert args.length == 1024 and args.batch_size == 12
    assert args.document_policy == "continuous-stream-v1"
    for extra in (["--phase", "resume"], ["--phase", "write", "--batch-size", "16"],
                  ["--phase", "write", "--reference-sha256", "b"*64]):
        with pytest.raises(SystemExit):
            parse_args(common+extra)
    resumed = parse_args(common+["--phase", "resume", "--reference-report", "reference.json",
        "--reference-sha256", "b"*64, "--expected-manifest-sha256", "c"*64])
    assert resumed.phase == "resume"
