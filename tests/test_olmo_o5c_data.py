"""Exact CE quotas, fresh document exclusion, conservative segmentation and cursors."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import olmo_o5c_data as data


def windows(lengths, first_doc=0):
    rows, offset = [], 0
    for document, count in enumerate(lengths, first_doc):
        rows.append([offset, count, document, 0]); offset += count
    return np.asarray(rows, dtype="<i8")


def targets(segments):
    return [(int(domain), int(window), int(start + token)) for domain, window, start, length, _ in segments
            for token in range(1, int(length))]


def test_exact_mixed_ce_exposure_with_different_row_counts_and_context_overlap():
    code, general = windows([6] * 30), windows([17] * 10, 100)
    pure, cp = data.build_plan(code, general, arm="code", start_code_window=2, updates=3, ce_per_update=12)
    mixed, mp = data.build_plan(code, general, arm="mixed", start_code_window=2, updates=3, ce_per_update=12)
    assert cp["batch_ce_prefix"] == mp["batch_ce_prefix"] == [0, 12, 24, 36]
    assert mp["domain_prefixes"]["code"]["ce_positions"] == [0, 6, 12, 18]
    assert mp["domain_prefixes"]["general"]["ce_positions"] == [0, 6, 12, 18]
    assert cp["batch_token_prefix"][-1] == 36 + len(pure)
    assert mp["batch_token_prefix"][-1] == 36 + len(mixed)
    assert len(set(targets(pure))) == len(targets(pure)) == 36
    assert len(set(targets(mixed))) == len(targets(mixed)) == 36
    assert [target for target in targets(mixed) if target[0] == 0] == targets(pure)[:18]
    mixed_code = mixed[mixed[:, 0] == 0]
    # The shared targets must also receive identical contexts, regardless of
    # whether two code chunks or code+general comprise an optimizer update.
    assert np.array_equal(mixed_code, pure[:len(mixed_code)])
    assert cp["segment_quota_ce"] == mp["segment_quota_ce"] == 6
    assert pure[:, 1].min() >= 2
    # The same source window can continue next update with only its context repeated.
    assert any(row[2] > 0 for row in pure)


@pytest.mark.parametrize("kwargs", [{"arm": "other"}, {"updates": 0}, {"ce_per_update": 3}, {"start_code_window": -1}])
def test_plan_rejects_invalid_recipe(kwargs):
    recipe = dict(arm="code", updates=3, ce_per_update=12, start_code_window=0)
    recipe.update(kwargs)
    with pytest.raises(ValueError): data.build_plan(windows([6] * 30), windows([17] * 10), **recipe)


def test_plan_refuses_source_exhaustion_instead_of_cycling():
    with pytest.raises(ValueError, match="Insufficient fresh"):
        data.build_plan(windows([4]), windows([8]), arm="code", start_code_window=0, updates=2, ce_per_update=4)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, models, pre_tokenizers
    lm = data.lm
    monkeypatch.setattr(lm, "EOS_ID", 3)
    monkeypatch.setattr(lm, "PAD_ID", 1)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "=": 2, "<|endoftext|>": 3,
                                            "alpha": 4, "beta": 5, "gamma": 6, "delta": 7,
                                            "Held": 8, "Fresh": 9}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer_path = tmp_path / "tokenizer.json"; tokenizer.save(str(tokenizer_path))
    token_hash = data.sha256_file(tokenizer_path)
    monkeypatch.setattr(lm, "FILE_SPECS", {"tokenizer.json": (tokenizer_path.stat().st_size, token_hash)})
    held = "= Held =\nalpha beta gamma\n"
    held_tokens = np.asarray([*tokenizer.encode(held, add_special_tokens=False).ids, 3], dtype="<u4")
    base_root = tmp_path / "base"; base_root.mkdir()
    table = windows([6] * 30)
    code_tokens = np.concatenate([np.asarray([10 + i, 4, 5, 6, 7, 3], dtype="<u4") for i in range(30)])
    train_docs = [{"document_id": i, "text_sha256": hashlib.sha256(f"code {i}".encode()).hexdigest(),
                   "token_sha256": hashlib.sha256(code_tokens[i * 6:(i + 1) * 6].tobytes()).hexdigest()} for i in range(30)]
    for split in lm.SPLITS:
        rows = train_docs if split == "train" else ([{"document_id": 30, "text_sha256": lm._text_hash(held),
                "token_sha256": hashlib.sha256(held_tokens.tobytes()).hexdigest()}] if split == "retention_dev" else [])
        (base_root / f"{split}.documents.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    base = SimpleNamespace(root=base_root, manifest_sha256="a" * 64, split_sizes={"train": len(table)},
                           _windows={"train": table}, _tokens={"train": code_tokens},
                           manifest={"files": {}, "tokenizer": {"sha256": token_hash},
                                     "preprocessing": {"wikitext": "original article rows"}, "licenses": {"wikitext": "test license"}})
    def load_base(root, expected_manifest_sha256=None):
        assert Path(root) == base_root
        if expected_manifest_sha256 is not None and expected_manifest_sha256 != base.manifest_sha256:
            raise ValueError("Base manifest differs")
        return base
    monkeypatch.setattr(lm, "load_lm_data", load_base)
    raw = tmp_path / "train.parquet"
    pq.write_table(pa.table({"text": ["= Held =\n", "alpha beta gamma\n", "= Fresh =\n", "alpha beta gamma delta " * 100,
                                             "= AliasA =\n", "alpha beta\n", "= AliasB =\n", "alpha beta\n"]}), raw)
    monkeypatch.setattr(data, "GENERAL_SPEC", (lm.WIKI_REPO, lm.WIKI_REVISION, "train.parquet", raw.stat().st_size, data.sha256_file(raw)))
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"schema": "olmo-o5b-arm-v1", "arm": "fbt", "status": "completed", "finished_utc": "2026-09-22",
                                  "configuration": {"data_manifest_sha256": base.manifest_sha256}, "data_cursor": 1,
                                  "counters": {"documents": 1, "input_tokens": 6, "ce_positions": 5}}))
    output = tmp_path / "prepared"
    args = dict(base_root=base_root, raw_general=raw, tokenizer_path=tokenizer_path, previous_report=parent,
                output_dir=output, updates=3, ce_per_update=12)
    manifest = data.prepare_data(**args)
    return SimpleNamespace(args=args, output=output, base=base, manifest=manifest)


def test_prepare_excludes_heldout_and_token_duplicates_without_evaluating_them(prepared):
    stats = prepared.manifest["general_statistics"]
    assert stats["source_documents"] == 4
    assert stats["byte_hash_duplicates_skipped"] == 1
    assert stats["token_hash_duplicates_skipped"] == 1
    assert stats["documents"] == 2
    assert prepared.manifest["test_evaluated"] is False
    assert prepared.manifest["fresh_code"]["start_window"] == 1


def test_load_batches_conserve_targets_masks_ids_and_resume_cursor(prepared):
    corpus = data.load_o5c_data(prepared.output, prepared.base.root,
                               expected_manifest_sha256=data.sha256_file(prepared.output / "manifest.json"))
    for arm in data.ARMS:
        cursor = 0
        for update in range(3):
            batch, cursor = corpus.next_train_batch(arm, cursor)
            assert batch.input_ids.shape[1] == 512
            valid = batch.valid_mask
            assert int((valid[:, 1:] & valid[:, :-1]).sum()) == 12
            assert batch.ce_mask is None and batch.latent_mask is None and batch.kl_mask is None
            assert bool((batch.document_ids[~valid] == -1).all())
            assert bool((batch.input_ids[~valid] == 1).all())
            assert all(len(set(row[mask].tolist())) == 1 for row, mask in zip(batch.document_ids, valid))
            repeated = corpus.batch_for_update(arm, update)
            assert bool(batch.input_ids.eq(repeated.input_ids).all())
        assert cursor == 3
        with pytest.raises(StopIteration): corpus.next_train_batch(arm, cursor)


def test_quota_boundary_does_not_invent_eos(prepared):
    corpus = data.load_o5c_data(prepared.output, prepared.base.root)
    found_crop = False
    for arm in data.ARMS:
        for index, segment in enumerate(corpus.segments[arm]):
            domain, window, start, length, _ = segment
            source = corpus.base._windows["train"] if domain == 0 else corpus.general_windows
            if start + length < source[window, 1]:
                batch = corpus.batch(arm, [index])
                assert int(batch.input_ids[0, length - 1]) != 3
                found_crop = True
    assert found_crop


def rewrite_manifest(prepared, change):
    path = prepared.output / "manifest.json"
    manifest = json.loads(path.read_text()); change(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("case", ["source", "raw_pin", "base", "inventory", "quota", "counter", "file_bytes", "symlink"])
def test_load_rejects_changed_provenance_or_schedule(prepared, case):
    if case == "source": rewrite_manifest(prepared, lambda m: m["source_hashes"].update({"scripts/olmo_o5c_data.py": "f" * 64}))
    elif case == "raw_pin": rewrite_manifest(prepared, lambda m: m["general_raw_source"].update(sha256="f" * 64))
    elif case == "base": rewrite_manifest(prepared, lambda m: m.update(base_manifest_sha256="f" * 64))
    elif case == "inventory": rewrite_manifest(prepared, lambda m: m["files"].pop("code.plan.npy"))
    elif case == "quota": rewrite_manifest(prepared, lambda m: m["plans"]["mixed"]["domain_prefixes"]["general"]["ce_positions"].__setitem__(1, 5))
    elif case == "counter": rewrite_manifest(prepared, lambda m: m["parent"].update(consumed_windows=0))
    elif case == "file_bytes":
        with (prepared.output / "general.tokens.bin").open("ab") as stream: stream.write(b"abcd")
    elif case == "symlink":
        path = prepared.output / "code.plan.npy"; moved = prepared.output / "moved.npy"
        path.rename(moved); path.symlink_to(moved)
    with pytest.raises(ValueError): data.load_o5c_data(prepared.output, prepared.base.root)


def test_loader_rejects_a_resigned_repeating_segment(prepared):
    path = prepared.output / "code.plan.npy"
    rows = np.load(path); rows[1] = rows[0]; np.save(path, rows, allow_pickle=False)
    rewrite_manifest(prepared, lambda m: m["files"][path.name].update(sha256=data.sha256_file(path), size_bytes=path.stat().st_size))
    with pytest.raises(ValueError, match="segmentation"):
        data.load_o5c_data(prepared.output, prepared.base.root)


def test_prepare_will_not_overwrite_existing_output(prepared):
    with pytest.raises(ValueError, match="already exists"):
        data.prepare_data(**prepared.args)


def test_new_data_must_start_after_a_complete_parent_document(prepared):
    args = {**prepared.args, "output_dir": prepared.output.parent / "second"}
    prepared.base._windows["train"][1, 2] = 0
    with pytest.raises(ValueError, match="previously consumed"):
        data.prepare_data(**args)
