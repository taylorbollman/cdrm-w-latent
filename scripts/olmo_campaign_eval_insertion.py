#!/usr/bin/env python3
"""Observe a no-jitter held-out forward between unchanged captured-DDP updates."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.distributed_checkpoint import _local_rng, _restore_local_rng
from cdrm.pretrained.document_shards import iter_documents, verify_document_shards
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks, _ce_chunk
from scripts import olmo_campaign_loop_guarded as guarded
from scripts.olmo_campaign_graph_probe import pointer_snapshot
from scripts.olmo_campaign_loop import LifecycleError
from scripts.olmo_lm_common import tree_digests

legacy = guarded.legacy
SCHEMA = "olmo-campaign-eval-insertion-v1"
FIXTURE_SCHEMA = "olmo-campaign-eval-fixture-v1"
PROTOCOL = ROOT / "docs/reports/olmo-campaign-lifecycle/eval-insertion-protocol.md"
CORPUS_SHA = "f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76"
SIZES = (16, 9)


def source_hashes():
    sources = guarded.source_hashes()
    for path in (Path(__file__), ROOT / "tests/test_campaign_eval_insertion.py", PROTOCOL):
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
    return dict(sorted(sources.items()))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fixture_from_documents(documents, *, corpus_sha256, vocab_size, eos_id, pad_id=1):
    """CPU selection from already verified disjoint document splits."""
    candidates = []
    for doc in documents:
        if doc.split == "dev" and len(doc.tokens) >= 16:
            identity = {"source": asdict(doc.source), "document_id": doc.document_id,
                        "split": doc.split, "text_sha256": doc.text_sha256}
            candidates.append((digest(identity), identity, doc.tokens))
    candidates.sort(key=lambda row: row[0])
    if len(candidates) < 2 or candidates[0][0] == candidates[1][0]:
        raise ValueError("Require two unique dev documents of at least16 tokens")
    records = []
    for rank, (key, identity, tokens) in enumerate(candidates[:2]):
        length = SIZES[rank]
        ids = [list(tokens[:length]) + [pad_id] * (16-length)]
        valid = [[True] * length + [False] * (16-length)]
        docs = [[rank] * length + [-1] * (16-length)]
        batch = {"input_ids": ids, "valid_mask": valid, "document_ids": docs,
                 "ce_mask": valid, "latent_mask": valid, "kl_mask": valid}
        records.append({"rank": rank, "document_key": key, "document": identity,
                        "full_tokens": len(tokens), "valid_tokens": length,
                        "slice_start": 0, "batch": batch, "batch_sha256": digest(batch)})
    return {"schema": FIXTURE_SCHEMA, "corpus_manifest_sha256": corpus_sha256,
            "vocab_size": vocab_size, "eos_id": eos_id, "pad_id": pad_id,
            "length": 16, "rows_per_rank": 1, "world_size": 2, "records": records,
            "input_tokens": 25, "ce_targets": 23,
            "scope": "Two distinct dev-document prefixes; right padding; zero intra-row document boundaries; no fabricated EOS"}


def prepare_fixture(corpus, output):
    corpus, output = Path(corpus), Path(output)
    if output.exists() or sha256_file(corpus/"manifest.json") != CORPUS_SHA:
        raise ValueError("Use fresh fixture destination and original pinned corpus")
    verified = verify_document_shards(corpus)
    if not verified["completed"]:
        raise ValueError("Only complete verified document shards are supported")
    config = json.loads((corpus/"config.json").read_text())
    fixture = fixture_from_documents(iter_documents(corpus, verify=False), corpus_sha256=CORPUS_SHA,
        vocab_size=config["vocab_size"], eos_id=config["eos_id"])
    if sha256_file(corpus/"manifest.json") != CORPUS_SHA:
        raise ValueError("Corpus changed during fixture preparation")
    write_json(output, fixture)
    load_fixture(output, sha256_file(output))
    return {"fixture": str(output), "sha256": sha256_file(output), "input_tokens": 25,
            "ce_targets": 23, "documents": [row["document"] for row in fixture["records"]]}


def load_fixture(path, expected_sha256):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 256*1024
            or sha256_file(path) != expected_sha256):
        raise ValueError("Evaluation fixture differs from its bounded SHA256 authority")
    fixture = json.loads(path.read_text())
    if (fixture.get("schema") != FIXTURE_SCHEMA or fixture.get("corpus_manifest_sha256") != CORPUS_SHA
            or fixture.get("world_size") != 2 or fixture.get("rows_per_rank") != 1
            or fixture.get("length") != 16 or fixture.get("input_tokens") != 25
            or fixture.get("ce_targets") != 23 or len(fixture.get("records", [])) != 2
            or type(fixture.get("vocab_size")) is not int or fixture["vocab_size"] < 3
            or type(fixture.get("pad_id")) is not int or not 0 <= fixture["pad_id"] < fixture["vocab_size"]
            or type(fixture.get("eos_id")) is not int or not 0 <= fixture["eos_id"] < fixture["vocab_size"]):
        raise ValueError("Evaluation fixture schema/counts/vocabulary differ")
    batches, keys = [], []
    fields = ("input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask")
    for rank, row in enumerate(fixture["records"]):
        doc, batch, size = row["document"], row["batch"], SIZES[rank]
        if (row.get("rank") != rank or row.get("valid_tokens") != size or row.get("slice_start") != 0
                or type(row.get("full_tokens")) is not int or row["full_tokens"] < 16
                or doc.get("split") != "dev" or row.get("document_key") != digest(doc)
                or row.get("batch_sha256") != digest(batch) or set(batch) != set(fields)):
            raise ValueError("Evaluation fixture identity or row layout differs")
        keys.append(row["document_key"])
        tensors = {}
        for name in fields:
            dtype = torch.long if name in ("input_ids", "document_ids") else torch.bool
            scalar = int if dtype == torch.long else bool
            values = batch[name]
            if (not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], list)
                    or len(values[0]) != 16 or any(type(v) is not scalar for v in values[0])):
                raise ValueError("Evaluation tensor dtype/shape differs")
            tensors[name] = torch.tensor(values, dtype=dtype)
        expected_valid = [[True]*size+[False]*(16-size)]
        if (any(batch[name] != expected_valid for name in ("valid_mask", "ce_mask", "latent_mask", "kl_mask"))
                or batch["document_ids"] != [[rank]*size+[-1]*(16-size)]
                or any(not 0 <= t < fixture["vocab_size"] for t in batch["input_ids"][0])
                or batch["input_ids"][0][size:] != [fixture["pad_id"]]*(16-size)):
            raise ValueError("Evaluation masks/padding/token bounds differ")
        value = NextLatBatch(**tensors)
        if int(build_nextlat_masks(value, document_policy=legacy.POLICY)["ce"].sum()) != size-1:
            raise ValueError("Evaluation CE denominator differs from actual masks")
        batches.append(value)
    if len(set(keys)) != 2 or sha256_file(path) != expected_sha256:
        raise ValueError("Evaluation documents duplicate or fixture changed while reading")
    return fixture, tuple(batches)


@contextmanager
def evaluation_scope(model, device, generators=None):
    modes = [(module, module.training) for module in model.modules()]
    rng = _local_rng(device, generators)
    try:
        model.eval()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device.type, enabled=False):
            yield
    finally:
        for module, training in modes:
            module.training = training
        _restore_local_rng(rng, device, generators)


def final_pass_ce(model, batch, mode):
    """No auxiliary calculation; final-pass CE is only a functionality statistic."""
    if model.training or torch.is_grad_enabled() or mode.feedback_jitter != 0:
        raise ValueError("Evaluation must be no-grad, eval mode, and explicitly jitter-free")
    output = model.backbone(batch.input_ids, attention_mask=batch.valid_mask,
        document_ids=batch.document_ids, mode=mode, feedback_noise=None,
        right_padded_causal=True, return_logits=False)
    masks = build_nextlat_masks(batch, document_policy=mode.document_policy)
    ce = masks["ce"]
    count = int(ce.sum())
    if count <= 0 or len(output.pass_hidden_states) != mode.num_passes:
        raise ValueError("Evaluation requires positive CE and unchanged pass count")
    total = _ce_chunk(output.pass_hidden_states[-1][:, :-1][ce],
                      model.backbone.readout_weight, batch.input_ids[:, 1:][ce])
    if not bool(torch.isfinite(total)):
        raise ValueError("Evaluation loss is nonfinite")
    return {"ce_sum": float(total), "ce_targets": count, "input_tokens": int(batch.valid_mask.sum()),
            "passes": len(output.pass_hidden_states), "hidden": tree_digests(output.pass_hidden_states[-1])}


def runner_observation(runner, boundary):
    adapter = runner.adapter
    buffers = {"inputs": adapter.owned_inputs(), "forward": tuple(adapter.forward_layout._owned_tensors()),
               "loss": {name: value for name, value in vars(adapter.loss_layout).items() if isinstance(value, torch.Tensor)}}
    return {"boundary": boundary(), "modes": {n:m.training for n,m in runner.model.named_modules()},
            "trainability": {n:p.requires_grad for n,p in runner.model.named_parameters()},
            "gradient_values": tree_digests({n:p.grad for n,p in runner.model.named_parameters()}),
            "buffers": tree_digests(buffers), "pointers": pointer_snapshot(runner),
            "graph_owners": tuple(id(getattr(runner, name, None)) for name in
                ("local_graph", "sync_graph", "ddp", "stream"))}


def local_evaluation(runner, batch, boundary, *, generators=None, forward=final_pass_ce):
    """No collectives here: caller coordinates ordinary failures before reduction."""
    runner.validate_execution()
    if not runner._at_update_boundary or runner._pending is not None:
        raise ValueError("Evaluation requires a completed captured optimizer boundary")
    if any(p.grad is not None and bool(p.grad.any()) for p in runner.model.parameters()):
        raise ValueError("Evaluation requires zeroed persistent gradient buffers")
    before = runner_observation(runner, boundary)
    original_input = tree_digests(vars(batch))
    device = next(runner.model.parameters()).device
    mode = replace(runner.adapter.mode, feedback_jitter=0.)
    if device.type == "cuda": torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with evaluation_scope(runner.model, device, generators):
            result = forward(runner.model, batch.to(device), mode)
    finally:
        if device.type == "cuda": torch.cuda.synchronize(device)
        runner.validate_execution()
    elapsed = time.perf_counter()-started
    after = runner_observation(runner, boundary)
    checks = {key: before[key] == after[key] for key in before}
    checks["fixture_unchanged"] = original_input == tree_digests(vars(batch))
    return {"result": result, "checks": checks, "passed": all(checks.values()),
            "elapsed_seconds": elapsed, "evaluation_mode": asdict(mode),
            "scope": "Eager unwrapped no-grad final-pass CE; not the combined training objective"}


class Observer:
    """Observe existing host seams; never mutate the frozen update implementation."""
    def __init__(self, options, fixture, batches):
        self.options, self.fixture, self.batches = options, fixture, batches
        self.data = self.runner = self.boundary_args = None
        self.report = self.tracker = self.stage_args = None

    def clear(self):
        self.data = self.runner = self.boundary_args = self.tracker = self.stage_args = None

    def live_boundary(self):
        model, optimizer, scheduler, counters, _, device, generators = self.boundary_args
        cursor = legacy.cursor_record(self.data, self.rank)
        return self.original_boundary(model, optimizer, scheduler, counters, cursor, device, generators)

    def evaluate(self, coordinator, completed):
        if self.runner is None or self.boundary_args is None or self.data is None:
            raise LifecycleError("Live evaluation boundary was not observed")
        local = coordinator.call("heldout local forward", lambda: local_evaluation(
            self.runner, self.batches[self.rank], self.live_boundary, generators=self.boundary_args[-1]))
        rows = coordinator.gather(local)
        # Preserve evidence before asserting an integrity gate.
        entry = {"after_update": completed, "by_rank": rows, "passed": all(r["passed"] for r in rows)}
        self.report.setdefault("evaluations", []).append(entry)
        coordinator.call("heldout observed evidence", lambda: write_json(self.stage_args.output_dir/"report.json", self.report), rank_zero=True)
        if not entry["passed"]:
            raise LifecycleError("Held-out evaluation changed captured training boundary")
        totals = torch.tensor([local["result"]["ce_sum"], local["result"]["ce_targets"]],
                              dtype=torch.float64, device=self.boundary_args[-2])
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=coordinator.group)
        ce_sum, targets = totals.cpu().tolist()
        if targets != 23 or not math.isfinite(ce_sum):
            raise LifecycleError("Held-out globally reduced targets or CE differ")
        entry.update(global_ce_sum=ce_sum, global_ce_targets=int(targets), global_ce_mean=ce_sum/targets,
                     max_rank_seconds=max(row["elapsed_seconds"] for row in rows))
        def publish():
            write_json(self.stage_args.output_dir/"report.json", self.report)
            # Installed W&B uses commit=False by default with an explicit
            # step; this merges with the following training log at that step.
            self.tracker.log({"update": completed,
                              "dev/functionality/final_pass_ce": entry["global_ce_mean"],
                              "dev/functionality/targets": targets,
                              "dev/functionality/max_rank_seconds": entry["max_rank_seconds"]}, step=completed)
        coordinator.call("heldout metrics", publish, rank_zero=True)

    @contextmanager
    def installed(self):
        self.original_boundary = legacy.boundary
        original_data, original_runner, original_loop = legacy.PackedCampaignData, legacy.CampaignDDPGraphTraining, legacy.run_loop
        original_stage, original_reference = legacy.run_stage, legacy.load_reference
        def data(*args, **kwargs):
            self.data = original_data(*args, **kwargs)
            if (self.data.manifest["corpus_manifest_sha256"] != self.fixture["corpus_manifest_sha256"]
                    or self.data.manifest["vocab_size"] != self.fixture["vocab_size"]
                    or self.data.split != "train"):
                self.data.close()
                raise ValueError("Evaluation corpus/vocabulary/split differs from training authority")
            return self.data
        def runner(*args, **kwargs):
            self.runner = original_runner(*args, **kwargs)
            return self.runner
        def boundary(*args, **kwargs):
            if kwargs or len(args) != 7:
                raise ValueError("Frozen live boundary signature changed")
            self.boundary_args = args
            return self.original_boundary(*args)
        def loop(**kwargs):
            original_update = kwargs["update"]
            def update():
                result = original_update()
                completed = kwargs["completed"]()
                if self.options.evaluation_policy == "insert" and completed == 1:
                    self.evaluate(kwargs["coordinator"], completed)
                return result
            return original_loop(**{**kwargs, "update": update})
        def reference(path, sha, sources, configuration):
            result = original_reference(path, sha, sources, configuration)
            adapter = result.get("evaluation_adapter", {})
            if (adapter.get("version") != SCHEMA or adapter.get("policy") != "reference"
                    or adapter.get("fixture_sha256") != self.options.fixture_sha256
                    or result.get("evaluations") != []):
                raise ValueError("Require the same-fixture no-evaluation reference")
            return result
        def stage(args, coordinator, device, runtime, determinism, report, tracker):
            self.rank, self.report, self.tracker, self.stage_args = coordinator.rank, report, tracker, args
            report.update(evaluations=[], evaluation_adapter={"version": SCHEMA,
                "policy": self.options.evaluation_policy, "fixture_sha256": self.options.fixture_sha256,
                "fixture": self.fixture, "insert_after_updates": [1] if self.options.evaluation_policy == "insert" else []})
            coordinator.call("heldout fixture snapshot", lambda: write_json(args.output_dir/"eval-fixture.json", self.fixture), rank_zero=True)
            try:
                result = original_stage(args, coordinator, device, runtime, determinism, report, tracker)
                expected = 1 if self.options.evaluation_policy == "insert" else 0
                if len(report["evaluations"]) != expected or sha256_file(self.options.fixture) != self.options.fixture_sha256:
                    raise LifecycleError("Evaluation count or immutable fixture changed")
                return result
            finally:
                self.clear()  # Graph owners must die before main tears down NCCL.
        with ExitStack() as stack:
            for name, value in (("source_hashes", source_hashes), ("PackedCampaignData", data),
                ("CampaignDDPGraphTraining", runner), ("boundary", boundary), ("run_loop", loop),
                ("run_stage", stage), ("load_reference", reference)):
                stack.enter_context(patch.object(legacy, name, value))
            yield


def parse_run(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluation-policy", required=True, choices=("reference", "insert"))
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--fixture-sha256", required=True)
    options, remaining = parser.parse_known_args(argv)
    args = legacy.parse_args(remaining)
    if (args.max_updates != 3 or args.arm != "NFR" or args.resume is not None
            or args.request_stop_after is not None or args.inject_log_error_at is not None or args.stop_file is not None
            or (options.evaluation_policy == "insert") != (args.reference_report is not None)):
        parser.error("Use exactly three fresh NFR updates; only insertion takes a pinned no-eval reference")
    if len(options.fixture_sha256) != 64 or any(c not in "0123456789abcdef" for c in options.fixture_sha256):
        parser.error("Fixture requires an independent lowercase SHA256")
    return options, remaining


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("prepare", "run"):
        raise SystemExit("Choose prepare (CPU fixture) or run (two-GPU guarded acceptance)")
    if argv[0] == "prepare":
        parser = argparse.ArgumentParser()
        parser.add_argument("--corpus", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        print(json.dumps(prepare_fixture(args.corpus, args.output), sort_keys=True))
        return
    options, remaining = parse_run(argv[1:])
    fixture, batches = load_fixture(options.fixture, options.fixture_sha256)
    with guarded.guarded_driver_scope():
        observer = Observer(options, fixture, batches)
        with observer.installed():
            return legacy.main(remaining)


if __name__ == "__main__":
    main()
