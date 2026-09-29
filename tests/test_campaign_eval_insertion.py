"""Real CPU forward/update oracles; CUDA graph/DDP execution remains a GPU gate."""
import copy
from dataclasses import asdict, replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cdrm.pretrained.campaign_data import SourcePin, TokenizedDocument
from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model, build_campaign_adamw, feedback_noise_for_rows
from cdrm.pretrained.campaign_training import CampaignObjective, CampaignGraphTraining
from cdrm.pretrained.lm_training import TrainingCounters, LMTrainingConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_campaign_eval_insertion as probe


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)
    torch.manual_seed(9237)
    with probe.sdpa_kernel(probe.SDPBackend.MATH):
        yield


def documents():
    source = SourcePin("fixture", "test://fixture", "test", "a"*64)
    return tuple(TokenizedDocument(source, str(index), "dev" if index < 3 else "train",
        str(index)*64, tuple((value+index) % 28+2 for value in range(23))+(31,)) for index in range(4))


def fixture(tmp_path):
    value = probe.fixture_from_documents(documents(), corpus_sha256=probe.CORPUS_SHA,
        vocab_size=32, eos_id=31)
    path = tmp_path/"fixture.json"
    probe.write_json(path, value)
    loaded, batches = probe.load_fixture(path, probe.sha256_file(path))
    return value, path, batches


def setup(tmp_path):
    value, _, batches = fixture(tmp_path)
    recipe = CampaignRecipe("NFR", sequence_length=16, rt_layers=(0, 1), document_policy=probe.legacy.POLICY)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32",
        ordinary_activation_checkpointing=True, tile_backend="eager", backward_tile_backend="eager",
        reuse_rope=True, kv_only_writes=True)
    model = build_campaign_model(base, recipe).train()
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    counters = TrainingCounters()
    train_batch = batches[0]
    noise = probe.legacy.feedback_noise_for_rows(recipe, ["training-fixture"], logical_update=0,
        sequence_length=16, width=model.config.model_dim, physical_batch_size=1)
    adapter = CampaignObjective(model, train_batch, mode=recipe.mode(), global_counts=model.counts(train_batch),
        feedback_noise=noise, config=LMTrainingConfig(precision="fp32", max_grad_norm=1.))
    runner = CampaignGraphTraining(adapter)
    # These slots stand in for DDP owner handles only, not CUDA/DDP computation.
    runner._pending = None
    runner.local_graph, runner.sync_graph, runner.ddp = object(), object(), object()
    generators = {"data": torch.Generator().manual_seed(58)}
    cursor = {"completed": 0}
    def boundary():
        return probe.tree_digests({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "counters": asdict(counters), "cursor": dict(cursor),
            "rng": probe._local_rng(torch.device("cpu"), generators)})
    def update():
        with probe.sdpa_kernel(probe.SDPBackend.MATH):
            result = runner.optimizer_step(optimizer, (train_batch,), feedback_noises=(noise,), counters=counters)
            runner.zero_grad()
        cursor["completed"] += 1
        return result
    update()
    return SimpleNamespace(model=model, runner=runner, optimizer=optimizer, counters=counters,
        batch=batches[1], recipe=recipe, generators=generators, boundary=boundary, update=update)


def test_fixture_uses_disjoint_dev_prefixes_actual_padding_and_exact_denominator(tmp_path):
    value, path, batches = fixture(tmp_path)
    assert [int(b.valid_mask.sum()) for b in batches] == [16, 9]
    assert len({r["document_key"] for r in value["records"]}) == 2
    assert all(r["document"]["split"] == "dev" for r in value["records"])
    assert all(int((b.input_ids[b.valid_mask] == 31).sum()) == 0 for b in batches)
    assert value["ce_targets"] == sum(int(probe.build_nextlat_masks(b, document_policy=probe.legacy.POLICY)["ce"].sum()) for b in batches)
    assert probe.load_fixture(path, probe.sha256_file(path))[0] == value


@pytest.mark.parametrize("mutation", ["split", "mask", "dtype", "duplicate", "padding", "count"])
def test_fixture_semantic_mutations_rejected_even_with_recomputed_file_pin(tmp_path, mutation):
    value, path, _ = fixture(tmp_path)
    if mutation == "split":
        value["records"][0]["document"]["split"] = "train"
        value["records"][0]["document_key"] = probe.digest(value["records"][0]["document"])
    elif mutation == "mask": value["records"][0]["batch"]["valid_mask"][0][0] = False
    elif mutation == "dtype": value["records"][0]["batch"]["input_ids"][0][0] = True
    elif mutation == "duplicate":
        value["records"][1]["document"] = copy.deepcopy(value["records"][0]["document"])
        value["records"][1]["document_key"] = value["records"][0]["document_key"]
    elif mutation == "padding": value["records"][1]["batch"]["input_ids"][0][-1] = 31
    else: value["ce_targets"] += 1
    for row in value["records"]: row["batch_sha256"] = probe.digest(row["batch"])
    probe.write_json(path, value)
    with pytest.raises(ValueError): probe.load_fixture(path, probe.sha256_file(path))


def test_actual_no_jitter_forward_matches_manual_ce_and_preserves_real_boundary(tmp_path):
    obj = setup(tmp_path)
    before = obj.boundary()
    calls = []
    def observed(model, batch, mode):
        assert not model.training and not torch.is_grad_enabled() and mode.feedback_jitter == 0
        assert mode.rt_mode == obj.recipe.mode().rt_mode and mode.num_passes == 4
        calls.append(mode)
        return probe.final_pass_ce(model, batch, mode)
    with probe.sdpa_kernel(probe.SDPBackend.MATH):
        row = probe.local_evaluation(obj.runner, obj.batch, obj.boundary, generators=obj.generators, forward=observed)
    assert row["passed"] and all(row["checks"].values()) and len(calls) == 1
    assert obj.boundary() == before and row["result"]["ce_targets"] == 8
    mode = replace(obj.recipe.mode(), feedback_jitter=0.)
    with probe.evaluation_scope(obj.model, torch.device("cpu"), obj.generators):
        output = obj.model.backbone(obj.batch.input_ids, attention_mask=obj.batch.valid_mask,
            document_ids=obj.batch.document_ids, mode=mode, return_logits=False, right_padded_causal=True)
        states = output.pass_hidden_states[-1][0, :8]
        logits = torch.nn.functional.linear(states, obj.model.backbone.readout_weight)
        ce = torch.nn.functional.cross_entropy(logits, obj.batch.input_ids[0, 1:9], reduction="sum")
        assert float(ce) == row["result"]["ce_sum"]
    assert obj.boundary() == before


def test_real_next_update_exact_with_and_without_eval(tmp_path):
    first = tmp_path/"first"; first.mkdir()
    obj = setup(first)
    # Restore the RNG and clone before the second trajectory's identical construction.
    saved_rng = probe._local_rng(torch.device("cpu"), obj.generators)
    with probe.sdpa_kernel(probe.SDPBackend.MATH):
        row = probe.local_evaluation(obj.runner, obj.batch, obj.boundary, generators=obj.generators)
    assert row["passed"]
    actual = obj.update()
    actual_boundary = obj.boundary()
    torch.manual_seed(9237)
    second = tmp_path/"second"; second.mkdir()
    reference = setup(second)
    probe._restore_local_rng(saved_rng, torch.device("cpu"), reference.generators)
    expected = reference.update()
    assert actual == expected and actual_boundary == reference.boundary()


def test_exception_restores_modes_named_rng_and_preserves_storage(tmp_path):
    obj = setup(tmp_path)
    before = obj.boundary()
    def fail(model, batch, mode):
        random.random(); np.random.random(); torch.rand(1)
        torch.rand(1, generator=obj.generators["data"])
        assert not torch.is_grad_enabled() and not model.training
        raise OSError("ordinary heldout forward failure")
    with probe.sdpa_kernel(probe.SDPBackend.MATH), pytest.raises(OSError, match="heldout"):
        probe.local_evaluation(obj.runner, obj.batch, obj.boundary, generators=obj.generators, forward=fail)
    assert obj.boundary() == before
    with probe.sdpa_kernel(probe.SDPBackend.MATH): obj.runner.validate_execution()


def test_eval_scope_restores_nonuniform_modes_and_requires_explicit_eval_contract(tmp_path):
    obj = setup(tmp_path)
    obj.model.predictor.eval()
    modes = {n:m.training for n,m in obj.model.named_modules()}
    with probe.evaluation_scope(obj.model, torch.device("cpu"), obj.generators):
        assert not any(m.training for m in obj.model.modules())
        with pytest.raises(ValueError, match="jitter-free"):
            probe.final_pass_ce(obj.model, obj.batch, obj.recipe.mode())
    assert modes == {n:m.training for n,m in obj.model.named_modules()}


def test_uncommitted_or_nonzero_gradient_boundary_rejected(tmp_path):
    obj = setup(tmp_path)
    obj.runner._pending = object()
    with probe.sdpa_kernel(probe.SDPBackend.MATH), pytest.raises(ValueError, match="completed"):
        probe.local_evaluation(obj.runner, obj.batch, obj.boundary)
    obj.runner._pending = None
    next(obj.model.parameters()).grad.fill_(1.)
    with probe.sdpa_kernel(probe.SDPBackend.MATH), pytest.raises(ValueError, match="zeroed"):
        probe.local_evaluation(obj.runner, obj.batch, obj.boundary)


def test_eval_detects_state_mutation_without_restoring_or_hiding_it(tmp_path):
    obj = setup(tmp_path)
    def mutate(model, batch, mode):
        result = probe.final_pass_ce(model, batch, mode)
        # Buffer is numerical model state, not a graph-owned input. Real scope
        # must report the mutation, never silently reset model/optimizer weights.
        model.backbone.fusion.output_scale.add_(.01)
        return result
    with probe.sdpa_kernel(probe.SDPBackend.MATH):
        with pytest.raises(ValueError, match="parameter/module|ownership|changed"):
            probe.local_evaluation(obj.runner, obj.batch, obj.boundary, forward=mutate)


def test_observer_wraps_only_after_committed_update_and_restores_all_bindings(tmp_path, monkeypatch):
    value, path, batches = fixture(tmp_path)
    options = SimpleNamespace(evaluation_policy="insert", fixture=path, fixture_sha256=probe.sha256_file(path))
    observer = probe.Observer(options, value, batches)
    events = []
    def fake_loop(**kwargs):
        assert kwargs["update"]() == {"metric": 1}
        events.append("loop returned")
        return 42
    monkeypatch.setattr(probe.legacy, "run_loop", fake_loop)
    monkeypatch.setattr(observer, "evaluate", lambda coordinator, completed: events.append(("eval", completed)))
    names = ("source_hashes", "PackedCampaignData", "CampaignDDPGraphTraining", "boundary", "run_loop", "run_stage", "load_reference")
    originals = {name:getattr(probe.legacy, name) for name in names}
    counter = [0]
    def update():
        counter[0] = 1; events.append("cursor committed")
        return {"metric": 1}
    with observer.installed():
        assert probe.legacy.run_loop(update=update, log=lambda _:None, completed=lambda: counter[0], coordinator=object()) == 42
    assert events == ["cursor committed", ("eval", 1), "loop returned"]
    assert all(getattr(probe.legacy, name) is value for name,value in originals.items())
    with pytest.raises(OSError):
        with observer.installed(): raise OSError("scope failure")
    assert all(getattr(probe.legacy, name) is value for name,value in originals.items())


def test_cli_rejects_resume_and_missing_reference_for_insertion(tmp_path):
    args = ["--evaluation-policy", "reference", "--fixture", "fixture.json", "--fixture-sha256", "a"*64,
        "--corpus", "corpus", "--index", "index", "--index-sha256", "b"*64,
        "--output-dir", str(probe.ROOT/".runtime/test-eval"),
        "--checkpoint-root", str(probe.ROOT/".runtime/test-eval-checkpoints")]
    assert probe.parse_run(args)[0].evaluation_policy == "reference"
    with pytest.raises(SystemExit): probe.parse_run(args+["--max-updates", "2"])
    with pytest.raises(SystemExit): probe.parse_run(args+["--resume", "x", "--resume-manifest-sha256", "c"*64])
    args[1] = "insert"
    with pytest.raises(SystemExit): probe.parse_run(args)
    assert probe.parse_run(args+["--reference-report", "ref.json", "--reference-sha256", "d"*64])[0].evaluation_policy == "insert"
