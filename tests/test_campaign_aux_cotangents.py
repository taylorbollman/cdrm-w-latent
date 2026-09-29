"""CPU contracts for fixed-state loss diagnostics; no GPU numerical claim."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, NextLatPredictor, build_nextlat_masks
from scripts.olmo_campaign_aux_cotangents import (
    component_cotangents, cotangent_comparison, decode_fixture, load_fixture,
    parse_args, relevant_weights, verify_reconstructed_model, write_fixture,
)
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def source_fixture(*, passes=4, activation_dtype=torch.float32, selected=False, empty_second=False):
    config = NextLatConfig(64, vocab_chunk_size=3, ce_chunk_size=5, lambda_latent=.7, lambda_kl=.4,
                           document_policy="continuous-stream-v1")
    predictor = NextLatPredictor(config)
    generator = torch.Generator().manual_seed(517)
    readout = torch.randn(23, 64, generator=generator)*.12
    model = SimpleNamespace(config=config, enabled=True, predictor=predictor,
                            backbone=SimpleNamespace(readout_weight=readout))
    records = []
    for index in range(2):
        ids = (torch.arange(5).reshape(1, 5)+index*3) % 23
        valid = torch.ones_like(ids, dtype=torch.bool)
        if index:
            valid[:, 3:] = False
        if index and empty_second:
            valid[:] = False
        docs = torch.full_like(ids, index)
        if index and not selected:
            docs[:, 2:] += 1  # True boundary makes CE/aux denominators differ.
        docs.masked_fill_(~valid, -1)
        masks = {term+"_mask": valid.clone() for term in ("ce", "latent", "kl")}
        if selected:
            for term in masks:
                masks[term][:] = False
                masks[term][:, 3] = valid[:, 3]
        batch = NextLatBatch(ids, valid, docs, **masks)
        hidden = tuple((torch.randn(1, 5, 64, generator=generator)*(.2+i*.1)).to(activation_dtype)
                       for i in range(passes))
        embeddings = (torch.randn(1, 5, 64, generator=generator)*.15).to(activation_dtype)
        records.append({"key": f"r{index}", "batch": vars(batch),
                        "token_embeddings": embeddings, "pass_hidden_states": hidden})
    provenance = {"source_checkpoint": {"sha256": "a"*64}, "recipe_sha256": "b"*64,
                  "execution": "synthetic CPU states; no GPU/backbone numerical claim"}
    return model, records, provenance


def exported(tmp_path, **kwargs):
    model, records, provenance = source_fixture(**kwargs)
    path = tmp_path/"fixture.json"
    receipt = write_fixture(path, model, records, provenance)
    return model, records, provenance, receipt, load_fixture(path, receipt["sha256"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_json_roundtrip_preserves_actual_activation_bytes_masks_and_weight_pins(tmp_path, dtype):
    model, records, provenance, receipt, fixture = exported(tmp_path, activation_dtype=dtype)
    assert receipt["size_bytes"] < 100_000
    assert fixture["weights"] == relevant_weights(model)
    assert fixture["provenance"] == provenance
    assert fixture["counts"] == {"ce": 6, "latent": 5, "kl": 3}
    for actual, source in zip(fixture["records"], records):
        assert tree_digests(vars(actual["batch"])) == tree_digests(source["batch"])
        assert tree_digests(actual["token_embeddings"]) == tree_digests(source["token_embeddings"])
        assert tree_digests(actual["pass_hidden_states"]) == tree_digests(source["pass_hidden_states"])
        assert all(not t.requires_grad and t.device.type == "cpu" for t in actual["pass_hidden_states"])
        assert actual["pass_hidden_states"][0].dtype == dtype
    with pytest.raises(FileExistsError):
        write_fixture(receipt["path"], model, records, provenance)
    with pytest.raises(ValueError, match="SHA256"):
        load_fixture(receipt["path"], "0"*64)


def test_reconstructed_weights_recipe_and_checkpoint_must_match(tmp_path):
    model, _, provenance, _, fixture = exported(tmp_path)
    recipe = SimpleNamespace(sha256=provenance["recipe_sha256"])
    verify_reconstructed_model(model, recipe, provenance["source_checkpoint"], fixture)
    with pytest.raises(ValueError, match="checkpoint/recipe"):
        verify_reconstructed_model(model, SimpleNamespace(sha256="c"*64), provenance["source_checkpoint"], fixture)
    with torch.no_grad():
        next(model.predictor.parameters()).view(-1)[0] += .01
    with pytest.raises(ValueError, match="readout/predictor bytes"):
        verify_reconstructed_model(model, recipe, provenance["source_checkpoint"], fixture)


@pytest.mark.parametrize("change", ["duplicate", "bytes", "shape", "empty", "passes", "provenance"])
def test_malformed_or_unbounded_fixtures_rejected(tmp_path, change):
    _, _, _, receipt, _ = exported(tmp_path)
    with open(receipt["path"]) as handle:
        payload = json.load(handle)
    if change == "duplicate":
        payload["records"][1]["key"] = payload["records"][0]["key"]
    elif change == "bytes":
        payload["records"][0]["pass_hidden_states"][0]["data_base64"] = "!!!!"
    elif change == "shape":
        payload["records"][0]["pass_hidden_states"][0]["shape"] = [2**40]
    elif change == "empty":
        payload["records"] = []
    elif change == "passes":
        payload["records"][0]["pass_hidden_states"] = payload["records"][0]["pass_hidden_states"][:2]
    else:
        payload["provenance"].pop("source_checkpoint")
    with pytest.raises(ValueError):
        decode_fixture(payload)


def manual_auxiliary(fixture, readout, predictor, objective):
    """Direct position-wise oracle, independent of either production gather path."""
    result, total = {}, 0.
    predictor.zero_grad(set_to_none=True)
    for row in fixture["records"]:
        hidden = tuple(t.float().clone().requires_grad_() for t in row["pass_hidden_states"])
        embeds = row["token_embeddings"].float().clone().requires_grad_()
        masks = build_nextlat_masks(row["batch"], document_policy=fixture["config"].document_policy)
        terms = []
        for state in hidden:
            predicted = predictor(state[:, :-1], embeds[:, 1:])
            if objective == "latent":
                loss = F.smooth_l1_loss(predicted, state[:, 1:].detach(), reduction="none").mean(-1)
                loss = loss[masks["latent"]].sum()
                weight = fixture["config"].lambda_latent
            else:
                with torch.no_grad():
                    teacher = F.log_softmax(F.linear(state[:, 1:-1], readout), dim=-1)
                student = F.log_softmax(F.linear(predicted[:, :-1], readout), dim=-1)
                loss = F.kl_div(student, teacher, log_target=True, reduction="none").sum(-1)
                loss = loss[masks["kl"]].sum()
                weight = fixture["config"].lambda_kl
            terms.append(loss)
        selected = sum(terms)/len(terms) * weight/fixture["counts"][objective]
        selected.backward()
        total += float(selected.detach())
        for index, tensor in enumerate(hidden):
            result[f"hidden/{row['key']}/pass_{index}"] = tensor.grad
        result[f"embedding/{row['key']}"] = embeds.grad
    result.update({"predictor/"+name: p.grad.clone() for name, p in predictor.named_parameters()})
    return total, result


@pytest.mark.parametrize("objective", ["latent", "kl"])
def test_both_layouts_match_direct_oracle_with_global_counts_pass_weights_and_no_state_mutation(tmp_path, objective):
    model, _, _, _, fixture = exported(tmp_path)
    weights = relevant_weights(model)
    inputs = tree_digests(fixture["records"])
    rng = torch.get_rng_state().clone()
    total, expected = manual_auxiliary(fixture, model.backbone.readout_weight, model.predictor, objective)
    for layout in ("sparse", "prepared"):
        metadata, actual = component_cotangents(fixture, model.backbone.readout_weight, model.predictor,
            precision="fp32", layout=layout, objective=objective)
        assert metadata["finite"] and not metadata["missing_predictor_gradients_zero_materialized"]
        assert metadata["objective"] == pytest.approx(total, rel=3e-6, abs=1e-7)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=1e-7, rtol=3e-5, msg=key)
    assert relevant_weights(model) == weights
    assert tree_digests(fixture["records"]) == inputs
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("objective,source", [("latent", 2), ("kl", 1)])
@pytest.mark.parametrize("layout", ["sparse", "prepared"])
def test_target_and_teacher_states_stay_detached_and_empty_record_cotangents_are_zero(tmp_path, objective, source, layout):
    model, _, _, _, fixture = exported(tmp_path, passes=1, selected=True, empty_second=True)
    metadata, gradients = component_cotangents(fixture, model.backbone.readout_weight, model.predictor,
        precision="fp32", layout=layout, objective=objective)
    assert metadata["finite"]
    hidden = gradients["hidden/r0/pass_0"]
    embeddings = gradients["embedding/r0"]
    assert torch.count_nonzero(hidden[:, source]) > 0
    assert torch.count_nonzero(embeddings[:, source+1]) > 0
    assert torch.count_nonzero(hidden[:, :source]) == 0
    assert torch.count_nonzero(hidden[:, source+1:]) == 0
    assert torch.count_nonzero(embeddings[:, :source+1]) == 0
    assert torch.count_nonzero(embeddings[:, source+2:]) == 0
    assert torch.count_nonzero(gradients["hidden/r1/pass_0"]) == 0
    assert torch.count_nonzero(gradients["embedding/r1"]) == 0


def test_bf16_execution_reuses_anchor_bytes_and_reports_finite_geometry(tmp_path):
    model, _, _, _, fixture = exported(tmp_path, passes=1, activation_dtype=torch.bfloat16)
    before = tree_digests(fixture["records"])
    _, reference = component_cotangents(fixture, model.backbone.readout_weight, model.predictor,
        precision="fp32", layout="sparse", objective="latent")
    for layout in ("sparse", "prepared"):
        metrics, actual = component_cotangents(fixture, model.backbone.readout_weight, model.predictor,
            precision="bf16_mixed", layout=layout, objective="latent")
        assert metrics["finite"]
        comparison = cotangent_comparison(actual, reference)
        assert all(row["tensors"] > 0 and row["reference_norm"] > 0 for row in comparison.values())
        assert all(row["relative_l2"] >= 0 for row in comparison.values())
    assert tree_digests(fixture["records"]) == before


def test_geometry_uses_reference_norm_and_reports_zero_reference_explicitly():
    reference = {"hidden/r0/pass_0": torch.tensor([1., 0.]), "embedding/r0": torch.zeros(2),
                 "predictor/weight": torch.tensor([0., 2.])}
    actual = {"hidden/r0/pass_0": torch.tensor([0., 1.]), "embedding/r0": torch.zeros(2),
              "predictor/weight": torch.tensor([0., 4.])}
    result = cotangent_comparison(actual, reference)
    assert result["hidden"]["cosine"] == pytest.approx(0.)
    assert result["hidden"]["relative_l2"] == pytest.approx(2**.5)
    assert result["predictor"]["norm_ratio"] == pytest.approx(2.)
    assert result["embedding"]["cosine"] is None and result["embedding"]["bitwise_equal"]
    assert result["all"]["relative_l2"] == pytest.approx((6/5)**.5)


@pytest.mark.parametrize("extra", [["--steps", "5"], ["--length", "1024"], ["--batch-size", "8"]])
def test_cli_rejects_unbounded_runtime_modes(extra):
    with pytest.raises(SystemExit):
        parse_args(["--fixture", "fixture.json", "--fixture-sha256", "a"*64,
                    "--output-dir", "/tmp/cotangents", *extra])
