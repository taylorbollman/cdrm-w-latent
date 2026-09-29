"""CPU ownership and independently observed loss-work oracles for the ledger."""
from dataclasses import replace
import json

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_losses import DynamicNextLatLayout, compute_dynamic_nextlat_loss_sums
from cdrm.pretrained.campaign_recipe import ARMS, CampaignRecipe, build_campaign_adamw, build_campaign_model
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, NextLatPredictor, compute_nextlat_loss_sums
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.resource_estimates import estimate_training_resources
from scripts import olmo_campaign_resource_ledger as ledger


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def tiny(arm):
    recipe = CampaignRecipe(arm, sequence_length=6, rt_layers=(0,1), document_policy="continuous-stream-v1")
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32")
    model = build_campaign_model(base, recipe)
    return model, recipe, build_campaign_adamw(model, recipe, fused=False)


def batch():
    ids = torch.tensor([[2,3,4,5,6,0], [7,8,0,0,0,0], [0,0,0,0,0,0]])
    valid = torch.tensor([[1,1,1,1,1,0],[1,1,0,0,0,0],[0,0,0,0,0,0]], dtype=torch.bool)
    docs = torch.tensor([[0,0,0,1,1,-1],[2,2,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1]])
    ce, latent = valid.clone(), valid.clone()
    ce[0,2] = False
    latent[0,1] = False
    return NextLatBatch(ids, valid, docs, ce, latent, valid.clone())


@pytest.mark.parametrize("arm", ARMS)
def test_actual_module_ownership_ties_and_optimizer_groups(arm):
    model, recipe, optimizer = tiny(arm)
    card = ledger.parameter_card(model, recipe, optimizer)
    assert all(card["checks"].values())
    groups, inventory = card["groups"], card["observed_inventory"]
    actual = {id(p):p for p in model.parameters()}
    assert inventory["registered_unique"] == sum(p.numel() for p in actual.values())
    assert groups["fusion"]["resident_parameters"] == 2*model.config.model_dim**2
    assert groups["fusion"]["branch_active_parameters"] == (groups["fusion"]["resident_parameters"] if recipe.feedback else 0)
    assert bool(groups["predictor"]["resident_parameters"]) == recipe.nextlat
    assert groups["predictor"]["deployable_parameters"] == 0
    embedding = model.backbone.token_embeddings.weight
    assert sum(p is embedding for g in optimizer.param_groups for p in g["params"]) == 1
    assert next(g for g in optimizer.param_groups if any(p is embedding for p in g["params"]))["weight_decay"] == 0
    assert inventory["registered_unique"]-inventory["trainable"] == (0 if recipe.feedback else 2*model.config.model_dim**2)


@pytest.mark.parametrize("arm", ARMS)
def test_per_pass_rt_selection_matches_real_stack_invocations(arm, monkeypatch):
    model, recipe, _ = tiny(arm)
    model.eval()
    calls = []
    original = model.backbone._stack
    def observed(embeddings, mode, **kwargs):
        calls.append(tuple(mode.selected_layers))
        return original(embeddings, mode, **kwargs)
    monkeypatch.setattr(model.backbone, "_stack", observed)
    with torch.no_grad():
        model.backbone(torch.tensor([[1,2,3,4,5,6]]), mode=recipe.mode(), return_logits=False)
    selected = {"ce":5,"latent":5 if recipe.nextlat else 0,"kl":4 if recipe.nextlat else 0,"predictor":5 if recipe.nextlat else 0}
    card = ledger.matrix_card(model.backbone.config, recipe, model.config, 1, selected, layout="dynamic_dense")
    assert len(calls) == (4 if recipe.feedback else 1)
    assert card["rt_block_calls_per_physical_slot"] == sum(len(c) for c in calls)
    assert card["ordinary_block_calls_per_physical_slot"] == sum(model.backbone.config.num_layers-len(c) for c in calls)


def test_independent_sparse_masks_and_dense_loss_forward_shapes(monkeypatch):
    from cdrm.pretrained import campaign_losses, nextlat
    data = batch()
    config = NextLatConfig(32, ce_chunk_size=128, vocab_chunk_size=128, document_policy="continuous-stream-v1")
    predictor = NextLatPredictor(config)
    h = torch.randn(3,6,32, requires_grad=True)
    e = torch.randn_like(h)
    w = torch.randn(64,32, requires_grad=True)
    selected = ledger.selected_work(data, nextlat=True, document_policy=config.document_policy)
    assert selected == {"ce":4,"latent":3,"kl":1,"predictor":4}
    seen = {"sparse":{}, "dynamic_dense":{}}
    old_sparse, old_dense = nextlat._chunked_sum, campaign_losses._weighted_chunked_sum
    def sparse(fn, states, *args, **kwargs):
        seen["sparse"]["ce" if fn is nextlat._ce_chunk else "kl"] = len(states)
        return old_sparse(fn, states, *args, **kwargs)
    def dense(fn, states, *args, **kwargs):
        seen["dynamic_dense"]["ce" if fn is campaign_losses._weighted_ce_chunk else "kl"] = len(states)
        return old_dense(fn, states, *args, **kwargs)
    monkeypatch.setattr(nextlat, "_chunked_sum", sparse)
    monkeypatch.setattr(campaign_losses, "_weighted_chunked_sum", dense)
    active = ["sparse"]
    handle = predictor.register_forward_pre_hook(lambda m,args: seen[active[0]].update(predictor=len(args[0])))
    compute_nextlat_loss_sums(h,e,w,data,predictor,config)
    active[0] = "dynamic_dense"
    layout = DynamicNextLatLayout.from_batch(data,config)
    compute_dynamic_nextlat_loss_sums(h,e,w,data.input_ids,predictor,config,layout)
    handle.remove()
    assert seen == {"sparse":{"ce":4,"kl":1,"predictor":4}, "dynamic_dense":{"ce":15,"kl":12,"predictor":15}}
    for kind, observed in seen.items():
        counts = ledger.executed_work(3,6,selected,nextlat=True,layout=kind)
        assert {k:counts[k] for k in observed} == observed


@pytest.mark.parametrize("arm", ARMS)
def test_full_row_ledger_matches_existing_matrix_formulas_exactly(arm):
    model, recipe, _ = tiny(arm)
    selected = {"ce":15,"latent":15 if recipe.nextlat else 0,"kl":12 if recipe.nextlat else 0,"predictor":15 if recipe.nextlat else 0}
    expected = estimate_training_resources(model.backbone.config,batch_size=3,sequence_length=6,mode=recipe.mode(),
        nextlat=model.config if recipe.nextlat else None,ordinary_checkpointing=True,backward_memory="recompute",kv_only_writes=True)
    for kind in ("sparse","dynamic_dense"):
        card = ledger.matrix_card(model.backbone.config,recipe,model.config,3,selected,layout=kind)
        assert card["matrix_flops_minimum"] == expected.matrix_flops_minimum
        assert card["matrix_flops_maximum"] == expected.matrix_flops_maximum
        assert {c["name"]:(c["minimum"],c["maximum"]) for c in card["components"]} == {c.name:(c.minimum,c.maximum) for c in expected.components}


def test_heterogeneous_slots_sum_allocated_and_supervised_work_without_world_size_double_count():
    model, recipe, optimizer = tiny("NFR")
    workload = {"ranks":[
        {"rank":0,"slots":2,"physical_rows":6,"valid_tokens":10,"padding_tokens":26,"empty_rows":3,
         "selected":{"ce":7,"latent":5,"kl":2,"predictor":6}},
        {"rank":1,"slots":1,"physical_rows":2,"valid_tokens":6,"padding_tokens":6,"empty_rows":1,
         "selected":{"ce":5,"latent":4,"kl":2,"predictor":4}}]}
    card = ledger.build_card(model,recipe,optimizer,workload)
    total = card["totals"]
    assert total["allocated_input_tokens"] == 48 and total["valid_input_tokens"] == 16
    assert total["allocated_pass_tokens"] == 192 and total["valid_pass_tokens"] == 64
    assert total["supervised_positions_once_per_update"] == {"ce":12,"latent":9,"kl":4,"predictor":10}
    assert total["dynamic_dense"]["executed_positions_once_per_pass"] == {"ce":40,"latent":40,"kl":32,"predictor":40}
    assert total["dynamic_dense"]["rt_block_invocations"] == 3*4*2
    assert total["dynamic_dense"]["matrix_flops_minimum"] == sum(r["dynamic_dense"]["matrix_flops_minimum"] for r in card["ranks"])
    sparse = {c["name"]:c for c in card["ranks"][0]["sparse"]["components"]}
    dense = {c["name"]:c for c in card["ranks"][0]["dynamic_dense"]["components"]}
    assert all(sparse[k] == dense[k] for k in sparse if k not in ledger.LOSS_COMPONENTS)
    assert sparse[ledger.LOSS_COMPONENTS[0]]["minimum"] < dense[ledger.LOSS_COMPONENTS[0]]["minimum"]


def test_empty_sparse_selection_still_counts_scheduled_backbone_and_dense_loss_capacity():
    model, recipe, _ = tiny("NF")
    counts = dict.fromkeys(ledger.TERMS,0)
    a = ledger.matrix_card(model.backbone.config,recipe,model.config,2,counts,layout="sparse")
    b = ledger.matrix_card(model.backbone.config,recipe,model.config,2,counts,layout="dynamic_dense")
    assert a["matrix_flops_minimum"] > 0 and b["matrix_flops_minimum"] > a["matrix_flops_minimum"]
    assert all(c["minimum"] == 0 for c in a["components"] if c["name"] in ledger.LOSS_COMPONENTS)


@pytest.mark.parametrize("change", ["negative","union","capacity","disabled","layout"])
def test_invalid_work_rejected(change):
    selected = {"ce":4,"latent":3,"kl":1,"predictor":4}
    enabled, layout = True,"sparse"
    if change == "negative": selected["ce"] = -1
    elif change == "union": selected["predictor"] = 2
    elif change == "capacity": selected["ce"] = 16
    elif change == "disabled": enabled = False
    else: layout = "unknown"
    with pytest.raises(ValueError):ledger.executed_work(3,6,selected,nextlat=enabled,layout=layout)


def test_pinned_workload_counts_are_independently_reconciled(tmp_path):
    rank = {"physical_rows":4,"packed_rows":3,"empty_rows":1,"valid_tokens":17,"padding_tokens":7,
        "ce_targets":14,"latent_pairs":12,"kl_triples":8,"excluded_boundary_latent_pairs":2}
    report = {"status":"passed","phase":"write","document_policy":"continuous-stream-v1","arm":"NFR",
        "length":6,"physical_batch_per_rank":2,"configuration":{"world_size":1},
        "updates":[{"label":"first_update","microbatches_per_rank":2,"rank_accounting":[rank],
                    "logical_counts":{k:rank[k] for k in ("valid_tokens","packed_rows","ce_targets","latent_pairs","kl_triples")}}]}
    path = tmp_path/"reference.json"
    path.write_text(json.dumps(report))
    got = ledger.workload_from_report(path,sha256_file(path))
    assert got["ranks"][0]["selected"] == {"ce":14,"latent":12,"kl":8,"predictor":12}
    old_sha = sha256_file(path)
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError,match="immutable"):ledger.workload_from_report(path,old_sha)
    report["updates"][0]["rank_accounting"][0]["padding_tokens"] -= 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="accounting"):ledger.workload_from_report(path,sha256_file(path))
