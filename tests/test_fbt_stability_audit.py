"""Independent JSON mutation oracles for clean F-only and pass-curve evidence."""
from copy import deepcopy
import ast
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import olmo_fbt_stability_audit as audit


def ownership():
    names = ["backbone.backbone.weight", "backbone.fusion.state_proj.weight", "backbone.fusion.token_gate.weight"]
    mode = {"enabled": True, "num_passes": 4, "beta": 1., "feedback_jitter": .02,
            "rt_mode": {"selected_layers": [], "alpha": 1.}}
    return {"arm": "F", "recipe": {"arm": "F", "feedback_jitter": .02,
            "fbt_passes": 4, "pass_loss_policy": "campaign_v1"},
        "model_contract": {"arm": "F", "weights": {"ce": 1., "latent": 0., "kl": 0.},
            "mode": mode, "tied_readout": True, "dormant_fusion_parameters": 0,
            "parameter_layout": [{"name": name, "requires_grad": True, "dtype": "torch.float32", "shape": [2, 2]}
                                 for name in names],
            "optimizer_ownership": [[names[0]], names[1:]], "resident_parameters": 12,
            "trainable_parameters": 12, "component_parameters": {"backbone": 4, "fusion": 8, "predictor": 0}}}


def test_clean_F_ownership_is_actual_and_does_not_mutate_input():
    payload = ownership(); before = deepcopy(payload)
    audit.model_check(audit.Audit(), payload, native=False)
    assert payload == before


@pytest.mark.parametrize("mutation", ["predictor", "RT", "auxiliary", "untied", "frozen_backbone",
    "duplicate_optimizer", "missing_optimizer", "component_count", "K2", "BF16master"])
def test_wrong_architecture_or_optimizer_ownership_fails(mutation):
    payload = ownership(); model = payload["model_contract"]
    if mutation == "predictor": model["parameter_layout"][0]["name"] = "predictor.weight"
    elif mutation == "RT": model["mode"]["rt_mode"]["selected_layers"] = [0]
    elif mutation == "auxiliary": model["weights"]["kl"] = .1
    elif mutation == "untied": model["tied_readout"] = False
    elif mutation == "frozen_backbone": model["parameter_layout"][0]["requires_grad"] = False
    elif mutation == "duplicate_optimizer": model["optimizer_ownership"][0] *= 2
    elif mutation == "missing_optimizer": model["optimizer_ownership"].pop()
    elif mutation == "component_count": model["component_parameters"]["fusion"] -= 1
    elif mutation == "K2": model["mode"]["num_passes"] = 2
    elif mutation == "BF16master": model["parameter_layout"][0]["dtype"] = "torch.bfloat16"
    with pytest.raises(ValueError): audit.model_check(audit.Audit(), payload, native=False)


def shared_origin():
    payload = ownership()
    payload.update(data={"ordered": "a" * 64}, plan={"first_cursor": {"next_chunk": 0},
        "updates": [{"update": n, "membership_sha256": str(n) * 64, "counts": {"ce": 30},
                     "allocation_by_arm": {"F": [1, 1]}} for n in range(1, 4)]},
        schedule={"valid_token_prefix": [0, 32, 64, 96], "lr_at_completed_boundaries": [.1, .2, .3, .4]})
    core = {"backbone.backbone.weight": {"sha256": "a" * 64}, "backbone.fusion.state_proj.weight": {"sha256": "b" * 64}}
    boundary = {"state": {"model": core, "optimizer": {"state": {}}}, "rng": {"seed": 123}}
    report = {"arm": "F", "configuration": {"source_checkpoint": {"sha256": "d" * 64},
        "execution_identity": {"payload": payload}}, "startup_import": {"receipt": {"sha": "e" * 64}},
        "origin_boundary_by_rank": [deepcopy(boundary), deepcopy(boundary)], "updates": {}, "observations": {}}
    old = deepcopy(report); old["arm"] = "NF"; old["loop"] = {"start_update": 0}
    p = old["configuration"]["execution_identity"]["payload"]
    p["recipe"]["arm"] = "NF"; p["plan"]["updates"].pop()
    for key in p["schedule"]: p["schedule"][key].pop()
    for entry in old["origin_boundary_by_rank"]: entry["state"]["model"]["predictor.weight"] = {"sha256": "f" * 64}
    return report, old


def test_clean_original_core_matches_shared_NF_start_but_predictor_is_excluded():
    report, old = shared_origin(); before = deepcopy((report, old))
    audit.baseline_check(audit.Audit(), report, old)
    assert (report, old) == before


@pytest.mark.parametrize("mutation", ["trained_NF_endpoint", "fresh_fusion", "data", "noise_seed",
    "order", "LR", "Adam_history", "RNG", "native_checkpoint", "import_receipt"])
def test_clean_origin_and_prefix_are_not_just_labels(mutation):
    report, old = shared_origin(); payload = report["configuration"]["execution_identity"]["payload"]
    if mutation == "trained_NF_endpoint": report["origin_boundary_by_rank"][0]["state"]["model"]["backbone.backbone.weight"]["sha256"] = "z" * 64
    elif mutation == "fresh_fusion": report["origin_boundary_by_rank"][0]["state"]["model"]["backbone.fusion.state_proj.weight"]["sha256"] = "z" * 64
    elif mutation == "data": payload["data"]["ordered"] = "z" * 64
    elif mutation == "noise_seed": payload["recipe"]["jitter_seed"] = 1234
    elif mutation == "order": payload["plan"]["updates"][0]["membership_sha256"] = "z" * 64
    elif mutation == "LR": payload["schedule"]["lr_at_completed_boundaries"][1] = .01
    elif mutation == "Adam_history": report["origin_boundary_by_rank"][0]["state"]["optimizer"]["state"] = {"weight": {"step": 32}}
    elif mutation == "RNG": report["origin_boundary_by_rank"][0]["rng"]["seed"] += 1
    elif mutation == "native_checkpoint": report["configuration"]["source_checkpoint"]["sha256"] = "z" * 64
    elif mutation == "import_receipt": report["startup_import"]["receipt"]["sha"] = "z" * 64
    with pytest.raises(ValueError): audit.baseline_check(audit.Audit(), report, old)


def curve_fixture():
    length, passes = 16, 32
    plan = {"length": length, "physical_batch_per_rank": 1, "deep_updates": [0], "deep_passes": passes,
            "shallow_passes": 8, "panel": {"index_manifest_sha256": "a" * 64,
            "fixed_plan": {"updates": [{"membership_sha256": "b" * 64,
                "counts": {"valid_tokens": 32, "ce_targets": 30, "packed_rows": 2},
                "allocation_by_arm": {"F": [{"valid_tokens": 16, "packed_rows": 1, "physical_rows": 1,
                    "padding_tokens": 0, "dummy_rows": 0, "microbatches": 1}] * 2}}]}}}
    physical = {"schema": audit.PROBE_SCHEMA, "policy": "common_fp32_no_jitter_v1", "beta": 1.,
                "input_tokens": 16, "ce_targets": 15, "passes": []}
    aggregate = {"schema": audit.PROBE_SCHEMA, "policy": physical["policy"], "beta": 1.,
                 "input_tokens": 32, "ce_targets": 30, "passes": []}
    for p in range(1, passes + 1):
        local_regions, total_regions = {}, {}
        for region in audit.REGIONS:
            # Independent explicit coordinate masks, not auditor _region_bounds.
            positions = list(range(length))
            if region.startswith("quarter_"):
                q = int(region[-1]) - 1; positions = positions[q * 4:(q + 1) * 4]
            elif region == "tail_128": positions = positions[-128:]
            elif region == "unsettled_suffix": positions = [i for i in positions if i >= p - 1]
            n, c = len(positions), len([i for i in positions if i < length - 1]); d = n if p > 1 else 0
            values = {"positions": n, "ce_targets": c, "difference_positions": d,
                "hidden_square_sum": 9. * n, "pre_norm_square_sum": 16. * n, "input_square_sum": .01 * n,
                "delta_square_sum": .25 * d, "previous_square_sum": 4. * d, "cosine_sum": .5 * d,
                "ce_sum": 2. * c, "entropy_sum": .8 * c}
            local_regions[region] = values
            sums = {k: 2 * v for k, v in values.items()}
            total_regions[region] = {"sums": sums, "metrics": {
                "hidden_rms": 3. if n else None, "pre_norm_rms": 4. if n else None,
                "input_rms": .1 if n else None, "delta_rms": .5 if d else None,
                "relative_delta_rms": .25 if d else None, "cosine": .5 if d else None,
                "ce": 2. if c else None, "entropy": sums["entropy_sum"] / (2 * c) if c else None}}
        physical["passes"].append({"pass": p, "regions": local_regions})
        aggregate["passes"].append({"pass": p, "regions": total_regions})
    preservation = {"checks": {key: True for key in audit.PRESERVATION_CHECKS}, "restored": True,
                    "integrity_passed": True, "precision": "fp32_math_eager", "feedback_jitter": 0.}
    part = {"rows": [physical], "preservation": preservation, "accounting": {
        "valid_tokens": 16, "ce_targets": 15, "packed_rows": 1, "physical_rows": 1, "padding_tokens": 0, "empty_rows": 0}}
    event = {"schema": audit.PROBE_SCHEMA, "after_update": 0, "num_passes": passes, "panel": "dev-main",
        "status": "completed", "training_boundary_exact_by_rank": [True, True], "total_seconds": 1.,
        "membership_sha256": "b" * 64, "index_manifest_sha256": "a" * 64,
        "by_rank": [deepcopy(part), deepcopy(part)], "result": aggregate}
    return event, plan


def test_curves_recompute_exact_metrics_and_exclude_the_settled_prefix():
    event, plan = curve_fixture(); before = deepcopy(event)
    audit.probe_event_check(audit.Audit(), event, plan, "probe")
    assert event == before
    assert event["result"]["passes"][15]["regions"]["unsettled_suffix"]["sums"]["positions"] == 2
    assert event["result"]["passes"][16]["regions"]["unsettled_suffix"]["metrics"]["relative_delta_rms"] is None


def scheduled_curves():
    event, plan = curve_fixture()
    plan.update(schema=audit.PROBE_SCHEMA, regions=list(audit.REGIONS), precision="common_fp32_no_jitter_v1",
        world_size=2, panel_rows=2, scheduled_updates=[0, 2, 3],
        acceptance_schedule="origin-live-graph-next-update-terminal-v1")
    events = [event]
    for number in (2, 3):
        row = deepcopy(event); row.update(after_update=number, num_passes=8)
        row["result"]["passes"] = row["result"]["passes"][:8]
        for part in row["by_rank"]:
            for batch in part["rows"]: batch["passes"] = batch["passes"][:8]
        events.append(row)
    report = {"scale": "tiny", "stability_probe_policy": plan, "stability_probes": events,
              "loop": {"start_update": 0, "completed_update": 3}}
    payload = {"probe_plan": deepcopy(plan), "recipe": {"sequence_length": 16}, "plan": {"updates": [{}, {}, {}]}}
    return report, payload


def test_tiny_acceptance_observes_a_live_graph_and_subsequent_update():
    report, payload = scheduled_curves()
    audit.probe_check(audit.Audit(), report, payload)
    resumed = deepcopy(report); resumed["loop"]["start_update"] = 2
    resumed["stability_probes"] = resumed["stability_probes"][1:]
    audit.probe_check(audit.Audit(), resumed, payload)


@pytest.mark.parametrize("mutation", ["omit_live_graph", "omit_terminal", "invent_boundary"])
def test_tiny_coverage_cannot_be_reduced_to_only_origin(mutation):
    report, payload = scheduled_curves()
    if mutation == "omit_live_graph": report["stability_probes"].pop(1)
    elif mutation == "omit_terminal": report["stability_probes"].pop()
    else: report["stability_probes"][1]["after_update"] = 1
    with pytest.raises(ValueError): audit.probe_check(audit.Audit(), report, payload)


@pytest.mark.parametrize("mutation", ["state", "RNG", "precision", "noise", "membership", "index",
    "pass_count", "tail_count", "settled_prefix", "wrong_denominator", "rank_mean", "fake_CE",
    "raw_count", "negative_square", "nonfinite", "first_difference", "invented_scope"])
def test_curve_corruption_and_false_preservation_fail(mutation):
    event, plan = curve_fixture(); part = event["by_rank"][0]
    raw = part["rows"][0]["passes"][1]["regions"]["unsettled_suffix"]
    metrics = event["result"]["passes"][1]["regions"]["unsettled_suffix"]["metrics"]
    if mutation == "state": event["training_boundary_exact_by_rank"][0] = False
    elif mutation == "RNG": part["preservation"]["checks"]["rng_restored"] = False
    elif mutation == "precision": part["preservation"]["precision"] = "bf16"
    elif mutation == "noise": part["preservation"]["feedback_jitter"] = .02
    elif mutation == "membership": event["membership_sha256"] = "c" * 64
    elif mutation == "index": event["index_manifest_sha256"] = "c" * 64
    elif mutation == "pass_count": event["num_passes"] = 4
    elif mutation == "tail_count": part["rows"][0]["passes"][1]["regions"]["tail_128"]["positions"] -= 1
    elif mutation == "settled_prefix": raw["positions"] += 1
    elif mutation == "wrong_denominator": metrics["relative_delta_rms"] = .5
    elif mutation == "rank_mean": event["result"]["passes"][1]["regions"]["all"]["sums"]["ce_sum"] /= 2
    elif mutation == "fake_CE": metrics["ce"] = 0.
    elif mutation == "raw_count": part["rows"][0]["ce_targets"] -= 1
    elif mutation == "negative_square": raw["delta_square_sum"] = -1.
    elif mutation == "nonfinite": raw["delta_square_sum"] = math.nan
    elif mutation == "first_difference": part["rows"][0]["passes"][0]["regions"]["all"]["delta_square_sum"] = 1.
    elif mutation == "invented_scope": raw["max_error"] = 0.
    with pytest.raises(ValueError): audit.probe_event_check(audit.Audit(), event, plan, "probe")


def test_training_validator_is_literal_accepted_validator_with_new_schema():
    def function(path, name):
        tree = ast.parse(path.read_text())
        return ast.dump(next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name), include_attributes=False)
    assert function(audit.ROOT / "scripts/olmo_pilot_async_audit.py", "training_check") == function(Path(audit.__file__), "training_check")


def test_import_is_JSON_only_and_does_not_initialize_torch_or_cloud():
    subprocess.run([sys.executable, "-c", "import sys; import scripts.olmo_fbt_stability_audit; assert 'torch' not in sys.modules; assert 'google.cloud.storage' not in sys.modules"],
                   cwd=audit.ROOT, check=True, capture_output=True, text=True)


def insertion_pair():
    from test_campaign_execution_audit import fixture
    old = fixture(arm="F")
    old["evaluations"] = []
    payload = old["configuration"]["execution_identity"]["payload"]
    payload["declaration"] = {"schema": audit.accepted_async.TINY_SCHEMA, "arm": "F", "data": {"pin": "a" * 64},
        "seed": 20260929, "evaluation": {"every_updates": 2}}
    current = deepcopy(old)
    p = current["configuration"]["execution_identity"]["payload"]
    p["declaration"]["schema"] = audit.TINY_SCHEMA
    p.update(probe_plan={"enabled": True}, checkpoint_updates=[], scope="new-explicit-scope")
    p["sources"]["scripts/olmo_fbt_stability_probe.py"] = "b" * 64
    return old, current


def test_probe_insertion_keeps_every_recorded_update_exact():
    old, current = insertion_pair()
    audit.paired_state_check(audit.Audit(), old, current, kind="insertion")


@pytest.mark.parametrize("mutation", ["state", "Adam", "gradient", "input", "RNG", "schedule", "seed", "historical_source"])
def test_probe_insertion_cannot_change_training(mutation):
    old, current = insertion_pair()
    row = current["updates"]["2"][0]
    if mutation == "state": row["boundary"]["state"]["model"]["weight"]["sha256"] = "0" * 64
    elif mutation == "Adam": row["boundary"]["state"]["optimizer"]["state"] = {}
    elif mutation == "gradient": row["raw_gradients"]["weight"]["sha256"] = "0" * 64
    elif mutation == "input": row["input"]["batches"]["sha256"] = "0" * 64
    elif mutation == "RNG": row["boundary"]["rng"]["python"] = [999]
    elif mutation == "schedule": row["metrics"]["lr_used"] = [.2]
    elif mutation == "seed": current["configuration"]["execution_identity"]["payload"]["declaration"]["seed"] += 1
    elif mutation == "historical_source": current["configuration"]["execution_identity"]["payload"]["sources"]["scripts/example.py"] = "0" * 64
    with pytest.raises(ValueError): audit.paired_state_check(audit.Audit(), old, current, kind="insertion")
