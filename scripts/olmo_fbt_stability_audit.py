#!/usr/bin/env python3
"""Independent JSON-only F-only startup, trajectory and diagnostic evidence audit.

Reuse the accepted training validator literally with this report schema. New
checks establish clean F ownership and lineage rather than numerical or quality
claims. Historical sources and input reports are never rewritten.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_pilot_execution_audit import (
    Audit, legacy, publication_metadata, expected_counts, expected_cursor,
    file_sha, bounded_json, digest, finite_json, boundary_check,
    OBSERVATION_SCHEMA, TERMS, HEAVY, IDENTITY_SCHEMA, CONFIG_SCHEMA)
from scripts import olmo_pilot_execution_audit as evaluation_audit
from scripts import olmo_pilot_async_audit as accepted_async
from scripts.olmo_pilot_async_audit import transport_check
from scripts.olmo_pilot_execution_audit_v2 import training_parity
SCHEMA = "olmo-fbt-stability-audit-v1"
REPORT_SCHEMA = "olmo-fbt-stability-execute-report-v1"
TINY_SCHEMA = "olmo-fbt-stability-tiny-acceptance-v1"
PROBE_SCHEMA = "olmo-fbt-stability-probe-v1"
REGIONS = ("all", "quarter_1", "quarter_2", "quarter_3", "quarter_4", "tail_128", "unsettled_suffix")
SUM_FIELDS = ("hidden_square_sum", "pre_norm_square_sum", "input_square_sum", "delta_square_sum",
              "previous_square_sum", "cosine_sum", "ce_sum", "entropy_sum")
COUNT_FIELDS = ("positions", "ce_targets", "difference_positions")
PRESERVATION_CHECKS = {"tensor_metadata_unchanged", "module_ownership_unchanged", "cache_generations_unchanged",
    "gradient_identity_unchanged", "gradient_values_remained_zero", "runtime_restored", "modes_restored",
    "rng_restored", "autocast_cache_restored"}
AUDIT_SOURCES = ("scripts/olmo_fbt_stability_audit.py",
    "scripts/olmo_pilot_async_audit.py", "scripts/olmo_pilot_execution_audit.py",
    "scripts/olmo_pilot_execution_audit_v2.py", "scripts/olmo_campaign_execution_audit.py")


def model_check(audit, payload, label="model", *, native=True):
    """Require actual F-only ownership, not just a requested arm label."""
    contract = payload["model_contract"]
    recipe = payload["recipe"]
    for field in (payload, recipe, contract):
        audit.equal(label + "/arm", field["arm"], "F")
    audit.equal(label + "/weights", contract["weights"], {"ce": 1., "latent": 0., "kl": 0.})
    mode = contract["mode"]
    audit.equal(label + "/feedback", (mode["enabled"], mode["num_passes"], mode["beta"]), (True, 4, 1.))
    audit.equal(label + "/no_temporal_RT", mode["rt_mode"], {"selected_layers": [], "alpha": 1.})
    audit.equal(label + "/feedback_jitter", mode["feedback_jitter"], recipe["feedback_jitter"])
    audit.equal(label + "/K4", recipe["fbt_passes"], 4)
    audit.equal(label + "/pass_reduction", recipe["pass_loss_policy"], "campaign_v1")
    audit.equal(label + "/tied_readout", contract["tied_readout"], True)
    audit.equal(label + "/no_dormant_fusion", contract["dormant_fusion_parameters"], 0)
    layout = contract["parameter_layout"]
    names = [row["name"] for row in layout]
    audit.require(label + "/unique_parameters", bool(names) and len(names) == len(set(names)))
    audit.require(label + "/all_active_FP32", all(row["requires_grad"] is True and row["dtype"] == "torch.float32" for row in layout))
    audit.require(label + "/only_core_parameters", all(name.startswith(("backbone.backbone.", "backbone.fusion.")) for name in names))
    fusion = {name for name in names if name.startswith("backbone.fusion.")}
    audit.equal(label + "/fusion_matrices", fusion, {"backbone.fusion.state_proj.weight", "backbone.fusion.token_gate.weight"})
    ownership = [name for group in contract["optimizer_ownership"] for name in group]
    audit.require(label + "/unique_optimizer_ownership", len(ownership) == len(set(ownership)))
    audit.equal(label + "/complete_optimizer_ownership", set(ownership), set(names))
    components = {"backbone": 0, "fusion": 0, "predictor": 0}
    for row in layout:
        shape = row["shape"]
        audit.require(label + "/positive_shape/" + row["name"], isinstance(shape, list) and bool(shape)
                      and all(type(value) is int and value > 0 for value in shape))
        key = "fusion" if row["name"] in fusion else "backbone"
        components[key] += math.prod(shape)
    audit.equal(label + "/component_counts", contract["component_parameters"], components)
    total = sum(components.values())
    audit.equal(label + "/all_resident_trainable", (contract["resident_parameters"], contract["trainable_parameters"]), (total, total))
    if native:
        audit.equal(label + "/native_component_counts", components,
                    {"backbone": 1_176_764_416, "fusion": 8_388_608, "predictor": 0})
        audit.equal(label + "/T1024", recipe["sequence_length"], 1024)
        audit.equal(label + "/effective_batch", recipe["effective_valid_tokens"], 524288)


def baseline_check(audit, report, baseline, label="shared_origin"):
    """Verify the clean F origin and unchanged logical prefix against retained NF.

    Physical allocation, optimizer group membership and the finite plan identity
    may differ. Data membership/order, loss target masks, token LR, keyed noise,
    and common initialized tensors may not. This is not trajectory equality.
    """
    payload = report["configuration"]["execution_identity"]["payload"]
    previous = baseline["configuration"]["execution_identity"]["payload"]
    audit.equal(label + "/reference_arm", baseline["arm"], "NF")
    audit.equal(label + "/reference_update_zero_origin", baseline["loop"]["start_update"], 0)
    audit.equal(label + "/native_source", report["configuration"]["source_checkpoint"], baseline["configuration"]["source_checkpoint"])
    audit.equal(label + "/data", payload["data"], previous["data"])
    audit.equal(label + "/recipe_except_arm", {k: v for k, v in payload["recipe"].items() if k != "arm"},
                {k: v for k, v in previous["recipe"].items() if k != "arm"})
    audit.equal(label + "/model_mode", payload["model_contract"]["mode"], previous["model_contract"]["mode"])
    old_plan, new_plan = previous["plan"], payload["plan"]
    audit.require(label + "/full_original_plan_prefix", len(new_plan["updates"]) >= len(old_plan["updates"]))
    audit.equal(label + "/first_cursor", new_plan["first_cursor"], old_plan["first_cursor"])
    for number, old_row in enumerate(old_plan["updates"]):
        audit.equal(label + f"/update{number + 1}/logical_plan",
                    {k: v for k, v in new_plan["updates"][number].items() if k != "allocation_by_arm"},
                    {k: v for k, v in old_row.items() if k != "allocation_by_arm"})
    n = len(old_plan["updates"]) + 1
    for key in ("valid_token_prefix", "lr_at_completed_boundaries"):
        audit.equal(label + "/schedule_prefix/" + key, payload["schedule"][key][:n], previous["schedule"][key])
    audit.equal(label + "/fusion_import_receipt", report["startup_import"]["receipt"], baseline["startup_import"]["receipt"])
    # A child-resume origin has learned weights. Its initial-state proof belongs
    # to the separately retained segment beginning at zero.
    if report.get("resume", {}).get("completed_update", 0) == 0:
        for rank, boundary in enumerate(report["origin_boundary_by_rank"]):
            old = baseline["origin_boundary_by_rank"][rank]
            audit.equal(label + f"/rank{rank}/core_weights", boundary["state"]["model"],
                        {k: v for k, v in old["state"]["model"].items() if not k.startswith("predictor.")})
            audit.equal(label + f"/rank{rank}/fresh_Adam", boundary["state"]["optimizer"]["state"], {})
            audit.equal(label + f"/rank{rank}/RNG", boundary["rng"], old["rng"])
    common_steps = set(report.get("updates", {})) & set(baseline.get("updates", {}))
    for step in sorted(common_steps, key=int):
        actual, old = report["updates"][step][0]["metrics"], baseline["updates"][step][0]["metrics"]
        for key in ("lr_used", "lr_next"):
            audit.equal(label + f"/update{step}/" + key, set(actual[key]), set(old[key]))
        # The cohort uses the same stable ordered membership and document masks;
        # physical padding and microbatch counts may legitimately change.
        for rank, row in enumerate(report["observations"][step]["rank_data"]):
            old_row = baseline["observations"][step]["rank_data"][rank]
            for key in ("valid_tokens", "ce_targets", "latent_pairs", "kl_triples", "packed_rows"):
                audit.equal(label + f"/update{step}/rank{rank}/" + key, row[key], old_row[key])

def training_check(audit, report, label, source_root=None):
    check = lambda name, condition: audit.require(label+'/'+name, condition)
    equal = lambda name, actual, expected: audit.equal(label+'/'+name, actual, expected)
    check('finite_json', finite_json(report))
    equal('schema', report['schema'], REPORT_SCHEMA)
    check('closed_success', report['status'] in ('completed_plan', 'stopped_at_boundary'))
    configuration = report['configuration']; identity = configuration['execution_identity']; payload = identity['payload']
    equal('identity_schema', identity['schema'], IDENTITY_SCHEMA)
    equal('identity_digest', identity['sha256'], digest(payload))
    equal('arm', report['arm'], payload['arm'])
    check('known_arm', payload['arm'] in ('B','N','F','R','NF','NR','FR','NFR'))
    equal('two_ranks', payload['partition']['world_size'], 2)
    equal('configuration_recipe', configuration['recipe'], payload['recipe'])
    equal('configuration_ownership', configuration['parameters'], payload['model_contract'])
    equal('configuration_precision', configuration['training']['precision'], payload['execution']['precision'])
    equal('sources', report['sources'], payload['sources'])
    check('nonempty_sources', isinstance(report['sources'], dict) and bool(report['sources']))
    fingerprint = report['source_fingerprint']
    equal('fingerprint_identity', fingerprint['execution_identity_sha256'], identity['sha256'])
    equal('fingerprint_digest', fingerprint['sha256'], identity['sha256'])
    equal('fingerprint_scope', fingerprint['scope'], 'execution_identity')
    equal('fingerprint_sources', fingerprint['sources'], report['sources'])
    equal('fingerprint_source', fingerprint['source_checkpoint'], configuration['source_checkpoint'])
    for name, pin in report['sources'].items():
        path = Path(name)
        check('source_safe/'+name, not path.is_absolute() and '..' not in path.parts and bool(name)
              and isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin) is not None)
        if source_root is not None:
            candidate = Path(source_root)/path
            check('source_snapshot/'+name, candidate.is_file() and not candidate.is_symlink() and file_sha(candidate) == pin)
    final = report['final_counters']['optimizer_updates']; start = report.get('resume', {}).get('completed_update', 0)
    check('integer_final_counters', all(type(v) is int for v in report['final_counters'].values()))
    expected = expected_counts(payload, final)
    equal('final_counters', report['final_counters'], expected)
    check('valid_segment', type(start) is int and 0 <= start <= final <= len(payload['plan']['updates']))
    equal('loop_start', report['loop']['start_update'], start)
    equal('loop_final', report['loop']['completed_update'], final)
    equal('plan_completed', report['plan_completed'], final == len(payload['plan']['updates']))
    equal('status_matches_plan', report['status'] == 'completed_plan', report['plan_completed'])
    updates = report.get('updates', {})
    equal('complete_update_membership', set(updates), {str(i) for i in range(start+1, final+1)})
    equal('graph_prepared', report['graph_prepared'], final > start)
    if report['graph_prepared']:
        equal('preparation_exact', report['preparation_boundary_exact'], [True]*2)
        check('runner_metadata', len(report['runner_by_rank']) == 2 and all(isinstance(x, dict) and x for x in report['runner_by_rank']))
    else:
        check('no_preparation_evidence', 'preparation_boundary_exact' not in report)
        equal('no_captured_runners', report['runner_by_rank'], [None, None])
    if 'resume' in report:
        equal('resume_no_reference_dependency', report['resume']['reference_report_required'], False)
        check('resume_pin', re.fullmatch('[0-9a-f]{64}', report['resume']['manifest_sha256']) is not None)
    boundary_check(audit, label+'/origin', report['origin_boundary_by_rank'], payload, start)
    boundary_check(audit, label+'/final', report['final_boundary_by_rank'], payload, final)
    mode = report['observation_mode']; check('observer_mode', mode in ('lean', 'acceptance'))
    active = {p['name'] for p in payload['model_contract']['parameter_layout'] if p['requires_grad']}
    check('active_parameters', bool(active))
    for label_clock, completed in (('origin_clocks',start),('final_clocks',final)):
        equal(label_clock, report[label_clock], {'adam_parameters':len(active) if completed else 0,
              'input_tokens':expected_counts(payload,completed)['input_tokens'],
              'optimizer_updates':completed,'scheduler_epoch':completed})
    equal('complete_accounting_membership', set(report.get('observations',{})), set(updates))
    for step in range(start+1, final+1):
        rank_data=report['observations'][str(step)]['rank_data']
        check(f'update{step}/accounting_ranks', isinstance(rank_data,list) and len(rank_data)==2)
        plan_row=payload['plan']['updates'][step-1]
        for key, value in plan_row['counts'].items():
            equal(f'update{step}/data_total/{key}',sum(row[key] for row in rank_data),value)
        for rank, allocated in enumerate(plan_row['allocation_by_arm'][payload['arm']]):
            for actual_key, planned_key in (('valid_tokens','valid_tokens'),('packed_rows','packed_rows'),
                    ('physical_rows','physical_rows'),('padding_tokens','padding_tokens'),('empty_rows','dummy_rows')):
                equal(f'update{step}/allocation/rank{rank}/{actual_key}',rank_data[rank][actual_key],allocated[planned_key])
        rows = updates[str(step)]; check(f'update{step}/rank_count', isinstance(rows, list) and len(rows) == 2)
        if mode == 'acceptance':
            boundary_check(audit, label+f'/update{step}/boundary', [row['boundary'] for row in rows], payload, step)
        before, after = expected_counts(payload, step-1), expected_counts(payload, step)
        for rank, row in enumerate(rows):
            prefix = f'update{step}/rank{rank}/'
            equal(prefix+'observation_fields',set(row), {'schema','mode','metrics','cursor','loss_means','clipping','observation_seconds'}
                  | (set(HEAVY) if mode=='acceptance' else set()))
            equal(prefix+'schema', row['schema'], OBSERVATION_SCHEMA)
            equal(prefix+'mode', row['mode'], mode)
            metrics = row['metrics']
            check(prefix+'complete_metrics', isinstance(metrics,dict) and
                  {'loss_sums','counts','objective','microbatches','documents','input_tokens',
                   'gradient_norm_before_clip','lr_used','lr_next','counters'} <= metrics.keys())
            check(prefix+'lr_groups', isinstance(metrics['lr_used'],list) and bool(metrics['lr_used'])
                  and isinstance(metrics['lr_next'],list) and len(metrics['lr_used']) == len(metrics['lr_next'])
                  and all(type(v) in (int,float) and v >= 0 for key in ('lr_used','lr_next') for v in metrics[key]))
            equal(prefix+'replica_metrics', metrics, rows[0]['metrics'])
            equal(prefix+'counters', metrics['counters'], after)
            equal(prefix+'cursor', row['cursor'], expected_cursor(payload, step, rank))
            for key in ('input_tokens','documents','microbatches'):
                equal(prefix+key, metrics[key], after[key]-before[key])
            counts = {term: after[key]-before[key] for term,key in zip(TERMS,('ce_positions','latent_pairs','kl_triples'))}
            equal(prefix+'loss_counts', metrics['counts'], counts)
            equal(prefix+'loss_terms', set(metrics['loss_sums']), set(TERMS))
            equal(prefix+'loss_means', row['loss_means'], {term: metrics['loss_sums'][term]/counts[term] if counts[term] else None for term in TERMS})
            limit = configuration['training']['max_grad_norm']; norm = metrics['gradient_norm_before_clip']
            check(prefix+'gradient_norm', type(norm) in (int,float) and norm >= 0)
            clipping = row['clipping']
            equal(prefix+'clip_limit', clipping['configured_limit'], limit)
            equal(prefix+'clip_exceeded', clipping['norm_exceeds_limit'], limit is not None and norm > limit)
            equal(prefix+'clip_coefficient', clipping['coefficient_estimate'], 1.0 if limit is None else min(1.0,limit/(norm+1e-6)))
            if mode == 'lean':
                check(prefix+'no_heavy_evidence', not set(HEAVY)&row.keys())
            else:
                check(prefix+'input_evidence', isinstance(row['input'], dict) and bool(row['input']))
                equal(prefix+'all_active_gradients', set(row['raw_gradients']), active)
                check(prefix+'gradient_digests', all(isinstance(v, dict) and {'shape','dtype','sha256'} == set(v)
                      and re.fullmatch('[0-9a-f]{64}', v['sha256']) is not None for v in row['raw_gradients'].values()))
                equal(prefix+'replica_gradients', row['raw_gradients'], rows[0]['raw_gradients'])
    if mode == 'acceptance' and final > start:
        equal('last_update_final_boundary', [row['boundary'] for row in updates[str(final)]], report['final_boundary_by_rank'])
    if final == start:
        equal('no_update_state_change', report['final_boundary_by_rank'], report['origin_boundary_by_rank'])
    return payload


def evaluation_check(audit, report, label):
    policy = report['evaluation_policy']; plan = policy['plan']; declaration = plan['declaration']
    payload = report['configuration']['execution_identity']['payload']
    audit.equal(label+'/resume_policy', policy['resume'], 'repeat_scheduled_restored_boundary_once_per_segment')
    audit.equal(label+'/failure_policy', policy['failure'],
                'earlier committed checkpoint remains authoritative; no save of unverified state')
    audit.equal(label+'/training_batch', policy['training_physical_batch_per_rank'], payload['partition']['physical_batch_per_rank'])
    if report['scale'] == 'native':
        audit.equal(label+'/bound_evaluation', payload['evaluation'], plan)
    else:
        audit.equal(label+'/tiny_schema', payload['declaration']['schema'], TINY_SCHEMA)
        audit.equal(label+'/tiny_evaluation', payload['declaration']['evaluation'], declaration)
        audit.equal(label+'/tiny_declaration_digest', payload['resolved_contract_sha256'], digest(payload['declaration']))
    if declaration['kind'] == 'deferred':
        audit.equal(label+'/deferred_fields', set(declaration), {'kind','reason'})
        audit.require(label+'/deferred_reason', isinstance(declaration['reason'],str) and bool(declaration['reason'].strip()))
        audit.equal(label+'/deferred_plan', set(plan), {'declaration','execution'})
        audit.equal(label+'/deferred_evaluations', report['evaluations'], [])
        audit.equal(label+'/deferred_batch', policy['evaluation_physical_batch_per_rank'], None)
        return
    audit.equal(label+'/policy_fields', set(declaration), {'kind','panels','physical_batch_by_arm','every_updates',
        'precision','feedback_jitter','report_passes','generation'})
    for key,value in {'kind':'ordered_named_dev_panels_v1','precision':'fp32','feedback_jitter':0,
                      'report_passes':'all_trained_passes','generation':'not_implemented'}.items():
        audit.equal(label+'/'+key, declaration[key], value)
    audit.equal(label+'/plan_schema', plan['schema'], 'olmo-pilot-evaluation-control-v1')
    names = [row['name'] for row in declaration['panels']]
    allowed = {'dev-main'} | {'dev-source/'+s for s in ('books','c4','cc_en_head','cc_en_middle','cc_en_tail','pes2o','reddit','stack','wiki')}
    audit.require(label+'/unique_dev_only', bool(names) and len(names)==len(set(names)) and set(names)<=allowed)
    audit.equal(label+'/declared_panel_inventory', set(plan['panels']), set(names))
    interval = declaration['every_updates']
    audit.require(label+'/positive_interval', type(interval) is int and interval>0)
    scheduled = list(range(interval, len(payload['plan']['updates'])+1, interval))
    scheduled = sorted(set(scheduled) | ({0, 100} if report['scale'] == 'native' else {0}))
    audit.equal(label+'/fixed_schedule', plan['scheduled_updates'], scheduled)
    start=report.get('resume',{}).get('completed_update',0); final=report['final_counters']['optimizer_updates']
    audit.equal(label+'/actual_schedule', [r['after_update'] for r in report['evaluations']], [s for s in scheduled if start<=s<=final])
    arm = payload['arm']; batch = declaration['physical_batch_by_arm'][arm]
    audit.require(label+'/positive_eval_batch', type(batch) is int and batch>0)
    audit.equal(label+'/eval_batch', policy['evaluation_physical_batch_per_rank'], batch)
    audit.equal(label+'/eval_partition', plan['partition_by_arm'][arm], {'world_size':2, 'physical_batch_per_rank':batch})
    audit.equal(label+'/overlap_policy', plan['overlap_policy'],
                'Report each panel separately; main/source overlap is not independent replication')
    for declared in declaration['panels']:
        name=declared['name']; panel=plan['panels'][name]; scope=label+'/'+name
        audit.require(scope+'/pin', re.fullmatch('[0-9a-f]{64}',panel['index_manifest_sha256']) is not None)
        audit.equal(scope+'/budget', panel['target_valid_tokens'], declared['target_valid_tokens'])
        audit.require(scope+'/positive_budget', type(panel['target_valid_tokens']) is int and panel['target_valid_tokens']>0)
        fixed=panel['fixed_plan']; audit.equal(scope+'/one_update',len(fixed['updates']),1)
        audit.equal(scope+'/origin',fixed['first_cursor'],{'manifest_sha256':panel['index_manifest_sha256'], 'split':'dev','next_chunk':0,'next_update':0})
        audit.equal(scope+'/planned_budget', fixed['updates'][0]['target_valid_tokens'], panel['target_valid_tokens'])
        audit.equal(scope+'/selection',panel['selection'],'fixed_ordered_prefix_from_chunk_zero')
        audit.require(scope+'/available_budget', panel['available_panel_tokens']>=panel['target_valid_tokens'])
        audit.equal(scope+'/allocation_rank_count', len(fixed['updates'][0]['allocation_by_arm'][arm]), 2)
        for rank, row in enumerate(fixed['updates'][0]['allocation_by_arm'][arm]):
            audit.equal(scope+f'/rank{rank}/batch_product',row['physical_rows'],row['microbatches']*batch)
    for row in report['evaluations']:
        scope=label+'/update'+str(row['after_update'])
        audit.equal(scope+'/schema',row['schema'],'olmo-pilot-evaluation-control-v1')
        audit.equal(scope+'/completed',row['status'],'completed')
        audit.equal(scope+'/training_boundary_preserved',row['training_boundary_exact_by_rank'],[True,True])
        audit.equal(scope+'/panels',set(row['panels']),set(names))
        for name, entry in row['panels'].items():
            panel=plan['panels'][name]
            audit.equal(scope+'/'+name+'/index_pin',entry['index_manifest_sha256'],panel['index_manifest_sha256'])
            audit.equal(scope+'/'+name+'/membership_pin',entry['membership_sha256'],panel['fixed_plan']['updates'][0]['membership_sha256'])
            evaluation_audit.panel_check(audit,entry,panel,payload,scope+'/'+name)


def _region_bounds(name, length, total_pass):
    if name == "all":
        return 0, length
    if name.startswith("quarter_"):
        quarter = int(name[-1]) - 1
        return quarter * length // 4, (quarter + 1) * length // 4
    if name == "tail_128":
        return max(0, length - 128), length
    if name == "unsettled_suffix":
        return min(length, total_pass - 1), length
    raise ValueError("Unknown stability region")


def probe_event_check(audit, event, plan, label):
    """Recompute derived metrics from rank-local sums without importing torch."""
    equal = lambda name, actual, expected: audit.equal(label + "/" + name, actual, expected)
    check = lambda name, condition: audit.require(label + "/" + name, condition)
    number = event["after_update"]
    count = plan["deep_passes"] if number in plan["deep_updates"] else plan["shallow_passes"]
    equal("schema", event["schema"], PROBE_SCHEMA)
    equal("completed", event["status"], "completed")
    equal("passes", event["num_passes"], count)
    equal("panel", event["panel"], "dev-main")
    equal("training_boundary", event["training_boundary_exact_by_rank"], [True, True])
    check("finite_wall", type(event["total_seconds"]) in (int, float) and event["total_seconds"] >= 0)
    panel = plan["panel"]
    fixed = panel["fixed_plan"]["updates"][0]
    equal("membership", event["membership_sha256"], fixed["membership_sha256"])
    equal("index", event["index_manifest_sha256"], panel["index_manifest_sha256"])
    equal("rank_count", len(event["by_rank"]), 2)
    physical_rows = []
    for rank, part in enumerate(event["by_rank"]):
        scope = f"rank{rank}/"
        preservation = part["preservation"]
        equal(scope + "preservation_check_membership", set(preservation["checks"]), PRESERVATION_CHECKS)
        check(scope + "all_preserved", preservation["restored"] is True
              and preservation["integrity_passed"] is True
              and all(value is True for value in preservation["checks"].values()))
        equal(scope + "precision", preservation["precision"], "fp32_math_eager")
        equal(scope + "no_jitter", preservation["feedback_jitter"], 0.)
        allocation = fixed["allocation_by_arm"]["F"][rank]
        equal(scope + "physical_batch_count", len(part["rows"]), allocation["microbatches"])
        for actual, planned in (("valid_tokens", "valid_tokens"), ("packed_rows", "packed_rows"),
                                ("physical_rows", "physical_rows"), ("padding_tokens", "padding_tokens"),
                                ("empty_rows", "dummy_rows")):
            equal(scope + "allocation/" + actual, part["accounting"][actual], allocation[planned])
        equal(scope + "input_sum", sum(row["input_tokens"] for row in part["rows"]), part["accounting"]["valid_tokens"])
        equal(scope + "target_sum", sum(row["ce_targets"] for row in part["rows"]), part["accounting"]["ce_targets"])
        physical_rows.extend(part["rows"])
    for key, value in fixed["counts"].items():
        equal("accounting/" + key, sum(part["accounting"][key] for part in event["by_rank"]), value)
    length = plan["length"]
    total_tokens = sum(row["input_tokens"] for row in physical_rows)
    total_targets = sum(row["ce_targets"] for row in physical_rows)
    equal("expected_tokens", total_tokens, fixed["counts"]["valid_tokens"])
    equal("expected_targets", total_targets, fixed["counts"]["ce_targets"])
    for index, row in enumerate(physical_rows):
        scope = f"physical{index}/"
        equal(scope + "schema", row["schema"], PROBE_SCHEMA)
        equal(scope + "policy", row["policy"], "common_fp32_no_jitter_v1")
        equal(scope + "beta", row["beta"], 1.)
        equal(scope + "pass_membership", [p["pass"] for p in row["passes"]], list(range(1, count + 1)))
        check(scope + "complete_or_dummy_rows", type(row["input_tokens"]) is int
              and row["input_tokens"] >= 0 and row["input_tokens"] % length == 0)
        actual_rows = row["input_tokens"] // length
        check(scope + "physical_capacity", actual_rows <= plan["physical_batch_per_rank"])
        equal(scope + "CE_within_chunk", row["ce_targets"], actual_rows * (length - 1))
        for p in row["passes"]:
            pass_scope = scope + f"pass{p['pass']}/"
            equal(pass_scope + "regions", set(p["regions"]), set(REGIONS))
            for region, values in p["regions"].items():
                metric_scope = pass_scope + region + "/"
                equal(metric_scope + "raw_membership", set(values), set(SUM_FIELDS) | set(COUNT_FIELDS))
                lo, hi = _region_bounds(region, length, p["pass"])
                n = actual_rows * (hi - lo)
                c = actual_rows * max(0, min(hi, length - 1) - lo)
                equal(metric_scope + "position_count", values["positions"], n)
                equal(metric_scope + "target_count", values["ce_targets"], c)
                equal(metric_scope + "difference_count", values["difference_positions"], n if p["pass"] > 1 else 0)
                check(metric_scope + "valid_sum_values", all(type(values[key]) in (int, float)
                      and math.isfinite(values[key]) and (key == "cosine_sum" or values[key] >= 0) for key in SUM_FIELDS))
                check(metric_scope + "bounded_cosine", abs(values["cosine_sum"]) <= n * 1.00001)
                if p["pass"] == 1:
                    equal(metric_scope + "undefined_first_difference", [values[key] for key in
                          ("delta_square_sum", "previous_square_sum", "cosine_sum")], [0., 0., 0.])
                if n == 0:
                    equal(metric_scope + "empty_region_sums", [values[key] for key in SUM_FIELDS], [0.] * len(SUM_FIELDS))
            for key in (*SUM_FIELDS, *COUNT_FIELDS):
                summed = math.fsum(p["regions"][f"quarter_{q}"][key] for q in range(1, 5))
                check(pass_scope + "quarter_partition/" + key,
                      math.isclose(summed, p["regions"]["all"][key], rel_tol=1e-12, abs_tol=1e-10))
    expected_passes = []
    for index in range(count):
        regions = {}
        for name in REGIONS:
            values = [row["passes"][index]["regions"][name] for row in physical_rows]
            total = {key: math.fsum(value[key] for value in values) for key in SUM_FIELDS}
            total.update({key: sum(value[key] for value in values) for key in COUNT_FIELDS})
            n, d, c = total["positions"], total["difference_positions"], total["ce_targets"]
            metrics = {key + "_rms": math.sqrt(total[key + "_square_sum"] / n) if n else None
                       for key in ("hidden", "pre_norm", "input")}
            metrics.update(delta_rms=math.sqrt(total["delta_square_sum"] / d) if d else None,
                relative_delta_rms=math.sqrt(total["delta_square_sum"] / max(total["previous_square_sum"], d * 1e-24)) if d else None,
                cosine=total["cosine_sum"] / d if d else None,
                ce=total["ce_sum"] / c if c else None, entropy=total["entropy_sum"] / c if c else None)
            regions[name] = {"sums": total, "metrics": metrics}
        expected_passes.append({"pass": index + 1, "regions": regions})
    equal("independently_reduced_statistics", event["result"], {"schema": PROBE_SCHEMA,
        "passes": expected_passes, "input_tokens": total_tokens, "ce_targets": total_targets,
        "beta": 1., "policy": "common_fp32_no_jitter_v1"})


def probe_check(audit, report, payload, label="stability"):
    plan = report["stability_probe_policy"]
    audit.equal(label + "/bound_plan", plan, payload["probe_plan"])
    audit.equal(label + "/schema", plan["schema"], PROBE_SCHEMA)
    audit.equal(label + "/regions", plan["regions"], list(REGIONS))
    audit.equal(label + "/precision", plan["precision"], "common_fp32_no_jitter_v1")
    audit.equal(label + "/length", plan["length"], payload["recipe"]["sequence_length"])
    audit.equal(label + "/world_size", plan["world_size"], 2)
    audit.equal(label + "/pass_counts", (plan["shallow_passes"], plan["deep_passes"]), (8, 32))
    audit.require(label + "/bounded_panel", type(plan["panel_rows"]) is int and 1 <= plan["panel_rows"] <= 8)
    audit.require(label + "/bounded_physical_batch", plan["physical_batch_per_rank"] in (1, 2))
    planned = len(payload["plan"]["updates"])
    deep = [n for n in (0, 32, 64, 96, 100, 128, 192) if n <= planned]
    audit.equal(label + "/deep_schedule", plan["deep_updates"], deep)
    scheduled = sorted({0, *range(8, planned + 1, 8), *deep})
    if report["scale"] == "tiny":
        audit.equal(label + "/tiny_scope", (plan["length"], planned), (16, 3))
        audit.equal(label + "/tiny_live_graph_schedule", plan["acceptance_schedule"],
                    "origin-live-graph-next-update-terminal-v1")
        scheduled = [0, 2, 3]
    audit.equal(label + "/fixed_schedule", plan["scheduled_updates"], scheduled)
    start, end = report["loop"]["start_update"], report["loop"]["completed_update"]
    events = report["stability_probes"]
    audit.equal(label + "/actual_schedule", [e["after_update"] for e in events], [n for n in scheduled if start <= n <= end])
    for event in events:
        probe_event_check(audit, event, plan, label + "/update" + str(event["after_update"]))


def startup_check(audit, report, payload):
    imported = report["startup_import"]
    transition = imported["transition"]
    required = {"core_state_exact", "core_parameter_identities_exact", "core_modes_exact",
        "predictor_absent", "no_predictor_parameters", "all_active_trainable", "tied_readout_preserved",
        "only_document_policy_changed"}
    audit.equal("startup/check_membership", set(transition["checks"]), required)
    audit.require("startup/all_checks", all(value is True for value in transition["checks"].values()))
    audit.equal("startup/no_inherited_Adam", transition["loaded_optimizer"], False)
    audit.equal("startup/target_recipe", transition["target_recipe"], payload["recipe"])
    audit.equal("startup/historical_arm", transition["historical_recipe"]["arm"], "NF")
    audit.equal("startup/historical_document_policy", transition["historical_recipe"].get("document_policy", "isolated-v1"), "isolated-v1")
    receipt = imported["receipt"]
    audit.equal("startup/import_check_membership", set(receipt["checks"]),
                {"complete_fusion_exact", "frozen_state_exact", "module_modes_preserved",
                 "parameter_identities_preserved", "tied_readout_preserved", "trainability_preserved"})
    audit.require("startup/import_preserved", all(value is True for value in receipt["checks"].values()))
    audit.equal("startup/checkpoint_SHA", receipt["checkpoint_sha256"], payload["startup"]["checkpoint"]["sha256"])
    audit.equal("startup/clean_weights_only_kind", payload["startup"]["kind"], "fusion128-fresh-F-all-adam-v1")
    audit.equal("startup/prefix_checks", report["prefix_checks"], {"shared_128_data_membership_exact": True,
        "shared_128_allocation_exact": True, "shared_128_token_prefix_exact": True, "shared_128_lr_prefix_exact": True})


def validate_report(audit, report, *, source_root=None, baseline=None, baseline_sha256=None,
                    check_transport=True):
    payload = training_check(audit, report, "training", source_root)
    audit.equal("configuration/schema", report["configuration"]["schema"], CONFIG_SCHEMA)
    native = report["scale"] == "native"
    model_check(audit, payload, native=native)
    if native:
        audit.require("baseline/provided", isinstance(baseline, dict))
        audit.require("baseline/SHA_provided", isinstance(baseline_sha256, str)
                      and re.fullmatch("[0-9a-f]{64}", baseline_sha256) is not None)
        audit.equal("baseline/declared_SHA", payload["declaration"]["parent_report"]["sha256"], baseline_sha256)
        startup_check(audit, report, payload)
        baseline_check(audit, report, baseline)
        audit.equal("plan/finite_ceiling", len(payload["plan"]["updates"]), 192)
        schedule = payload["schedule"]
        prefix = [expected_counts(payload, n)["input_tokens"] for n in range(193)]
        audit.equal("schedule/token_prefix", schedule["valid_token_prefix"], prefix)
        recipe = payload["recipe"]
        expected_lrs = [recipe["plateau_lr"] * (recipe["warmup_start_fraction"]
            + (1 - recipe["warmup_start_fraction"]) * min(1., tokens / recipe["warmup_tokens"])) for tokens in prefix]
        audit.equal("schedule/independent_LR", schedule["lr_at_completed_boundaries"], expected_lrs)
        for step, rows in report["updates"].items():
            for key, index in (("lr_used", int(step) - 1), ("lr_next", int(step))):
                actual = rows[0]["metrics"][key]
                audit.equal(f"schedule/update{step}/" + key, actual, [expected_lrs[index]] * len(actual))
    else:
        audit.equal("tiny/scale", report["scale"], "tiny")
        audit.equal("tiny/no_native_import", report["startup_import"], {})
    evaluation_check(audit, report, "evaluation")
    probe_check(audit, report, payload)
    if check_transport:
        transport_check(audit, report, "transport")
    return payload


def _result(audit, failure=None, **extra):
    result = audit.result(failure)
    result.update(schema=SCHEMA, audit_sources={name: file_sha(ROOT / name) for name in AUDIT_SOURCES},
        scope="Independent JSON, source, startup, accounting and raw-statistic consistency; no tensor reload, BF16 equivalence, contraction proof or quality claim.",
        **extra)
    return result


def audit_report(report, *, source_root=None, baseline=None, baseline_sha256=None, check_transport=True):
    audit = Audit()
    try:
        validate_report(audit, report, source_root=source_root, baseline=baseline,
                        baseline_sha256=baseline_sha256, check_transport=check_transport)
    except (KeyError, TypeError, ValueError, IndexError, OSError, ZeroDivisionError) as error:
        return _result(audit, f"{type(error).__name__}: {error}")
    return _result(audit)


def paired_state_check(audit, reference, actual, *, kind, publication=None):
    """Exact training-state oracle for probe insertion or same-child restart."""
    for report in (reference, actual):
        audit.equal("pair/tiny_acceptance", (report["scale"], report["observation_mode"]), ("tiny", "acceptance"))
    rp = reference["configuration"]["execution_identity"]["payload"]
    ap = actual["configuration"]["execution_identity"]["payload"]
    if kind == "insertion":
        audit.require("insertion/no_resume", "resume" not in reference and "resume" not in actual)
        exceptions = {"scope", "resolved_contract_sha256", "sources", "evaluation", "declaration", "probe_plan", "checkpoint_updates"}
        audit.equal("insertion/shared_payload_fields", set(ap) - exceptions, set(rp) - exceptions)
        for key in set(ap) - exceptions:
            audit.equal("insertion/shared_payload/" + key, ap[key], rp[key])
        audit.require("insertion/pinned_historical_sources", all(ap["sources"].get(key) == value for key, value in rp["sources"].items()))
        for key in ("data", "seed", "arm", "evaluation"):
            audit.equal("insertion/declaration/" + key, ap["declaration"][key], rp["declaration"][key])
        audit.equal("insertion/old_schema", rp["declaration"]["schema"], accepted_async.TINY_SCHEMA)
        audit.equal("insertion/new_schema", ap["declaration"]["schema"], TINY_SCHEMA)
        audit.equal("insertion/config_except_identity", {k: v for k, v in actual["configuration"].items() if k != "execution_identity"},
                    {k: v for k, v in reference["configuration"].items() if k != "execution_identity"})
    elif kind == "restart":
        audit.require("restart/committed_resume", "resume" in actual and publication is not None)
        for key in ("configuration", "sources", "source_fingerprint", "arm", "scale", "declaration_sha256",
                    "resolved_sha256", "startup_import", "evaluation_policy", "stability_probe_policy"):
            audit.equal("restart/" + key, actual.get(key), reference.get(key))
        manifest, _, identity = publication_metadata(publication)
        audit.equal("restart/published_identity", identity, actual["configuration"]["execution_identity"])
        audit.equal("restart/published_configuration", manifest["metadata"]["configuration"], actual["configuration"])
        audit.equal("restart/manifest", actual["resume"]["manifest_sha256"], publication["manifest_sha256"])
        completed = actual["resume"]["completed_update"]
        audit.equal("restart/counters", manifest["counters"], expected_counts(ap, completed))
        audit.equal("restart/cursors", manifest["rank_cursors"], [expected_cursor(ap, completed, rank) for rank in range(2)])
    else:
        raise ValueError("Unknown paired acceptance kind")
    training_parity(audit, reference, actual)
    prior = {row["after_update"]: row for row in reference["evaluations"]}
    for event in actual["evaluations"]:
        if kind == "insertion" and event["after_update"] == 0:
            continue
        old = prior[event["after_update"]]
        audit.equal("pair/eval_panels", set(event["panels"]), set(old["panels"]))
        for name, panel in event["panels"].items():
            previous = old["panels"][name]
            for key in ("result", "index_manifest_sha256", "membership_sha256"):
                audit.equal(f"pair/eval{event['after_update']}/{name}/" + key, panel[key], previous[key])
            for rank, part in enumerate(panel["by_rank"]):
                for key in ("rows", "accounting", "cursor"):
                    audit.equal(f"pair/eval{event['after_update']}/{name}/rank{rank}/" + key, part[key], previous["by_rank"][rank][key])
    if kind == "restart":
        prior = {row["after_update"]: row for row in reference["stability_probes"]}
        for event in actual["stability_probes"]:
            old = prior[event["after_update"]]
            for key in ("result", "membership_sha256", "index_manifest_sha256", "num_passes"):
                audit.equal(f"restart/probe{event['after_update']}/" + key, event[key], old[key])


def audit_pair(reference, actual, *, kind, publication=None, source_roots=None, check_transport=True):
    audit = Audit(); source_roots = source_roots or {}
    try:
        if kind == "insertion":
            accepted_async.validate_report(audit, reference, "reference", source_roots.get("reference"))
        else:
            validate_report(audit, reference, source_root=source_roots.get("reference"), check_transport=check_transport)
        validate_report(audit, actual, source_root=source_roots.get("actual"), check_transport=check_transport)
        paired_state_check(audit, reference, actual, kind=kind, publication=publication)
    except (KeyError, TypeError, ValueError, IndexError, OSError, ZeroDivisionError) as error:
        return _result(audit, f"{type(error).__name__}: {error}", comparison_kind=kind)
    return _result(audit, comparison_kind=kind)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--kind", choices=("report", "insertion", "restart"), default="report")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--publication-sha256")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.baseline is None) != (args.baseline_sha256 is None):
        parser.error("Baseline report and independent SHA256 must be supplied together")
    for path, pin in ((args.reference, args.reference_sha256), (args.publication, args.publication_sha256)):
        if (path is None) != (pin is None):
            parser.error("Reference/publication paths require independent SHA256 pins")
    if args.kind != "report" and args.reference is None:
        parser.error("Paired comparison requires a reference")
    if args.kind == "restart" and args.publication is None:
        parser.error("Restart comparison requires a pinned publication")
    report, report_pin = bounded_json(args.report, args.report_sha256)
    baseline, baseline_pin = (None, None) if args.baseline is None else bounded_json(args.baseline, args.baseline_sha256)
    extra_pins = {}
    if args.kind == "report":
        result = audit_report(report, source_root=args.source_root, baseline=baseline, baseline_sha256=args.baseline_sha256)
    else:
        reference, extra_pins["reference"] = bounded_json(args.reference, args.reference_sha256)
        publication = None
        if args.publication is not None:
            publication, extra_pins["publication"] = bounded_json(args.publication, args.publication_sha256)
        result = audit_pair(reference, report, kind=args.kind, publication=publication,
                            source_roots={"actual": args.source_root})
    result["inputs"] = {"report": report_pin, **({"baseline": baseline_pin} if baseline_pin else {}), **extra_pins}
    for value in result["inputs"].values():
        if file_sha(value["path"]) != value["sha256"]:
            raise ValueError("Audit input changed while being read")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]), "failures": result["failures"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
