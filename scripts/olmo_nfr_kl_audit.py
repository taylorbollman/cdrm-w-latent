#!/usr/bin/env python3
"""Independent narrow NFR authorization around the accepted KL-pair audit.

The validator and pair comparisons are literal accepted implementations. The
branch scope changes only NF -> NFR at native update32 ->64; a separately pinned
NFR declaration must be bound through all 215 execution source identities.
There is no report relabeling, module-global patching or tensor execution.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import olmo_kl_continuation_audit as original
from scripts.olmo_kl_continuation_audit import (Audit, legacy, publication_metadata,
    training_check, _parent_check, _same_except_kl, evaluation_check,
    _raw_evaluation, expected_counts, transport_check, training_parity,
    file_sha, bounded_json, _recipe_check, _configuration_receipt,
    BRANCH_SCHEMA, CONFIG_SCHEMA)

SCHEMA = "olmo-nfr-kl-audit-v1"
SCOPE_SCHEMA = "olmo-nfr-kl-scope-v1"
AUDIT_SOURCES = ("scripts/olmo_nfr_kl_audit.py", "scripts/olmo_kl_continuation_audit.py",
    "scripts/olmo_kl_continuation_audit_v2.py", "scripts/olmo_pilot_async_audit.py",
    "scripts/olmo_pilot_execution_audit.py", "scripts/olmo_pilot_execution_audit_v2.py",
    "scripts/olmo_campaign_execution_audit.py")


def declaration_check(audit, scope, scope_sha256, source_pins, label="scope"):
    audit.require(label + "/provided", isinstance(scope, dict) and isinstance(source_pins, dict))
    audit.equal(label + "/fields", set(scope), {"schema", "activation", "arm", "parent_update", "review_stop",
        "planned_updates", "kl_weights", "declaration", "resolved", "parent_report", "parent_checkpoint", "storage_prefix"})
    expected = {"schema": SCOPE_SCHEMA, "activation": "conditional_after_fbt_only_assessment", "arm": "NFR",
        "parent_update": 32, "review_stop": 64, "planned_updates": 128, "kl_weights": [1., .1]}
    for key, value in expected.items():
        audit.equal(label + "/" + key, scope[key], value)
    audit.require(label + "/SHA", isinstance(scope_sha256, str) and re.fullmatch("[0-9a-f]{64}", scope_sha256) is not None)
    audit.equal(label + "/source_count", len(source_pins), 215)
    matches = [name for name, pin in source_pins.items() if pin == scope_sha256 and name.endswith(".json")]
    audit.equal(label + "/unique_bound_scope", len(matches), 1)
    for key in ("declaration", "resolved", "parent_report", "parent_checkpoint"):
        row = scope[key]
        audit.equal(label + "/" + key + "/fields", set(row), {"path", "sha256"})
        audit.require(label + "/" + key + "/pin", isinstance(row["sha256"], str)
                      and re.fullmatch("[0-9a-f]{64}", row["sha256"]) is not None)
        audit.require(label + "/" + key + "/path", isinstance(row["path"], str) and bool(row["path"]))
    audit.require(label + "/storage", isinstance(scope["storage_prefix"], str)
                  and scope["storage_prefix"].startswith("gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/"))


def scope_check(audit, report, scope, scope_sha256, source_pins, label):
    declaration_check(audit, scope, scope_sha256, source_pins, label + "/scope")
    audit.equal(label + "/native_only", report["scale"], "native")
    audit.equal(label + "/NFR_only", report["arm"], "NFR")
    audit.equal(label + "/scope_record", report["nfr_scope"],
                {"schema": "olmo-nfr-kl-execution-scope-v1", "declaration": scope, "sha256": scope_sha256})
    audit.equal(label + "/independent_source_authority", report["sources"], source_pins)
    payload = report["configuration"]["execution_identity"]["payload"]
    audit.equal(label + "/identity_sources", payload["sources"], source_pins)
    audit.equal(label + "/finite_plan", len(payload["plan"]["updates"]), 128)
    audit.equal(label + "/native_RT", payload["model_contract"]["mode"]["rt_mode"], {"selected_layers": [0, 15], "alpha": 1.})
    audit.equal(label + "/K4", payload["model_contract"]["mode"]["num_passes"], 4)
    audit.equal(label + "/latent_weight_unchanged", payload["model_contract"]["weights"]["latent"], 1.)
    for key in ("declaration", "resolved"):
        audit.equal(label + "/" + key + "_pin", report[key + "_sha256"], scope[key]["sha256"])
    audit.equal(label + "/parent_report_pin", report["branch"]["parent_report_sha256"], scope["parent_report"]["sha256"])
    audit.equal(label + "/parent_checkpoint_pin", report["branch"]["parent_manifest_sha256"], scope["parent_checkpoint"]["sha256"])


def preflight_check(audit, scope, scope_sha256, source_pins, dry, parent, parent_sha256, *, source_root=None):
    """Independent metadata audit; full tensor restoration remains unperformed."""
    declaration_check(audit, scope, scope_sha256, source_pins)
    audit.equal("dry/schema", dry["schema"], "olmo-nfr-kl-dry-resolution-v1")
    audit.equal("dry/status", dry["status"], "validated_not_launched")
    audit.equal("dry/scope", dry["scope"], scope)
    audit.equal("dry/scope_SHA", dry["scope_sha256"], scope_sha256)
    audit.equal("dry/sources", dry["sources"], source_pins)
    audit.equal("dry/boundaries", (dry["origin_update"], dry["stop_update"]), (32, 64))
    audit.equal("dry/new_exposure", dry["new_input_tokens_per_branch"], 32 * 524288)
    audit.equal("parent/report_pin", parent_sha256, scope["parent_report"]["sha256"])
    audit.equal("parent/arm", parent["arm"], "NFR")
    audit.require("parent/closed_success", parent["status"] in ("completed_plan", "stopped_at_boundary"))
    config = parent["configuration"]; identity = config["execution_identity"]; payload = identity["payload"]
    audit.equal("parent/identity", identity["sha256"], original.digest(payload))
    audit.equal("dry/parent_identity", dry["parent_identity_sha256"], identity["sha256"])
    audit.equal("dry/plan", dry["plan_sha256"], original.digest(payload["plan"]))
    audit.equal("dry/schedule", dry["schedule"], config["schedule"])
    audit.equal("parent/schedule_length", config["schedule"]["planned_updates"], 128)
    audit.equal("parent/K4", payload["model_contract"]["mode"]["num_passes"], 4)
    audit.equal("parent/native_RT", payload["model_contract"]["mode"]["rt_mode"], {"selected_layers": [0, 15], "alpha": 1.})
    audit.equal("parent/weights", payload["model_contract"]["weights"], {"ce": 1., "latent": 1., "kl": 1.})
    published = [p for p in parent["published_checkpoints"] if p["manifest_sha256"] == scope["parent_checkpoint"]["sha256"]]
    audit.equal("parent/unique_publication", len(published), 1)
    manifest, _, published_identity = publication_metadata(published[0])
    audit.equal("parent/published_identity", published_identity, identity)
    audit.equal("parent/published_configuration", manifest["metadata"]["configuration"], config)
    audit.equal("parent/update", manifest["counters"]["optimizer_updates"], 32)
    boundary = legacy.boundary_at(parent, 32)
    for rank, row in enumerate(boundary):
        audit.equal(f"parent/rank{rank}/counters", row["state"]["counters"], manifest["counters"])
        audit.equal(f"parent/rank{rank}/cursor", row["cursor"], manifest["rank_cursors"][rank])
        active = sum(p["requires_grad"] for p in payload["model_contract"]["parameter_layout"])
        audit.equal(f"parent/rank{rank}/populated_Adam", len(row["state"]["optimizer"]["state"]), active)
        audit.equal(f"parent/rank{rank}/scheduler", row["state"]["scheduler"]["last_epoch"], 32)
    for name, pin in parent["sources"].items():
        audit.equal("sources/historical_parent/" + name, source_pins.get(name), pin)
    if source_root is not None:
        for name, pin in source_pins.items():
            path = Path(name)
            audit.require("sources/safe/" + name, not path.is_absolute() and ".." not in path.parts)
            candidate = Path(source_root) / path
            audit.require("sources/bytes/" + name, candidate.is_file() and not candidate.is_symlink() and file_sha(candidate) == pin)


def _result(audit, error=None, **extra):
    result = original._result(audit, error, **extra)
    result["schema"] = SCHEMA
    result["audit_sources"] = {name: file_sha(ROOT / name) for name in AUDIT_SOURCES}
    result["scope_extension"] = "Native NFR32-to64 under independently pinned 215-source scope; unchanged accepted KL checks"
    return result

def _branch_check(audit, report, label, *, expected_stop):
    branch = report['branch']; configuration = report['configuration']
    payload = configuration['execution_identity']['payload']
    audit.equal(label+'/branch_fields', set(branch), {'schema','kl_weight','parent_manifest_sha256',
        'parent_identity_sha256','parent_update','review_stop','recipe_as_declared','parent_report_sha256'})
    audit.equal(label+'/branch_schema', branch['schema'], BRANCH_SCHEMA)
    audit.require(label+'/supported_weight', type(branch['kl_weight']) in (int,float)
                  and branch['kl_weight'] in (1., .1))
    audit.require(label+'/valid_review', type(branch['parent_update']) is int
                  and 0 < branch['parent_update'] < branch['review_stop'])
    audit.equal(label+'/review_stop', branch['review_stop'], expected_stop)
    audit.equal(label+'/completed_review', report['final_counters']['optimizer_updates'], expected_stop)
    for key in ('parent_manifest_sha256','parent_identity_sha256','parent_report_sha256'):
        audit.require(label+'/'+key, isinstance(branch[key],str) and re.fullmatch('[0-9a-f]{64}', branch[key]) is not None)
    audit.equal(label+'/configuration_schema', configuration['schema'], CONFIG_SCHEMA)
    audit.equal(label+'/configuration_branch', configuration['objective_branch'], branch)
    audit.equal(label+'/identity_branch', payload['objective_branch'], branch)
    weights = {'ce':1., 'latent':1., 'kl':branch['kl_weight']}
    audit.equal(label+'/actual_model_KL', configuration['model']['lambda_kl'], branch['kl_weight'])
    audit.equal(label+'/actual_weights', configuration['parameters']['weights'], weights)
    audit.equal(label+'/declared_recipe_KL', configuration['recipe']['auxiliary']['kl'], branch['kl_weight'])
    audit.equal(label+'/operative_recipe', branch['recipe_as_declared'], configuration['recipe'])
    _recipe_check(audit, configuration['recipe'], report['original_configuration']['recipe'], branch,
                  label+'/recipe_KL_and_inherited_metadata_only')
    for key in ('parent_manifest_sha256','parent_report_sha256'):
        audit.equal(label+'/reported_'+key, report[key], branch[key])
    audit.equal(label+'/original_identity', report['original_configuration']['execution_identity']['sha256'],
                branch['parent_identity_sha256'])
    _same_except_kl(audit, configuration, report['original_configuration'], label+'/only_KL_math_changed')
    audit.equal(label+'/Adam_resident_before_DDP', report['adam_resident_before_ddp'], True)
    if report.get('branch_resume'):
        audit.equal(label+'/child_resume_flag', report['branch_resume'], True)
        audit.require(label+'/no_second_transition', 'objective_transition' not in report)
        audit.require(label+'/child_resume_boundary', report['resume']['completed_update'] > branch['parent_update'])
    else:
        audit.require(label+'/no_spurious_resume_flag', 'branch_resume' not in report)
        audit.equal(label+'/fork_origin', report['resume']['completed_update'], branch['parent_update'])
        audit.equal(label+'/parent_pin', report['resume']['manifest_sha256'], branch['parent_manifest_sha256'])
        receipt = report['objective_transition']
        audit.equal(label+'/transition_fields', set(receipt), {'parent_manifest_sha256','before_boundary_by_rank',
                    'after_boundary_by_rank','configuration_change_by_rank'})
        audit.equal(label+'/transition_parent', receipt['parent_manifest_sha256'], branch['parent_manifest_sha256'])
        audit.equal(label+'/complete_state_preserved', receipt['before_boundary_by_rank'], receipt['after_boundary_by_rank'])
        audit.equal(label+'/transition_is_origin', receipt['after_boundary_by_rank'], report['origin_boundary_by_rank'])
        audit.require(label+'/two_configuration_receipts', len(receipt['configuration_change_by_rank']) == 2)
        for rank,change in enumerate(receipt['configuration_change_by_rank']):
            _configuration_receipt(audit,change,report,label+f'/configuration_change/rank{rank}')
        audit.equal(label+'/replicated_configuration_change',receipt['configuration_change_by_rank'][0],
                    receipt['configuration_change_by_rank'][1])
    if report['scale'] == 'native':
        audit.equal(label+'/native_scope', (report['arm'],branch['parent_update'],expected_stop), ('NFR',32,64))
        audit.equal(label+'/native_length', payload['recipe']['sequence_length'], 1024)
        audit.equal(label+'/native_effective_batch', payload['recipe']['effective_valid_tokens'], 524288)
        audit.equal(label+'/native_physical_batch', payload['partition']['physical_batch_per_rank'], 12)
    else:
        audit.equal(label+'/tiny_scale', report['scale'], 'tiny')


def validate_report(audit,report,label,*,expected_stop,source_root=None,check_transport=True):
    payload=training_check(audit,report,label,source_root)
    _branch_check(audit,report,label,expected_stop=expected_stop)
    evaluation_check(audit,report,label+'/evaluation')
    objective=report['evaluation_policy']['objective']
    audit.equal(label+'/evaluation_weights',objective['weights'],payload['model_contract']['weights'])
    audit.equal(label+'/evaluation_objective_schema',objective['schema'],'olmo-kl-continuation-evaluation-v1')
    prefix=[expected_counts(payload,n)['input_tokens'] for n in range(len(payload['plan']['updates'])+1)]
    if report['scale']=='native':
        schedule=payload['schedule']
        audit.equal(label+'/schedule_count',len(schedule['lr_at_completed_boundaries']),len(prefix))
        audit.equal(label+'/token_prefix',schedule['valid_token_prefix'],prefix)
        for step,rows in report.get('updates',{}).items():
            for key,index in (('lr_used',int(step)-1),('lr_next',int(step))):
                actual=rows[0]['metrics'][key]
                audit.equal(label+f'/update{step}/'+key,actual,[schedule['lr_at_completed_boundaries'][index]]*len(actual))
    else:
        schedule=report['configuration']['schedule']
        audit.equal(label+'/tiny_schedule_schema',schedule['schema'],'campaign-token-schedule-v1')
        audit.equal(label+'/tiny_planned_tokens',schedule['planned_tokens'],prefix[-1])
        audit.equal(label+'/tiny_planned_updates',schedule['planned_updates'],len(prefix)-1)
        audit.equal(label+'/tiny_exact_inherited_schedule',schedule,report['original_configuration']['schedule'])
        # Tiny has no LR lookup in payload; it does have exact saved schedule
        # state and a KL1 accepted trajectory. Pair/restart parity below checks
        # every actual LR value as well as token/optimizer clocks.
        audit.require(label+'/tiny_no_invented_payload_schedule','schedule' not in payload)
    if check_transport:transport_check(audit,report,label+'/transport')
    return payload


def audit_pair(parent_report, control_report, reduced_report, *, expected_stop,
               source_roots=None, check_transport=True, parent_report_sha256=None, scope=None, scope_sha256=None, source_pins=None):
    """Audit two new fork reports against the exact retained common parent."""
    audit = Audit(); source_roots = source_roots or {}
    try:
        reports = {'control':control_report, 'reduced':reduced_report}
        for label, report in reports.items():
            scope_check(audit, report, scope, scope_sha256, source_pins, label)
        for label, report in reports.items():
            validate_report(audit, report, label, expected_stop=expected_stop,
                            source_root=source_roots.get(label), check_transport=check_transport)
            audit.require(label+'/is_first_fork_segment', not report.get('branch_resume',False))
            _parent_check(audit, parent_report, report, label, parent_report_sha256)
        a,b = control_report,reduced_report
        audit.equal('pair/weights', (a['branch']['kl_weight'],b['branch']['kl_weight']), (1.,.1))
        for field in a['branch']:
            if field not in ('kl_weight','recipe_as_declared'):
                audit.equal('pair/branch/'+field, a['branch'][field], b['branch'][field])
        _same_except_kl(audit, a['configuration'], b['configuration'], 'pair/only_declared_KL_change')
        audit.equal('pair/origin_complete_state', a['origin_boundary_by_rank'], b['origin_boundary_by_rank'])
        ap,bp = (r['configuration']['execution_identity']['payload'] for r in (a,b))
        for key in ('data','partition','plan','schedule','execution','sources','runtime','determinism','storage_policy'):
            audit.equal('pair/'+key, (key in ap,ap.get(key)), (key in bp,bp.get(key)))
        audit.equal('pair/evaluation_plan', a['evaluation_policy']['plan'], b['evaluation_policy']['plan'])
        first = a['branch']['parent_update']+1
        for key in ('loss_sums','counts'):
            audit.equal('pair/first_forward_'+key, a['updates'][str(first)][0]['metrics'][key],
                        b['updates'][str(first)][0]['metrics'][key])
        for step in range(first,expected_stop+1):
            for key in ('lr_used','lr_next'):
                audit.equal(f'pair/update{step}/'+key,a['updates'][str(step)][0]['metrics'][key],
                            b['updates'][str(step)][0]['metrics'][key])
            audit.equal(f'pair/update{step}/accounting', a['observations'][str(step)]['rank_data'],
                        b['observations'][str(step)]['rank_data'])
            if a['observation_mode'] == b['observation_mode'] == 'acceptance':
                audit.equal(f'pair/update{step}/inputs_noise', [r['input'] for r in a['updates'][str(step)]],
                            [r['input'] for r in b['updates'][str(step)]])
        if a['scale']=='tiny':
            audit.equal('control/acceptance',a['observation_mode'],'acceptance')
            training_parity(audit,parent_report,a)
            previous={row['after_update']:row for row in parent_report['evaluations']}
            for row in a['evaluations']:
                audit.equal('control/raw_evaluation'+str(row['after_update']),_raw_evaluation(row),
                            _raw_evaluation(previous[row['after_update']]))
        origin = first-1
        ae = {r['after_update']:r for r in a['evaluations']}
        be = {r['after_update']:r for r in b['evaluations']}
        if origin in ae or origin in be:
            audit.require('pair/origin_eval_both', origin in ae and origin in be)
            audit.equal('pair/origin_raw_evaluation', _raw_evaluation(ae[origin]), _raw_evaluation(be[origin]))
    except (KeyError,TypeError,ValueError,IndexError,OSError) as error:
        return _result(audit, f'{type(error).__name__}: {error}', comparison_kind='paired_fork')
    return _result(audit, comparison_kind='paired_fork')


def audit_preflight(scope, scope_sha256, source_pins, dry, parent, parent_sha256, *, source_root=None):
    audit = Audit()
    try:
        preflight_check(audit, scope, scope_sha256, source_pins, dry, parent, parent_sha256, source_root=source_root)
    except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
        return _result(audit, f"{type(error).__name__}: {error}", comparison_kind="metadata_preflight_not_training")
    return _result(audit, comparison_kind="metadata_preflight_not_training")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("preflight", "pair"), required=True)
    parser.add_argument("--input", action="append", nargs=3, metavar=("LABEL", "PATH", "SHA256"), required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    reports, pins = {}, {}
    for label, path, pin in args.input:
        if label in reports:
            parser.error("Duplicate input label")
        reports[label], pins[label] = bounded_json(Path(path), pin)
    required = {"scope", "sources", "parent", "dry"} if args.kind == "preflight" else {"scope", "sources", "parent", "control", "reduced"}
    if set(reports) != required:
        parser.error("Require exact input labels " + str(sorted(required)))
    if args.kind == "preflight":
        result = audit_preflight(reports["scope"], pins["scope"]["sha256"], reports["sources"], reports["dry"],
            reports["parent"], pins["parent"]["sha256"], source_root=args.source_root)
    else:
        result = audit_pair(reports["parent"], reports["control"], reports["reduced"], expected_stop=64,
            scope=reports["scope"], scope_sha256=pins["scope"]["sha256"], source_pins=reports["sources"],
            parent_report_sha256=pins["parent"]["sha256"], source_roots={
                label: Path(pins[label]["path"]).parent / "source-snapshot" for label in ("control", "reduced")})
    for pin in pins.values():
        if file_sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Input changed during audit")
    result["inputs"] = pins
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]), "failures": result["failures"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
