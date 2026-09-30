"""Saved-weight feedback diagnostics; never a distributed training resume."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.distributed_checkpoint import inspect_distributed_checkpoint, _local_rng
from cdrm.pretrained.campaign_training import CampaignObjective
from scripts.olmo_feedback_gradient_probe import feedback_gradient_probe
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, parameter_layout
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_evaluation import evaluation_runtime
from scripts.olmo_campaign_fp32_localize import configure_full_fp32
from scripts.olmo_campaign_recurrence_precision import state_pins
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_feedback_fixture import load_fixture
from scripts.olmo_pilot_async_execute import construct
from scripts import olmo_pilot_execution_contract as contract
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu
from scripts.olmo_lm_common import tree_digests

ROOT = Path(__file__).resolve().parents[1]
DECL = '259fc4b84f0b5356bd3a038cee9142ca73fe65efed1cdb7a56ed63a325222109'
RESOLVED = '44d0e8ca4bf8b4d7dbd9a767980faca7c1951375f701e5f6dc0d4ff4ff238a8d'
FIXTURE = 'b83df92927f56c0528bef44ff17a5928e1579d3834cb7e98d5c09f33afb99c7f'


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)


def validate_weights_payload(payload, manifest, model):
    """Validate model ownership and committed provenance without restoring Adam/RNG."""
    if payload['metadata'] != manifest['metadata'] or payload['counters'] != manifest['counters']:
        raise ValueError('Payload metadata/counters disagree with committed manifest')
    TrainingCounters(**payload['counters'])
    ranks = payload['rank_states']
    if ([r['rank'] for r in ranks] != list(range(manifest['world_size']))
            or [r['data_cursor'] for r in ranks] != manifest['rank_cursors']):
        raise ValueError('Saved rank mapping/cursors disagree')
    modes = payload['module_training']
    if set(modes) != set(dict(model.named_modules())) or any(type(v) is not bool for v in modes.values()):
        raise ValueError('Module ownership disagrees')
    current, saved = model.state_dict(), payload['model']
    if set(current) != set(saved):
        raise ValueError('Saved model keys disagree')
    for name, value in current.items():
        other = saved[name]
        if not isinstance(other, torch.Tensor) or value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Saved tensor shape/dtype disagrees: '+name)
    layout = manifest['metadata']['parameter_layout']
    if layout != parameter_layout(model):
        raise ValueError('Complete unique parameter/alias layout disagrees')
    actual = dict(model.named_parameters())
    if {r['name'] for r in layout} != set(actual):
        raise ValueError('Unique parameter ownership disagrees')
    aliases = dict(model.named_parameters(remove_duplicate=False))
    for row in layout:
        parameter = actual[row['name']]
        if (list(parameter.shape) != row['shape'] or str(parameter.dtype) != row['dtype']
                or parameter.requires_grad != row['requires_grad']):
            raise ValueError('Saved parameter layout disagrees')
        if any(aliases[name] is not parameter for name in row['aliases']):
            raise ValueError('Current aliases do not share ownership')
        if any(not torch.equal(saved[row['aliases'][0]], saved[name]) for name in row['aliases'][1:]):
            raise ValueError('Saved tied aliases disagree')


def import_saved_weights(model, spec, directory, manifest_sha256):
    manifest = inspect_distributed_checkpoint(directory, expected_manifest_sha256=manifest_sha256,
                                               verify_state=True)
    update = manifest['counters']['optimizer_updates']
    if (update not in (0,32) or manifest['counters']['input_tokens'] != update*524288
            or any(r['cursor']['next_update'] != update or r['cursor']['next_chunk'] != update*512
                   for r in manifest['rank_cursors'])):
        raise ValueError('Checkpoint is outside the declared origin0/endpoint32 scope')
    config = manifest['metadata']['configuration']
    plain_recipe = json.loads(json.dumps(spec['recipe'].to_dict()))
    if (config['recipe'] != plain_recipe or config['backbone'] != model.backbone.backbone.config.to_dict()
            or config['model'] != model.config.to_dict()):
        raise ValueError('Saved architecture/recipe differs from authenticated declaration')
    payload = torch.load(Path(directory)/manifest['state']['filename'], map_location='cpu',
                         weights_only=True, mmap=True)
    validate_weights_payload(payload, manifest, model)
    identities = {n: id(p) for n, p in model.named_parameters()}
    model.load_state_dict(payload['model'], strict=True, assign=False)
    if identities != {n: id(p) for n, p in model.named_parameters()}:
        raise ValueError('Import replaced parameter ownership')
    del payload
    gc.collect()
    return {'manifest_sha256': manifest_sha256, 'state': manifest['state'],
        'counters': manifest['counters'], 'saved_world_size': manifest['world_size'],
        'mode': 'strict weights-only diagnostic import; no Adam/scheduler/RNG restoration',
        'optimizer_updates': 0}


def source_inventory():
    frozen_path = ROOT/'.runtime/olmo-pilot-async/runtime-sources.json'
    frozen = json.loads(frozen_path.read_text())
    if len(frozen) != 200 or sha256_file(frozen_path) != 'cfa54e38b245fc72b993093b1e7978de07f529c86a28886771be727941bc2162':
        raise ValueError('Frozen inventory authority changed')
    for name, expected in frozen.items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError('Frozen runtime changed: '+name)
    added = ('scripts/olmo_feedback_diagnostic.py', 'scripts/olmo_feedback_fixture.py',
             'scripts/olmo_feedback_gradient_probe.py', 'scripts/olmo_feedback_forward_probe.py',
             'docs/reports/olmo-feedback-diagnostic/protocol.md')
    return {'frozen_count': len(frozen), 'frozen_inventory_sha256': sha256_file(frozen_path),
            'new_sources': {n: sha256_file(ROOT/n) for n in added}}


def diagnostic_spec(arm):
    """Read the pinned accepted plan, without rediscovering a training inventory.

    New observational modules extend the package glob. They are separately pinned
    and do not migrate the accepted training declaration or its resolved plan.
    """
    directory = ROOT/'.runtime/olmo-adaptation-pilot/declarations-01'
    declaration = contract.legacy.read_json(directory/'nf-nfr-declaration.json', DECL)
    resolved = contract.legacy.read_json(directory/'nf-nfr-resolved.json', RESOLVED, limit=128*1024**2)
    sources = declaration['implementation_sources']
    frozen = json.loads((ROOT/'.runtime/olmo-pilot-async/runtime-sources.json').read_text())
    if any(frozen.get(n) != pin or sha256_file(ROOT/n) != pin for n,pin in sources.items()):
        raise ValueError('Declared accepted implementation changed')
    contract.validate_declaration(declaration, sources=sources)
    contract.validate_resolved(resolved)
    if resolved['declaration'] != declaration or resolved['sources'] != sources:
        raise ValueError('Pinned plan/declaration disagree')
    startup = contract.startup_plan(resolved, arm)
    return {'kind': 'native', 'declaration': declaration, 'resolved': resolved,
            'manifest': declaration['planning_manifest'], 'startup': startup,
            'recipe': contract.recipe_from_dict(startup['target_recipe']),
            'plan': resolved['planning']['plan'], 'checkpoint_mode': 'async'}


def forward_summary(rows):
    result = []
    for index in range(4):
        passes = [r['passes'][index] for r in rows]
        counts = {t: sum(p['counts'][t] for p in passes) for t in ('ce','latent','kl')}
        means = {t: sum(p['sums'][t] for p in passes)/counts[t] for t in counts}
        entropies = {key: sum(p[key]['sum_nats'] for p in passes)/sum(p[key]['count'] for p in passes)
                     for key in ('teacher_entropy_on_kl_positions','student_entropy_on_kl_positions')}
        result.append({'pass': index+1, 'counts': counts, 'means': means, **entropies})
    return result


def run(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'olmo-feedback-diagnostic-v1', 'status': 'preflight', 'arm': args.arm,
              'stage': args.stage, 'selection': args.selection, 'cases': [], 'optimizer_updates': 0}
    tracker = None
    started = time.monotonic()
    def publish(event=None):
        report['elapsed_seconds'] = time.monotonic()-started
        if event is not None:
            report.setdefault('progress', []).append(event)
            print(json.dumps(event, allow_nan=False), flush=True)
        write_json(output/'report.json', report)
    try:
        report['source_authorities'] = source_inventory()
        frozen = json.loads((ROOT/'.runtime/olmo-pilot-async/runtime-sources.json').read_text())
        report['sources'] = frozen | report['source_authorities']['new_sources']
        for name in report['sources']:
            destination = output/'source-snapshot'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, destination)
        spec = diagnostic_spec(args.arm)
        batch, provenance = load_fixture(ROOT/'.runtime/olmo-feedback-diagnostic/fixture-01',
            expected_report_sha256=FIXTURE, selection=args.selection)
        report['fixture'] = provenance
        if (args.stage == 'forward') != (args.selection == 'dev'):
            raise ValueError('Forward uses dev; gradient uses a declared training batch')
        configure_determinism(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.set_num_threads(8)
        report['runtime'] = require_container_gpu()
        tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', output_dir=output,
            group='feedback-diagnostic', name=output.name, preserve_state=preserve_local_rng)
        tracker.start({'arm': args.arm, 'stage': args.stage, 'selection': args.selection,
                       'precision': 'fp32_math_eager', 'feedback_jitter': 0, 'optimizer_updates': 0,
                       'manifest_sha256': args.manifest_sha256})
        report['tracking'] = tracker.record
        publish({'phase': 'constructing_model', 'wandb': tracker.record['run_url']})
        model, _, _ = construct(spec, torch.device('cuda:0'))
        report['checkpoint'] = import_saved_weights(model, spec, args.checkpoint, args.manifest_sha256)
        report['diagnostic_runtime'] = configure_full_fp32(model)
        before = state_pins(model)
        rng_before = tree_digests(_local_rng(torch.device('cuda:0'), None))
        publish({'phase': 'weights_loaded', 'checkpoint_update': report['checkpoint']['counters']['optimizer_updates']})
        if args.stage == 'forward':
            from scripts.olmo_feedback_forward_probe import feedback_forward_probe
            for beta in args.betas:
                rows = []
                with evaluation_runtime(model) as runtime_evidence:
                    for index in range(batch.input_ids.shape[0]):
                        one = NextLatBatch(**{n: None if v is None else v[index:index+1]
                                               for n,v in batch.__dict__.items()}).to('cuda:0')
                        row = feedback_forward_probe(model, spec['recipe'], one, beta)
                        if report['cases'] and row['passes'][0] != report['cases'][0]['rows'][index]['passes'][0]:
                            raise ValueError('First pass changed across beta controls')
                        rows.append(row)
                        publish({'phase': 'forward_row', 'beta': beta, 'row': index})
                case = {'beta': beta, 'rows': rows, 'runtime_integrity': runtime_evidence,
                        'per_pass': forward_summary(rows)}
                report['cases'].append(case)
                write_json(output/f'beta-{beta:g}.json', case)
                tracker.log({'beta': beta, **scalar_metrics({f'pass{p["pass"]}': p for p in case['per_pass']}, 'dev')})
                publish({'phase': 'forward_beta_complete', 'beta': beta})
        else:
            model.train()
            mode = replace(spec['recipe'].mode(), feedback_jitter=0)
            masks = build_nextlat_masks(batch, document_policy=model.config.document_policy)
            counts = {n: int(v.sum()) for n,v in masks.items()}
            with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', enabled=False, cache_enabled=False):
                adapter = CampaignObjective(model, batch, mode=mode, global_counts=counts,
                    config=LMTrainingConfig(precision='fp32'))
                result = feedback_gradient_probe(adapter, publish=publish)
            report['gradient'] = result
            write_json(output/'gradient.json', result)
            tracker.log(scalar_metrics(result, 'gradient'))
            if not result['passed']:
                raise ValueError('Gradient reconstruction/structural checks failed')
        report['weights_unchanged'] = before == state_pins(model)
        report['grad_buffers_untouched'] = all(p.grad is None for p in model.parameters())
        report['rng_unchanged'] = rng_before == tree_digests(_local_rng(torch.device('cuda:0'), None))
        if not all(report[k] for k in ('weights_unchanged','grad_buffers_untouched','rng_unchanged')):
            raise ValueError('Diagnostic mutated model state')
        if source_inventory() != report['source_authorities']:
            raise ValueError('Diagnostic helper sources changed during execution')
        report['status'] = 'complete'
        tracker.summary({'completed': True, 'optimizer_updates': 0})
        tracker.finish(succeeded=True)
        publish({'phase': 'complete'})
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        if tracker is not None:
            tracker.finish(succeeded=False)
        publish()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('NF','NFR'), required=True)
    parser.add_argument('--stage', choices=('forward','gradient'), required=True)
    parser.add_argument('--selection', choices=('dev','train_primary','train_conditional'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--betas', type=float, nargs='+', default=[1.,0.,.5])
    args = parser.parse_args()
    if any(beta not in (0.,.5,1.) for beta in args.betas):
        parser.error('Only the three fixed diagnostic beta controls are declared')
    run(args)


if __name__ == '__main__':
    main()
