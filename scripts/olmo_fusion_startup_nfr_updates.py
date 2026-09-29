#!/usr/bin/env python3
"""Four matched full-parameter NFR updates per precision, with CPU boundaries."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_recipe import build_campaign_adamw, CampaignTokenSchedule
from cdrm.pretrained.lm_training import (TrainingCounters, _rng_state, _restore_rng,
    optimizer_ownership, parameter_layout, save_training_checkpoint, load_training_checkpoint)
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct, global_fixture_metadata, advance_counters
from scripts.olmo_campaign_precision_bridge import configure_path
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, TERMS, component_backward, pass_sums, gradient_geometry
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, fixture_pins, state_pins
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_restart import checkpoint_disk_preflight
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_component_probe import enable_native_rt, objective_contract
from scripts.olmo_fusion_startup_data import StartupData, DEFAULT_ROOT, DEFAULT_MANIFEST_SHA256
from scripts.olmo_fusion_startup_long_probe import load_long_fixture
from scripts.olmo_fusion_startup_rt_strength import source_hashes as strength_sources, load_component_reference
from scripts.olmo_fusion_startup_train import load_fusion_checkpoint, retain_checkpoint
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA = 'olmo-fusion-startup-nfr-updates-v1'
PLAN = {'train_indices': [144, 145, 146, 147], 'updates_per_precision': 4,
        'length': 128, 'physical_batch': 8, 'ce_targets_per_update': 8192,
        'paths': [FP32, BF16], 'checkpoint_seconds': 600}


def source_hashes():
    result = strength_sources()
    for name in ('scripts/olmo_fusion_startup_nfr_updates.py', 'tests/test_fusion_startup_nfr_updates.py',
                 'docs/reports/olmo-fusion-startup/nfr-updates-protocol.md'):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def cpu_copy(value):
    """No CPU snapshot tensor may alias the live model, Adam or another branch."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def expected_counters(metadata, completed):
    if type(completed) is not int or not 0 <= completed <= len(metadata):
        raise ValueError('Completed update lies outside the fixed diagnostic plan')
    counters = TrainingCounters()
    for row in metadata[:completed]:
        advance_counters(counters, row)
    return counters


def validate_boundary(model, optimizer, scheduler, snapshot, metadata):
    completed = snapshot['counters']['optimizer_updates']
    if (snapshot.get('schema') != SCHEMA or snapshot['counters'] != asdict(expected_counters(metadata, completed))
            or snapshot['ownership'] != optimizer_ownership(model, optimizer)
            or snapshot['layout'] != parameter_layout(model)
            or snapshot['modes'] != {n: m.training for n, m in model.named_modules()}
            or snapshot['scheduler']['last_epoch'] != completed):
        raise ValueError('CPU boundary ownership, modes or data clocks differ')
    for key in ('token_prefix', 'warmup_tokens', 'start_fraction', 'plan_sha256', 'base_lrs'):
        if snapshot['scheduler'][key] != getattr(scheduler, key):
            raise ValueError('CPU boundary token schedule differs')
    current = model.state_dict()
    if current.keys() != snapshot['model'].keys():
        raise ValueError('CPU boundary model inventory differs')
    for name, value in snapshot['model'].items():
        if (value.device.type != 'cpu' or value.shape != current[name].shape or value.dtype != current[name].dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('CPU boundary model shape/dtype/finiteness differs')
    for row in snapshot['layout']:
        if any(not torch.equal(snapshot['model'][row['aliases'][0]], snapshot['model'][alias]) for alias in row['aliases'][1:]):
            raise ValueError('CPU boundary tied parameter aliases differ')
    groups = snapshot['optimizer']['param_groups']
    if len(groups) != len(optimizer.param_groups):
        raise ValueError('CPU boundary optimizer groups differ')
    if (snapshot['scheduler'].get('_step_count') != completed+1
            or snapshot['scheduler'].get('_last_lr') != [group['lr'] for group in groups]):
        raise ValueError('CPU boundary serialized scheduler history differs')
    seen = set()
    for group, live, names in zip(groups, optimizer.param_groups, snapshot['ownership']):
        if group.get('param_names') != names or len(group['params']) != len(live['params']):
            raise ValueError('CPU boundary Adam ownership differs')
        for key in ('betas', 'eps', 'weight_decay', 'fused', 'foreach', 'initial_lr', 'component', 'decay_policy', 'lr_multiplier'):
            if group.get(key) != live.get(key):
                raise ValueError('CPU boundary Adam settings differ')
        exposure = snapshot['scheduler']['token_prefix'][completed]
        warmup = snapshot['scheduler']['warmup_tokens']
        fraction = snapshot['scheduler']['start_fraction']
        lr = group['initial_lr']*(fraction+(1-fraction)*min(1., exposure/warmup) if warmup else 1.)
        if group['lr'] != lr:
            raise ValueError('CPU boundary Adam LR differs from token clock')
        for identifier, parameter in zip(group['params'], live['params']):
            if identifier in seen:
                raise ValueError('CPU boundary duplicate Adam ownership')
            seen.add(identifier)
            state = snapshot['optimizer']['state'].get(identifier, {})
            if completed == 0:
                if state:
                    raise ValueError('Fresh diagnostic Adam must have no inherited moments')
            else:
                if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != completed:
                    raise ValueError('CPU boundary Adam step/moment inventory differs')
                for key in ('exp_avg', 'exp_avg_sq'):
                    value = state[key]
                    if value.device.type != 'cpu' or value.shape != parameter.shape or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all()):
                        raise ValueError('CPU boundary Adam moments differ or are nonfinite')
                    if key == 'exp_avg_sq' and bool((value < 0).any()):
                        raise ValueError('CPU boundary Adam variance is negative')
    if set(snapshot['optimizer']['state']) != (seen if completed else set()):
        raise ValueError('CPU boundary has missing or foreign Adam state')
    return completed


def capture_boundary(model, optimizer, scheduler, counters, metadata):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Capture only a completed boundary with cleared gradients')
    snapshot = {'schema': SCHEMA, 'model': cpu_copy(model.state_dict()),
        'optimizer': cpu_copy(optimizer.state_dict()), 'scheduler': cpu_copy(scheduler.state_dict()),
        'rng': cpu_copy(_rng_state(None)), 'counters': asdict(counters),
        'ownership': optimizer_ownership(model, optimizer), 'layout': parameter_layout(model),
        'modes': {n: m.training for n, m in model.named_modules()}}
    validate_boundary(model, optimizer, scheduler, snapshot, metadata)
    return snapshot


def restore_boundary(model, optimizer, scheduler, snapshot, metadata):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Restore only between completed updates with no live gradients')
    validate_boundary(model, optimizer, scheduler, snapshot, metadata)
    identities = {name: id(p) for name, p in model.named_parameters()}
    model.load_state_dict(snapshot['model'], strict=True, assign=False)
    # On CPU, Adam's loader may reuse CPU tensor storage. Clone explicitly so a
    # subsequent optimizer step cannot modify the retained branch boundary.
    optimizer.load_state_dict(cpu_copy(snapshot['optimizer']))
    scheduler.load_state_dict(copy.deepcopy(snapshot['scheduler']))
    _restore_rng(snapshot['rng'], None)
    if identities != {name: id(p) for name, p in model.named_parameters()}:
        raise AssertionError('Boundary restore replaced parameter objects')
    return TrainingCounters(**snapshot['counters'])


def parameter_values(model, *, gradient=False):
    values = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradient else parameter
        if not parameter.requires_grad or value is None or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Full NFR requires every FP32 parameter and gradient to participate finitely')
        values[name] = value.detach().to(device='cpu', copy=True)
    return values


def moment_values(snapshot):
    result = {'exp_avg': {}, 'exp_avg_sq': {}}
    for group, names in zip(snapshot['optimizer']['param_groups'], snapshot['ownership']):
        for identifier, name in zip(group['params'], names):
            for key in result:
                result[key][name] = snapshot['optimizer']['state'][identifier][key]
    return result


def snapshot_parameters(model, snapshot):
    return {name: snapshot['model'][name] for name, _ in model.named_parameters()}


class DeltaView(Mapping):
    """Exact master subtraction generated per parameter, never full FP64 state."""
    def __init__(self, actual, reference):
        if actual.keys() != reference.keys():
            raise ValueError('Parameter inventories differ')
        self.actual, self.reference = actual, reference

    def __iter__(self):
        return iter(self.actual)

    def __len__(self):
        return len(self.actual)

    def __getitem__(self, name):
        return self.actual[name].double()-self.reference[name].double()


def summary(values):
    # Existing per-parameter streamed FP64 geometry; no whole-model FP64 clone.
    return gradient_geometry(values, values)


def update(model, recipe, fixtures, optimizer, scheduler, counters, *, path, original_flags, metadata):
    objective_contract(model, recipe, 'combined')
    actual = global_fixture_metadata(model, fixtures)
    if actual != metadata[counters.optimizer_updates]:
        raise ValueError('Update inputs/counts differ from frozen plan')
    scheduler.validate_next_update(actual['input_tokens'])
    rates = [group['lr'] for group in optimizer.param_groups]
    rng = tree_digests(_rng_state(None))
    execution = configure_path(model, original_flags, path)
    device = next(model.parameters()).device
    backend = SDPBackend.FLASH_ATTENTION if path == BF16 and device.type == 'cuda' else SDPBackend.MATH
    started = time.monotonic()
    with sdpa_kernel(backend), torch.autocast(device.type, enabled=False):
        metrics = component_backward(model, recipe, fixtures, precision=execution['precision'], layout='sparse', objective='combined')
    if not all(math.isfinite(value) for value in (metrics['objective'], *metrics['loss_sums'].values())):
        raise FloatingPointError('Combined objective or component sum is nonfinite before Adam')
    raw = parameter_values(model, gradient=True)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.max_grad_norm, error_if_nonfinite=True, foreach=False)
    clipped = parameter_values(model, gradient=True)
    optimizer.step(); scheduler.step(); advance_counters(counters, metrics)
    model.zero_grad(set_to_none=True)
    if rng != tree_digests(_rng_state(None)):
        raise AssertionError('Keyed deterministic training unexpectedly consumed global RNG')
    return {'metrics': metrics, 'path': path, 'execution': execution, 'lr_used': rates,
        'lr_next': [group['lr'] for group in optimizer.param_groups], 'gradient_norm_before_clip': float(norm),
        'clip_scale': min(1., recipe.max_grad_norm/(float(norm)+1e-6)),
        'raw_gradient_pins': tree_digests(raw), 'clipped_gradient_pins': tree_digests(clipped),
        'elapsed_seconds_including_observation': time.monotonic()-started}, raw, clipped


def evaluate_fp32(model, recipe, fixtures, *, original_flags):
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Held-out evaluation requires cleared gradients')
    before, rng, inputs = state_pins(model), tree_digests(_rng_state(None)), fixture_pins(fixtures)
    flags = {n: getattr(model.backbone.backbone, n) for n in RUNTIME_FLAGS}
    modes = {n: m.training for n, m in model.named_modules()}
    cache = torch.is_autocast_cache_enabled()
    metadata = global_fixture_metadata(model, fixtures)
    totals = dict.fromkeys(TERMS, 0.)
    device = next(model.parameters()).device
    try:
        configure_path(model, original_flags, FP32)
        with preserve_local_rng(), torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device.type, enabled=False):
            for batches, noises in fixtures:
                for batch, noise in zip(batches, noises):
                    losses = model.loss_sums(batch.to(device), backbone_kwargs={'mode': recipe.mode(),
                        'feedback_noise': tuple(v.to(device) for v in noise), 'right_padded_causal': True})
                    sums = pass_sums(tuple(p.sums for p in losses.pass_losses))
                    for term in TERMS:
                        totals[term] += float(sums[term])
    finally:
        for name, value in flags.items():
            setattr(model.backbone.backbone, name, value)
    checks = {'state_unchanged': before == state_pins(model), 'rng_unchanged': rng == tree_digests(_rng_state(None)),
        'fixture_unchanged': inputs == fixture_pins(fixtures),
        'modes_unchanged': modes == {n: m.training for n, m in model.named_modules()},
        'runtime_restored': flags == {n: getattr(model.backbone.backbone, n) for n in RUNTIME_FLAGS},
        'autocast_cache_restored': cache == torch.is_autocast_cache_enabled(),
        'gradients_absent': all(p.grad is None for p in model.parameters()),
        'finite': all(bool(torch.isfinite(torch.tensor(v))) for v in totals.values())}
    if not all(checks.values()):
        raise AssertionError('FP32 held-out evaluation changed state or produced nonfinite losses')
    means = {term: totals[term]/metadata['counts'][term] for term in TERMS}
    return {'precision': FP32, 'loss_sums': totals, 'loss_means': means,
        'combined_objective': sum(means.values()), 'counts': metadata['counts'], 'checks': checks}


def run_pair(model, recipe, metadata, data_for_step, heldout, *, original_flags, publish, checkpoint):
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    scheduler = CampaignTokenSchedule(optimizer, [row['input_tokens'] for row in metadata],
        warmup_tokens=recipe.warmup_tokens, start_fraction=recipe.warmup_start_fraction)
    initial = capture_boundary(model, optimizer, scheduler, TrainingCounters(), metadata)
    states = {FP32: initial, BF16: cpu_copy(initial)}
    evaluations = {'0': {FP32: evaluate_fp32(model, recipe, heldout, original_flags=original_flags)}}
    evaluations['0'][BF16] = copy.deepcopy(evaluations['0'][FP32])
    rows, last_save = [], time.monotonic()
    try:
        for step in range(len(metadata)):
            fixtures = data_for_step(step)
            input_pins = fixture_pins(fixtures)
            observations, pair_rows = {}, []
            for path in (FP32, BF16):
                old = states[path]
                before_pin = tree_digests(old)
                counters = restore_boundary(model, optimizer, scheduler, old, metadata)
                row, raw, clipped = update(model, recipe, fixtures, optimizer, scheduler, counters,
                    path=path, original_flags=original_flags, metadata=metadata)
                new = capture_boundary(model, optimizer, scheduler, counters, metadata)
                if tree_digests(old) != before_pin:
                    raise AssertionError('Updating a branch mutated its saved CPU boundary')
                delta = DeltaView(snapshot_parameters(model, new), snapshot_parameters(model, old))
                values = {'raw_gradient': raw, 'clipped_gradient': clipped, 'actual_master_delta': delta,
                          **moment_values(new)}
                row.update(update=step+1, train_data_index=PLAN['train_indices'][step], input_pins=input_pins,
                    start_boundary_pins=before_pin, final_boundary_pins=tree_digests(new),
                    observations={name: summary(value) for name, value in values.items()},
                    memory=memory() if next(model.parameters()).device.type == 'cuda' else {'scope': 'explicit_cpu_test'})
                if path == BF16:
                    row['bf16_vs_fp32'] = {name: gradient_geometry(value, observations[name]) for name, value in values.items()}
                    row['parameter_separation_vs_fp32'] = gradient_geometry(snapshot_parameters(model, new), snapshot_parameters(model, states[FP32]))
                else:
                    observations = values
                states[path] = new
                pair_rows.append(row); publish(row)
                del raw, clipped, delta, values, old
            if (pair_rows[0]['input_pins'] != pair_rows[1]['input_pins']
                    or pair_rows[0]['lr_used'] != pair_rows[1]['lr_used']
                    or pair_rows[0]['metrics']['counts'] != pair_rows[1]['metrics']['counts']
                    or states[FP32]['rng'] is states[BF16]['rng']
                    or tree_digests(states[FP32]['rng']) != tree_digests(states[BF16]['rng'])
                    or fixture_pins(fixtures) != input_pins):
                raise AssertionError('Paired inputs, RNG, schedule or fixture ownership differ')
            rows.append(pair_rows)
            observations = None; del fixtures; gc.collect()
            completed = step+1
            if completed == len(metadata) or time.monotonic()-last_save >= PLAN['checkpoint_seconds']:
                for path in (FP32, BF16):
                    counters = restore_boundary(model, optimizer, scheduler, states[path], metadata)
                    checkpoint(path, model, optimizer, scheduler, counters, states[path])
                last_save = time.monotonic()
        evaluations[str(len(metadata))] = {}
        for path in (FP32, BF16):
            restore_boundary(model, optimizer, scheduler, states[path], metadata)
            evaluations[str(len(metadata))][path] = evaluate_fp32(model, recipe, heldout, original_flags=original_flags)
        return {'rows': rows, 'evaluations': evaluations, 'states': states,
            'optimizer_calls': 2*len(metadata), 'initial_boundary_pins': tree_digests(initial),
            'schedule': scheduler.checkpoint_contract(), 'ownership': optimizer_ownership(model, optimizer)}
    finally:
        model.zero_grad(set_to_none=True)
        configure_path(model, original_flags, BF16)


def checkpoint_configuration(common, path):
    if path not in (FP32, BF16):
        raise ValueError('Unknown diagnostic trajectory')
    return {**common, 'trajectory': path}


def save_endpoint(path, model, optimizer, scheduler, counters, *, configuration, source_fingerprint, data_cursor):
    checkpoint_disk_preflight(path, model)
    return save_training_checkpoint(path, model, optimizer, scheduler=scheduler, counters=counters,
        configuration=configuration, source_fingerprint=source_fingerprint, data_cursor=data_cursor)


def load_endpoint(path, model, optimizer, scheduler, *, configuration, source_fingerprint,
                  expected_sha256, metadata, expected_cursor):
    # Generic loader retains its complete source/configuration/ownership guards.
    # Additional diagnostic clock checks precede any live-state mutation.
    if sha256_file(path) != expected_sha256:
        raise ValueError('NFR endpoint checkpoint SHA differs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    step = payload['counters']['optimizer_updates']
    snapshot = {'schema': SCHEMA, 'model': payload['model'], 'optimizer': payload['optimizer'],
        'scheduler': payload['scheduler'], 'rng': payload['rng'], 'counters': payload['counters'],
        'ownership': payload['optimizer_ownership'], 'layout': payload['parameter_layout'], 'modes': payload['module_training']}
    validate_boundary(model, optimizer, scheduler, snapshot, metadata)
    if payload['data_cursor'] != expected_cursor(step):
        raise ValueError('NFR endpoint data cursor differs from fixed plan')
    del payload, snapshot
    return load_training_checkpoint(path, model, optimizer, scheduler=scheduler,
        configuration=configuration, source_fingerprint=source_fingerprint, expected_sha256=expected_sha256)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'fixture', 'component-report'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--artifacts', type=Path, default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--data-manifest-sha256', default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--storage-prefix', required=True)
    args = parser.parse_args(argv)
    for value in (args.checkpoint_sha256, args.fixture_sha256, args.component_report_sha256, args.data_manifest_sha256):
        if len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            parser.error('Require independent lowercase SHA256 pins')
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error('Full checkpoints and evidence must stay on persistent project storage')
    return args


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True); runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError('Paired diagnostic is one GPU/process without DDP or graphs')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    for name in sources:
        target = args.output_dir/'source-snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
    report = {'schema': SCHEMA, 'status': 'running', 'passed': False, 'sources': sources,
        'runtime': runtime, 'determinism': determinism, 'plan': PLAN, 'rows': [], 'checkpoints': [],
        'origin_checkpoint_sha256': args.checkpoint_sha256, 'fixture_sha256': args.fixture_sha256,
        'component_report_sha256': args.component_report_sha256, 'optimizer_state': 'fresh_all_components',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'qualification': 'Four tiny-budget NFR-combined updates per precision; no quality or BF16 production clearance; later differences include trajectory divergence'}
    tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', output_dir=args.output_dir,
        group='olmo-fusion-startup', name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None
    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/'report.json', report)
    try:
        tracker.start({key: report[key] for key in ('plan', 'origin_checkpoint_sha256', 'fixture_sha256', 'qualification')})
        reference, _ = load_component_reference(args.component_report, args.component_report_sha256, sources,
            fixture_sha256=args.fixture_sha256, checkpoint_sha256=args.checkpoint_sha256)
        persist('construct_original_nf')
        model, recipe, source, _, _ = construct(SimpleNamespace(scale='pretrained', length=16, artifacts=args.artifacts), 'NF', torch.device('cuda'))
        original_flags = {name: getattr(model.backbone.backbone, name) for name in RUNTIME_FLAGS}
        report['import'] = load_fusion_checkpoint(model, args.checkpoint, source, expected_sha256=args.checkpoint_sha256)
        if report['import']['counters']['optimizer_updates'] != 128 or state_pins(model) != reference['initial_state'] or tree_digests(source) != tree_digests(reference['source_checkpoint']):
            raise ValueError('Require original backbone/predictor and exact saved NF fusion128')
        data = StartupData.from_prepared(args.data_root, expected_manifest_sha256=args.data_manifest_sha256)
        if data.manifest_sha256 != report['import']['configuration']['data_manifest_sha256']:
            raise ValueError('Training data differ from original startup authority')
        heldout, provenance = load_long_fixture(args.fixture, expected_sha256=args.fixture_sha256, recipe=recipe, width=model.config.model_dim)
        recipe, transition = enable_native_rt(model, recipe)
        if recipe.rt_layers != (0, 15) or not all(p.requires_grad for p in model.parameters()):
            raise ValueError('Diagnostic requires full trainability and native RT layers0/15')
        metadata, data_selections = [], []
        for index in PLAN['train_indices']:
            row = data.update_metadata(index)
            data_selections.append({'source_training_index': index, **row})
            metadata.append({key: row[key] for key in ('counts', 'microbatches', 'documents', 'input_tokens')})
        if any(row['counts']['ce'] != 8192 for row in metadata):
            raise ValueError('Actual training selections differ from 8192 CE-target budget')
        report.update(recipe=recipe.to_dict(), transition=transition, contract=objective_contract(model, recipe, 'combined'),
            initial_state=state_pins(model), fixture_pins=fixture_pins(heldout), fixture_provenance=provenance,
            data_manifest_sha256=data.manifest_sha256, training_metadata=metadata,
            training_data_selections=data_selections,
            data_cursor_scope='Fresh diagnostic optimizer clock0..4; source selection cursor144..148; prior fusion Adam discarded')
        common = {'schema': SCHEMA, 'plan': PLAN, 'recipe': recipe.to_dict(), 'sources': sources,
            'runtime': runtime, 'determinism': determinism, 'origin_checkpoint_sha256': args.checkpoint_sha256,
            'source_checkpoint': source, 'data_manifest_sha256': data.manifest_sha256,
            'fixture_sha256': args.fixture_sha256, 'training_metadata': metadata,
            'optimizer': {'implementation': 'build_campaign_adamw', 'fused': False, 'fresh': True},
            'schedule': {'warmup_tokens': recipe.warmup_tokens, 'start_fraction': recipe.warmup_start_fraction,
                         'update_valid_tokens': [row['input_tokens'] for row in metadata]}}
        fingerprint = {'checkpoint_sha256': args.checkpoint_sha256, 'base': source, 'sources': sources}
        report['checkpoint_configuration'] = common
        def data_for_step(step):
            batches, noises = data.update_batches(PLAN['train_indices'][step], recipe, model.config.model_dim)
            return [(batches, noises)]
        def publish(row):
            report['rows'].append(row); persist('update'+str(row['update'])+'/'+row['path'])
            tracker.log(scalar_metrics(row, 'nfr_updates/'+row['path']), step=len(report['rows']))
            print({'update': row['update'], 'path': row['path'], 'objective': row['metrics']['objective']}, flush=True)
        def checkpoint(path, model, optimizer, scheduler, counters, state):
            name = path+'-update-'+str(counters.optimizer_updates).zfill(6)+'.pt'
            before = tree_digests(state)
            receipt = save_endpoint(args.output_dir/name, model, optimizer, scheduler, counters,
                configuration=checkpoint_configuration(common, path), source_fingerprint=fingerprint,
                data_cursor=data.cursor(144+counters.optimizer_updates))
            report['checkpoints'].append({'trajectory': path, **receipt}); persist('checkpoint_local/'+name)
            receipt['gcs'] = retain_checkpoint(args.output_dir/name, args.storage_prefix, receipt['sha256'])
            report['checkpoints'][-1] = {'trajectory': path, **receipt}
            if tree_digests(capture_boundary(model, optimizer, scheduler, counters, metadata)) != before:
                raise AssertionError('Checkpoint/retention changed completed model/Adam/RNG boundary')
            persist('checkpoint_retained/'+name)
        outcome = run_pair(model, recipe, metadata, data_for_step, heldout, original_flags=original_flags,
            publish=publish, checkpoint=checkpoint)
        report.update(evaluations=outcome['evaluations'], optimizer_calls=outcome['optimizer_calls'],
            initial_boundary_pins=outcome['initial_boundary_pins'], schedule=outcome['schedule'], ownership=outcome['ownership'],
            final_boundary_pins={path: tree_digests(state) for path, state in outcome['states'].items()})
        report['integrity'] = {'sources_unchanged': sources == source_hashes(),
            'origin_checkpoint_unchanged': sha256_file(args.checkpoint) == args.checkpoint_sha256,
            'fixture_unchanged': sha256_file(args.fixture) == args.fixture_sha256,
            'component_authority_unchanged': sha256_file(args.component_report) == args.component_report_sha256,
            'eight_optimizer_calls': outcome['optimizer_calls'] == 8,
            'both_endpoints_saved': all(any(r['trajectory'] == path and r['optimizer_updates'] == 4 and r.get('gcs') for r in report['checkpoints']) for path in (FP32, BF16)),
            'gradients_cleared': all(p.grad is None for p in model.parameters()),
            'production_flags_restored': all(getattr(model.backbone.backbone, name) == value for name, value in original_flags.items())}
        if not all(report['integrity'].values()):
            raise AssertionError('Paired NFR final integrity failed')
        report.update(status='passed_bounded_functionality', passed=True); persist('complete')
    except BaseException as error:
        failure = error
        report.update(status='failed', passed=False, error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat(); persist(report.get('stage', 'setup'))
        try:
            tracker.finish(succeeded=report['passed'])
        except BaseException as error:
            report.update(status='failed', passed=False, tracking_finish_error={'type': type(error).__name__})
            if failure is None: raise
        finally: persist(report.get('stage', 'setup'))


if __name__ == '__main__':
    main()
