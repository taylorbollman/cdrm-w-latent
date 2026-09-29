#!/usr/bin/env python3
"""Fixed alpha0/.25 combined precision observations at saved NF fusion128."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import json
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
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from cdrm.pretrained.recurrent import RTMode
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct, global_fixture_metadata
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_position_geometry import position_geometry
from scripts.olmo_campaign_precision_bridge import capture_passes, configure_path, forward_geometry
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, TERMS, component_backward, record_gradients
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, arm_contract, fixture_pins, state_pins, gradient_norm_summary
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_component_probe import case_health, reference_checks, source_hashes as component_sources
from scripts.olmo_fusion_startup_long_probe import load_long_fixture
from scripts.olmo_fusion_startup_train import load_fusion_checkpoint
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA = 'olmo-fusion-startup-rt-strength-v1'
ALPHAS = (0.0, 0.25)


@dataclass(frozen=True)
class AlphaDiagnostic:
    """Explicit mode-only override; the checkpoint's NF recipe stays immutable.

    This is deliberately not a CampaignRecipe arm_contract. Alpha1 is accepted
    only to test equivalence with the frozen NFR helper; the GPU plan is fixed.
    """
    original_recipe: CampaignRecipe
    alpha: float

    def __post_init__(self):
        recipe = self.original_recipe
        if (not isinstance(recipe, CampaignRecipe) or recipe.arm != 'NF'
                or recipe.document_policy != 'isolated-v1'
                or recipe.mode().rt_mode.selected_layers or not recipe.rt_layers):
            raise ValueError('RT strength override requires the original isolated NF recipe')
        if type(self.alpha) not in (int, float) or self.alpha not in (*ALPHAS, 1.0):
            raise ValueError('Only alpha0/.25 and the CPU alpha1 oracle are supported')
        object.__setattr__(self, 'alpha', float(self.alpha))

    @property
    def document_policy(self):
        return self.original_recipe.document_policy

    def mode(self):
        return replace(self.original_recipe.mode(),
            rt_mode=RTMode(self.original_recipe.rt_layers, self.alpha))


def diagnostic_contract(model, diagnostic):
    if not isinstance(diagnostic, AlphaDiagnostic):
        raise TypeError('Use an explicit immutable AlphaDiagnostic')
    original = arm_contract(model, diagnostic.original_recipe)
    mode = diagnostic.mode()
    if (model.backbone.backbone.rt_implementation != 'native'
            or replace(mode, rt_mode=diagnostic.original_recipe.mode().rt_mode)
                != diagnostic.original_recipe.mode()):
        raise ValueError('RT strength may override only the native recurrence selection and alpha')
    return {'schema': SCHEMA, 'original_nf_recipe': diagnostic.original_recipe.to_dict(),
        'original_nf_import_contract': original, 'effective_arm': 'NFR',
        'actual_mode': asdict(mode), 'objective': 'combined',
        'objective_weights': dict.fromkeys(TERMS, 1.0),
        'ce_pass_weights': [.5, 1/6, 1/6, 1/6], 'auxiliary_pass_weights': [.25]*4,
        'auxiliary_cotangents': {'latent': 1.0, 'kl': 1.0},
        'denominator_scope': original['denominator_scope'],
        'scope': 'Fixed diagnostic mode override after strict NF import; not an alpha1 campaign arm contract'}


def source_hashes():
    result = component_sources()
    for name in ('scripts/olmo_fusion_startup_rt_strength.py',
                 'tests/test_fusion_startup_rt_strength.py',
                 'docs/reports/olmo-fusion-startup/rt-strength-protocol.md'):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def measure_case(model, diagnostic, fixtures, *, path, original_flags,
                 reference_gradients=None, reference_forward=None):
    if (reference_gradients is None) != (reference_forward is None):
        raise ValueError('Supply both gradient and forward references')
    contract = diagnostic_contract(model, diagnostic)
    metadata = global_fixture_metadata(model, fixtures)
    if metadata['microbatches'] != 2 or any(metadata['counts'][term] <= 0 for term in TERMS):
        raise ValueError('Two physical records with positive combined-loss targets required')
    before = rng_snapshot()
    execution = configure_path(model, original_flags, path)
    device = next(model.parameters()).device
    backend = SDPBackend.FLASH_ATTENTION if path == BF16 and device.type == 'cuda' else SDPBackend.MATH
    started = time.monotonic()
    with capture_passes(model, fixtures) as observed, sdpa_kernel(backend), torch.autocast(device.type, enabled=False):
        # Combined delegates unchanged to canonical_backward. Its only recipe
        # access is mode(); all weighting, masks and stop-gradients stay frozen.
        metrics = component_backward(model, diagnostic, fixtures, precision=execution['precision'],
                                     layout='sparse', objective='combined')
    gradients, snapshot = record_gradients(model, reference_gradients, save_cpu=reference_gradients is None,
        scope='BF16/FP32 at identical fixed alpha and state; descriptive, no acceptance-threshold change')
    forward = forward_geometry(observed, reference_forward)
    health = case_health(metrics, gradients, forward, observed, metadata, 'combined')
    health['rng_unchanged'] = rng_unchanged(before)
    row = {'alpha': diagnostic.alpha, 'effective_arm': 'NFR', 'objective': 'combined',
        'path': path, 'execution': execution, 'contract': contract, 'metrics': metrics,
        'normalized_loss_means': {term: metrics['loss_sums'][term]/metrics['counts'][term] for term in TERMS},
        'gradients' if reference_gradients is None else 'gradients_vs_fp32': gradients,
        'gradient_norms_descriptive_only': gradient_norm_summary(gradients),
        'forward_vs_fp32': forward,
        'position_geometry': position_geometry(observed, reference_forward, document_policy=diagnostic.document_policy),
        'forward_fingerprints': tree_digests([{key: record[key] for key in
            ('batch', 'token_embeddings', 'pass_hidden_states')} for record in observed]),
        'health': health, 'passed': all(health.values()),
        'elapsed_seconds_including_observation': time.monotonic()-started}
    if not row['passed']:
        raise AssertionError('RT strength case failed operational health')
    return row, snapshot, observed


def measure_bridge(model, recipe, fixtures, *, original_flags, publish):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Start from cleared gradients')
    initial, inputs, before = state_pins(model), fixture_pins(fixtures), rng_snapshot()
    recipe_pin = tree_digests(recipe.to_dict())
    trainability = {n: p.requires_grad for n, p in model.named_parameters()}
    modes = {n: m.training for n, m in model.named_modules()}
    rows = []
    reference_gradients = reference_forward = None
    try:
        for alpha in ALPHAS:
            diagnostic = AlphaDiagnostic(recipe, alpha)
            reference_gradients = reference_forward = None
            for path in (FP32, BF16):
                row, snapshot, observed = measure_case(model, diagnostic, fixtures, path=path,
                    original_flags=original_flags, reference_gradients=reference_gradients,
                    reference_forward=reference_forward)
                rows.append(row); publish(row)
                if path == FP32:
                    reference_gradients, reference_forward = snapshot, observed
            model.zero_grad(set_to_none=True)
            reference_gradients = reference_forward = snapshot = observed = None
            gc.collect()
        integrity = {'state_unchanged': state_pins(model) == initial,
            'fixture_tensors_unchanged': fixture_pins(fixtures) == inputs,
            'original_recipe_unchanged': tree_digests(recipe.to_dict()) == recipe_pin,
            'trainability_unchanged': trainability == {n: p.requires_grad for n, p in model.named_parameters()},
            'training_modes_unchanged': modes == {n: m.training for n, m in model.named_modules()},
            'rng_unchanged': rng_unchanged(before), 'four_cases_completed': len(rows) == 4}
        if not all(integrity.values()):
            raise AssertionError('RT strength diagnostic altered fixed state or fixture')
        return {'rows': rows, 'integrity': integrity}
    finally:
        model.zero_grad(set_to_none=True)
        configure_path(model, original_flags, BF16)
        reference_gradients = reference_forward = None
        gc.collect()


def load_component_reference(path, digest, sources, *, fixture_sha256, checkpoint_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024*1024 or sha256_file(path) != digest:
        raise ValueError('Component reference differs from independent bounded SHA pin')
    report = json.loads(path.read_text())
    checks = (report.get('integrity'), report.get('nf_reference_checks'), report.get('bridge', {}).get('integrity'))
    if (sha256_file(path) != digest or report.get('schema') != 'olmo-fusion-startup-component-probe-v1'
            or report.get('status') != 'passed_operational_diagnostic' or report.get('passed') is not True
            or report.get('fixture_sha256') != fixture_sha256 or report.get('checkpoint_sha256') != checkpoint_sha256
            or report.get('optimizer_updates') != 0 or report.get('aggregate_backwards') != 6
            or report.get('physical_backwards') != 12 or len(report.get('rows', [])) != 6
            or report.get('import', {}).get('counters', {}).get('optimizer_updates') != 128
            or report.get('determinism', {}).get('deterministic_algorithms') is not True
            or not all(isinstance(check, dict) and check and all(check.values()) for check in checks)
            or not report.get('sources') or any(sources.get(k) != v for k, v in report['sources'].items())):
        raise ValueError('Require complete deterministic same-state fusion128 component authority')
    anchors = {}
    for arm, objective in (('NF', 'combined'), ('NFR', 'ce'), ('NFR', 'combined')):
        for path_name in (FP32, BF16):
            matches = [row for row in report['rows'] if (row.get('arm'), row.get('objective'), row.get('path')) == (arm, objective, path_name)]
            if len(matches) != 1 or not matches[0].get('passed') or not matches[0].get('health') or not all(matches[0]['health'].values()):
                raise ValueError('Component authority lacks a unique healthy endpoint')
            row = matches[0]
            if row.get('contract', {}).get('mode', {}).get('rt_mode', {}).get('alpha') != 1.0:
                raise ValueError('Component authority must retain its declared alpha1 mode')
            anchors[arm+'/'+objective+'/'+path_name] = row
    return report, anchors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'fixture', 'component-report'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--artifacts', type=Path, default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    for value in (args.checkpoint_sha256, args.fixture_sha256, args.component_report_sha256):
        if len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            parser.error('Require independent lowercase SHA256 pins')
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error('Evidence must stay within the persistent project')
    return args


def main(argv=None):
    args = parse_args(argv)
    deterministic = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError('RT strength probe is single-GPU without DDP')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    for name in sources:
        destination = args.output_dir/'source-snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, destination)
    report = {'schema': SCHEMA, 'status': 'running', 'passed': False, 'sources': sources,
        'runtime': runtime, 'determinism': deterministic, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint_sha256': args.checkpoint_sha256, 'fixture_sha256': args.fixture_sha256,
        'component_report_sha256': args.component_report_sha256, 'alphas': list(ALPHAS), 'rows': [],
        'aggregate_backwards': 4, 'physical_backwards': 8, 'optimizer_updates': 0,
        'math_sdpa_reduced_precision_reduction': torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        'bf16_matmul_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'qualification': 'Fixed-state combined-loss RT strength diagnostic; alpha0 still scans selected layers; no BF16, training or quality clearance'}
    tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', output_dir=args.output_dir,
        group='olmo-fusion-startup', name=args.output_dir.name, preserve_state=preserve_local_rng)
    start, failure = time.monotonic(), None
    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-start, wandb=tracker.record)
        write_json(args.output_dir/'report.json', report)
    try:
        tracker.start({k: report[k] for k in ('checkpoint_sha256', 'fixture_sha256', 'component_report_sha256', 'qualification')})
        reference, anchors = load_component_reference(args.component_report, args.component_report_sha256, sources,
            fixture_sha256=args.fixture_sha256, checkpoint_sha256=args.checkpoint_sha256)
        persist('construct_original_nf')
        model, recipe, source, _, _ = construct(SimpleNamespace(scale='pretrained', length=16, artifacts=args.artifacts), 'NF', torch.device('cuda'))
        if recipe.rt_layers != (0, 15):
            raise ValueError('Production RT strength diagnostic requires layers0/15')
        original = {name: getattr(model.backbone.backbone, name) for name in RUNTIME_FLAGS}
        cold = state_pins(model)
        if any(cold[group] != reference['initial_state'][group] for group in ('backbone', 'predictor')):
            raise AssertionError('Cold backbone/predictor differ from saved component authority')
        report['import'] = load_fusion_checkpoint(model, args.checkpoint, source, expected_sha256=args.checkpoint_sha256)
        if report['import']['counters']['optimizer_updates'] != 128:
            raise ValueError('Require saved NF fusion128')
        fixtures, metadata = load_long_fixture(args.fixture, expected_sha256=args.fixture_sha256, recipe=recipe, width=model.config.model_dim)
        report.update(source_checkpoint=source, initial_state=state_pins(model), contract=arm_contract(model, recipe),
            fixture_pins=fixture_pins(fixtures), fixture_metadata=global_fixture_metadata(model, fixtures), fixture_provenance=metadata)
        report['component_reference_checks'] = reference_checks(report, reference)
        if not all(report['component_reference_checks'].values()):
            raise AssertionError('RT strength state/data/runtime differs from component authority')
        report['retained_reference_summaries'] = {key: {k: row[k] for k in ('contract', 'normalized_loss_means',
            'gradient_norms_descriptive_only', 'forward_fingerprints')}
            | ({'gradient_geometry': row['gradients_vs_fp32']['geometry']} if row['path'] == BF16 else {})
            for key, row in anchors.items() if row['objective'] == 'combined'}
        def publish(row):
            row['memory'] = memory(); report['rows'].append(row)
            stage = 'alpha'+str(row['alpha'])+'/'+row['path']; persist(stage)
            tracker.log(scalar_metrics(row, 'rt_strength/'+stage), step=len(report['rows']))
            print({'alpha': row['alpha'], 'path': row['path'], 'passed': row['passed']}, flush=True)
        bridge = measure_bridge(model, recipe, fixtures, original_flags=original, publish=publish)
        report['bridge_integrity'] = bridge['integrity']
        report['integrity'] = {'sources_unchanged': source_hashes() == sources,
            'checkpoint_unchanged': sha256_file(args.checkpoint) == args.checkpoint_sha256,
            'fixture_unchanged': sha256_file(args.fixture) == args.fixture_sha256,
            'component_reference_unchanged': sha256_file(args.component_report) == args.component_report_sha256,
            'state_unchanged': state_pins(model) == report['initial_state'],
            'gradients_cleared': all(p.grad is None for p in model.parameters()),
            'four_cases_completed': len(report['rows']) == 4,
            'production_flags_restored': all(getattr(model.backbone.backbone, key) == value for key, value in original.items())}
        if not all(report['integrity'].values()):
            raise AssertionError('RT strength final integrity failed')
        report.update(status='passed_operational_diagnostic', passed=True); persist('complete')
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
