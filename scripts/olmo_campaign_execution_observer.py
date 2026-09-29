"""Explicit lean or exact-acceptance observations outside the training math path.

The caller owns backward, clipping/Adam, cursor commitment, collective ordering,
checkpoint guards and durable logging. No forward or optimizer call occurs here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import math
from pathlib import Path
import time

import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.distributed_checkpoint import _local_rng, _restore_local_rng
from cdrm.pretrained.lm_training import TERMS, TrainingCounters
from scripts.olmo_campaign_restart import boundary
from scripts.olmo_lm_common import tree_digests

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/reports/olmo-campaign-execution/observation.md'
SCHEMA = 'olmo-campaign-execution-observation-v1'


def source_hashes():
    paths = (Path(__file__), ROOT/'tests/test_campaign_execution_observer.py', PROTOCOL,
             ROOT/'cdrm/pretrained/distributed_checkpoint.py', ROOT/'cdrm/pretrained/lm_training.py',
             ROOT/'scripts/olmo_campaign_restart.py', ROOT/'scripts/olmo_f1_common.py',
             ROOT/'scripts/olmo_lm_common.py')
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def json_copy(value):
    """Copy ordinary report values without converting/scanning any tensors."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Observation contains a nonfinite scalar')
        return value
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError('Observation keys must be strings')
        return {key: json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_copy(item) for item in value]
    raise TypeError('Observation expects JSON-native scalars, not tensors or state objects')


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def validate_metrics(metrics, counters):
    """Validate returned step telemetry without inspecting model/Adam tensors."""
    if not isinstance(counters, TrainingCounters):
        raise TypeError('Pass the current TrainingCounters explicitly')
    current = asdict(counters)
    TrainingCounters(**current)
    result = json_copy(metrics)
    if not isinstance(result, dict):
        raise TypeError('Completed step metrics must be a mapping')
    required = {'loss_sums', 'counts', 'objective', 'microbatches', 'documents', 'input_tokens',
                'gradient_norm_before_clip', 'lr_used', 'lr_next', 'counters'}
    if not required <= result.keys():
        raise ValueError('Completed step telemetry is incomplete')
    if (set(result['counts']) != set(TERMS) or set(result['loss_sums']) != set(TERMS)
            or any(not _nonnegative_int(result['counts'][term]) for term in TERMS)
            or any(type(result['loss_sums'][term]) not in (int, float) for term in TERMS)
            or result['counts']['ce'] <= 0):
        raise ValueError('Completed objective counts or loss sums differ from campaign contract')
    if (any(not _nonnegative_int(result[key]) for key in ('microbatches','documents','input_tokens'))
            or result['microbatches'] <= 0 or result['input_tokens'] <= 0
            or any(value > result['input_tokens'] for value in result['counts'].values())
            or result['documents'] > result['input_tokens']):
        raise ValueError('Completed update counts are invalid')
    if (type(result['objective']) not in (int, float)
            or type(result['gradient_norm_before_clip']) not in (int, float)
            or result['gradient_norm_before_clip'] < 0):
        raise ValueError('Completed objective or preclip norm is invalid')
    for key in ('lr_used','lr_next'):
        if not isinstance(result[key], list) or not result[key] or any(
                type(value) not in (int, float) or value < 0 for value in result[key]):
            raise ValueError('Completed learning rates are invalid')
    if len(result['lr_used']) != len(result['lr_next']):
        raise ValueError('Optimizer group learning-rate counts differ')
    if result['counters'] != current or current['optimizer_updates'] <= 0:
        raise ValueError('Completed metric counters differ from current committed counters')
    pairs = {'microbatches':'microbatches','documents':'documents','input_tokens':'input_tokens'}
    if any(result[name] > current[field] for name, field in pairs.items()) or any(
            result['counts'][term] > current[field] for term,field in
            (('ce','ce_positions'),('latent','latent_pairs'),('kl','kl_triples'))):
        raise ValueError('Per-update counts exceed cumulative counters')
    return result


@contextmanager
def preserve_observation_state(model, device, generators):
    """Restore local and named RNG plus heterogeneous module modes on errors too."""
    rng = _local_rng(device, generators)
    modes = tuple((module, module.training) for module in model.modules())
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training
        _restore_local_rng(rng, device, generators)


class ExecutionObserver:
    def __init__(self, mode, *, model, optimizer, scheduler, device, generators=None,
                 max_grad_norm=None):
        if mode not in ('lean','acceptance'):
            raise ValueError('Choose lean or acceptance observation explicitly')
        if max_grad_norm is not None and (type(max_grad_norm) not in (int,float)
                or not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError('Clip limit must be positive finite or None')
        self.mode, self.model, self.optimizer, self.scheduler = mode, model, optimizer, scheduler
        self.device = torch.device(device)
        if self.device.type not in ('cpu','cuda'):
            raise ValueError('Observation supports CPU or the current rank CUDA device')
        self.generators = {} if generators is None else dict(generators)
        self.max_grad_norm = max_grad_norm
        self._backward_seconds = None

    def after_backward(self):
        """Read preclip gradients in acceptance; lean does not enumerate tensors."""
        started = time.perf_counter()
        try:
            if self.mode == 'lean':
                return None
            with preserve_observation_state(self.model,self.device,self.generators):
                return tree_digests({name: parameter.grad for name,parameter in self.model.named_parameters()
                                     if parameter.requires_grad})
        finally:
            self._backward_seconds = time.perf_counter()-started

    def after_update(self, metrics, *, counters, cursor, input_record=None, raw_gradients=None):
        """Call after the same step and committed cursor in either observation mode."""
        started = time.perf_counter()
        if self._backward_seconds is None:
            raise ValueError('Call after_backward before observing the completed update')
        result = validate_metrics(metrics,counters)
        cursor = json_copy(cursor)
        if not isinstance(cursor,dict):
            raise TypeError('Completed cursor must be a JSON mapping')
        row = {'schema':SCHEMA,'mode':self.mode,'metrics':result,'cursor':cursor,
               'loss_means': {term: result['loss_sums'][term]/result['counts'][term]
                              if result['counts'][term] else None for term in TERMS},
               'clipping': {'configured_limit':self.max_grad_norm,
                    'norm_exceeds_limit':self.max_grad_norm is not None and result['gradient_norm_before_clip'] > self.max_grad_norm,
                    'coefficient_estimate':1.0 if self.max_grad_norm is None else min(1.0,
                        self.max_grad_norm/(result['gradient_norm_before_clip']+1e-6)),
                    'scope':'Host estimate from returned norm; actual clipping remains the unchanged runner.step'}}
        if self.mode == 'acceptance':
            if not isinstance(input_record,dict) or not input_record or not isinstance(raw_gradients,dict) or not raw_gradients:
                raise ValueError('Acceptance needs caller input evidence and preclip gradient hashes')
            row['input'] = json_copy(input_record)
            row['raw_gradients'] = json_copy(raw_gradients)
            with preserve_observation_state(self.model,self.device,self.generators):
                row['boundary'] = boundary(self.model,self.optimizer,self.scheduler,counters,
                                           cursor,self.device,self.generators)
        elif input_record is not None or raw_gradients is not None:
            raise ValueError('Lean observation must not receive acceptance input or gradient hashes')
        row['observation_seconds'] = {'after_backward':self._backward_seconds,
                                      'after_update':time.perf_counter()-started}
        self._backward_seconds = None
        return row

    def assert_reference(self, observation, expected):
        """Compare exact semantic evidence; caller must persist before this gate."""
        if self.mode != 'acceptance' or observation.get('mode') != 'acceptance' or expected.get('mode') != 'acceptance':
            raise ValueError('Exact reference comparison requires acceptance observations')
        required = {'metrics','cursor','input','raw_gradients','boundary','loss_means','clipping'}
        if any(row.get('schema') != SCHEMA or not required <= row.keys() for row in (observation,expected)):
            raise ValueError('Exact reference comparison requires complete acceptance evidence')
        left = {key:value for key,value in observation.items() if key != 'observation_seconds'}
        right = {key:value for key,value in expected.items() if key != 'observation_seconds'}
        if left != right:
            raise ValueError('Acceptance observation differs from its exact reference')
