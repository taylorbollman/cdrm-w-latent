"""CPU provenance and parameter checks before loading an RT evaluation checkpoint."""
from __future__ import annotations

from collections.abc import Mapping
import json
import re

import torch


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{label} must be a lowercase SHA256 digest')
    return value


def _metadata(value):
    """Compare checkpoint metadata using the JSON report's tuple/list convention."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('Checkpoint/report identity metadata must be finite JSON values') from error


def validate_evaluation_contract(
    training_report,
    state,
    *,
    checkpoint_sha256,
    heldout_manifest_sha256,
    current_source_sha256,
    protocol_sha256,
    expected_parameters,
):
    """Require a completed training endpoint and its original data/source identity.

    ``expected_parameters`` is ``dict(model.named_parameters())`` from the
    evaluator's one fresh model, before loading weights. Only its names, shapes
    and dtypes are read; the saved tensors must be CPU FP32 and finite. No
    accelerator operation is performed by this helper.

    The caller hashes the supplied completed training report for its evaluation
    record and hashes the frozen protocol, checkpoint and held-out manifest.
    The current source mapping must include every recorded olmo Python source;
    any other source present in both training and evaluation is checked too.
    """
    if training_report.get('schema') != 'rt-precision-training-v1' or training_report.get('status') != 'complete':
        raise ValueError('Require a completed RT precision training report')
    if state.get('schema') != 'rt-precision-state-v1':
        raise ValueError('Require a training checkpoint, not a diagnostic arm packet')
    for value, label in [(checkpoint_sha256, 'checkpoint'),
                         (heldout_manifest_sha256, 'heldout manifest'), (protocol_sha256, 'protocol')]:
        _digest(value, label)
    matching = [record for record in training_report.get('checkpoints', [])
                if record.get('sha256') == checkpoint_sha256]
    if len(matching) != 1:
        raise ValueError('Checkpoint digest must identify exactly one recorded training checkpoint')
    checkpoint = matching[0]
    completed = state.get('completed_updates')
    if (isinstance(completed, bool) or not isinstance(completed, int) or completed not in (100, 500)
            or completed != checkpoint.get('completed_updates')
            or completed != training_report.get('completed_updates')
            or completed != training_report.get('endpoint')):
        raise ValueError('Require the completed 100/500-update report endpoint, not step zero or an intermediate checkpoint')
    if state.get('next_data_row') != completed * 512 or checkpoint.get('next_data_row') != completed * 512:
        raise ValueError('Checkpoint data position does not match the completed endpoint')
    policy = training_report.get('policy')
    if policy not in ('legacy', 'bf16_fp32_state') or state.get('policy') != policy:
        raise ValueError('Training policy and checkpoint policy differ')
    if training_report.get('precision') != 'bf16':
        raise ValueError('Only the two native BF16 training arms are evaluation endpoints')
    seed = training_report.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or state.get('seed') != seed:
        raise ValueError('Training seed and checkpoint seed differ')
    config = _metadata(state.get('model_config'))
    if not isinstance(config, Mapping) or config != _metadata(training_report.get('model_config')):
        raise ValueError('Checkpoint model configuration differs from training authority')
    if config.get('recurrent_precision_policy') != policy:
        raise ValueError('Model precision policy differs from recorded training policy')
    for key, expected in [('protocol_sha256', protocol_sha256),
                          ('heldout_manifest_sha256', heldout_manifest_sha256)]:
        if training_report.get(key) != expected or state.get(key) != expected:
            raise ValueError(f'Checkpoint or training {key} differs from the frozen authority')
    contract = _metadata(training_report.get('resume_contract'))
    if not isinstance(contract, Mapping) or _metadata(state.get('resume_contract')) != contract:
        raise ValueError('Checkpoint resume identity differs from completed training authority')
    data = contract.get('data', {})
    _digest(data.get('manifest_sha256'), 'training manifest')
    if (data.get('heldout_exclusion_manifest_sha256') != heldout_manifest_sha256
            or state.get('data_manifest_sha256') != data.get('manifest_sha256')):
        raise ValueError('Held-out manifest is not the exclusion authority of this training corpus')
    if contract.get('protocol_sha256') != protocol_sha256:
        raise ValueError('Frozen protocol differs from checkpoint resume identity')
    if contract.get('model_config') != config or contract.get('seed') != state.get('seed'):
        raise ValueError('Model/seed fields disagree with the checkpoint resume identity')
    recorded_sources = training_report.get('source_sha256')
    if not isinstance(recorded_sources, Mapping) or state.get('source_sha256') != recorded_sources:
        raise ValueError('Checkpoint source identity differs from training authority')
    if contract.get('source_sha256') != recorded_sources:
        raise ValueError('Resume source identity differs from training authority')
    prefix = 'recurrent-transformer/olmo/'
    expected_model_sources = {name for name in recorded_sources if name.startswith(prefix)}
    actual_model_sources = {name for name in current_source_sha256 if name.startswith(prefix)}
    if not expected_model_sources or expected_model_sources != actual_model_sources:
        raise ValueError('Current model source coverage differs from training authority')
    checked_sources = {}
    for name in sorted(set(recorded_sources) & set(current_source_sha256)):
        _digest(recorded_sources[name], f'source {name}')
        if current_source_sha256[name] != recorded_sources[name]:
            raise ValueError(f'Current evaluation source differs from training: {name}')
        checked_sources[name] = recorded_sources[name]
    saved_parameters = state.get('model')
    names = state.get('optimizer_parameter_names')
    if (not isinstance(expected_parameters, Mapping) or not expected_parameters
            or not isinstance(saved_parameters, Mapping)
            or set(saved_parameters) != set(expected_parameters)):
        raise ValueError('Checkpoint parameter names do not cover the actual model exactly')
    if (not isinstance(names, list) or len(names) != len(set(names))
            or set(names) != set(expected_parameters)
            or contract.get('optimizer_parameter_names') != names):
        raise ValueError('Checkpoint optimizer parameter names do not match canonical model coverage')
    parameter_count = 0
    for name, target in expected_parameters.items():
        value = saved_parameters[name]
        if (not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor)
                or value.shape != target.shape or value.dtype != torch.float32
                or target.dtype != torch.float32 or value.device.type != 'cpu'):
            raise ValueError(f'Checkpoint parameter must have canonical shape and CPU FP32 dtype: {name}')
        if not torch.isfinite(value).all().item():
            raise ValueError(f'Checkpoint parameter is nonfinite: {name}')
        parameter_count += value.numel()
    return {'status': 'verified', 'checkpoint_sha256': checkpoint_sha256,
            'recorded_checkpoint_path': checkpoint.get('path'), 'completed_updates': completed,
            'seed': state['seed'], 'training_policy': policy, 'protocol_sha256': protocol_sha256,
            'heldout_manifest_sha256': heldout_manifest_sha256,
            'training_manifest_sha256': data['manifest_sha256'], 'checked_source_sha256': checked_sources,
            'parameter_tensors': len(expected_parameters), 'parameter_count': parameter_count,
            'saved_parameters_cpu_fp32_finite': True, 'canonical_parameter_coverage': True}
