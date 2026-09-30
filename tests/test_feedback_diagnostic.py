"""CPU ownership/provenance checks for diagnostic weights-only imports."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cdrm.pretrained.lm_training import TrainingCounters, parameter_layout
from scripts import olmo_feedback_diagnostic as diagnostic


class Config:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return deepcopy(self.value)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.backbone = nn.Module()
        self.backbone.backbone.config = Config({'width': 3})
        self.backbone.backbone.embedding = nn.Embedding(5, 3)
        self.backbone.fusion = nn.Linear(3, 3, bias=False)
        self.readout = nn.Linear(3, 5, bias=False)
        self.readout.weight = self.backbone.backbone.embedding.weight
        self.config = Config({'latent': True})


def checkpoint():
    model = Model().float().train()
    recipe = {'arm': 'NF'}
    counters = asdict(TrainingCounters(optimizer_updates=32, microbatches=1408,
        documents=16384, input_tokens=16777216, ce_positions=16760832,
        latent_pairs=16730817, kl_triples=16684487))
    metadata = {'parameter_layout': parameter_layout(model), 'world_size': 2,
        'configuration': {'recipe': recipe, 'backbone': model.backbone.backbone.config.to_dict(),
                          'model': model.config.to_dict()}}
    cursors = [{'rank': rank, 'world_size': 2, 'physical_batch_per_rank': 12,
        'schema': 'olmo-campaign-execution-cursor-v1',
        'cursor': {'manifest_sha256': 'a'*64, 'next_chunk': 16384,
                   'next_update': 32, 'split': 'train'}} for rank in range(2)]
    manifest = {'metadata': deepcopy(metadata), 'counters': deepcopy(counters), 'world_size': 2,
                'rank_cursors': deepcopy(cursors), 'state': {'filename': 'state.pt'}}
    payload = {'metadata': deepcopy(metadata), 'counters': deepcopy(counters),
               'rank_states': [{'rank': rank, 'data_cursor': deepcopy(cursors[rank])} for rank in range(2)],
               'module_training': {name: module.training for name, module in model.named_modules()},
               'model': {name: value.clone() for name, value in model.state_dict().items()},
               'optimizer': {'must_not_be_restored': 'sentinel'},
               'scheduler': {'must_not_be_restored': 'sentinel'}}
    spec = {'recipe': Config(recipe)}
    return model, payload, manifest, spec


def test_valid_payload_preserves_parameter_ownership_without_touching_model():
    model, payload, manifest, _ = checkpoint()
    before = {name: (id(value), value.detach().clone()) for name, value in model.named_parameters()}
    assert diagnostic.validate_weights_payload(payload, manifest, model) is None
    for name, parameter in model.named_parameters():
        assert id(parameter) == before[name][0]
        assert torch.equal(parameter, before[name][1])
        assert parameter.grad is None


@pytest.mark.parametrize('mutation', ['metadata', 'counters', 'rank_order', 'rank_cursor',
    'module_keys', 'module_bool', 'tensor_keys', 'tensor_shape', 'tensor_dtype',
    'layout_name', 'layout_dtype', 'layout_requires_grad', 'tied_values'])
def test_rejects_ownership_shape_dtype_rank_counter_and_alias_disagreement(mutation):
    model, payload, manifest, _ = checkpoint()
    first = next(iter(payload['model']))
    if mutation == 'metadata':
        payload['metadata']['extra'] = True
    elif mutation == 'counters':
        payload['counters']['optimizer_updates'] = 31
    elif mutation == 'rank_order':
        payload['rank_states'].reverse()
    elif mutation == 'rank_cursor':
        payload['rank_states'][0]['data_cursor']['cursor']['next_chunk'] += 1
    elif mutation == 'module_keys':
        del payload['module_training']['readout']
    elif mutation == 'module_bool':
        payload['module_training']['readout'] = 1
    elif mutation == 'tensor_keys':
        del payload['model'][first]
    elif mutation == 'tensor_shape':
        payload['model'][first] = payload['model'][first].flatten()
    elif mutation == 'tensor_dtype':
        payload['model'][first] = payload['model'][first].bfloat16()
    elif mutation.startswith('layout_'):
        key = mutation.removeprefix('layout_')
        value = {'name': 'foreign', 'dtype': 'torch.bfloat16', 'requires_grad': False}[key]
        manifest['metadata']['parameter_layout'][0][key] = value
        payload['metadata'] = deepcopy(manifest['metadata'])
    else:
        payload['model']['readout.weight'].add_(1)
    with pytest.raises(ValueError):
        diagnostic.validate_weights_payload(payload, manifest, model)


@pytest.mark.parametrize('mutation', ['empty_aliases', 'omitted_alias', 'duplicate_layout',
                                     'negative_counter', 'bool_counter'])
def test_rejects_consistently_malformed_metadata_not_just_mismatched_copies(mutation):
    model, payload, manifest, _ = checkpoint()
    layout = manifest['metadata']['parameter_layout']
    if mutation == 'empty_aliases':
        layout[0]['aliases'] = []
    elif mutation == 'omitted_alias':
        layout[0]['aliases'] = layout[0]['aliases'][:1]
    elif mutation == 'duplicate_layout':
        layout.append(deepcopy(layout[0]))
    else:
        manifest['counters']['optimizer_updates'] = -1 if mutation == 'negative_counter' else True
    payload['metadata'] = deepcopy(manifest['metadata'])
    payload['counters'] = deepcopy(manifest['counters'])
    with pytest.raises(ValueError):
        diagnostic.validate_weights_payload(payload, manifest, model)


def test_weights_only_import_copies_saved_weights_without_replacing_parameters_or_modes(tmp_path, monkeypatch):
    model, payload, manifest, spec = checkpoint()
    # Make saved tied values distinct from the currently constructed state.
    for value in payload['model'].values():
        value.add_(4)
    torch.save(payload, tmp_path/'state.pt')
    inspection = []
    def inspect(directory, **kwargs):
        inspection.append((directory, kwargs))
        return manifest
    monkeypatch.setattr(diagnostic, 'inspect_distributed_checkpoint', inspect)
    names = {name: id(value) for name, value in model.named_parameters()}
    torch.manual_seed(744)
    rng = torch.get_rng_state().clone()
    result = diagnostic.import_saved_weights(model, spec, tmp_path, 'a'*64)
    assert inspection == [(tmp_path, {'expected_manifest_sha256': 'a'*64, 'verify_state': True})]
    assert names == {name: id(value) for name, value in model.named_parameters()}
    assert model.readout.weight is model.backbone.backbone.embedding.weight
    assert all(torch.equal(value, payload['model'][name]) for name, value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training and all(parameter.grad is None for parameter in model.parameters())
    assert result['optimizer_updates'] == 0 and result['saved_world_size'] == 2


@pytest.mark.parametrize('field', ['recipe', 'backbone', 'model'])
def test_import_rejects_checkpoint_architecture_or_recipe_before_loading_payload(tmp_path, monkeypatch, field):
    model, _, manifest, spec = checkpoint()
    manifest['metadata']['configuration'][field] = {'wrong': True}
    monkeypatch.setattr(diagnostic, 'inspect_distributed_checkpoint', lambda *args, **kwargs: manifest)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('Invalid architecture loaded a payload'))
    with pytest.raises(ValueError, match='architecture/recipe'):
        diagnostic.import_saved_weights(model, spec, tmp_path, 'a'*64)


@pytest.mark.parametrize('field', ['update', 'inputs', 'cursor_update', 'cursor_chunk'])
def test_import_rejects_wrong_diagnostic_boundary_before_loading_payload(tmp_path, monkeypatch, field):
    model, _, manifest, spec = checkpoint()
    if field == 'update':
        manifest['counters']['optimizer_updates'] = 16
    elif field == 'inputs':
        manifest['counters']['input_tokens'] -= 1
    elif field == 'cursor_update':
        manifest['rank_cursors'][1]['cursor']['next_update'] = 31
    else:
        manifest['rank_cursors'][1]['cursor']['next_chunk'] -= 1
    monkeypatch.setattr(diagnostic, 'inspect_distributed_checkpoint', lambda *args, **kwargs: manifest)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('Invalid boundary loaded a payload'))
    with pytest.raises(ValueError, match='origin0/endpoint32'):
        diagnostic.import_saved_weights(model, spec, tmp_path, 'a'*64)


def test_source_inventory_checks_every_frozen_source_and_records_new_helper_pins(tmp_path, monkeypatch):
    frozen = {}
    for index in range(200):
        name = f'frozen/source-{index}.py'
        path = tmp_path/name
        path.parent.mkdir(exist_ok=True)
        path.write_text(f'# source {index}\n')
        frozen[name] = diagnostic.sha256_file(path)
    inventory = tmp_path/'.runtime/olmo-pilot-async/runtime-sources.json'
    inventory.parent.mkdir(parents=True)
    inventory.write_text(json.dumps(frozen))
    added = ('scripts/olmo_feedback_diagnostic.py', 'scripts/olmo_feedback_fixture.py',
             'cdrm/pretrained/feedback_gradient_probe.py', 'cdrm/pretrained/feedback_forward_probe.py',
             'docs/reports/olmo-feedback-diagnostic/protocol.md')
    for name in added:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name+'\n')
    monkeypatch.setattr(diagnostic, 'ROOT', tmp_path)
    observed = diagnostic.source_inventory()
    assert observed['frozen_count'] == 200
    assert observed['frozen_inventory_sha256'] == diagnostic.sha256_file(inventory)
    assert observed['new_sources'] == {name: diagnostic.sha256_file(tmp_path/name) for name in added}
    (tmp_path/'frozen/source-199.py').write_text('mutated final source\n')
    with pytest.raises(ValueError, match='Frozen runtime changed'):
        diagnostic.source_inventory()
