"""Explicit RT mode overrides preserve the established combined objective."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from scripts import olmo_fusion_startup_component_probe as old
from scripts import olmo_fusion_startup_rt_strength as probe
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, state_pins, fixture_pins


@pytest.fixture(autouse=True)
def cpu_rng(monkeypatch):
    torch.set_num_threads(1)
    for module in (old, probe):
        monkeypatch.setattr(module, 'rng_snapshot', lambda: torch.get_rng_state().clone())
        monkeypatch.setattr(module, 'rng_unchanged', lambda before: torch.equal(before, torch.get_rng_state()))


def tiny():
    model, recipe, _, ids, eos = construct(SimpleNamespace(scale='tiny', length=8), 'NF', torch.device('cpu'))
    model.backbone.backbone.attention_precision = 'mixed'
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0, length=8,
        token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
    flags = {name: getattr(model.backbone.backbone, name) for name in RUNTIME_FLAGS}
    return model, recipe, fixtures, flags


def test_override_is_immutable_explicit_and_does_not_rewrite_import_recipe():
    model, recipe, _, _ = tiny()
    before = recipe.to_dict()
    override = probe.AlphaDiagnostic(recipe, .25)
    contract = probe.diagnostic_contract(model, override)
    assert contract['original_nf_recipe'] == before and recipe.to_dict() == before
    assert contract['original_nf_import_contract']['mode']['rt_mode']['selected_layers'] == ()
    assert contract['actual_mode']['rt_mode'] == {'selected_layers': (0, 1), 'alpha': .25}
    assert contract['objective'] == 'combined' and contract['auxiliary_cotangents'] == {'latent': 1., 'kl': 1.}
    assert replace(override.mode(), rt_mode=recipe.mode().rt_mode) == recipe.mode()
    with pytest.raises(FrozenInstanceError):
        override.alpha = 1.
    with pytest.raises(ValueError, match='original isolated NF'):
        probe.AlphaDiagnostic(replace(recipe, arm='NFR'), .25)
    for alpha in (True, -.1, .1, 2., float('nan')):
        with pytest.raises(ValueError):
            probe.AlphaDiagnostic(recipe, alpha)


def test_alpha1_wrapper_exactly_reproduces_old_nfr_combined_forward_and_every_gradient():
    model, recipe, fixtures, flags = tiny()
    nfr, _ = old.enable_native_rt(model, recipe)
    expected, expected_grads, _ = old.measure_case(model, nfr, fixtures, objective='combined', path=FP32, original_flags=flags)
    actual, actual_grads, _ = probe.measure_case(model, probe.AlphaDiagnostic(recipe, 1.), fixtures, path=FP32, original_flags=flags)
    assert actual['metrics'] == expected['metrics']
    assert actual['forward_fingerprints'] == expected['forward_fingerprints']
    assert actual_grads.keys() == expected_grads.keys()
    assert all(torch.equal(actual_grads[name], expected_grads[name]) for name in actual_grads)


def test_alpha0_matches_ordinary_limit_at_existing_fp32_roundoff_budgets():
    model, recipe, fixtures, flags = tiny()
    expected, expected_grads, expected_forward = old.measure_case(model, recipe, fixtures,
        objective='combined', path=FP32, original_flags=flags)
    actual, actual_grads, observed = probe.measure_case(model, probe.AlphaDiagnostic(recipe, 0.),
        fixtures, path=FP32, original_flags=flags)
    assert actual['contract']['actual_mode']['rt_mode']['selected_layers'] == (0, 1)
    assert actual['metrics']['objective'] == pytest.approx(expected['metrics']['objective'], rel=4e-6, abs=2e-6)
    for new, old_record in zip(observed, expected_forward):
        for hidden, expected_hidden in zip(new['pass_hidden_states'], old_record['pass_hidden_states']):
            torch.testing.assert_close(hidden, expected_hidden, rtol=4e-6, atol=2e-6)
    # Existing tiled RT parameter-gradient FP32 budget, not a new BF16 budget.
    for name in actual_grads:
        torch.testing.assert_close(actual_grads[name], expected_grads[name], rtol=1e-4, atol=8e-6, msg=name)


def test_four_cases_use_requested_modes_without_changing_state_or_fixtures(monkeypatch):
    model, recipe, fixtures, flags = tiny()
    before, inputs, recipe_before = state_pins(model), fixture_pins(fixtures), recipe.to_dict()
    calls = []
    backward = probe.component_backward
    def record(model, actual_recipe, *args, **kwargs):
        calls.append((actual_recipe.mode().rt_mode.alpha, actual_recipe.mode().rt_mode.selected_layers,
                      kwargs['objective'], kwargs['precision']))
        return backward(model, actual_recipe, *args, **kwargs)
    monkeypatch.setattr(probe, 'component_backward', record)
    result = probe.measure_bridge(model, recipe, fixtures, original_flags=flags, publish=lambda row: None)
    assert calls == [(alpha, (0, 1), 'combined', precision) for alpha in (0., .25) for precision in ('fp32', 'bf16_mixed')]
    assert all(result['integrity'].values()) and len(result['rows']) == 4
    assert all(row['passed'] and row['gradient_norms_descriptive_only']['predictor'] > 0 for row in result['rows'])
    assert state_pins(model) == before and fixture_pins(fixtures) == inputs and recipe.to_dict() == recipe_before
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(getattr(model.backbone.backbone, name) == value for name, value in flags.items())


def test_publish_failure_restores_runtime_and_clears_partial_gradients():
    model, recipe, fixtures, flags = tiny()
    def fail(_):
        raise OSError('deliberate logging failure')
    with pytest.raises(OSError):
        probe.measure_bridge(model, recipe, fixtures, original_flags=flags, publish=fail)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(getattr(model.backbone.backbone, name) == value for name, value in flags.items())


def reference():
    return {'schema': 'olmo-fusion-startup-component-probe-v1', 'status': 'passed_operational_diagnostic',
        'passed': True, 'checkpoint_sha256': 'a'*64, 'fixture_sha256': 'b'*64,
        'optimizer_updates': 0, 'aggregate_backwards': 6, 'physical_backwards': 12,
        'import': {'counters': {'optimizer_updates': 128}}, 'determinism': {'deterministic_algorithms': True},
        'integrity': {'state': True}, 'nf_reference_checks': {'state': True}, 'bridge': {'integrity': {'state': True}},
        'sources': {'old.py': 'c'*64}, 'rows': [{'arm': arm, 'objective': objective, 'path': path,
            'contract': {'mode': {'rt_mode': {'alpha': 1.0}}}, 'passed': True, 'health': {'finite': True}}
            for arm, objective in (('NF', 'combined'), ('NFR', 'ce'), ('NFR', 'combined')) for path in (FP32, BF16)]}


@pytest.mark.parametrize('mutation', [None, 'checkpoint', 'fixture', 'source', 'count', 'endpoint',
    'health', 'integrity', 'bridge', 'determinism', 'duplicate', 'alpha'])
def test_component_reference_pins_all_case_and_endpoint_authorities(tmp_path, mutation):
    report = reference()
    if mutation == 'checkpoint': report['checkpoint_sha256'] = 'd'*64
    elif mutation == 'fixture': report['fixture_sha256'] = 'd'*64
    elif mutation == 'source': report['sources']['old.py'] = 'd'*64
    elif mutation == 'count': report['physical_backwards'] = 8
    elif mutation == 'endpoint': report['import']['counters']['optimizer_updates'] = 32
    elif mutation == 'health': report['rows'][0]['health'] = {}
    elif mutation == 'integrity': report['integrity']['state'] = False
    elif mutation == 'bridge': report['bridge']['integrity']['state'] = False
    elif mutation == 'determinism': report['determinism']['deterministic_algorithms'] = False
    elif mutation == 'duplicate': report['rows'][1] = copy.deepcopy(report['rows'][0])
    elif mutation == 'alpha': report['rows'][0]['contract']['mode']['rt_mode']['alpha'] = .25
    path = tmp_path/'reference.json'; path.write_text(json.dumps(report))
    args = (path, sha256_file(path), {'old.py': 'c'*64, 'new.py': 'e'*64})
    kwargs = {'fixture_sha256': 'b'*64, 'checkpoint_sha256': 'a'*64}
    if mutation:
        with pytest.raises(ValueError):
            probe.load_component_reference(*args, **kwargs)
    else:
        actual, anchors = probe.load_component_reference(*args, **kwargs)
        assert actual == report and len(anchors) == 6
