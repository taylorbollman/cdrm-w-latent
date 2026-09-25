"""Actual tiny ordinary execution and shared single/DDP/ZeRO1 sweep selection."""
from types import SimpleNamespace

import pytest
import torch

from scripts import olmo_two_gpu_graph as graph
from scripts import olmo_two_gpu_single_reference as single
from scripts import olmo_two_gpu_zero1_graph as zero1
from scripts.olmo_f1_common import active_names
from scripts.olmo_two_gpu_validate import construct
from scripts.olmo_rt_large_batch import ARMS


HARNESSES = [graph, single, zero1]


def arguments(harness, case='ordinary', *, tiny=False, rope=None):
    argv = ['--case', case, '--batch-size', '2', '--length', '8' if tiny else '512',
            '--output-dir', str(harness.ROOT/'.runtime/ordinary-selection-unit')]
    if harness is graph: argv += ['--stage', 'correctness']
    if tiny: argv.append('--tiny')
    if rope is not None: argv += ['--ordinary-rope-backend', rope]
    return argv


@pytest.mark.parametrize('harness', HARNESSES)
def test_actual_tiny_ordinary_loss_executes_no_rt_fusion_or_predictor(harness, monkeypatch):
    torch.set_num_threads(1)
    args = harness.parse_args(arguments(harness, tiny=True))
    case = graph.performance_case(args)
    model = construct(case, args, torch.device('cpu'))
    options = graph.configure_performance_model(model, args)
    assert options['ordinary_rope_backend'] == 'native'
    assert case.rt_layers == () and not case.fbt and not case.nextlat
    assert not case.mode().enabled and case.mode().num_passes == 1 and not model.enabled
    def forbidden(*args, **kwargs):
        raise AssertionError('Ordinary execution entered an optional model branch')
    import cdrm.pretrained.olmo_tiled as tiled
    monkeypatch.setattr(tiled, 'tiled_recurrent_layer', forbidden)
    monkeypatch.setattr(model.backbone.fusion, 'forward', forbidden)
    assert model.predictor is None
    batch = graph.fixed_batch(case, None, 0, 0, tiny=True)
    loss = model.loss_sums(batch, backbone_kwargs={'mode': case.mode()})
    assert len(loss.pass_losses) == 1
    assert loss.counts['ce'] > 0 and loss.counts['latent'] == loss.counts['kl'] == 0
    loss.sums['ce'].backward()
    names = active_names(model, case.mode())
    assert names and all(name.startswith('backbone.backbone.') for name in names)
    assert {name for name, p in model.named_parameters() if p.grad is not None} == names
    assert not any(name.startswith('predictor.') for name, _ in model.named_parameters())
    assert all(p.grad is None for p in model.backbone.fusion.parameters())


@pytest.mark.parametrize('harness', HARNESSES)
@pytest.mark.parametrize('case', ['rt', 'combined'])
@pytest.mark.parametrize('tiny', [False, True])
def test_existing_rt_case_and_native_defaults_are_preserved(harness, case, tiny):
    args = harness.parse_args(arguments(harness, case, tiny=tiny))
    selected = graph.performance_case(args)
    assert selected.rt_layers == (0, 1 if tiny else 15)
    assert selected.fbt == selected.nextlat == (case == 'combined')
    assert graph.performance_options(args) == {
        'ordinary_rope_backend': 'native', 'optimizer_arm': 'compiled-native', 'fused_adam': True}


@pytest.mark.parametrize('harness', HARNESSES)
@pytest.mark.parametrize('case,tiny', [('rt', False), ('combined', False), ('ordinary', True)])
def test_dao_is_rejected_outside_full_size_ordinary(harness, case, tiny):
    with pytest.raises(SystemExit):
        harness.parse_args(arguments(harness, case, tiny=tiny, rope='dao'))


@pytest.mark.parametrize('harness', HARNESSES)
@pytest.mark.parametrize('rope', ['native', 'dao'])
def test_rope_only_model_change_optimizer_annotation_and_dependency_pin(harness, rope, monkeypatch):
    args = harness.parse_args(arguments(harness, rope=rope))
    base = SimpleNamespace(ordinary_rope_backend='native', ordinary_pointwise_backend='compiled',
        ordinary_attention_backend='sdpa', ordinary_activation_checkpointing=True,
        tile_backend='triton', backward_tile_backend='triton', backward_memory='recompute',
        cast_weights_once=True, reuse_rope=True, kv_only_writes=True)
    original = vars(base).copy()
    model = SimpleNamespace(backbone=SimpleNamespace(backbone=base))
    options = graph.configure_performance_model(model, args)
    assert vars(base) == {**original, 'ordinary_rope_backend': rope}
    assert options['optimizer_arm'] == ('optimized' if rope == 'dao' else 'compiled-native')
    assert options['fused_adam'] is True and ARMS[options['optimizer_arm']]['fused_adam'] is True
    seen = []
    def record(directory, **kwargs):
        seen.append((directory, kwargs)); return {'pinned': True}
    monkeypatch.setattr(graph, 'dependency_record', record)
    assert graph.performance_dependencies(args) == {'pinned': True}
    assert seen == [(args.output_dir, {'include_dao': rope == 'dao', 'include_fa4': False})]


def test_ordinary_additive_protocol_preserves_original_and_other_cases(tmp_path, monkeypatch):
    base = tmp_path/'base-protocol.md'; base.write_text('base')
    extra = tmp_path/'docs/reports/olmo-ordinary-two-gpu/protocol.md'
    extra.parent.mkdir(parents=True); extra.write_text('ordinary')
    monkeypatch.setattr(graph, 'ROOT', tmp_path)
    monkeypatch.setattr(graph, 'PROTOCOL', base)
    assert graph.performance_protocols(SimpleNamespace(case='ordinary')) == [base, extra]
    assert graph.performance_protocols(SimpleNamespace(case='rt')) == [base]
    extra.unlink()
    assert graph.performance_protocols(SimpleNamespace(case='ordinary')) == [base]
