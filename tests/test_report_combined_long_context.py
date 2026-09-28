"""Evidence checks reject mismatched objectives, dispatch, timing and tampering."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import report_combined_long_context as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)


def fixture(tmp_path, *, batch=16, historical=False):
    directory = tmp_path/('historical' if historical else 'new')
    length = 512 if historical else 2048
    counts = dict(ce=batch*(length-1), latent=batch*(length-1), kl=batch*length//2)
    tokens = batch*length
    metric = dict(counts=counts, objective_weights=dict(ce=1., latent=1., kl=1.),
                  update_completed=True, loss_means=dict(ce=1., latent=2., kl=3.))
    model = dict(nextlat_enabled=True, counts_per_pass=counts, objective_weights=metric['objective_weights'], pass_loss_gamma=1.,
        backbone=dict(model_dim=2048, num_layers=16, num_heads=16, mlp_intermediate_size=8192, vocab_size=50304, max_context_length=2048),
        nextlat=dict(model_dim=2048, proj_factor=1.6, lambda_latent=1., lambda_kl=1., vocab_chunk_size=128, ce_chunk_size=2048),
        ordinary=dict(ordinary_attention_backend='sdpa', ordinary_rope_backend='native', ordinary_pointwise_backend='compiled',
                      ordinary_activation_checkpointing=True, ordinary_checkpoint_layers=None),
        native_rt=dict(rt_implementation='native', attention_precision='mixed', tile_backend='triton', backward_tile_backend='triton',
                       backward_memory='recompute', cast_weights_once=True, reuse_rope=True, kv_only_writes=True))
    report = dict(schema='olmo-two-gpu-single-reference-v1', status='passed', stage='complete',
        case=dict(name='combined', batch_size=batch, length=length, fbt=True, nextlat=True, passes=2, rt_layers=[0,15], alpha=1., beta=1.),
        mode=dict(enabled=True, num_passes=2, beta=1., rt_mode=dict(selected_layers=[0,15], alpha=1.)),
        configuration=dict(world_size=1, ddp_buckets=False, accumulation_steps=1, tiny=False,
                           ordinary_rope_backend='native', ordinary_attention_backend='sdpa'),
        model_contract=model, checkpoint=dict(sha256=audit.CHECKPOINT_SHA), physical_updates=8,
        preparation_updates=[deepcopy(metric) for _ in range(3)],
        timed_updates=[dict(seconds=2., input_tokens=tokens, metrics=deepcopy(metric)) for _ in range(5)],
        throughput=dict(global_tokens_per_second=tokens/2, seconds_per_update=2.),
        checks=[dict(name=k, passed=True) for k in sorted(audit.OWN_CHECKS if historical else audit.NEW_CHECKS)],
        resources=dict(analytic_matrix_work=dict(ordinary_block_calls_per_microbatch=30, rt_block_calls_per_microbatch=2,
            parameter_counts=audit.PARAMETERS, input_tokens_per_update=tokens, pass_token_work_per_update=tokens*2,
            objective_positions_per_update={**counts, 'predictor':counts['latent']},
            objective_positions_across_passes={**{k:2*v for k,v in counts.items()}, 'predictor':counts['latent']*2}),
            observed_parameters={**dict.fromkeys(('registered_unique','trainable','executed_declared','gradient_participating','optimizer_owned'),
                audit.PARAMETERS['training_architecture']), 'deployable_inference_declared':audit.PARAMETERS['deployable_inference']}),
        setup_memory=dict(peak_allocated_gib=30., peak_reserved_gib=40.),
        steady_memory=dict(allocated_gib=19., reserved_gib=35., device_free_gib=40.),
        dependencies=dict(packages={'torch':'fixture'}, dao_sources={}, fa4_sources={}), runtime={}, sources={})
    if not historical:
        gate = next(r for r in report['checks'] if r['name'] == 'combined_dispatch_native_rt_and_ordinary')
        expected = dict(audit.Counter(f'{i & -i}x{i & -i}' for i in range(1,2048)))
        expected = {k:2*v for k,v in expected.items()}
        gate.update(ordinary=dict(passed=True), native_rt=dict(passed=True,
            forward_tiles_by_shape=expected, forward_triton_tiles_by_shape={k:v for k,v in expected.items() if int(k.split('x')[0]) <= 256},
            forward_eager_tiles_by_shape={'512x512':4, '1024x1024':2},
            backward_recomputed_triton_calls=4094, expected_backward_recomputed_triton_calls=4094))
    for source in audit.REQUIRED_SOURCES | (set() if historical else {audit.PROTOCOL}):
        path = directory/'source-snapshot'/source
        write(path, 'source fixture\n')
        report['sources'][source] = audit.digest(path)
    write(directory/'report.json', report)
    return directory, report


def test_full_contract_and_pending_retention_are_plot_eligible(tmp_path):
    directory, _ = fixture(tmp_path)
    row = audit.flatten_stage(directory)
    assert row['plot_eligible'] and row['gates']['complete_and_passed']
    assert row['dispatch']['passed'] and not row['retention']['verified']


@pytest.mark.parametrize('alter', ['kl_off','wrong_rt_layers','wrong_pass_count','wrong_rope','bad_timing','missing_gate','bad_dispatch','failed_status','source_tamper'])
def test_ineligible_evidence_is_retained_but_excluded(tmp_path, alter):
    directory, report = fixture(tmp_path)
    if alter == 'kl_off': report['model_contract']['nextlat']['lambda_kl'] = 0.
    elif alter == 'wrong_rt_layers': report['mode']['rt_mode']['selected_layers'] = [0]
    elif alter == 'wrong_pass_count': report['mode']['num_passes'] = 3
    elif alter == 'wrong_rope': report['configuration']['ordinary_rope_backend'] = 'dao'
    elif alter == 'bad_timing': report['timed_updates'][0]['seconds'] = 20.
    elif alter == 'missing_gate': report['checks'].pop()
    elif alter == 'bad_dispatch':
        gate = next(c for c in report['checks'] if c['name'] == 'combined_dispatch_native_rt_and_ordinary')
        gate['native_rt']['forward_eager_tiles_by_shape']['1024x1024'] = 0
    elif alter == 'failed_status': report['status'] = 'failed'
    elif alter == 'source_tamper': (directory/'source-snapshot/cdrm/pretrained/olmo_tiled.py').write_text('tampered')
    write(directory/'report.json', report)
    row = audit.flatten_stage(directory)
    assert not row['plot_eligible'] and row['plot_exclusion_reasons']


def test_b2_diagnostic_excluded_even_when_all_checks_pass(tmp_path):
    directory, _ = fixture(tmp_path, batch=2)
    row = audit.flatten_stage(directory)
    assert row['combined_contract']['passed'] and row['gates']['complete_and_passed']
    assert row['diagnostic'] and not row['plot_eligible']


def test_pool_tokens_over_seconds_not_average_rates(tmp_path):
    directory, _ = fixture(tmp_path)
    one = audit.flatten_stage(directory)
    two = deepcopy(one)
    two['stage'] = 'repeat'
    two['total_timed_seconds'] *= 2
    two['tokens_per_second'] /= 2
    pooled = audit.pooled_measurements([one,two])[0]
    assert pooled['runs'] == 2
    assert pooled['tokens_per_second'] == one['tokens_per_second']*2/3
    assert pooled['tokens_per_second'] != (one['tokens_per_second']+two['tokens_per_second'])/2


def test_historical_frozen_backend_is_not_inferred_from_name(tmp_path):
    directory, report = fixture(tmp_path, batch=128, historical=True)
    report['configuration'].pop('ordinary_rope_backend')
    report['configuration'].pop('ordinary_attention_backend')
    script_sources = {
        'scripts/olmo_two_gpu_validate.py': "def construct(case, args, device):\n    set_arm(model, 'compiled-native')\n",
        'scripts/olmo_rt_large_batch.py': "ARMS = {'compiled-native': {'attention':'sdpa', 'rope':'native'}}\n",
        'scripts/olmo_two_gpu_single_reference.py': "def run(args):\n    model = construct(args)\n",
    }
    for source, body in script_sources.items():
        path = directory/'source-snapshot'/source; write(path, body)
        report['sources'][source] = audit.digest(path)
    write(directory/'report.json', report)
    row = audit.flatten_stage(directory, historical=True)
    assert row['plot_eligible'] and row['origin'] == 'historical_T512_reference'
    path = directory/'source-snapshot/scripts/olmo_two_gpu_single_reference.py'
    path.write_text("model.ordinary_rope_backend = 'dao'\n")
    report['sources']['scripts/olmo_two_gpu_single_reference.py'] = audit.digest(path)
    write(directory/'report.json', report)
    assert not audit.flatten_stage(directory, historical=True)['plot_eligible']


def test_dependency_snapshot_tamper_detected(tmp_path):
    directory, report = fixture(tmp_path)
    path = directory/'dependency-snapshot/flash_attn/cute/interface.py'
    write(path, 'imported bytes')
    report['dependencies']['fa4_sources'] = {'interface.py':{'sha256':audit.digest(path)}}
    path.write_text('different bytes')
    assert audit.dependency_check(directory, report)['issues']


def test_output_cannot_overwrite_raw_stage(tmp_path):
    directory, _ = fixture(tmp_path)
    with pytest.raises(SystemExit):
        audit.main(['--evidence-dir',str(tmp_path),'--historical-dir',str(tmp_path/'absent'),
                    '--output-dir',str(directory/'derived')])
