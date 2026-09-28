"""Long-context benchmark selection and failure-retention guards; no GPU work."""
from types import SimpleNamespace

import pytest

from scripts import olmo_two_gpu_graph as graph
from scripts import olmo_two_gpu_single_reference as single


def argv(*extra, case='ordinary', batch=16, length=2048):
    return ['--case',case,'--batch-size',str(batch),'--length',str(length),
        '--output-dir',str(single.ROOT/'.runtime/ordinary-long-context-unit'),*extra]


def test_long_context_ordinary_preserves_actual_base_model_selection():
    args=single.parse_args(argv('--ordinary-rope-backend','dao'))
    case=graph.performance_case(args)
    assert single.long_context_ordinary(args)
    assert case.rt_layers==() and not case.fbt and not case.nextlat
    assert case.mode().num_passes==1 and not case.mode().enabled
    assert args.ordinary_attention_backend=='sdpa'
    assert not args.check_attention_parity


def test_long_context_does_not_expand_rt_only_scope():
    with pytest.raises(SystemExit): single.parse_args(argv(case='rt'))


def test_t512_defaults_keep_original_group_and_no_new_probes():
    args=single.parse_args(argv(length=512))
    assert not single.long_context_ordinary(args)
    assert args.ordinary_rope_backend=='native' and args.ordinary_attention_backend=='sdpa'
    assert not args.check_attention_parity and not args.continue_after_compatibility_miss


@pytest.mark.parametrize('extra,batch,length',[
    (['--check-attention-parity'],2,2048),
    (['--ordinary-rope-backend','dao','--ordinary-attention-backend','fa4',
      '--check-attention-parity'],16,2048),
    (['--ordinary-rope-backend','dao','--ordinary-attention-backend','fa4',
      '--check-attention-parity'],2,512),
    (['--continue-after-compatibility-miss'],2,2048),
    (['--ordinary-attention-backend','fa4'],2,2048),
    (['--ordinary-rope-backend','dao','--ordinary-attention-backend','fa4'],16,512),
])
def test_bounded_screen_and_matched_backend_guards(extra,batch,length):
    with pytest.raises(SystemExit): single.parse_args(argv(*extra,batch=batch,length=length))


def test_fa4_selection_and_both_dependency_snapshots(monkeypatch):
    args=single.parse_args(argv('--ordinary-rope-backend','dao',
        '--ordinary-attention-backend','fa4','--check-attention-parity',
        '--continue-after-compatibility-miss',batch=2))
    base=SimpleNamespace(ordinary_attention_backend='sdpa',ordinary_rope_backend='native',
        ordinary_pointwise_backend='compiled',ordinary_activation_checkpointing=True,
        unchanged=object())
    before=vars(base).copy()
    model=SimpleNamespace(backbone=SimpleNamespace(backbone=base))
    options=graph.configure_performance_model(model,args)
    assert vars(base)=={**before,'ordinary_rope_backend':'dao','ordinary_attention_backend':'fa4'}
    assert options['fused_adam'] is True and options['optimizer_arm']=='optimized'
    observed=[]
    def record(directory,**kwargs):
        observed.append((directory,kwargs));return {'pinned':True}
    monkeypatch.setattr(graph,'dependency_record',record)
    assert graph.performance_dependencies(args)=={'pinned':True}
    assert observed==[(args.output_dir,{'include_dao':True,'include_fa4':True})]


def test_protocol_addition_applies_only_to_long_ordinary(tmp_path,monkeypatch):
    base=tmp_path/'base.md';base.write_text('base')
    paths=[tmp_path/f'docs/reports/{name}/protocol.md' for name in
           ('olmo-ordinary-two-gpu','olmo-ordinary-long-context')]
    for path in paths:
        path.parent.mkdir(parents=True);path.write_text('protocol')
    monkeypatch.setattr(graph,'ROOT',tmp_path);monkeypatch.setattr(graph,'PROTOCOL',base)
    assert graph.performance_protocols(SimpleNamespace(case='ordinary',length=2048))==[base,*paths]
    assert graph.performance_protocols(SimpleNamespace(case='ordinary',length=512))==[base,paths[0]]
    assert graph.performance_protocols(SimpleNamespace(case='rt',length=2048))==[base]
    paths[1].unlink()
    with pytest.raises(FileNotFoundError):
        graph.performance_protocols(SimpleNamespace(case='ordinary',length=2048))


def failed_numeric():
    return {'name':'same_state_candidate_vs_reference','passed':False,
        'ownership_matches':True,'finite':True,'counts_equal':True,
        'losses':{'ce':{}},'outputs':{'hidden':{}},'gradients':{'weight':{}}}


def test_numeric_failure_is_retained_without_promoting_final_status():
    report={'checks':[]};persisted=[]
    check=failed_numeric()
    single.publish_check(report,check,lambda:persisted.append(len(report['checks'])),
        allow_numerical_miss=True)
    single.publish_check(report,{'name':'own_graph','passed':True},lambda:None)
    assert report['checks'][0] is check and check['passed'] is False and persisted==[1]
    assert single.final_check_summary(report['checks'])=={
        'status':'failed','numerical_compatibility_passed':False,'operational_checks_passed':True}


@pytest.mark.parametrize('mutation,enabled',[
    ({},False),({'finite':False},True),({'ownership_matches':False},True),
    ({'counts_equal':False},True),({'name':'ordinary_dispatch_no_fallback'},True),
])
def test_other_failures_remain_immediately_fatal_and_persisted(mutation,enabled):
    check={**failed_numeric(),**mutation};report={'checks':[]};persisted=[]
    with pytest.raises(AssertionError):
        single.publish_check(report,check,lambda:persisted.append(True),
            allow_numerical_miss=enabled)
    assert report['checks']==[check] and persisted==[True]
