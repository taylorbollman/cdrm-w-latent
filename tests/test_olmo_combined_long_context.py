"""Combined T2048 benchmark guards and actual backend dispatch evidence."""
from types import SimpleNamespace

import pytest

from scripts import olmo_two_gpu_graph as graph
from scripts import olmo_two_gpu_single_reference as single
from scripts.olmo_f1_common import IntegrationCase


def argv(*extra,length=2048,batch=2):
    return ['--case','combined','--length',str(length),'--batch-size',str(batch),
        '--output-dir',str(single.ROOT/'.runtime/combined-long-context-unit'),*extra]


def test_combined_long_context_keeps_actual_native_rt_nextlat_k2_contract():
    args=single.parse_args(argv())
    case=graph.performance_case(args)
    mode=case.mode()
    assert single.long_context_combined(args) and not single.long_context_ordinary(args)
    assert case.rt_layers==(0,15) and case.fbt and case.nextlat
    assert mode.enabled and mode.num_passes==2 and mode.beta==mode.rt_mode.alpha==1
    assert graph.performance_options(args)=={
        'ordinary_rope_backend':'native','ordinary_attention_backend':'sdpa',
        'optimizer_arm':'compiled-native','fused_adam':True}


@pytest.mark.parametrize('extra,length,batch',[
    (['--ordinary-rope-backend','dao'],2048,2),
    (['--ordinary-attention-backend','fa4'],512,2),
    (['--ordinary-attention-backend','fa4'],2048,2),
    (['--check-attention-parity'],2048,2),
    (['--ordinary-attention-backend','fa4','--check-attention-parity'],2048,4),
    (['--tiny','--ordinary-attention-backend','fa4'],2048,2),
])
def test_combined_rejects_new_unmatched_or_unbounded_scope(extra,length,batch):
    with pytest.raises(SystemExit): single.parse_args(argv(*extra,length=length,batch=batch))


def test_combined_preserves_all_runtime_flags_and_pins_no_new_backend(monkeypatch):
    args=single.parse_args(argv())
    base=SimpleNamespace(ordinary_rope_backend='native',ordinary_attention_backend='sdpa',
        tile_backend='triton',backward_tile_backend='triton',backward_memory='recompute',
        cast_weights_once=True,reuse_rope=True,kv_only_writes=True)
    before=vars(base).copy()
    model=SimpleNamespace(backbone=SimpleNamespace(backbone=base))
    graph.configure_performance_model(model,args)
    assert vars(base)==before
    observed=[]
    monkeypatch.setattr(graph,'dependency_record',lambda directory,**kwargs:observed.append(kwargs))
    graph.performance_dependencies(args)
    assert observed==[{'include_dao':False,'include_fa4':False}]


def test_combined_protocol_required_only_for_new_context(tmp_path,monkeypatch):
    base=tmp_path/'base.md';base.write_text('base')
    monkeypatch.setattr(graph,'ROOT',tmp_path);monkeypatch.setattr(graph,'PROTOCOL',base)
    assert graph.performance_protocols(SimpleNamespace(case='combined',length=512))==[base]
    with pytest.raises(FileNotFoundError,match='Combined long-context'):
        graph.performance_protocols(SimpleNamespace(case='combined',length=2048))
    extra=tmp_path/'docs/reports/olmo-combined-long-context/protocol.md'
    extra.parent.mkdir(parents=True);extra.write_text('frozen')
    assert graph.performance_protocols(SimpleNamespace(case='combined',length=2048))==[base,extra]


@pytest.mark.parametrize('omit_backward,ordinary_passed',[(False,True),(True,True),(False,False)])
def test_dispatch_requires_real_kernel_calls_and_ordinary_gate(monkeypatch,omit_backward,ordinary_passed):
    from cdrm.pretrained import olmo_tiled,olmo_rt_kernels,olmo_rt_recompute_kernels
    initialized=[]
    mode=IntegrationCase('combined',fbt=True,nextlat=True,rt_layers=(0,15),length=2048).mode()
    plan=SimpleNamespace(batch=SimpleNamespace(input_ids=SimpleNamespace(shape=(2,2048))),
        mode=mode,initialize_gradients=lambda:initialized.append(True))
    monkeypatch.setattr(olmo_rt_kernels,'add_tile',lambda *args,**kwargs:None)
    monkeypatch.setattr(olmo_rt_recompute_kernels,'backward_recomputed_tile',lambda *args,**kwargs:None)
    def simulated_forward(query,key):
        if max(query.shape[-2],key.shape[-2])<=256:
            olmo_rt_kernels.add_tile(query,key)
    monkeypatch.setattr(olmo_tiled,'_add_tile',simulated_forward)
    def simulated_dispatch(plan,arm):
        assert initialized==[True]
        for _ in range(2):
            for boundary in range(1,2048):
                width=boundary & -boundary
                tensor=SimpleNamespace(shape=(2,16,width,128))
                olmo_tiled._add_tile(tensor,tensor)
                if not omit_backward:
                    olmo_rt_recompute_kernels.backward_recomputed_tile()
        return {'passed':ordinary_passed}
    monkeypatch.setattr(single,'dispatch_probe',simulated_dispatch)
    check=single.combined_dispatch_probe(plan,'compiled-native')
    assert check['passed'] is (ordinary_passed and not omit_backward)
    native=check['native_rt']
    assert sum(native['forward_tiles_by_shape'].values())==4094
    assert sum(native['forward_triton_tiles_by_shape'].values())==4088
    assert native['forward_eager_tiles_by_shape']=={'512x512':4,'1024x1024':2}
    assert native['backward_recomputed_triton_calls']==(0 if omit_backward else 4094)
