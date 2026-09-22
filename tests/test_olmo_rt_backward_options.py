"""Historical adjoints agree with independent attention autograd; dispatch is explicit."""
from dataclasses import replace
from types import SimpleNamespace
import math
import pytest
import torch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained import olmo_tiled as tiled


@pytest.mark.parametrize("length,history,scale", [(3,5,1.), (9,7,10.), (1,3,.001)])
def test_fp32_historical_adjoint_matches_independent_attention_autograd(length,history,scale):
    torch.set_num_threads(1)
    torch.manual_seed(73317)
    config=replace(OLMoConfig.tiny(),num_heads=2)
    q=torch.randn(2,2,length,16)
    k=torch.randn(2,2,history,16,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    mask=torch.arange(history)%3!=1
    p=(q@k.transpose(-1,-2)/4.).masked_fill(~mask,-torch.inf).softmax(-1)
    a=p@v
    g=torch.randn_like(a)*scale
    expected=torch.autograd.grad(a,(k,v),g)
    spec=tiled._Invocation(config,1.,"fp32",False,torch.bfloat16)
    with torch.no_grad():
        actual=tiled._historical_backward_tile(p,g,v,q,(g*a).sum(-1),spec,torch.float32)
    for got,want in zip(actual,expected):
        torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-6)
        assert torch.count_nonzero(got[:,:,~mask])==0


@pytest.mark.parametrize("value", [None,False,"flash","TRITON"])
def test_bad_backward_backend_rejected_at_construction(value):
    with pytest.raises(ValueError,match="backward_tile_backend"):
        tiled.OLMoTiledRTForCausalLM(OLMoConfig.tiny(),backward_tile_backend=value)


def test_requested_backward_triton_rejects_cpu_before_execution():
    model=tiled.OLMoTiledRTForCausalLM(OLMoConfig.tiny(),backward_tile_backend="triton")
    from cdrm.pretrained.recurrent import RTMode
    with pytest.raises(ValueError,match="backward tiles require CUDA"):
        model(torch.tensor([[2,3]]),mode=RTMode((0,)))


@pytest.mark.parametrize("precision,qdtype,pdtype,dim,rows,cols,expected", [
    ("mixed",torch.bfloat16,torch.float32,16,3,5,"triton"),
    ("fp32",torch.bfloat16,torch.float32,16,3,5,"eager"),
    ("mixed",torch.float32,torch.float32,16,3,5,"eager"),
    ("mixed",torch.bfloat16,torch.bfloat16,16,3,5,"eager"),
    ("mixed",torch.bfloat16,torch.float32,8,3,5,"eager"),
    ("mixed",torch.bfloat16,torch.float32,16,257,5,"eager"),
    ("mixed",torch.bfloat16,torch.float32,16,3,257,"eager"),
])
def test_backward_dispatch_metadata_and_fallback(monkeypatch,precision,qdtype,pdtype,dim,rows,cols,expected):
    class Metadata:
        device=SimpleNamespace(type="cuda")
        def __init__(self,shape,dtype):self.shape,self.dtype=shape,dtype
        def transpose(self,*args):return self
    class EagerReached(Exception):pass
    def eager(*args):raise EagerReached
    from cdrm.pretrained import olmo_rt_backward_kernels
    monkeypatch.setattr(tiled,"_mm",eager)
    monkeypatch.setattr(olmo_rt_backward_kernels,"backward_tile",lambda *args:"triton")
    spec=SimpleNamespace(backward_tile_backend="triton",attention_precision=precision,
                         config=SimpleNamespace(head_dim=dim))
    args=(Metadata((1,2,rows,cols),pdtype),Metadata((1,2,rows,dim),torch.float32),
          Metadata((1,2,cols,dim),qdtype),Metadata((1,2,rows,dim),qdtype),
          Metadata((1,2,rows),torch.float32),spec,qdtype)
    if expected=="triton":assert tiled._historical_backward_tile(*args)=="triton"
    else:
        with pytest.raises(EagerReached):tiled._historical_backward_tile(*args)
