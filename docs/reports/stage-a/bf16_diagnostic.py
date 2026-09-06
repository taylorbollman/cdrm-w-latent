import sys, os, json, math
from pathlib import Path
ROOT=Path('/workspace/cdrm-w-latent')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'recurrent-transformer')]
os.environ['CDRM_TEST_DEVICE']='cuda'
import torch
from stage_a_common import require_cuda_container, seed_all
from tests.test_recurrent_foundation import tiny
from olmo.model import OLMo
from olmo.checkpoint_conversion import convert_model
require_cuda_container()
seed_all(937)
torch.set_float32_matmul_precision('highest')
naive=OLMo(tiny(recurrent_layers=[3],include_bias=False,bias_for_layer_norm=False))
tiled=OLMo(tiny(recurrent_layers=[3],recurrent_backend='tiled',include_bias=False,bias_for_layer_norm=False))
convert_model(naive,tiled)
tokens=torch.randint(1,32,(2,9),device='cuda')
# Preserve the first failing test's BF16 cotangent exactly.
with torch.autocast('cuda',dtype=torch.bfloat16):
    a=naive(tokens).logits
    b=tiled(tokens).logits
cot=torch.randn_like(a)
outs={}
grads={}
for name,model,bf16 in [('fp32',naive,False),('naive_bf16',naive,True),('tiled_bf16',tiled,True)]:
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
        out=model(tokens).logits
    out.backward(cot.to(out.dtype))
    outs[name]=out.detach().float().cpu()
    grads[name]={n:p.grad.detach().float().cpu() for n,p in model.named_parameters()}

def metrics(ref,value):
    d=(ref-value).double(); ref=ref.double()
    rms=ref.square().mean().sqrt().item()
    rel=d.norm().item()/max(ref.norm().item(),1e-30)
    return dict(max_abs=d.abs().max().item(), reference_rms=rms,relative_l2=rel,
        max_normalized_abs=d.abs().max().item()/max(rms,1e-30),
        first_tolerance_violations=int((d.abs()>.003+.03*ref.abs()).sum()),numel=ref.numel())
report={'cotangent':'same BF16 random cotangent for all three modes','seed':937,'comparisons':{}}
for mode in ['naive_bf16','tiled_bf16']:
    rows={n:metrics(grads['fp32'][n],g) for n,g in grads[mode].items()}
    report['comparisons'][mode]={'logits':metrics(outs['fp32'],outs[mode]),'parameter_gradients':rows}
    print(mode,'max l2',max((m['relative_l2'],n) for n,m in rows.items()),'max rms',max((m['max_normalized_abs'],n) for n,m in rows.items()))
    print('head',rows['transformer.ff_out.weight'])
rows={n:metrics(grads['naive_bf16'][n],g) for n,g in grads['tiled_bf16'].items()}
report['comparisons']['between_bf16']={'parameter_gradients':rows}
print('between','max l2',max((m['relative_l2'],n) for n,m in rows.items()),'max rms',max((m['max_normalized_abs'],n) for n,m in rows.items()))
print('head',rows['transformer.ff_out.weight'])
(ROOT/'docs/reports/stage-a/bf16_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
