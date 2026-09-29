"""CPU authority checks for the versioned SSD executor; no CUDA execution."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
import torch
from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import ARMS,build_campaign_adamw
from scripts import olmo_campaign_ssd_execute as cli
from test_campaign_eval_execute import declaration,args
from test_campaign_eval_control import evaluation_paths


@pytest.mark.parametrize('arm',ARMS)
def test_storage_identity_preserves_model_contract_and_binds_keep_policy(declaration,arm):
    torch.set_num_threads(1)
    value,path=declaration;value['arm']=arm;path.write_text(json.dumps(value))
    spec=cli.load_spec(args(path,arm))
    model,_,_=cli.construct(spec,torch.device('cpu'))
    optimizer=build_campaign_adamw(model,spec['recipe'],fused=False)
    from scripts import olmo_campaign_eval_execute as old
    parameters=(spec,spec['recipe'],model,optimizer,{'device':'cpu'},{'test':True},{'fixture':'0'*64})
    identity,ownership=cli.construct_identity(*parameters)
    previous,old_ownership=old.construct_identity(*parameters)
    expected=deepcopy(previous['payload']);expected['storage_policy']=cli.storage_policy(spec)
    assert identity['payload']==expected and ownership==old_ownership
    assert identity['sha256']!=previous['sha256'] and not optimizer.state
    changed=deepcopy(spec);changed['manifest']['retention']['keep_local_completed']=3
    assert cli.construct_identity(changed,*parameters[1:])[0]['sha256']!=identity['sha256']


@pytest.mark.parametrize('keep',[True,0,1,-1,2.0,'2'])
def test_storage_policy_rejects_missing_recovery_headroom(keep):
    with pytest.raises(ValueError):cli.storage_policy({'manifest':{'retention':{'keep_local_completed':keep}}})


def test_cli_delegates_paths_before_execution_and_requires_pinned_resume(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(cli,'validate_storage_paths',lambda c,e:calls.append((c,e)))
    base=['--declaration',str(tmp_path/'d.json'),'--declaration-sha256','a'*64,
        '--output-dir',str(tmp_path/'new'),'--checkpoint-root','/mnt/localssd/cdrm-checkpoints/test/new','--arm','NFR']
    parsed=cli.parse_args(base)
    assert parsed.checkpoint_root.as_posix()=='/mnt/localssd/cdrm-checkpoints/test/new' and len(calls)==1
    with pytest.raises(SystemExit):cli.parse_args(base+['--resume','/missing'])
    def reject(*a):raise ValueError('SSD mount missing')
    monkeypatch.setattr(cli,'validate_storage_paths',reject)
    with pytest.raises(SystemExit):cli.parse_args(base)
