"""Exact paired-fixture checks; no GPU execution or scaling claim."""
from dataclasses import replace

import pytest
import torch

from scripts.olmo_f1_common import IntegrationCase
from scripts.olmo_two_gpu_graph import fixed_batch
from scripts.olmo_two_gpu_single_reference import paired_batch,parse_args


@pytest.mark.parametrize('batch_size',[2,4,8])
@pytest.mark.parametrize('update',[0,2,7])
def test_single_fixture_is_exact_rank_concatenation(batch_size,update):
    case=IntegrationCase('combined',fbt=True,nextlat=True,rt_layers=(0,1),batch_size=batch_size,length=8)
    actual=paired_batch(case,None,update,tiny=True)
    shard_case=replace(case,batch_size=batch_size//2)
    shards=[fixed_batch(shard_case,None,update,rank,tiny=True) for rank in (0,1)]
    for name in ('input_ids','valid_mask','ce_mask','latent_mask','kl_mask'):
        assert torch.equal(getattr(actual,name),torch.cat([getattr(s,name) for s in shards]))
    assert torch.equal(actual.document_ids[:,0],torch.arange(batch_size))
    assert all(row.unique().numel()==1 for row in actual.document_ids)


def test_masks_remain_fixed_across_changed_tokens():
    case=IntegrationCase('rt',rt_layers=(0,1),batch_size=4,length=8)
    first=paired_batch(case,None,0,tiny=True)
    changed=paired_batch(case,None,5,tiny=True)
    assert not torch.equal(first.input_ids,changed.input_ids)
    for name in ('valid_mask','document_ids','ce_mask','latent_mask','kl_mask'):
        assert torch.equal(getattr(first,name),getattr(changed,name))


@pytest.mark.parametrize('batch_size',[0,1,3])
def test_cli_rejects_nonpaired_batch(batch_size):
    with pytest.raises(SystemExit):
        parse_args(['--case','rt','--batch-size',str(batch_size),'--output-dir','.runtime/test'])
