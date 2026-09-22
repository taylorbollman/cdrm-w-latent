"""Execution bookkeeping safeguards for the bounded fusion-only experiment."""
import pytest
import torch
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.lm_training import TrainingCounters
from scripts.olmo_o5c_train import microbatches, exposure, check_counters, health_failure


def test_variable_final_microbatch_preserves_masks_and_rows():
    ids = torch.arange(35*7).reshape(35,7)
    batch = NextLatBatch(ids, torch.ones_like(ids,dtype=torch.bool), torch.zeros_like(ids))
    parts = microbatches(batch,16)
    assert [p.input_ids.shape[0] for p in parts] == [16,16,3]
    assert torch.equal(torch.cat([p.input_ids for p in parts]), ids)
    assert all(p.ce_mask is None for p in parts)


@pytest.mark.parametrize('physical',[0,-1,True,1.5])
def test_bad_physical_batch_rejected(physical):
    with pytest.raises(ValueError): microbatches(None,physical)


def test_exact_counter_plan_and_domain_exposure():
    plan = {'batch_ce_prefix':[0,8], 'batch_token_prefix':[0,11], 'batch_row_prefix':[0,3],
            'domain_prefixes':{'code':{'ce_positions':[0,4], 'input_tokens':[0,6], 'documents':[0,2]},
                               'general':{'ce_positions':[0,4], 'input_tokens':[0,5], 'documents':[0,1]}}}
    c = TrainingCounters(optimizer_updates=1,input_tokens=11,ce_positions=8,documents=3)
    check_counters(plan,c)
    assert exposure(plan,1)['general']['input_tokens'] == 5
    c.ce_positions = 9
    with pytest.raises(ValueError): check_counters(plan,c)


def test_retention_only_health_deterioration_triggers():
    def values(code,ret):
        return {'dev':{'passes':[{'mean_nll':2.},{'mean_nll':code}]},
                'retention_dev':{'passes':[{'mean_nll':3.},{'mean_nll':ret}]}}
    initial = values(2.,4.)
    assert health_failure(values(1.8,5.6),initial,1.5)
    assert not health_failure(values(1.8,5.5),initial,1.5)
    assert health_failure(values(3.6,4.),initial,1.5)
