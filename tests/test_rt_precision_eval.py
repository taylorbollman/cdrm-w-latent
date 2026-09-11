from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rt_precision_eval import aggregate_documents, paired_document_interval


def test_shifted_supervision_maps_document_transitions_and_row_starts():
    # Full stream has target weights [0,1,2,3,0,4,5,6]; a boundary inside
    # the first row does not mask that document's first supervised token.
    ce=np.array([[1.,2.,3.],[4.,5.,6.]])
    docs=[{'token_begin':0,'token_end':2,'text_sha256':'a'},
          {'token_begin':2,'token_end':5,'text_sha256':'b'},
          {'token_begin':5,'token_end':10,'text_sha256':'c'}]
    assert aggregate_documents(ce,docs)==[
        {'text_sha256':'a','targets':1,'ce_sum':1.},
        {'text_sha256':'b','targets':2,'ce_sum':5.},
        {'text_sha256':'c','targets':3,'ce_sum':15.}]


@pytest.mark.parametrize('boundaries',[
    [{'token_begin':1,'token_end':4,'text_sha256':'a'}],
    [{'token_begin':0,'token_end':3,'text_sha256':'a'}]])
def test_incomplete_or_gapped_supervision_rejected(boundaries):
    with pytest.raises(ValueError):aggregate_documents(np.ones((1,3)),boundaries)


def test_paired_bootstrap_uses_token_weighting_and_document_identity():
    a=[{'text_sha256':'a','targets':10,'ce_sum':10.},
       {'text_sha256':'b','targets':30,'ce_sum':30.}]
    b=[dict(row,ce_sum=row['ce_sum']+.004*row['targets']) for row in a]
    result=paired_document_interval(a,b,resamples=100)
    assert result['candidate_minus_reference_nats_per_target']==pytest.approx(.004)
    assert result['one_sided_95_upper']==pytest.approx(.004)
    b[0]['text_sha256']='wrong'
    with pytest.raises(ValueError):paired_document_interval(a,b,resamples=100)


def test_unequal_document_effects_are_token_weighted_and_reproducible():
    a=[{'text_sha256':'a','targets':10,'ce_sum':10.},
       {'text_sha256':'b','targets':30,'ce_sum':30.}]
    b=[dict(a[0],ce_sum=11.),dict(a[1],ce_sum=27.)]
    result=paired_document_interval(a,b,resamples=500,seed=42)
    assert result['candidate_minus_reference_nats_per_target']==pytest.approx(-.05)
    assert result==paired_document_interval(a,b,resamples=500,seed=42)
