"""Anchored diagnostic restores and failure/provenance persistence, CPU only."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.lm_training import TrainingCounters,build_warmup_scheduler
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import (anchor_to_reference,update_snapshot,
    check_update,finalize_report,persist_checkpoint_reference,MANIFEST_FILENAME)


def trained_fixture():
    torch.manual_seed(381)
    model=torch.nn.Linear(3,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.03,eps=1e-5)
    scheduler=build_warmup_scheduler(optimizer,warmup_updates=2)
    counters=TrainingCounters()
    return model,optimizer,scheduler,counters


def step(model,optimizer,scheduler,counters,scale=1.):
    model.zero_grad(set_to_none=True)
    model(torch.tensor([[1.,2.,3.]])).square().mean().mul(scale).backward()
    optimizer.step();scheduler.step();model.zero_grad(set_to_none=True)
    counters.optimizer_updates+=1;counters.microbatches+=4;counters.documents+=4;counters.input_tokens+=32


def test_anchor_resets_weights_moments_scheduler_counters_without_parameter_replacement():
    model,optimizer,scheduler,counters=trained_fixture()
    step(model,optimizer,scheduler,counters)
    canonical=update_snapshot(model,optimizer,scheduler,counters)
    before_digest=tree_digests(canonical)
    pointers={n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}
    step(model,optimizer,scheduler,counters,scale=2.)
    candidate=update_snapshot(model,optimizer,scheduler,counters)
    assert not check_update(model,optimizer,scheduler,counters,canonical)['passed']
    anchored=anchor_to_reference(model,optimizer,scheduler,counters,canonical)
    assert anchored['passed'] and anchored['canonical_state_exact']
    assert counters.optimizer_updates==1
    assert tree_digests(update_snapshot(model,optimizer,scheduler,counters))==before_digest
    assert {n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}==pointers
    # Recorded candidate outcome is independent of the later diagnostic reset.
    assert candidate['counters']['optimizer_updates']==2
    assert tree_digests(candidate)!=before_digest
    step(model,optimizer,scheduler,counters)
    assert tree_digests(canonical)==before_digest  # Adam state cannot alias saved CPU tensors.


def test_anchor_rejects_pending_gradient_boundary_without_mutation():
    model,optimizer,scheduler,counters=trained_fixture()
    step(model,optimizer,scheduler,counters)
    reference=update_snapshot(model,optimizer,scheduler,counters)
    model(torch.ones(1,3)).sum().backward()
    before=tree_digests(model.state_dict())
    with pytest.raises(ValueError,match='cleared-gradient'):
        anchor_to_reference(model,optimizer,scheduler,counters,reference)
    assert tree_digests(model.state_dict())==before


def test_anchored_next_update_matches_canonical_starting_state():
    model,optimizer,scheduler,counters=trained_fixture()
    step(model,optimizer,scheduler,counters)
    boundary=update_snapshot(model,optimizer,scheduler,counters)
    step(model,optimizer,scheduler,counters,scale=1.3)
    expected=update_snapshot(model,optimizer,scheduler,counters)
    # Move to a different trajectory, then restore the canonical preceding step.
    step(model,optimizer,scheduler,counters,scale=7.)
    assert anchor_to_reference(model,optimizer,scheduler,counters,boundary)['passed']
    step(model,optimizer,scheduler,counters,scale=1.3)
    assert tree_digests(update_snapshot(model,optimizer,scheduler,counters))==tree_digests(expected)


def test_primary_report_exists_before_tracker_finish(tmp_path):
    report={'passed':True,'cases':[{'result':'saved'}]}
    def finish(*,succeeded):
        assert succeeded is True
        assert json.loads((tmp_path/'report.json').read_text())==report
    finalize_report(tmp_path,report,SimpleNamespace(finish=finish))


def test_tracking_error_does_not_replace_primary_failure(tmp_path):
    primary=ValueError('primary numerical failure')
    report={'passed':False,'error':{'type':'ValueError','message':str(primary)}}
    def finish(**unused): raise RuntimeError('tracking finalization failed')
    finalize_report(tmp_path,report,SimpleNamespace(finish=finish),original_error=primary)
    saved=json.loads((tmp_path/'report.json').read_text())
    assert saved['error']['message']=='primary numerical failure'
    assert saved['tracking_finish_error']['type']=='RuntimeError'
    assert not saved['passed'] and primary.__notes__


def test_tracking_only_failure_invalidates_passing_report(tmp_path):
    def finish(**unused): raise RuntimeError('tracking finalization failed')
    report={'passed':True}
    with pytest.raises(RuntimeError,match='tracking finalization'):
        finalize_report(tmp_path,report,SimpleNamespace(finish=finish))
    saved=json.loads((tmp_path/'report.json').read_text())
    assert not saved['passed'] and saved['error']['type']=='RuntimeError'


def test_checkpoint_reference_copies_original_manifest_bytes(monkeypatch,tmp_path):
    import scripts.olmo_two_gpu_validate as module
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    contents=b'{"checkpoint": {"sha256": "retained", "path": "native/model.pt"}}\n'
    (artifacts/MANIFEST_FILENAME).write_bytes(contents)
    monkeypatch.setattr(module,'validate_prepared_manifest',lambda path:json.loads(contents))
    output=tmp_path/'output';output.mkdir()
    record=persist_checkpoint_reference(artifacts,output)
    assert (output/record['manifest_path']).read_bytes()==contents
    assert record['manifest_size_bytes']==len(contents)
    assert record['checkpoint']['sha256']=='retained'
    assert len(record['manifest_sha256'])==64
