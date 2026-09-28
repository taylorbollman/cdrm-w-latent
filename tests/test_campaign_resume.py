"""Real fresh-process CPU recovery for the new objective and token clock.

This small fixture qualifies portable checkpoint contracts, not distributed or
GPU restart. Child processes inherit the explicitly CPU-only test container.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from cdrm.pretrained.campaign_data import CampaignData, SourcePin, TokenizerPin, TokenizedDocument
from cdrm.pretrained.campaign_recipe import CampaignRecipe, CampaignTokenSchedule, build_campaign_model, build_campaign_adamw, feedback_noise_for_rows
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, optimizer_step, save_training_checkpoint, load_training_checkpoint
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_lm_common import tree_digests


def _setup():
    torch.set_num_threads(1)
    torch.manual_seed(624)
    r = CampaignRecipe("NFR", sequence_length=8, rt_layers=(0, 1), warmup_tokens=50,
                       effective_valid_tokens=14)
    source = SourcePin("fixture", "fixture://text", "fixture-v1", "a" * 64)
    tokenizer = TokenizerPin("fixture", "fixture-v1", "b" * 64)
    documents = [TokenizedDocument(source, str(i), "train", hashlib.sha256(str(i).encode()).hexdigest(),
                                  tuple(range(2 + i, 2 + i + (3 + i % 4)))) for i in range(12)]
    data = CampaignData(documents, tokenizer=tokenizer, length=8, eos_id=63, vocab_size=64)
    cursor = data.cursor("train")
    updates = []
    while cursor.next_window < len(data.windows("train")):
        update = data.next_update(cursor, r.effective_valid_tokens)
        updates.append(update)
        cursor = update.next_cursor
    core = OLMoTiledRTForCausalLM(replace(OLMoConfig.tiny(), vocab_size=64), attention_backend="math", attention_precision="fp32")
    model = build_campaign_model(core, r)
    opt = build_campaign_adamw(model, r, fused=False)
    schedule = CampaignTokenSchedule(opt, [u.counts.presented_tokens for u in updates], warmup_tokens=r.warmup_tokens)
    config = {"recipe": r.to_dict(), "mode": asdict(r.mode()), "data_manifest": data.manifest_sha256,
              "schedule": schedule.checkpoint_contract(), "execution": "intentional tiny CPU fixture"}
    return r, data, updates, model, opt, schedule, config


def _update(r, data, updates, model, opt, schedule, counters):
    u = updates[counters.optimizer_updates]
    schedule.validate_next_update(u.counts.presented_tokens)
    noise = feedback_noise_for_rows(r, [row.key for row in u.rows],
              logical_update=counters.optimizer_updates, sequence_length=8, width=model.config.model_dim)
    return optimizer_step(model, opt, [data.batch(u.rows, pad_to=8)], config=LMTrainingConfig(),
                          backbone_kwargs={"mode": r.mode(), "feedback_noise": noise},
                          scheduler=schedule, counters=counters)


def _worker(directory, resume):
    directory = Path(directory)
    r, data, updates, model, opt, schedule, config = _setup()
    fingerprint = {"checkpoint_sha256": "c" * 64, "scope": "random tiny CPU fixture only"}
    checkpoint = directory / "boundary.pt"
    if resume:
        saved = load_training_checkpoint(checkpoint, model, opt, scheduler=schedule,
                         configuration=config, source_fingerprint=fingerprint)
        counters = saved["counters"]
        cursor = data.restore_cursor(saved["data_cursor"])
        assert cursor == updates[counters.optimizer_updates].start_cursor
    else:
        counters = TrainingCounters()
        _update(r, data, updates, model, opt, schedule, counters)
        save_training_checkpoint(checkpoint, model, opt, scheduler=schedule, counters=counters,
             data_cursor=asdict(updates[0].next_cursor), configuration=config, source_fingerprint=fingerprint)
    metrics = _update(r, data, updates, model, opt, schedule, counters)
    result = {"model": tree_digests(model.state_dict()), "optimizer": tree_digests(opt.state_dict()),
              "schedule": schedule.state_dict(), "counters": asdict(counters), "metrics": metrics,
              "rng_next": torch.rand(4).tolist(), "data_cursor": asdict(updates[1].next_cursor)}
    (directory / ("restored.json" if resume else "reference.json")).write_text(json.dumps(result, sort_keys=True))


def test_fresh_process_mid_warmup_checkpoint_and_next_update_are_exact(tmp_path):
    here = str(Path(__file__).resolve())
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_worker'](sys.argv[2], sys.argv[3]=='resume')"
    for phase in ("reference", "resume"):
        result = subprocess.run([sys.executable, "-c", code, here, str(tmp_path), phase],
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "reference.json").read_text()) == json.loads((tmp_path / "restored.json").read_text())
