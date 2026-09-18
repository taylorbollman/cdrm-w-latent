"""CPU structure/initialization checks; no training or model inference."""
from dataclasses import asdict

import pytest
import torch

from olmo.model import OLMo, OLMoRecurrentAutogradBlock, OLMoRecurrentBlockTiled
from scripts.rt_a5_common import canonical_parameter_sha256, model_config
from scripts.rt_a5_l1r_depth import backbone_parameter_count, build_model, configuration
from scripts.rt_a5_nextlat import _parameter_sha256, build_nextlat_model
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def test_six_total_blocks_match_canonical_mitchell_at_actual_depth():
    original = build_nextlat_model("rt", width=128)
    before_rng = torch.get_rng_state().clone()
    model = build_model(width=128)
    assert torch.equal(before_rng, torch.get_rng_state())
    blocks = model.backbone.transformer.blocks
    assert len(blocks) == 6 and type(blocks[0]) is WindowTwoRecurrentBlockTiled
    assert all(type(block) is OLMoRecurrentBlockTiled for block in blocks[1:])
    assert [block.layer_id for block in blocks] == list(range(6))
    source_config = model_config("seq", 128)
    source_config.n_layers = 6
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        canonical = OLMo(source_config).float()
    assert canonical_parameter_sha256(model.backbone) == canonical_parameter_sha256(canonical)
    assert canonical_parameter_sha256(model.backbone) != canonical_parameter_sha256(original.backbone)
    before, after = asdict(original.backbone.config), asdict(model.backbone.config)
    assert {key for key in before if before[key] != after[key]} == {"n_layers"}
    assert model.nextlat_config == original.nextlat_config
    assert _parameter_sha256(model.predictor) == _parameter_sha256(original.predictor)
    assert model.nextlat_initialization["two_layer_backbone_initialization_paired"] is False
    assert model.nextlat_initialization["window_replacement_changed_parameter_slices"] == []
    assert model.backbone.config.alibi and not model.backbone.config.rope
    assert "wpe" not in model.backbone.transformer
    assert all(parameter.dtype == torch.float32 and parameter.device.type == "cpu" for parameter in model.parameters())
    for block in blocks:
        assert block.pre_attention_block.q_proj is block.q_proj
        assert block.pre_attention_block.kv_proj is block.kv_proj
        assert block.post_attention_block.ff_proj is block.ff_proj


def test_actual_width512_six_layer_counts_and_original_predictor_digest():
    model = build_model()
    assert sum(parameter.numel() for parameter in model.backbone.parameters()) == 18948608
    assert sum(parameter.numel() for parameter in model.parameters()) == 19998208
    assert len(list(model.backbone.parameters())) == 57
    assert len(list(model.parameters())) == 61
    assert model.nextlat_initialization["predictor_parameter_count"] == 1049600
    assert model.nextlat_initialization["predictor_sha256"] == "3a1fccdd63a53e50328c010b03ecd81d7cf8782229b63cf1c843d1c658731e58"


def test_only_first_block_supplies_two_key_mask_and_preserves_allowed_alibi(monkeypatch):
    model = build_model(width=128)
    observed = []
    def capture(block, x, attention_bias=None):
        observed.append((block.layer_id, attention_bias))
        return x
    monkeypatch.setattr(OLMoRecurrentBlockTiled, "_real_forward", capture)
    x = torch.zeros(1, 4, 128)
    query, key = torch.arange(4)[:, None], torch.arange(4)[None, :]
    bias = (key - query).float().masked_fill(key > query, float("-inf"))[None, None]
    saved = bias.clone()
    for block in model.backbone.transformer.blocks:
        block._real_forward(x, bias)
    assert torch.equal(bias, saved)
    first = observed[0][1]
    for q in range(4):
        for k in range(4):
            if max(0, q-1) <= k <= q:
                assert first[0, 0, q, k] == bias[0, 0, q, k]
            else:
                assert first[0, 0, q, k] == float("-inf")
    assert all(value is bias for _, value in observed[1:])


def test_predictor_seed_is_independent_of_backbone_seed_and_depth():
    left = build_model(width=128, n_layers=5, seed=7)
    right = build_model(width=128, n_layers=6, seed=9)
    other_predictor = build_model(width=128, n_layers=6, seed=9, predictor_seed=10)
    assert _parameter_sha256(left.predictor) == _parameter_sha256(right.predictor)
    assert _parameter_sha256(right.predictor) != _parameter_sha256(other_predictor.predictor)
    assert canonical_parameter_sha256(right.backbone) == canonical_parameter_sha256(other_predictor.backbone)
    assert sum(p.numel() for p in left.backbone.parameters()) == backbone_parameter_count(128, 5)
    assert len(left.backbone.transformer.blocks) == 5
    assert len(right.experiment_config["attention"]["layers"]) == 6
    assert configuration()["full_recurrent_layers"] == 5


def test_naive_backend_retains_same_structure_and_depth_initialization():
    tiled = build_model(width=128)
    naive = build_model(width=128, backend="naive")
    assert type(naive.backbone.transformer.blocks[0]) is WindowTwoRecurrentAutogradBlock
    assert all(type(block) is OLMoRecurrentAutogradBlock for block in naive.backbone.transformer.blocks[1:])
    assert canonical_parameter_sha256(naive.backbone) == canonical_parameter_sha256(tiled.backbone)
    assert _parameter_sha256(naive.predictor) == _parameter_sha256(tiled.predictor)


@pytest.mark.parametrize("kwargs", [{"n_layers": 1}, {"n_layers": True}, {"width": 65}, {"predictor_seed": True}])
def test_invalid_shape_or_seed_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)
