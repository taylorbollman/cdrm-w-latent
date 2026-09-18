"""Bounded CPU model checks for four-block SEQ and first-layer-window RT."""
import copy
from dataclasses import asdict

import pytest
import torch

from olmo.model import OLMo, OLMoRecurrentAutogradBlock, OLMoRecurrentBlockTiled, OLMoSequentialBlock
from scripts.rt_a5_common import canonical_parameter_sha256, configure_fp32_runtime, fp32_context, make_optimizer, model_config
from scripts.rt_a5_depth_order import backbone_parameter_count, build_model, configuration
from scripts.rt_a5_nextlat import _parameter_sha256, build_nextlat_model, nextlat_objective
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    configure_fp32_runtime()


@pytest.mark.parametrize("width",[128,512])
def test_seq4_is_fresh_canonical_four_layer_mitchell_with_original_predictor(width):
    original = build_nextlat_model("seq",width=width)
    rng = torch.get_rng_state().clone()
    model = build_model("seq4_alibi",width=width)
    assert torch.equal(rng,torch.get_rng_state())
    config = model_config("seq",width)
    config.n_layers = 4
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        expected = OLMo(config).float()
    assert len(model.backbone.transformer.blocks) == 4
    assert all(type(block) is OLMoSequentialBlock for block in model.backbone.transformer.blocks)
    assert all(torch.equal(value,model.backbone.state_dict()[name]) for name,value in expected.state_dict().items())
    assert canonical_parameter_sha256(model.backbone) == canonical_parameter_sha256(expected)
    before,after = asdict(original.backbone.config),asdict(model.backbone.config)
    assert {key for key in before if before[key] != after[key]} == {"n_layers"}
    assert model.nextlat_config == original.nextlat_config
    assert _parameter_sha256(model.predictor) == _parameter_sha256(original.predictor)
    assert model.nextlat_initialization["exact_two_layer_learned_initialization"] is False
    assert model.nextlat_initialization["canonical_sha256"] != original.nextlat_initialization["canonical_sha256"]
    assert not model.backbone.config.rope and "wpe" not in model.backbone.transformer
    assert sum(p.numel() for p in model.backbone.parameters()) == backbone_parameter_count(width,4)
    if width == 512:
        assert sum(p.numel() for p in model.backbone.parameters()) == 12653056
        assert sum(p.numel() for p in model.parameters()) == 13702656
        assert len(list(model.parameters())) == 39


@pytest.mark.parametrize("width",[128,512])
def test_rt_window_first_preserves_every_original_learned_tensor_and_layer_slot(width):
    original = build_nextlat_model("rt",width=width,backend="naive")
    rng = torch.get_rng_state().clone()
    model = build_model("rt_window2_first",width=width,backend="naive")
    assert torch.equal(rng,torch.get_rng_state())
    assert type(model.backbone.transformer.blocks[0]) is WindowTwoRecurrentAutogradBlock
    assert type(model.backbone.transformer.blocks[1]) is OLMoRecurrentAutogradBlock
    assert list(model.state_dict()) == list(original.state_dict())
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in original.state_dict().items())
    assert asdict(model.backbone.config) == asdict(original.backbone.config)
    assert model.nextlat_config == original.nextlat_config
    assert model.nextlat_initialization["reference_two_layer_initialization"] == original.nextlat_initialization
    assert model.nextlat_initialization["exact_two_layer_learned_initialization"] is True
    assert model.nextlat_initialization["changed_parameter_slices"] == []
    if width == 512:
        assert model.nextlat_initialization["canonical_sha256"] == "0380e2fd4cdd4db63ce6d732a0834ad054e185c2276da55d733839276d8c55ad"
        assert sum(p.numel() for p in model.parameters()) == 7407104


def test_tiled_first_window_owns_its_original_parameter_views_without_cuda():
    model = build_model("rt_window2_first",width=128)
    first,second = model.backbone.transformer.blocks
    assert type(first) is WindowTwoRecurrentBlockTiled and type(second) is OLMoRecurrentBlockTiled
    assert first.layer_id == 0 and second.layer_id == 1
    assert first.pre_attention_block.q_proj is first.q_proj
    assert first.pre_attention_block.kv_proj is first.kv_proj
    assert first.post_attention_block.ff_proj is first.ff_proj
    parameters = [p for group in make_optimizer(model).param_groups for p in group["params"]]
    assert len(parameters) == len(set(map(id,parameters))) == 25
    assert "wpe" not in model.backbone.transformer


@pytest.mark.parametrize("variant",["seq4_alibi","rt_window2_first"])
def test_joint_gradients_prefix_causality_and_backbone_only_evaluation(variant):
    model = build_model(variant,width=128,backend="naive").eval()
    tokens = torch.tensor([[0,7,3,9,1,11],[2,1,5,0,4,16]])
    changed = tokens.clone()
    changed[:,3:] = (changed[:,3:]+1)%60
    with fp32_context("cpu"):
        result = nextlat_objective(model,tokens,tokens)
        result["loss"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with torch.no_grad():
            original,altered = model(tokens).logits,model(changed).logits
    torch.testing.assert_close(original[:,:3],altered[:,:3],rtol=2e-5,atol=2e-6)
    def reject_predictor(*_args,**_kwargs):
        raise AssertionError("Accuracy evaluation must not invoke predictor")
    handle = model.predictor.register_forward_pre_hook(reject_predictor)
    try:
        with torch.no_grad(),fp32_context("cpu"):
            assert torch.equal(model(tokens).logits,original)
    finally:
        handle.remove()


def test_first_window_restricts_later_reads_but_keeps_recurrent_credit():
    full = build_nextlat_model("rt",width=128,backend="naive")
    model = build_model("rt_window2_first",width=128,backend="naive")
    tokens = torch.tensor([[0,7,3,9,1,11]])
    with torch.no_grad(),fp32_context("cpu"):
        a,b = full(tokens).logits,model(tokens).logits
    assert torch.equal(a[:,:2],b[:,:2])
    assert not torch.equal(a[:,2:],b[:,2:])
    x = torch.randn(1,6,128,generator=torch.Generator().manual_seed(99),requires_grad=True)
    with fp32_context("cpu"):
        output = model.backbone.transformer.blocks[0]._real_forward(x)
        output[:,-1].square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad[:,0].abs().max().item() > 0


@pytest.mark.parametrize("kwargs",[{"variant":"rt_window2_second"},{"variant":"seq4_alibi","width":65},
                                     {"variant":"seq4_alibi","seed":-1},{"variant":"rt_window2_first","predictor_seed":True}])
def test_unsupported_requests_rejected(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)
