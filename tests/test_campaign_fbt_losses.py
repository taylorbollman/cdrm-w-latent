"""Campaign weighting checks against independent scalar and model objectives.

CPU tests exercise tensor execution and simulated rank averaging, not CUDA
graphs, DDP collectives, or mixed precision.
"""
import copy
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.ddp_graph_training import PreparedDDPObjective
from cdrm.pretrained.distributed_training import ObjectiveForwardAdapter
from cdrm.pretrained.fbt_training import FBTNextLatLM, aggregate_pass_losses
from cdrm.pretrained.nextlat import (
    NextLatBatch, NextLatConfig, NextLatLosses, compute_nextlat_loss_sums,
)
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.resource_estimates import estimate_training_resources
from cdrm.pretrained.static_training import StaticFBTTraining, normalized_objective


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(9028)


@pytest.mark.parametrize("passes", [1, 2, 4])
def test_campaign_scalar_formula_and_independent_gradients(passes):
    # Unequal denominators and non-unit auxiliary coefficients ensure the
    # contract weights per-term means, not a combined numerator/denominator.
    counts = {"ce": 7, "latent": 5, "kl": 3}
    weights = {"ce": 1., "latent": .3, "kl": .7}
    leaves = torch.arange(1, passes*3+1, dtype=torch.float64).reshape(passes, 3).requires_grad_()
    records = tuple(NextLatLosses(dict(zip(counts, row.square())), counts, weights)
                    for row in leaves)
    result = aggregate_pass_losses(records, pass_loss_policy="campaign_v1")
    means = leaves.square() / torch.tensor([7., 5., 3.], dtype=torch.float64)
    ce = means[0, 0] if passes == 1 else .5*means[0, 0] + .5*means[1:, 0].mean()
    expected = ce + .3*means[:, 1].mean() + .7*means[:, 2].mean()
    torch.testing.assert_close(result.total, expected)
    actual_grad = torch.autograd.grad(result.total, leaves, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, leaves)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert result.counts == counts
    assert result.weights == weights
    assert result.pass_loss_policy == "campaign_v1"
    assert result.term_pass_coefficients["latent"] == (1/passes,)*passes
    assert result.term_pass_coefficients["kl"] == (1/passes,)*passes
    assert sum(result.pass_coefficients) == pytest.approx(1.)


@pytest.mark.parametrize("passes,gamma", [(1, 0.), (2, .3), (4, 1.), (4, 0.)])
def test_legacy_default_retains_operation_order_and_coefficients(passes, gamma):
    leaves = [torch.tensor(float(k+1), requires_grad=True) for k in range(passes)]
    records = [NextLatLosses({t: x for t in ("ce", "latent", "kl")},
                 {"ce": 3, "latent": 2, "kl": 1}, {"ce": 1., "latent": 1., "kl": 1.})
               for x in leaves]
    default = aggregate_pass_losses(records, gamma=gamma)
    explicit = aggregate_pass_losses(records, gamma=gamma, pass_loss_policy="legacy")
    coefficients = (1.,) if passes == 1 else (1.,) + (gamma/(passes-1),)*(passes-1)
    for term in default.sums:
        old = sum(coefficient*record.sums[term] for coefficient, record in zip(coefficients, records))
        assert torch.equal(default.sums[term], old)
        assert torch.equal(default.sums[term], explicit.sums[term])
        assert default.term_pass_coefficients[term] == coefficients
    assert default.pass_coefficients == coefficients


def make_model(*, nextlat=True, policy="campaign_v1"):
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=True,
        reuse_rope=True, kv_only_writes=True)
    return FBTNextLatLM(OLMoFBT(base), NextLatConfig(base.config.model_dim,
        proj_factor=2, lambda_latent=.3, lambda_kl=.7, vocab_chunk_size=4),
        enabled=nextlat, pass_loss_policy=policy).train()


def make_batch():
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 8, 9, 2, 1, 1], [4, 7, 1, 1, 1, 1]])
    valid = torch.tensor([[True]*6, [True]*4+[False]*2, [True]*2+[False]*4])
    docs = torch.arange(3)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    ce, latent, kl = [valid.clone() for _ in range(3)]
    ce[0, 2] = False
    latent[0, 3] = False
    kl[0, 4] = False
    latent[1:] = False  # One simulated rank has no auxiliary targets.
    kl[1:] = False
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def assert_gradients_equal(left, right):
    for (name, actual), (_, expected) in zip(left.named_parameters(), right.named_parameters()):
        if actual.grad is None or expected.grad is None:
            assert actual.grad is expected.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=4e-4, msg=name)


@pytest.mark.parametrize("fbt,rt,nextlat", [(f,r,n) for f in (False,True)
                                         for r in (False,True) for n in (False,True)])
def test_all_eight_campaign_conditions_match_explicit_loss_and_static_gradients(fbt, rt, nextlat):
    canonical = make_model(nextlat=nextlat)
    manual, static = copy.deepcopy(canonical), copy.deepcopy(canonical)
    batch = make_batch()
    mode = FBTMode(enabled=fbt, num_passes=4, beta=.35,
        rt_mode=RTMode((0, 1) if rt else (), .37), first_pass_policy="configured-rt-v1")
    actual = canonical.loss_sums(batch, backbone_kwargs={"mode": mode})
    actual.total.backward()

    # The reference computes all pass objectives directly, without calling
    # aggregate_pass_losses or reading its coefficient metadata.
    embeddings = manual.backbone.token_embeddings(batch.input_ids)
    output = manual.backbone(inputs_embeds=embeddings, attention_mask=batch.valid_mask,
        document_ids=batch.document_ids, mode=mode, return_logits=False)
    terms = [compute_nextlat_loss_sums(hidden, embeddings, manual.backbone.readout_weight,
        batch, manual.predictor, manual.config, enabled=nextlat).means
        for hidden in output.pass_hidden_states]
    ce = terms[0]["ce"] if not fbt else .5*terms[0]["ce"] + sum(x["ce"] for x in terms[1:])/6
    objective = ce + .3*sum(x["latent"] for x in terms)/len(terms) + .7*sum(x["kl"] for x in terms)/len(terms)
    torch.testing.assert_close(actual.total, objective, atol=2e-6, rtol=2e-6)
    objective.backward()
    assert_gradients_equal(canonical, manual)

    prepared = StaticFBTTraining(static, batch, mode=mode)
    static_result = prepared.backward()
    torch.testing.assert_close(normalized_objective(static_result), actual.total,
                               atol=2e-6, rtol=2e-6)
    assert static_result.term_pass_coefficients == actual.term_pass_coefficients
    assert_gradients_equal(static, canonical)


def test_unequal_microbatch_accumulation_and_simulated_rank_averaging_preserve_objective():
    reference = make_model()
    accumulated = copy.deepcopy(reference)
    ranks = [copy.deepcopy(reference) for _ in range(2)]
    batch = make_batch()
    mode = FBTMode(num_passes=4, rt_mode=RTMode((0, 1), .37),
                   first_pass_policy="configured-rt-v1")
    expected = reference.loss_sums(batch, backbone_kwargs={"mode": mode})
    expected.total.backward()
    # A rank with more rows can still have fewer valid tokens and zero KL.
    parts = [NextLatBatch(**{name: value[part] for name, value in vars(batch).items()})
             for part in (slice(0,1), slice(1,3))]
    for part in parts:
        result = accumulated.loss_sums(part, backbone_kwargs={"mode": mode})
        sum(result.sums[t]*result.weights[t]/expected.counts[t]
            for t in result.sums if expected.counts[t]).backward()
    assert_gradients_equal(accumulated, reference)
    objectives = []
    for rank, part in zip(ranks, parts):
        output = ObjectiveForwardAdapter(rank)(part, global_counts=expected.counts,
                                               world_size=2, backbone_kwargs={"mode": mode})
        output["objective"].backward()
        objectives.append(output["objective"].detach())
        assert output["term_pass_coefficients"] == expected.term_pass_coefficients
    torch.testing.assert_close(sum(objectives)/2, expected.total, atol=2e-6, rtol=2e-6)
    for (_, target), *parameters in zip(reference.named_parameters(),
            *(rank.named_parameters() for rank in ranks)):
        grads = [parameter.grad for _, parameter in parameters if parameter.grad is not None]
        if not grads:
            assert target.grad is None
        else:
            torch.testing.assert_close(sum(grads)/2, target.grad, atol=2e-6, rtol=4e-4)


def test_prepared_ddp_adapter_carries_policy_and_rejects_changed_objective():
    model, batch = make_model(), make_batch()
    adapter = PreparedDDPObjective(model, batch, mode=FBTMode(num_passes=4),
        global_counts=model.counts(batch), world_size=1)
    output = adapter()
    assert output["pass_loss_policy"] == "campaign_v1"
    assert output["term_pass_coefficients"]["ce"] == (.5, 1/6, 1/6, 1/6)
    assert output["term_pass_coefficients"]["latent"] == (.25,)*4
    model._pass_loss_policy = "legacy"
    with pytest.raises(ValueError, match="objective configuration"):
        adapter.validate_execution()


@pytest.mark.parametrize("policy", ["unknown", None, True])
def test_unknown_policy_rejected(policy):
    with pytest.raises(ValueError, match="pass_loss_policy"):
        make_model(policy=policy)


@pytest.mark.parametrize("gamma", [0., .3, 2.])
def test_campaign_rejects_ambiguous_legacy_gamma(gamma):
    with pytest.raises(ValueError, match="gamma=1"):
        aggregate_pass_losses([], gamma=gamma, pass_loss_policy="campaign_v1")
    with pytest.raises(ValueError, match="gamma=1"):
        FBTNextLatLM(None, NextLatConfig(64), gamma=gamma, pass_loss_policy="campaign_v1")


def test_public_policy_cannot_be_reassigned():
    model = make_model()
    with pytest.raises(AttributeError):
        model.pass_loss_policy = "legacy"


def test_graph_training_rejects_jitter_until_noise_loading_is_qualified():
    model, batch = make_model(), make_batch()
    mode = FBTMode(num_passes=4, feedback_jitter=.02)
    with pytest.raises(ValueError, match="feedback jitter"):
        StaticFBTTraining(model, batch, mode=mode)
    with pytest.raises(ValueError, match="feedback jitter"):
        PreparedDDPObjective(model, batch, mode=mode,
            global_counts=model.counts(batch), world_size=1)


@pytest.mark.parametrize("enabled,passes,rt_calls", [(False, 4, 2), (True, 1, 2),
                                                     (True, 2, 4), (True, 4, 8)])
def test_resource_estimate_counts_rt_in_first_pass(enabled, passes, rt_calls):
    config = replace(OLMoConfig.tiny(), num_layers=4)
    mode = FBTMode(enabled=enabled, num_passes=passes, rt_mode=RTMode((0,3)),
                   first_pass_policy="configured-rt-v1")
    estimate = estimate_training_resources(config, batch_size=2, sequence_length=5, mode=mode)
    assert estimate.rt_block_calls_per_microbatch == rt_calls
    assert estimate.ordinary_block_calls_per_microbatch == (passes if enabled else 1)*4-rt_calls
