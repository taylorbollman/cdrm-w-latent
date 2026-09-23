# Bounded author-backend integration

Prospective protocol. This follows the authorized native-efficiency and
author-derived comparison plan, conditional on a repeatable useful isolated
advantage. Initial isolated B128 measurements favor the author-derived path in
both throughput and setup memory; repeated and multi-block results must accompany
any final decision. This is functionality/efficiency work, not quality training.

## Fixed model and execution

Use the unchanged OLMo-1B step60000 checkpoint,16layers,D2048,H16,head128,
SwiGLU8192 per branch, native LayerNorm/RoPE, no Q/K normalization and tied
50304-token embeddings/readout. RT selects layers0/15 at full strength.
The two cases are RT-only and K2 FBT+RT+NextLat, with the existing ordinary
bootstrap pass and unchanged CE/latent/KL semantics, CE2048/KL128.

Native control enables Stage A RoPE reuse and K/V-only writes, cast reuse,
Triton historical forward/backward kernels and probability recomputation.
Author candidate uses the already validated isolated port, author_legacy
arithmetic, compiled helper boundaries, four local MLP chunks, invocation-local
cast reuse and materialized backward attention. Preserve exact initial native,
fusion and predictor parameters across arms. Record all seeds and actual counts.

The opt-in selector must preserve default native behavior, checkpoint keys,
parameter identities and ordinary execution. Reject unsupported author RT:
fractional strength, padding, prefixes/exported caches, unsupported precision or
missing prepared tables. Static signatures guard backend/options. Ordinary
bootstrap and surrounding layers keep deterministic PyTorch Flash SDPA and
ordinary activation checkpointing. Author RT itself does not call Flash.

## Gates

Before GPU integration, run tiny FP32 comparisons for ordinary bypass, RT-only,
K2 combined and K3 shared-block reuse. Check pass outputs/losses, all raw
parameter gradients, tied/shared ownership, scope guards and static metadata.
These complement the isolated native-width GPU FP32 reference already checked;
they do not replace actual BF16/Flash/graph validation.

At B8/T512 in BF16 mixed, compare native and author with identical checkpoint,
auxiliary weights, source examples and full-CE loss layout. Changed-input and
Adam-parity batches retain that same full-CE mask. The tiny standalone Flash
observer is a separate dispatch fixture. Record every individual loss
and raw parameter tensor before clipping. Preserve existing screens: aggregate
gradient L2<=1/64, per-tensor L2<=1/32, maximum error normalized by the reference
tensor peak<=1/16, and the existing Stage A output/loss budgets. Do not widen a
budget after observing a result. Retain finite screen misses explicitly while
running own-candidate operational checks; they do not clear replacement.

The author candidate must have exact own-backend eager/graph loss/raw gradients,
changed-input overwrite, repeated replay, three eager versus three graph Adam
updates with identical model/moments/scheduler/counters, and changed-weight
replay. Nonfinite or operational failures block performance interpretation and
need diagnosis. Verify ordinary Flash dispatch in a separate bounded observation,
outside timed updates. No timing result establishes language-model quality.

## Timing and evidence

After operational gates, run matched native and author full-CE timing at B64/T512
for RT-only and combined, using B32 if setup memory is not comfortable. Each
fresh process has three preparation updates, ten backward warmups, CUDA-graph
capture, five timed complete optimizer updates and three graph-only replays.
Keep helper compilation/setup outside steady timing, check compiler fallback,
and record setup/steady allocated and reserved memory. Repeat close differences
in reverse order before claiming a gain. No broad batch-size optimization.

Record actual resident/active/trainable/optimizer/deployable parameters and
audited matrix FLOPs. Replace the selected native RT ledger contribution with
the author schedule, including its materialized attention; ordinary layers,
fusion and objectives keep their existing accounting. Report distinct input
tokens and pass work for K2, not an inflated throughput numerator.

Freeze source/protocol before the GPU queue. Use the project container and one
H10080GB. Log online to W&B group`olmo-rt-author-integration` under
`taylorbollman/pretrained-fbt-rt-nextlat`. Retain completed and failed reports,
exact source/protocol snapshots and small plots under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-rt-author-integration/`.
Reference the existing native checkpoint instead of uploading diagnostic weights.

Existing F4 RT+FBT and BF16/FP32 qualifications remain open. This bounded scope
does not establish padding/cache support, save/resume, multiple GPUs, long-run
numerical stability or a production default change. Integration results decide
whether to expose the author backend as an experimental option, keep native, or
port specific improvements separately.
