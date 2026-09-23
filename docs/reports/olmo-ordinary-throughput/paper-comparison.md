# Ordinary throughput: audit of the paper comparison

Read-only paper and source audit, 2026-09-23. This is the architectural and
execution context for the [bounded native benchmark](protocol.md), not a
reproduction claim or a record of its results.

## Hardware and reported measurement

[The Recurrent Transformer, Section 6 and Appendix D.1](https://arxiv.org/pdf/2604.21215)
states that training uses one H100 at a time. D.1 reports ordinary/RT throughput
of 132k/42k tokens per second for the twelve-layer model and 153k/49k for the
six-layer model. The six-layer dimensions are width2048, MLP8192 and32heads;
the sequence length is512. The H100 variant/capacity and an exact saved run or
timing interval for these appendix throughput figures are not specified.

Section6 separately describes single-layer latency measurements that exclude
embedding, unembedding and loss. Those exclusions must not be silently applied
to the D.1 full-model throughput figures. Its batch-size discussion motivates
physical512 for RT; the ordinary source recipe actually uses accumulation.

## Published recipe versus native OLMo

The table distinguishes the paper description from its published source
recipe. We do not have the exact resolved configuration behind each reported
throughput value, so the source is supporting evidence rather than proof that
every published run used precisely these settings.

| Property | Published six-layer ordinary recipe | Native six-layer diagnostic | Prior native F4 ordinary |
| --- | --- | --- | --- |
| Layers | 6 | 6 | 16 |
| Width | 2048 | 2048 | 2048 |
| Heads / head width | 32 / 64 | 32 / 64; bridge uses16 /128 | 16 /128 |
| MLP | GELU8192; two matrices | SwiGLU8192 per branch; three matrices | Same native SwiGLU |
| Positions | ALiBi | RoPE | RoPE |
| Normalization | Affine layer and Q/K normalization | Native nonaffine LayerNorm; no Q/K normalization | Same native normalization |
| Vocabulary / matrix rows | T5-base32100 /32128 | 50304 matrix rows | 50304 matrix rows |
| Input/readout weights | Untied | Tied | Tied |
| Batch | Global512, physical32,16 microbatches | Explicit physical batches; no implicit accumulation | Physical64 and96 |
| Ordinary block checkpointing | Null in base recipe | Initially enabled; bounded off arm separate | Enabled |
| Compilation / graphs | `torch.compile` default; no graph override in ordinary sweep | CUDA graphs; no compiler | CUDA graphs; no compiler |
| Precision | BF16 AMP | BF16 AMP; explicit FP32 weights/gradients/Adam states | Same native precision policy |
| CE supervision | Full next-token training | Half-target bridge; separate full-target arm | Final256 positions per row |

The [base recipe](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/configs/kempner/base-c4-t5.yaml)
specifies GELU, the tokenizer and embedding choices, BF16 AMP, null outer
checkpointing and default compilation. The
[six-layer overlay](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/configs/kempner/models/300m_6.yaml)
sets the dimensions. The
[ordinary sweep](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/sweep_olmo_512.yaml)
selects single-device execution, ALiBi, length512, physical microbatch32 and
global batch512. Thus an ordinary physical-B512 test is useful, but it does
not itself reproduce that batching recipe.

For contrast, the
[RT sweep](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/sweep_recurrent_512.yaml)
sets physical/global batch512, logit microbatch2, whole CUDA graphs and null
outer compilation. RT's custom backward recomputes intermediates internally;
null outer activation checkpointing is not evidence that it retains everything.

## Matrix accounting

These are exact arithmetic counts of dense block weights, excluding norm
vectors and embeddings. At widthD and intermediateM, ordinary multihead QKV
and output projections contribute4D². GELU contributes2DM, whereas native
SwiGLU with intermediate widthM **per branch** contributes3DM. Here D=2048 and
M=8192. No GQA reduction applies.

| Model | Dense block weights per layer | Layers | Dense body weights |
| --- | ---: | ---: | ---: |
| Paper six-layer GELU | 50,331,648 | 6 | 301,989,888 |
| Native six-layer SwiGLU | 67,108,864 | 6 | 402,653,184 |
| Native sixteen-layer SwiGLU | 67,108,864 | 16 | 1,073,741,824 |

Native six-layer therefore has4/3 the dense body of the paper six-layer model;
native sixteen-layer has32/9, approximately3.556 times the dense body. Adding
native tied embeddings gives505,675,776 and1,176,764,416 active parameters,
respectively. The paper recipe's two32128-by2048 embedding/readout matrices
add131,596,288 parameters, yielding433,586,176 matrix parameters before its
small affine normalization vectors. Frozen fusion allocations retained by our
benchmark wrapper are separate from these active native model counts.

The packed native SwiGLU input projection has16384 outputs before splitting.
Do not read `mlp_hidden_size:8192` in the authors' GELU configuration as the
same packed matrix shape, or assume it means a SwiGLU branch width4096.
The published
[activation and block implementation](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/olmo/model.py)
defines this distinction directly.

Whole-block activation checkpointing also adds an extra forward evaluation
to the ordinary block backward. For dense operations this changes the usual
rough three-forward-equivalent training work to four. Combining that effect
with the3.556 body-size ratio can plausibly explain a substantial fraction of
the raw throughput gap. This is **analytic context, not measured attribution**:
readout work, CE coverage/chunking, optimizer work, fusion, launch overhead and
hardware utilization still matter. It is not evidence that the native path is
already efficient.

## Timer and source provenance

The published
[training code](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/olmo/train.py)
starts its speed monitor immediately before `train_step` and records per-device
input tokens. The step includes transfer, forward/backward, clipping, Adam and
loss synchronization; fetching the current batch occurs before that timer.
This supports treating its training metric as broader than standalone layer
latency. It still does not prove the appendix numbers came from that exact
metric or establish matching logging/setup exclusions.

Source pin: `a21b42d2bc292edb86ed1b62cee4bcab809a9d21`, upstream
`geniucos/recurrent-transformer`. Local fork HEAD at audit:
`824767f9a6f8e29959a0c486d169ce10b7194d41`. The base recipe, six-layer overlay
and both512 sweeps are unchanged between those revisions. The twelve-layer
overlay in this source differs from the paper's stated twelve-layer dimensions;
the six-layer overlay matches. Do not silently generalize the six-layer audit
to an exact twelve-layer paper reproduction.

No GPU was used for this audit. The resulting benchmark should report its
own complete configuration, true physical batch, attention dispatch, timer
boundary and memory, before comparing its rate to the paper.
