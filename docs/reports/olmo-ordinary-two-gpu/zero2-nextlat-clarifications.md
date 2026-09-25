# ZeRO-2, accumulation and the current NextLat objective

2026-09-25. This answers the user's follow-up during the ordinary-model sweep.
It clarifies priorities; ZeRO-2 has not been implemented or benchmarked here.

## ZeRO-2 is deferred, not ruled out for throughput

ZeRO-1 partitions optimizer state; ZeRO-2 additionally partitions retained
gradients. Saving memory can permit a larger physical microbatch, reduce the
number of accumulation rounds for a fixed global batch, and improve throughput
where the model benefits from larger physical batches. Communication overlap
and optimizer implementation also matter. There is no measured ZeRO-2 result
in this project yet.

Accumulation means more examples contribute to an update, not a larger model
gradient tensor. For example, four microbatches still produce one gradient per
parameter. Communication depends on when those gradients are synchronized:

- DDP with correctly scoped `no_sync`, or ZeRO-1 with deferred reduction, can
  accumulate locally and synchronize at the update boundary.
- Standard DeepSpeed ZeRO-2 reduces/partitions after each backward so it can
  retain only the owned gradient partition. Four microbatches can therefore
  entail four gradient-reduction rounds, even with only one optimizer update.
- DeepSpeed currently documents `coalesce_grad_reduction()` to defer those
  rounds. Its tradeoff is holding a full local gradient during the accumulation
  window; ordinary `no_sync()` is not supported for ZeRO-2/3. This feature has
  not been installed, integrated or tested with our graph/precision policy.

These are reduction rounds, not necessarily individual NCCL calls: bucketing
may produce multiple calls per round. See the official
[DeepSpeed training API](https://deepspeed.readthedocs.io/en/stable/training.html#coalesced-gradient-reduction)
for the current behavior and constraints. Our existing ZeRO-1 implementation
uses PyTorch's optimizer sharding, not a DeepSpeed engine.

The ordinary-model measurements use **one physical microbatch per update**,
without accumulation. The selected B64/GPU point leaves about 44.9 GiB free;
B128/B192 add only about 1% throughput. Memory capacity is consequently not a
compelling reason to add ZeRO-2 for this particular baseline. Ideal two-way
sharding of its 4.384 GiB FP32 gradient tensor would save about 2.192 GiB/rank
relative to a full gradient; actual memory depends on buckets, temporary
buffers and precision policy. This is distinct from sharing DDP gradient/bucket
storage, which can remove a local duplicate.

The RT/FBT/NextLat combinations have different memory requirements. Keep a
bounded ZeRO-1 versus ZeRO-2 comparison available when the intended workload
benefits from more physical batch or substantial accumulation. Match precision,
global batch, loss normalization and optimizer settings; record physical batch,
accumulation count, reduction rounds/bytes, full-update throughput and memory.
Check CUDA-graph compatibility and one complete update before interpreting speed.
Existing eager accumulation checks do not validate graph accumulation: the
current graph training path zeros gradients on each replay and executes one
physical microbatch per update.

## NextLat includes KL in the current OLMo implementation

For NextLat-enabled OLMo passes, the current coefficients are
`lambda_latent=1.0` and `lambda_kl=1.0`, giving

\[
L = L_{\mathrm{CE}} + L_{\mathrm{SmoothL1}} + L_{\mathrm{KL}}.
\]

The predictor uses the current hidden state and actual next-token embedding to
estimate the next hidden state. Smooth L1 compares that prediction with a
detached backbone target. The KL is teacher-to-prediction over the full output
vocabulary:

\[
D_{\mathrm{KL}}\left(
\operatorname{softmax}(\operatorname{sg}(W)\operatorname{sg}(h_{t+1}))
\;\middle\|\;
\operatorname{softmax}(\operatorname{sg}(W)\hat h_{t+1})\right).
\]

The auxiliary readout and teacher branch are detached; gradients reach the
predictor and its conditioning inputs. Tied embeddings still train through
lookup/backbone and ordinary CE. This follows the direction and stop-gradient
convention in [NextLat equations 4–5](https://arxiv.org/html/2511.05963v4#S3).
Our present prediction horizon is one; FBT's pass count is a separate setting.

Implementation: `cdrm/pretrained/nextlat.py` (`NextLatConfig`, `_kl_chunk`,
`compute_nextlat_loss_sums`) and `cdrm/pretrained/static_nextlat.py` share the
KL calculation. Masks select valid same-document triples and explicit KL
targets. The retained `zero1-combined-b128-01` T512 report has 65,536 global
KL targets per update and a first-preparation-update KL mean of 19.1186,
summing its two equally weighted FBT passes. Thus this is an active term, not
just an available option. Per-loss gradient probes were recorded previously.

The ordinary-only sweep disables NextLat entirely and uses CE alone. Historical
A5 recipes included KL-disabled configurations; the confirmation above is for
the current OLMo language-model implementation, not every historical run.
