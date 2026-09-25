# Ordinary OLMo batch and throughput protocol

2026-09-25, before measurements. User requests useful batch sizes, memory and
throughput for the ordinary base model on two H10080GB GPUs, before further
experiments, and clarification of bucket views/dynamic graph layouts.

## Fixed model and execution

Original OLMo-1B step60000 (~252B tokens),16layers,D2048,H16,head128,
SwiGLU8192 per branch,tied50304vocabulary. Ordinary causal attention in every
layer: no RT layers, no FBT feedback/fusion execution, no NextLat predictor or
auxiliary objectives. The existing wrapper may retain a frozen unused fusion
module (8,388,608parameters/32MiB); report it separately from the active backbone.
No checkpoint/architecture/optimizer hyperparameter changes.

T512, full-valid CE with position chunks2048, one physical microbatch per rank.
BF16 mixed with FP32 parameters/gradients/residuals/norms/Adam state; TF32 off,
autocast weight caching off. All ordinary blocks checkpointed; Flash SDPA,
rounded compiled SwiGLU and fused AdamW. Primary ordinary RoPE is the previously
validated Dao native-FP32 option; native RoPE is an explicitly labeled matched
control. No bucket views, new attention backend, sharding beyond ZeRO-1, or
model-quality experiment. CUDA graphs capture actual forward/loss/backward and
DDP/NCCL. Clipping, Adam, scheduler and health checks remain outside capture.

## Bounded adaptive sequence

1. Small actual ordinary Dao DDP complete-update graph/eager check at B1/rank.
   Reuse established graph and optimizer validation; no broad precision campaign.
2. Matched native/Dao DDP B64/rank, then Dao B128/rank. The native control keeps
   an execution-matched reference to the prior RT/combined measurements.
3. If safe, Dao B192/rank and B256/rank; choose follow-ups from measured memory
   and incremental throughput. Do not seek an OOM or the final gigabyte. A larger
   batch is useful only if it improves throughput enough for the lost headroom.
4. Test ZeRO-1 at a useful chosen batch if DDP headroom limits the recommendation.
   Extend to one nearby larger point only if worthwhile. Capacity need not reach
   paperB512. If the curve is flat, prefer a smaller physical batch.
5. One matched single-GPU ordinary Dao reference if feasible to quantify scaling;
   repeat the recommended DDP point in a fresh process to check direction/stability.

Every capacity row retains three Adam preparation updates, eleven DDP warmup
backwards before capture, five timed complete updates and exact own initial/
changed-weight terminal eager/graph raw checks. Replica/state ownership checks
remain. Failures and OOMs are retained; no tolerance relaxed. Each stage uses a
fresh process and an external timeout. Exclude fixture construction, report/
hash/checkpoint work from timed updates. Count original input tokens once.

## Evidence and recommendations

Report local/global batches, aggregate input tokens/s, seconds/update, GPU-time/
token, setup peak allocated/reserved, steady graph-pool reservation and sampled
free memory per GPU. Sampled free is not a continuous minimum. Separate actual
parameter/gradient/Adam bytes from reserved graph pools and communication storage.
No claim of quality, optimizer-batch equivalence, data-loader throughput or a
maximum supported batch follows from short synthetic/repeated-text timings.

Keep W&B online in pretrained-fbt-rt-nextlat, use ordinary-prefixed stage names,
retain each final stage via existing verified GCS helper and preserve source/
dependency hashes. Checkpoints are not necessary for these disposable eight-update
capacity probes; original pretrained weights remain retained by O1. Persist work
and evidence throughout. Capture settings and source remain frozen within a stage.

Current two-GPU harness freezes its earlier protocol too. This additive protocol
specifies ordinary-only options; retain it beside every new report as additional
prospective evidence without modifying the earlier frozen protocol.
