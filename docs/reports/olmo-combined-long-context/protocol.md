# Combined OLMo T2048: prospective single-H100 benchmark

2026-09-28. The user authorizes the combined FBT + native RT + NextLat model at
sequence length 2048 against its saved length-512 performance. The latest
instruction narrows this milestone to **Flash SDPA only, then review**. Do not
run FA4 or rerun T512. This is a compute and functionality benchmark, not a
quality-training experiment or a new numerical-equivalence campaign.

## Fixed model and execution

Original OLMo-1B step60000 (~252B pretraining tokens): 16 layers, width2048,
16 heads of dimension128, SwiGLU8192 per branch, tied50304 vocabulary, native
nonaffine normalization, no added Q/K normalization. K2 FBT means an ordinary
bootstrap pass followed by a feedback pass with native RT at indices0 and15.
Feedback stays attached; alpha=beta=gamma=1. Both passes contribute CE,
NextLat SmoothL1 and KL with coefficient1 each; pass losses are summed.

Preserve the saved combined model's native ordinary RoPE, rounded compiled
ordinary SwiGLU, fused AdamW, and all ordinary-layer activation checkpointing.
Use native RT's per-invocation weight-cast/RoPE reuse, K/V-only permanent writes,
Triton historical tiles and recompute backward. BF16 mixed uses FP32 parameters,
gradients, residuals, norms and Adam state; TF32 and autocast weight cache off.
Forced deterministic PyTorch Flash SDPA is ordinary attention. It does not
replace the native RT kernels. No production arithmetic/default changes.

Active training parameters1,267,879,936 = backbone1,176,764,416 +
fusion8,388,608 + training-only NextLat predictor82,726,912. Deployable model
with fusion1,185,153,024. CE position chunks2048; KL position chunks128.
Independent full-valid row documents; full CE/latent masks, response-half KL
mask. One physical batch/update, one GPU, no accumulation. Input tokens count
once despite two forward passes. At B32/T2048:65,536inputtokens/update and
per-pass CE/latent/KL counts65,504/65,504/32,768.

## Reference and bounded measurements

Use `.runtime/olmo-two-gpu/single-combined-b128-01/report.json` as the historical
reference: B128/T512,65,536inputtokens/update,12,361.8197tokens/s,
5.30148486s/update,58.095GiB setup allocated,65.113GiB reserved,
12.891GiB sampled free. Same paired fixture and capture/timing procedure.
Label it historical rather than same-day interleaved. Its per-pass counts are
65,408/65,408/32,768. Older large-batch repeats support it but are not pooled.

1. Confirm container, idle H10080GB, installed dependency namespace, checkpoint
   identity, persistent disk headroom and W&B. Freeze sources/protocol before
   GPU work; retain sources and imported dependencies.
2. Extend only harness selection/reporting to combined T2048 with native RoPE
   and Flash SDPA. Preserve existing ordinary options and unsupported guards.
   Run focused CPU tests, then an initial B2/T2048 operational check.
3. Measure B16/T2048, then B32 if setup/capture headroom supports it. B8 is an
   optional fallback if B16 is unsuitable. B32 matches the historical input
   tokens/update; it does not match physical batch or sequential recurrence
   depth. Repeat the useful selected batch to check timing stability. At most
   two further capacity attempts if needed to identify a comfortable setting;
   no last-GB/OOM hunt or physicalB128/T2048 requirement.
4. Every fresh-process row starts from the same pretrained checkpoint, with
   three real Adam preparation updates, eleven backward warmups and five timed
   complete updates. CUDA graphs capture forward/loss/backward; clipping,
   fused AdamW, scheduler and health checks remain outside. Include input
   validation/copy, exclude compilation/logging/artifact I/O from timing.
5. Check finite losses, raw gradients and updates; exact own initial and
   changed-weight terminal eager/graph parity; active objective/parameter
   counts; selected ordinary/RT dispatch; runtime/dependency end integrity.
   Preserve every attempt. Nonfinite, structural, dispatch or own parity failure
   stops dependent capacity runs until understood. This is not a new independent
   full-Adam trajectory comparison and does not clear older BF16 qualifications.

T2048 has a known execution qualification: historical forward rectangles with
side<=256 use Triton, while sides512/1024 use existing eager BF16 matmuls.
Across RT0/15, six such larger calls cover75.04% of historical pair area, not
75.04% of model time. Recompute backward is fused through2048. Record actual
dispatch rather than silently claiming every forward tile is fused. With fixed
input tokens, RT batch falls128 to32 and sequential length grows4x; ordinary-only
throughput scaling is not a prediction for this combined model.

## Memory, evidence and closeout

Report setup and steady allocated/reserved separately, sampled device-free
memory (not a continuously measured minimum), parameter/gradient/optimizer
inventories, counts and the existing analytic FLOP estimate. Store eager
references on CPU before capture, release transient cache before capture, and
release graph memory before terminal eager verification as in the reference.

W&B entity`taylorbollman`, project`pretrained-fbt-rt-nextlat`,
group`olmo-combined-long-context`. Local `.runtime/olmo-combined-long-context/`;
retain reports/logs/source/dependency snapshots and final derived evidence in
`gs://fast-chunks` with the existing helper under a fresh timestamp in
`cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/`. O1 starting weights are already
retained; these disposable timing updates require no new full checkpoint.

Report whether T2048 maintains the saved T512 throughput, quantify memory and
batch tradeoffs, retain qualifications and update interruption handoffs. Pause
for user review before FA4 or kernel optimization. The previously reported
absolute7.23e-5nats/target was FA4 minus SDPA at the same T2048, not a T512-versus-
T2048 quality difference. OLMo's training context does not alter the strict
relative-loss equivalence budget.
