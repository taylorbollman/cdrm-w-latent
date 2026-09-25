# Single-GPU distributed-training preparation protocol

Frozen before execution. This checks the objective adapter and accumulation on
one H100; it establishes no DDP, NCCL, two-GPU, sharding or distributed graph
compatibility. No quality or throughput claim is made.

Use the existing original OLMo-1B step60000 checkpoint and pinned tokenizer.
Check RT-only and combined K2 FBT+NextLat separately, with RT at indices0/15.
Primary shape is physical B2/T512. B1 or T32 are explicitly labeled diagnostic
alternatives, never silent fallbacks. Full model, native FP32 RoPE, ordinary
compiled rounded SwiGLU, deterministic Flash SDPA, all ordinary layers
checkpointed, native Triton RT forward/backward, backward recomputation,
cast reuse, RoPE reuse and KV-only writes. CE chunks2048/KL128, BF16 mixed with
FP32 parameters, gradients and Adam moments, TF32 off, autocast cache off.
Seed20260922. No CUDA graphs in this preparation harness.

Every canonical and adapter forward, including both complete-update branches,
uses the explicit `full_valid_causal=True` eager option. The existing input and
document checks run first; all tokens must be valid, caches and packed documents
remain unsupported, and supplied RoPE positions are preserved. Only after this
proof is the redundant all-valid mask omitted from every fresh stack pass so
deterministic Flash SDPA can dispatch. Default historical eager behavior remains
unchanged. The original `rt-b2-t512-01` attempt passed an explicit all-valid mask
to ordinary Flash SDPA, which rejects non-null masks in this environment; it
failed before any adapter comparison or optimizer update. Preserve that raw
failed report. This revision changes dispatch preparation, not the precision
budget or objective semantics; CPU explicit-math forward/gradient equivalence
and rejection tests accompany it.

The adapter calls the canonical model.loss_sums through its forward method and
returns one attached scalar objective plus detached diagnostics. Canonical
FBT pass weighting remains base + gamma*mean(extra passes); position counts
remain independent CE positions, latent pairs and KL triples, without a K
multiplier. Counts refer to the whole optimizer update. Default DDP gradient
averaging will require world_size/global_count on each local sum; this GPU
harness uses world_size1. CPU tests simulate rank averaging algebraically,
including locally empty objectives, without pretending to exercise collectives.

Checks, in order:

1. On one full-valid-CE batch at unchanged weights, compare canonical
   loss_sums plus explicit normalization against adapter forward. All local
   and per-pass losses/counts and every raw gradient must be bitwise equal;
   gradient ownership must agree and match the independently specified active
   parameter-name set for the case, and all gradients must be finite.
2. Construct two B2/T512 microbatches from the established real-text fixtures.
   Tokens change; physical shapes and validity stay fixed. CE, latent and KL
   masks have different counts; the second microbatch has no auxiliary targets.
   Both use global objective-specific denominators summed across the two.
3. Independently compute canonical VJPs for each microbatch, clearing gradients
   between them, then add their completed FP32 CPU gradient tensors. Separately
   accumulate canonical backwards in microbatch order, then adapter backwards
   in the same order. Canonical versus adapter accumulation must be bitwise
   exact for all raw gradients and detached loss records, and the accumulated
   gradient-name set must match all expected case-active parameters. The first
   microbatch retains positive targets for every enabled objective, so the
   expected union is the full case-active set despite the second's empty aux.
4. Compare adapter accumulated gradients to the independent CPU sum. Different
   FP32 addition grouping is permitted only within prospectively fixed bounds:
   global gradient relative L2 <=2e-6, every tensor relative L2 <=2e-6 and
   maximum absolute error/reference tensor peak <=1e-5. A zero reference
   requires zero error. This check deliberately keeps BF16 physical shapes
   identical, avoiding a large-batch/microbatch kernel comparison.
5. Two canonical accumulated updates using the existing `optimizer_step`, then
   two adapter accumulated updates from identical initial weights and RNG.
   Each uses two same-shaped microbatches/update, independent global denominators,
   and clipping to1 after both backwards. Existing Adam policy: fused AdamW,
   LR1e-5, betas(0.9,0.95), epsilon1e-8, decay0.1 on matrices, two-update warmup.
   The canonical and adapter branches both disable only the autocast weight
   cache around the complete update; their forward-only enabled autocast
   boundaries stay intact, and nested RT backward replay inherits the same
   cache policy. W&B operations preserve RNG.
6. Keep the initial model state on CPU. After the canonical branch, release its
   optimizer/scheduler before restoring weights in place, then reconstruct the
   fresh adapter optimizer/scheduler. Verify the initial model/RNG boundary
   exactly. Never retain duplicate GPU models or optimizer states. At each
   corresponding update require exact loss/count/metric equality, identical
   input batches, and exact full model/optimizer/scheduler/counter/RNG digests.
   Independently check expected gradient participation before every step and
   changed probe weights plus finite parameters/moments after every step.
   The primary diagnostic performs **four physical updates** across two
   branches and reaches a **two-update logical endpoint**. The one-update CLI
   option means two physical updates and one logical endpoint; it is a labeled
   shorter diagnostic. The primary case has15 required gates.

All gates retain failures and stop dependent work; no automatic budget changes.
Physical optimizer steps and per-branch counts are recorded by a successful-step hook, including when
a later scheduler/health/logging operation fails. Backwards for references are
not optimizer steps. No inference or training-quality interpretation is made
from these changed weights, so no new long-term quality checkpoint is needed.

W&B: taylorbollman/pretrained-fbt-rt-nextlat, group olmo-distributed-prepare.
Create-only output directories retain raw reports, failed status, source and
protocol snapshots, hashes, dependency versions and the original checkpoint
reference. Verify all frozen sources/dependencies at completion. CPU references
are temporary RAM state. The milestone owner retains final reports/receipts
alongside the existing checkpoint lineage before two-GPU testing.
