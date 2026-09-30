# Proposed next milestone: replicate the KL comparison with native RT present

The NF continuation gives a useful directional result: reducing the KL weight
from 1 to 0.1 improved raw development CE in every pass and reduced gradient
norms. It also increased raw latent and KL losses relative to the control in
every pass at update 64. Later passes remain substantially worse than pass 1.
This is a loss tradeoff with improved CE, not a complete fix for feedback or
evidence that all NextLat objectives improved.

I recommend one bounded, matched **NFR** replication before changing another
design element. This is a proposal; no NFR continuation was launched as part
of the current milestone.

## Paired comparison

Start both branches from the same complete NFR update-32 checkpoint from
`native-nfr12-first32-01`. Its saved distributed manifest SHA256 is
`1c83b37812005b9d87a3ce06a1156b18789498e0d4a1ce7964cb16c5338bc82c`.
Authenticate its original publication and report before launch. Preserve its
own model, populated Adam state, scheduler, counters, cursors and per-rank RNG.
Do not import NF weights or reset the NFR optimizer to make this comparison.

Run KL 1 versus KL 0.1 through update 64, keeping latent weight 1, K4 FBT,
feedback beta 1, jitter 0.02 and native RT at layers 0 and 15 unchanged. Retain
the accepted T1024 data plan, physical batch 12 per rank on two H100s, effective
524,288 input tokens per update, masks, BF16 production path, CUDA graphs and
the original 128-update scheduler with its 100-update warmup. Compare 32
additional matched updates per branch. This is the same intervention tested
in NF, now with RT present; it does not isolate the causal value of RT across
the independently adapted NF and NFR histories.

Use a clearly declared objective fork and exact saved-parent loading, then
strict branch identities for child restarts. The current launcher deliberately
permits native NF only, so extend that scope explicitly with tests and fresh
source pins. Do not silently reuse an NF declaration for NFR. Existing tiny
NFR acceptance already exercises the KL transition and exact child restart
with RT present; reuse this evidence where the implementation remains
unchanged rather than starting a broad new numerical campaign.

## Measurements and decision

Repeat common FP32/no-jitter evaluation at restored update 32 and at updates
48 and 64 on the same named development prefix. Require matching raw
update-32 losses across branches. Track:

- Absolute per-pass CE and later-minus-first-pass gaps. Better gaps alone can
  be caused by deterioration of the first pass.
- Unweighted latent and KL losses per pass, with the unchanged eligible-target
  denominators. The smaller branch-specific total objective is not evidence
  of improvement.
- Pre-clip norms, clipping coefficients/frequency, finite updates and unchanged
  graph/evaluation preservation checks.
- Input exposure, LR, scoped throughput, memory and retained checkpoint
  authority, using the same reporting conventions as this milestone.

If lower KL again improves CE while preserving first-pass behavior, it becomes
a reasonable candidate for the next small combined-model pilot. That would
still leave the later-pass deficit and auxiliary tradeoff to assess. If NFR
does not show the same direction, examine the matched trajectories and scope
of the difference before making another intervention. If both branches remain
poor, return to feedback startup or fusion adaptation rather than widening the
loss/precision/RT-placement search immediately.

No outcome from this short comparison establishes long-run quality, native RT
benefit, representation collapse, or new BF16 numerical clearance. Stop at 64
and review. Do not automatically extend training or remove another loss.

## Runtime and recovery

The prior NFR first-32 stage took about 114 minutes including its evaluation,
checkpoint and retention costs. Budget roughly **3–4 hours for the two
branches**, with additional preparation/verification overhead possible; this
is a planning estimate, not a throughput guarantee. Use both GPUs for one
branch at a time, with the accepted periodic local checkpoint plus asynchronous
CPU upload/readback workflow. Preserve both parent recovery inputs and the
latest verified child publication. Unexpected VM loss may roll back work
after the latest verified cloud checkpoint.

Continue retaining progress every 20–30 minutes or sooner. Reuse the existing
failure stops for nonfinite values, state/configuration mismatch, failed
preservation or failed publication. Heavy clipping by itself remains a
measurement to interpret, not an added automatic stop rule.
