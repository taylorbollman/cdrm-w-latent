# NFR endpoint settling and continuation to update 128

Authorized by the user on 2026-09-30 after review of the paired KL experiment.
NFR means NextLat + four-pass full-bandwidth feedback (FBT) + native RT at
layers 0 and 15. This is a bounded functionality/adaptation milestone.

## Questions and sequence

1. Measure matched K1–32 pass curves at the saved NFR64 KL1 and KL0.1
   endpoints. Use the existing eight packed development rows, T1024, FP32,
   no fusion jitter and no optimizer updates. Authenticate these newer states
   explicitly rather than changing the older diagnostic's checkpoint scope.
2. Inspect finite values, hidden scales, tail changes, entropy and CE. Emphasize
   practical K4/K8 residuals and the K4-to-K32 CE difference. Reaching the FP32
   numerical floor before pass five is not an acceptance requirement. Settling
   to a poor prediction is different from useful refinement.
3. If the endpoints remain bounded and settle, continue **only the KL0.1
   branch from 64 to 128**. Preserve its populated Adam state, rank RNG, data
   order, original token/LR plan and all model/loss settings. Observe the
   existing update-100 warmup boundary and retain 96/100/128 for inspection.
4. Stop at 128 and review. An extension beyond 128, a matched KL1 continuation,
   or a new NF-only experiment is not part of this execution.

Nonfinite values, persistent growing/oscillating deep-pass states, failed
state preservation or a checkpoint/resume mismatch require localization before
continuation. Slow but decaying changes, large finite clipped gradients or
later-pass CE worse than pass one do not alone indicate an execution failure.
They must remain visible in the report rather than being relabeled as success.

## Fixed training setup

- OLMo-1B checkpoint at step 60,000 (roughly 252B pretrained tokens), with the
  existing fusion preparation and subsequent NFR training lineage.
- BF16 mixed training with FP32 parameters/Adam; existing native RT, ordinary
  Flash SDPA, activation checkpointing, CUDA graphs and fused AdamW.
- Four total FBT passes, full fusion strength, jitter 0.02, NextLat latent
  coefficient 1 and KL coefficient 0.1. CE weights remain 1/2 for the first
  pass and 1/6 for each later pass; auxiliary losses are averaged over passes.
- T1024, physical batch 12 per rank on two H100s, 524,288 real input tokens per
  optimizer update; 22 accumulation slots per rank, with unchanged dummy rows.
- The existing 128-update schedule warms from 10% to 100% of peak LR 2e-4
  through update 100. Continuing does not restart warmup.
- Added exposure: 64 updates / 33,554,432 input tokens. Cumulative exposure at
  128: 67,108,864 inputs, excluding earlier fusion preparation and pretraining.

NextLat predicts the next **token-position** representation within each pass,
not the next feedback iteration. Its shared predictor may encounter different
distributions across passes; whether this materially impedes learning remains
a hypothesis, not an established explanation.

## Comparison and interpretation

Use the same regular 64-row FP32/no-jitter panel for training progress and the
separate eight-row panel for deep-pass curves. Report their metrics separately.
Keep absolute first/later-pass CE, raw latent/KL losses, preclip norms and
clipping frequency visible. A smaller weighted objective is not evidence of
better predictive quality. The unextended KL1 checkpoint at 64 is not a
matched control for the continued branch at 128.

Paper Figure 3 uses an absolute hidden-state norm; our principal curve uses
relative RMS changes on declared position masks. Different y-axis thresholds
are not directly comparable. Convergence of successive passes is neither
proof of useful refinement nor general generation/exact-online equivalence.
This milestone also does not resolve the prior BF16 optimization-equivalence
qualification or establish RT's causal benefit.

## Execution and interruption recovery

The endpoint probes can run independently on one GPU each. Release both GPUs
before two-rank training. The continuation is expected to take roughly four
hours, subject to startup and storage overhead.

Keep the accepted immutable SSD checkpoint plus asynchronous verified GCS
publication policy. Save progress approximately every 20–30 minutes or sooner
under the existing trigger. Training may continue during upload; interruption
before verification can lose work since the last verified cloud checkpoint.
Drain uploads at the terminal boundary. Preserve original parent32 metadata
and files where required by strict child-resume authentication.

Log graphable measurements online under `taylorbollman/pretrained-fbt-rt-nextlat`.
Store resumable states under local SSD and `gs://fast-chunks`; retain code,
source snapshots and small evidence on persistent storage and in Git/cloud.
After an interruption, inspect active containers, reports and publication
receipts before launching or resuming work.

No architecture, fusion ramp, pass-frequency schedule, Q/K normalization,
precision policy or kernel optimization is introduced here. New execution and
diagnostic authority must preserve the existing frozen training-source lineage.
