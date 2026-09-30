# Proposed next comparison — not launched

The current evidence does not justify replacing kernels, changing Q/K
normalization, weakening numerical checks, or abandoning RT. NF already has the
deficit without RT, and the controls work with RT present. It also does not
justify assuming that more training alone will fix the feedback route.

I recommend a short paired NF continuation from the same update-32 checkpoint:
keep the existing objective as control and reduce the KL weight from 1 to 0.1
in one explicitly new branch. Keep latent weight 1, full-strength feedback,
K4, jitter, masks, data order, effective batch, precision and learning-rate
schedule unchanged. Preserve the saved backbone, fusion, predictor and Adam
moments in both branches; declare the changed objective rather than treating it
as an exact same-configuration resume. Do not reset optimizer state.

KL is the largest measured auxiliary gradient contribution, so this changes
one interpretable quantity while retaining NextLat training. The negative
first-pass alignment is local and the combined gradient is not uniformly
opposed to CE; this is a hypothesis test, not a demonstrated fix. A CE-only or
predictor-only fork would change more pathways at once and can wait.

For review, propose 32 additional matched updates per branch, with common
scheduled full-panel FP32/no-jitter evaluations at 48 and 64. Keep all per-pass
CEs and their gap to pass 1 visible alongside latent/KL and clipping. Reuse the
asynchronous SSD-to-GCS checkpoint path. Finalize new identities and resource
cost before launch; observed prior NF compute suggests tens of minutes per
branch, with checkpoint/evaluation overhead additional. Updates through64
would still be inside the100-update warmup, so this remains directional.

If the control improves comparably, ordinary adaptation remains plausible.
If reduced KL protects first-pass CE and improves later-pass CE, replicate the
specific finding with NFR next. If both remain poor, revisit feedback startup
or fusion-focused adaptation; half-strength interpolation alone did not help.
Do not expand to a beta/loss/RT-placement grid based on these small probes.
