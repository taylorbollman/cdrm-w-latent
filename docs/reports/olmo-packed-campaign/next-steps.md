# Recommended next milestone: bounded numerical localization

The packed-data/update/recovery milestone now passes same-runtime, two-H100
T1024/B12 recovery under explicit deterministic backend controls. The corrected
cloud-restored continuation passed all 10 gates, with input/noise, raw gradients,
metrics and model/Adam/cursor/RNG state bitwise equal on both ranks. The original
failed attempt is retained; see [repeatability investigation](restart-repeatability.md).

Numerical localization remains the next priority before quality training:
~86% BF16/full-FP32 combined gradient difference on initial NFR T16 and the
separate BF16 sparse/prepared discrepancies are unresolved. Exact checkpoint
continuation establishes repeatability under the fixed execution contract,
not agreement with a different precision/backend implementation. Preserve these
qualifications and avoid a threshold change.

1. Freeze the next bridge to **six backwards**: CE-only and combined objectives
   on each of three paths: full-FP32 math-SDPA/eager-native-RT, BF16
   math-SDPA/eager-native-RT, and BF16 Flash-SDPA/Triton-native-RT. Reuse the same
   pinned initial NFR state, T16 rows, jitter, pass weights and normalization.
   Configure deterministic execution before CUDA initialization in this new
   diagnostic and retain TF32-off controls on all paths. Compare forward
   states by FBT pass as well as full raw-gradient norms, angles and errors.

   Build the BF16 bridge from the existing BF16 runtime flags: change the RT
   forward/backward tile backends to `eager` and force ordinary SDPA to `MATH`,
   while retaining **`attention_precision="mixed"`**, BF16 autocast and FP32
   master parameters. Keep recomputation, cast-once, RoPE reuse and KV-only
   writes unchanged. Merely calling `configure_full_fp32()` and re-enabling
   autocast would leave RT attention at FP32 and introduce an unintended fourth
   precision regime.

   This bridge separates the combined backend change from precision effects;
   it does not individually attribute a gap to Flash versus Triton. Add only
   one crossed-backend condition if that distinction is needed after seeing
   these results. No optimizer updates, new training run, all-length sweep or
   new BF16 acceptance threshold is part of this diagnostic.
2. In a separate small loss-level diagnostic, reuse identical detached hidden
   states, embeddings, readout and predictor values for sparse/prepared losses.
   Compare the incoming hidden-state cotangents for latent and KL in FP32 and
   BF16, then propagate a common cotangent through a fixed backbone. The
   existing 3.40224% layout difference localizes to auxiliary participation,
   but component gradient norms do not add linearly under rounded backwards.
3. Follow the resulting branch: change one kernel/precision boundary at a time,
   or record activation/cotangent scales by FBT pass and RT layer. If necessary,
   add only the smallest feature ablation that separates RT, feedback and their
   interaction. Check a bounded real packed-data fixture before generalizing
   from T16. Avoid an all-arms/all-lengths numerical sweep without a hypothesis.
4. Prefer a narrow implementation or precision correction if supported. A
   Q/K-normalization transition changes the model and should follow evidence,
   not serve as an unexplained fix for the present discrepancy. Reuse frozen
   input/source pins and rerun only checks affected by a fix.

No quality-training campaign, production mixture, H200 estimate or new model
architecture is chosen here. H200 readiness still needs short actual-hardware
NCCL/capture/restart and memory checks. A production loader will additionally
need the selected source mixture/order/shuffle/tail policy, held-out evaluation
fixtures, and the agreed common token-clock calibration; the present finite
coverage corpus and recovery checkpoint are not that campaign.

Before turning this bounded readiness runner into a longer-lived campaign
runner, coordinate rank-zero logging failures as well as data/optimizer
failures. Its current `tracker.log()` is outside the coordinated error path:
a logging exception can leave another rank waiting until torchrun tears the
job down. This does not invalidate successful acceptance evidence; preserve
the frozen tested runtime and address it with campaign-loop hardening.
