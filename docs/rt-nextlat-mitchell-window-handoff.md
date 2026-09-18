# RT + NextLat: Mitchell positions and second-layer window pilots

Authorized on 2026-09-14: restore original Mitchell initialization, first run
fixed sinusoidal positions with full RT, then choose between those positions
and the original ALiBi pilot and run a second pilot with window length two
in the second recurrent block only. The user explicitly authorized proceeding
through both pilots without an intervening approval.

## Scope and lineage

Lineage: `.runtime/rt-a5/20260914T175404Z-nextlat-mitchell-window/`.
The prospective `protocol.json` freezes the two-stage procedure. Historical
identity-centered code and artifacts remain unchanged; the new active factory
starts from the original Mitchell backbone and changes no learned tensor.

Both pilots are fresh 0→10,000 runs with checkpoints 0/1k/5k/10k, physical
B1024, length12 training, and the same frozen A5 data/order as the original
RT+NextLat pilot. The comparator is
`.runtime/rt-a5/20260911T191702Z-nextlat/train-rt-nextlat/`.
The new directories are `train-sinusoidal` and `train-window2`.

Architecture: two tiled recurrent blocks, rho1, D512/H8/GELU FFN2048,
LayerNorm plus learned full-width Q/K normalization, vocabulary60, untied
embedding/head, dropout0. The backbone has6,357,504 parameters and the
unchanged NextLat predictor1,049,600, totaling7,407,104 in25 tensors.

All new learned initial tensors must equal the original step0 checkpoint
bitwise. Backbone canonical SHA256 is
`0380e2fd4cdd4db63ce6d732a0834ad054e185c2276da55d733839276d8c55ad`.
Backbone/data-order seed1234 and predictor seed1235 remain unchanged.
The NextLat predictor uses its original initialization, not the backbone's
Mitchell distribution. No identity-centered initialization remains in these arms.

AdamW LR1e-4, betas(.9,.95), epsilon1e-8, matrix decay.01/vector decay0,
global clip1, fullFP32/math attention, no autocast/TF32/compile/CUDA graphs.
The same original function objects perform the joint CE+SmoothL1 update,
teacher-conditioned diagnostics, evaluation and word ordering. Prediction
targets remain post-final-LayerNorm latents, detached in the target role;
source latents and raw next-operation embeddings remain attached. No
autonomous NextLat predictor rollout is used.

## Positional selection

Pilot1 restores original learned initialization and replaces ALiBi with
fixed unit-amplitude sinusoids, base10000, origin0, added once to unscaled
backbone token embeddings. No RoPE or learned positional parameters are used.
The predictor conditioning embedding remains the raw token embedding.

Selection compares the fixed10k endpoints on the same102,400 OOD-development
words. The prospective score is

\[
S=\frac1{24}\sum_{t=13}^{36}E(t),
\qquad E(t)=P(\text{every state through }t\text{ is correct}).
\]

This is unconditional: words with an error in their first12 operations
contribute zero, rather than being discarded. Equivalently,24S is the
expected number of additional fully correct prefix steps beyond12, capped
at36. Integer counts are used for selection; exact ties retain ALiBi.
Short-development accuracy, individual E(t), A(t), M(t) and1k/5k learning
curves remain descriptive. This is one-seed development selection, not a
confirmed population ranking or a best-checkpoint search.

`scripts/rt_a5_window_select.py` verifies source snapshots, paired contracts,
all10k minibatch-order hashes and endpoint checkpoint hashes, then writes
`position-selection.json`. Its prospective source copy is retained in
`selector-source/`. The second pilot uses that selected position scheme
from fresh original initialization, rather than continuing the first pilot.

## Exact layer-2 window semantics

For second-block input x_t and output z_t, the direct attention records are

\[
\mathcal J_t=\{t-1,t\}\cap\{0,\ldots,t\},\qquad
r_{t,t}=x_t,\qquad r_{t,t-1}=z_{t-1}.
\]

The self record is provisional input-derived K/V. The immediately preceding
record is permanent output-derived K/V, including that preceding step's
attention, residual and MLP computation. The block applies the original
normalization, Q/K/V projections, per-head softmax, output projection,
residual and MLP, restricted to these records. At t=0, self is the only record.

Layer1 remains full-prefix RT. The preceding layer2 output can already encode
earlier tokens, and gradients remain attached through every recurrent update.
Thus window2 limits direct reads; it does not truncate the effective history
to two tokens or implement truncated backpropagation through time.

The new block subclass applies an additive -infinity exclusion mask inside
layer2, after OLMo's global mask preprocessing. It preserves ALiBi values on
the permitted keys if ALiBi wins. The mask is passed to the original tiled
forward and backward reconstruction. Every query retains a finite self
contribution, including cross-tiles with no permitted permanent keys.
No sparse-kernel speed improvement is claimed: the original tiled arithmetic
still executes its existing tile schedule.

## Source identity and checks

New training files: `scripts/rt_a5_window.py`,
`scripts/rt_a5_window_train.py`, and `configs/rt_a5_window/base.json`.
The model reuses the existing parameter-free `FixedSinusoidalPositions`
class, without invoking the old identity-initialization factory.
New checkpoint schema: `rt-a5-window-training-v1`; the strict contract
includes `experiment_config` with positions and layer2 window.

The frozen training manifest includes the historical49 files plus those
three files (52 total), SHA256
`9d12d61e994bdad57567cb1d30fd36f1ea3296c94f7aa401d574ecbb34d339c2`.
Both pilots must retain this source identity. Historical NextLat46 and
identity/sinusoidal49 identities are unchanged. New validation, selection
and reporting programs have separately retained provenance.

Before training:27 model,21 trainer and12 independent-reference CPU tests
passed. The direct reference physically constructs just the two permitted
records and independently computes its projection/normalization/MLP math.
It agrees with masked naive RT at lengths1,2,3,12, and tests finite nonzero
temporal gradients through earlier permanent writes. T1/T2 reproduce full
RT. Actual GPU checks are in `preflight-sinusoidal` and `preflight-window2`;
the latter compares the joint-loss tiled model with the independent scan.
Each actual-shape smoke uses25 discarded B1024/T12/D512 updates.

Saved-state auditing is CPU-only and verifies actual step0 tensors against
the original checkpoint, plus final FP32 parameter/Adam state, counters,
source identity and complete paired data-order histories. Final confirmation
remains unevaluated.

## Tracking and retention

All GPU execution uses the project Docker launcher with container path and
`nvidia-smi` checks. CPU artifact checks use `CDRM_DOCKER_GPUS=none`.
W&B project is `taylorbollman/rt-a5-state-tracking`, group
`20260914T175404Z-nextlat-mitchell-window`.

[Mitchell + sinusoidal training](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/oe69tg2n).
The lineage's launch scripts, logs, source freeze and protocol preserve exact
commands. Checkpoints/evidence go to
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260914T175404Z-nextlat-mitchell-window/`.
The retention helper verifies uploaded checksums and records object generations.
Archive only after both runs, reports and audits finish; archive-command logs
must stay outside the archived lineage.

## Completed outcomes (2026-09-14)

Both authorized pilots finished exactly 10,000 updates, with online W&B
synced and no training left running for this task. The second pilot was a
fresh start, with no continuation from the sinusoidal run.

The prospective selection chose **ALiBi**: mean E(13–36) was
0.07219523111979166 versus 0.02473876953125 for sinusoids. The recorded
`position-selection.json` binds this decision to both endpoint reports,
checkpoints, protocol and the selector source frozen before outcomes.

The ALiBi window-2 endpoint scored mean E(13–36) 0.04581461588541667,
below the full-attention ALiBi control. Its E(13)/E(14) were
79.6963%/26.0879%, versus 96.8652%/62.2373% for that control. Its short
development whole-word accuracy was 93.0303%, versus 98.3047%. Sinusoidal
full RT scored 54.2559%/4.9277% at E(13)/E(14), with 97.8350% short whole-word
accuracy. All three had zero observed E(36). These are fixed-endpoint,
single-seed development comparisons, without confirmation evaluation or
claims about an architecture's eventual capability.

See [the outcome summary](reports/rt-a5/nextlat-mitchell-window-10k/outcome.md)
and [complete report](reports/rt-a5/nextlat-mitchell-window-10k/report.md).
Full and boundary plots were visually inspected and use identical metric
rows. The reporter and independent artifact audit passed. W&B:
[window-2 training](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/6c59kpwp)
and [three-arm comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/u991ouce).

All 79 CPU tests and both GPU preflights passed. The window preflight compared
combined-loss tiled gradients with the independent physical two-record scan
at T2/T12/T36; maximum absolute parameter-gradient error was 5.07e-7.
Saved-state audits passed 34/34 checks for each sinusoidal init/final state
and 39/39 for each window init/final state. The independent artifact audit
verified 432 metric rows, 18 evaluations, 12 checkpoint hashes, 150 source
checks, 33 report artifact hashes and all 10,000 shared minibatch orders.
Evidence is in `final-evidence-audit.json` and the per-arm state-validation
JSON/logs within the lineage.

All eight new checkpoints are retained in GCS with verified checksums.
The final evidence archive is created after these notes and all artifacts
are stable; `evidence-storage.json` and its separate readback receipt are
the authoritative closure records. Archive/readback command logs live in
`/tmp`, outside the archived lineage. Do not modify archived members after
closure; use a new lineage for future experiments. The original 52-file
training source identity remains unchanged.
