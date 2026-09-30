# Saved-state feedback diagnostic

This milestone diagnoses the large later-pass CE deficit in the completed
32-update B/NF/NFR pilot. It makes **no optimizer update**, extends no training
run, and does not select an architecture by task performance. The user approved
the bounded investigation proposed in
[the preceding next steps](../olmo-adaptation-pilot/next-steps.md).

The completed cohort authority is
`.runtime/olmo-adaptation-pilot/summary-cohort-01/report.json`, SHA256
`ffb6e4eae228e4ae8e57394bc117a90ce2918eab3d3adb7820f64084d3e1178c`.
Its checkpoint publications, manifests, declarations and runtime-source pins
remain authoritative. Diagnostic helpers and their snapshots are additional
sources; they do not change the frozen 200-file training runtime.

## Question and scope

At update 32, NF development CE was 2.96039/7.39349/7.42586/7.43514;
NFR was 3.04301/7.02627/7.05529/7.06382. Both clipped all 32 updates.
These are full-panel observations, not the smaller diagnostic panel below.
They motivate separating three possibilities: incomplete feedback adaptation
already present at the imported origin, sensitivity to full-strength feedback,
and competing objective gradients. None is established by scalar losses alone.

NF means NextLat plus four-pass FBT; NFR additionally uses native RT in layers
0 and 15. Both retain the original predictor, fusion, tied backbone and
`continuous-stream-v1` semantics. NF is examined first because its deficit
does not require RT. NFR provides a bounded confirmation of the decisive
finding. No new beta schedule, loss-weight fork, placement sweep, precision
grid or automatic extension beyond update 32 is part of this milestone.

## Saved-state loading and execution

Use copied weights from the retained NF origin and NF/NFR update-32 checkpoints.
Do not reconstruct the origin predictor from a fresh random seed or substitute
the earlier fusion-only checkpoint for the actual paired origin.

Reuse the accepted native declaration/configuration constructor. Verify the
checkpoint publication and manifest SHA256, state-file size and SHA256, saved
metadata, counters and rank cursors before importing weights. Require complete
model keys, matching tensor shapes/dtypes and preserved tied aliases. Load all
model buffers, including the fusion output scale, with strict state-dict loading
and `assign=False` so destination parameter ownership remains intact.

This is an explicitly labeled **weights-only diagnostic import**, not an exact
training resume. The distributed resume loader requires the original two-rank
optimizer/RNG/scheduler contract and is inappropriate for this local probe.
Do not construct or load Adam, consume the training cursor, or overwrite any
checkpoint. Record checkpoint authority, source/model/configuration pins and
before/after model integrity for each independently saved case.

The NF origin was restored with the unchanged
`scripts.olmo_pilot_execution_restore` helper to
`/mnt/localssd/cdrm-checkpoints/feedback-diagnostic/nf0-restore-02`.
Restore evidence is `.runtime/olmo-feedback-diagnostic/restore-nf0-02/report.json`,
SHA256 `05dc4e1e14ffa724e055f5059ee2ba276aa2082c9b624f0adf3467a5a90af247`;
checkpoint manifest SHA256 is
`c9d834ba744073f6fcce108181df1825a07fd7b7cad70625bc31b5c5ffb4848d`.
This verifies generation-bound assets only, before diagnostic tensor loading.

Run on one GPU inside the required container, with FP32 master parameters,
autocast and TF32 disabled, forced SDPA math, eager pointwise/RT helpers, native
RoPE and zero feedback jitter. Configure these flags before preparing layouts.
No DDP, CUDA graph capture or throughput claim is involved. The established
`evaluation_runtime` context is suitable for forward-only cases; it includes
`no_grad`. The gradient probe instead requires grad-enabled training mode with
zero dropout, the same explicit FP32 settings, and `CampaignObjective`.

## Fixed data

The separately pinned `olmo-feedback-fixture-v1` fixture preserves original
tokens, document identities, valid masks, row identities and declaration/data
source bytes. It reuses `OrderedCampaignData.batch` and `build_nextlat_masks`.
The materialized fixture is
`.runtime/olmo-feedback-diagnostic/fixture-01/report.json`, SHA256
`b83df92927f56c0528bef44ff17a5928e1579d3834cb7e98d5c09f33afb99c7f`.

| Group | Ordered chunk ordinals | Use |
| --- | --- | --- |
| Development | 0 through 7 | First eight complete T1024 rows from the existing 64-row fixed development prefix |
| Training primary | 16384 and 16385 | First two rows of the next unused logical update, update 33 |
| Training conditional | 16386 and 16387 | Next two rows of that same update, only if a repeat is justified |

The recorded update-32 next cursor must equal update-33 start cursor:
`next_chunk=16384`, `next_update=32`. Retain metadata membership for all 512
rows of update 33 (ending at chunk 16896) and the original 64-row dev prefix.
This fixes selection before observing the diagnostic results. It establishes
row identity, not a new claim about corpus-level independence or overlap.

CE supervises each valid adjacent pair, including true document boundaries.
Latent pairs must remain inside one document; KL triples must remain inside
one document. Document IDs, not the presence of an EOS token alone, determine
auxiliary eligibility. Feedback eligibility under the continuous policy uses
valid adjacency, including document crossings. Do not accidentally apply the
stricter latent mask to fusion inputs. There is no cross-chunk target.

Independently check masks and counts from these identities. Full rows imply
8,184 dev CE targets and 2,046 primary training CE targets; latent/KL counts
are determined by actual document boundaries. Normalize each loss by its own
selected-batch count, not by input tokens or by a shared CE denominator.

The materialized counts are:

| Group | Input tokens | CE targets | Latent pairs | KL triples |
| --- | ---: | ---: | ---: | ---: |
| Development | 8,192 | 8,184 | 8,172 | 8,152 |
| Training primary | 2,048 | 2,046 | 2,044 | 2,040 |
| Training conditional | 2,048 | 2,046 | 2,046 | 2,044 |

## Forward cases and exact controls

Run NF origin at beta 1, then NF endpoint at beta 1, 0 and 0.5. Retain K4,
`enabled=True`, `first_pass_policy="configured-rt-v1"`, zero jitter and all
other settings. Run NFR endpoint beta 1 and beta 0, adding beta 0.5 if that is
the decisive NF comparison; retain native RT at 0/15 on every pass.

For eligible positions after the first token, the actual fusion is

\[
f(h,e)=s\,\mathrm{RMSNorm}\left(P_hh\odot
\sigma(P_e\mathrm{RMSNorm}(e))\right),\qquad
x=(1-\beta)e+\beta f(h,e).
\]

Here `s` is the stored embedding output scale, the previous-pass state at
position `t-1` feeds the new input at position `t`, and each pass executes a
fresh stack. Beta 1 is normalized feedback replacement, not a small additive
residual. The first token and ineligible positions retain the embedding.

The independent functional expectations are:

- Pass 1 is invariant across beta controls at a fixed checkpoint.
- At beta 0, every pass has the same original inputs, weights and RT mode and
  should reproduce pass 1. Compare hidden states and per-pass losses, recording
  exact equality and any maximum difference. A nonzero discrepancy is a
  localization issue before interpreting beta as an architectural result.
- Beta 0 bypasses the fusion module entirely. A fusion hook must have no calls;
  raw-fusion statistics are absent, not zeros or values retained from a previous
  case. Actual injected inputs equal embeddings. At beta 0.5, distinguish raw
  fusion output from the interpolated input the upper stack actually receives.
- Aggregate per-pass numerators and eligible counts across the eight rows
  before division. Do not average unequal local auxiliary means.

Record CE and the existing latent/KL terms for each pass. Additional summaries
are limited to final-normalized hidden-state variation, fusion/embedding RMS
and cosine, and aligned teacher/predictor next-token entropy. Hooks observe
and return no replacement output; they must not change tensor paths or RNG.

For representation variation, specify the selected valid positions and report
their count. Forward observation uses physical B1; centered coordinate variance
within that row is
`(sum(||h||²)/N - ||sum(h)/N||²) / D`; use stable accumulation and distinguish
within-row variation from an eight-row pooled quantity. The implementation can
compute centered differences directly rather than subtracting close moments;
do not label a mean of row RMS values as pooled variance. RMS alone is weak
evidence because fusion normalizes its output. Fusion cosine compares flattened
eligible position-feature arrays, not mean token cosine or cosine of pooled
means; preserve that label when aggregating raw dot products and squared sums.

Teacher/student entropy uses the full native readout vocabulary and the same
KL-eligible triples. The teacher state `h[t+1]` and prediction of that state
must be aligned; both readouts predict the following token. Position chunks
may bound memory, but each softmax normalizes over the complete vocabulary.
Require finite entropy in `[0, log(vocabulary_size)]` within roundoff and
explicit counts. Missing eligible positions yield absent statistics, not a
fabricated zero. The predictor readout is a diagnostic recomputation on selected
positions; regrouping its GEMM may introduce small FP32 roundoff relative to
the original loss computation. It is not an exact KL reconstruction. Neither
low entropy nor low variance alone proves collapse.

## Per-loss gradient decomposition

At NF update 32, use the primary two-row batch at beta 1. Keep every branch
enabled, retain actual objective weights and detached targets, and evaluate
the existing prepared loss functions. With each term already normalized by
its own selected-batch count, the four contributions are:

1. First-pass CE: `0.5 * CE1`.
2. Later-pass CE: `(CE2 + CE3 + CE4) / 6`.
3. Latent regression: `(A1 + A2 + A3 + A4) / 4`, including its configured weight.
4. KL: `(Q1 + Q2 + Q3 + Q4) / 4`, including its configured weight.

Latent regression uses coordinate-mean SmoothL1 against detached actual next
states. KL is `KL(teacher || student)`, with a detached teacher state and
detached readout weights. The predicted-state branch remains differentiable;
its embedding input can still send gradients into the shared embedding.
Do not accidentally remove that route or add teacher/readout gradients. There
is no supervised token CE directly on predicted states.

`feedback_gradient_probe` computes four separate VJPs and a fifth fresh forward
through the actual joint `CampaignObjective` as an independent reconstruction
reference. It uses local world size one and the two-row counts, not the saved
large-batch denominators. It does not write `.grad`, clip gradients or apply
Adam. Report norms, Gram entries and cosines for unique parameters grouped as
backbone, fusion and predictor, plus their union; tied readout ownership must
not be counted twice. In addition to pairwise geometry, report the combined
CE gradient versus combined auxiliary gradient on shared parameter groups.

Structural expectations are zero predictor gradient from either CE term and
zero fusion gradient from first-pass CE. Zero-vector cosines are undefined and
reported as null. The sum of measured components is compared with the joint
gradient using bounded FP64 CPU reductions of FP32 vectors. Gate only global
relative reconstruction L2 at `1e-4`; per-group residuals and maximum absolute
differences remain descriptive. This checks the decomposition, not BF16
equivalence or the original large-batch update.

If the first batch shows clear shared-parameter opposition, repeat on the
predeclared conditional batch and perform the decisive endpoint NFR probe.
Record the reason before running the conditional case. If no such opposition
appears, do not launch a wider gradient grid. A single saved-origin joint
gradient on the same batch can be considered only to distinguish startup scale
from endpoint scale, as in the approved next-step recommendation.

## Persistence, stop conditions and interpretation

Save each completed case independently, with explicit checkpoint/fixture/helper
pins, precision/runtime settings, pass structure, counts and preservation
checks. Persist scalar progress after each gradient contribution and retain
completed artifacts every 20–30 minutes. Raw gradient vectors are temporary
CPU working data; if interruption happens mid-decomposition, rerun that small
case from the same weights. Existing training checkpoints remain untouched.
Retain reports and useful evidence in `gs://fast-chunks`; do not treat volatile
SSD copies as durable recovery points.

Stop for a pin mismatch, changed frozen source, nonfinite quantity, incorrect
beta-zero/pass-one control, failed joint reconstruction, or changed weights,
buffers, gradient ownership or RNG. Diagnose the specific failure before
starting another case. Do not weaken a check to complete the planned matrix.

Compare origin and endpoint on this same fixed subset, not directly against
the cohort's larger panel. Interpret opposition as local geometry, not proof
that an auxiliary caused the training deficit. These two-row FP32 norms are
not numerically comparable with 524,288-input BF16 training norms. The cohort
still reflects a fresh optimizer, new predictor, early 100-update warmup,
single seed and limited fixed-prefix strata. An interpolation benefit can
motivate a later training proposal; it is not permission to change the recipe
or evidence that a new recipe has already trained successfully.

The milestone ends with a short diagnosis and the smallest proposed next
learning experiment, if warranted. It does not automatically launch that
experiment or continue any arm past update 32.
