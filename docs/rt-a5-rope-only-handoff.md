# A5 RoPE-only control, 50k updates

## Authorization and scope

While the released-GPT control was still training to 100k, the user explicitly
authorized step 3 without an intervening pause: train **Our Transformer with
RoPE instead of ALiBi** to **50,000 updates**, then pause for review. Complete
the already-running reference experiment first. Do not change the RT model,
add NextLat, or extend this control beyond 50k.

The new frozen protocol is
`.runtime/rt-a5/20260911T210811Z-rope-only/protocol.json`.
The earlier 100k controls and their original protocol remain preserved in
`.runtime/rt-a5/20260911T201820Z-transformer-reference/`, with an explicit
authorization amendment documenting the new follow-up.

## The isolated change

Use the original OLMo sequential model's existing RoPE implementation.
Construct the RoPE configuration and strictly copy every learned parameter
from the original SEQ initialization. The entire resolved model configuration
must differ only in `alibi: true -> false` and `rope: false -> true`.

Preserve two layers, width 512, eight heads, head dimension 64, 60 classes,
GELU FFN width 2048, LayerNorm, learned full-width Q/K normalization, Mitchell
initialization, untied embedding/head, zero dropout and **6,357,504 parameters**.
The original Q/K normalization still runs before RoPE. The native RoPE path
uses its existing configuration defaults; it does not import RMSNorm,
SwiGLU or the released GPT's initialization. This is a SEQ control, not an
attempt to insert RoPE into the recurrent kernel.

The seed-1234 parameter hash must remain exactly
`0380e2fd4cdd4db63ce6d732a0834ad054e185c2276da55d733839276d8c55ad`.
The initial states are paired; training starts fresh at update zero, rather
than switching positional encoding in a trained ALiBi checkpoint.

New files are `scripts/rt_a5_rope.py`, `scripts/rt_a5_rope_train.py`, and
`configs/rt_a5_rope/base.json`. Keep the historical 43 executed files and
the reference control's 46 executed files unchanged. The new driver imports
the same historical update, evaluator, data-order, batch, loss and optimizer
functions. Its schema is `rt-a5-rope-training-v1`, architecture `seq_rope`.

## Execution and comparison

Use the original frozen `.runtime/rt-a5/20260911T154748Z/data`, batch 1024,
seed/order seed 1234, training length 12, full FP32, math SDPA, TF32 disabled,
eager execution and the unchanged AdamW settings. Save initialization plus
10k/25k/50k checkpoints; evaluate 102,400 dev and OOD words at each trained
checkpoint, with 4,096-row checks every 5k. Confirmation remains untouched.

Only run GPU work through the project Docker launcher after checking the
container, cwd and `nvidia-smi`. CPU checks explicitly disable GPU passthrough.
The RoPE GPU run follows the reference GPT run sequentially. Use a disposable
bounded actual-shape check before starting fresh training.

- Run directory: lineage `train-seq-rope/`.
- W&B: `taylorbollman/rt-a5-state-tracking`, group `20260911T210811Z-rope-only`.
- GCS: `gs://fast-chunks/cdrm-w-latent/rt-a5/20260911T210811Z-rope-only/`.
- Report: `docs/reports/rt-a5/rope-only-50k/`.

The primary causal comparison is **SEQ 50k versus SEQ+RoPE 50k**. Use the
already-saved SEQ checkpoint from the earlier 100k continuation. Include
released GPT 50k and RT 50k as context. The other jobs' 100k endpoints do not
change which checkpoints or data-order prefixes belong in this comparison.
For learning curves and training time, use the first 50k updates of every
arm. Keep the released GPT's different initialization explicitly unpaired.

Report E(t), A(t), and M(t) on the same length-36 OOD words, with a full 1–36
view, a clearly labeled 10–18 zoom and an E-only view. Preserve 10k/25k results
as diagnostics. Do not choose a favorable checkpoint after viewing results.

Both ordinary attention-only positional schemes share the repeated-initial-
operation ambiguity described in
[the previous handoff](rt-a5-transformer-reference-handoff.md#shared-repeated-prefix-limitation).
That small common ceiling does not explain a large later-position difference.

## Completion

Validate exact initialization/config isolation, common source and batch-order
identity, full evaluation counts, and finite final FP32 model/Adam state.
Retain all four new checkpoints and source/report evidence in GCS with
verified checksums. Record endpoint results and links below, then pause at
50k for user review.

RoPE-only training completed successfully and stopped at exactly **50,000
updates**, after the reference GPT completed 100k. Eight model and twelve
trainer CPU checks passed, including exact
configuration/initialization isolation and a manual attention-order check.
The disposable actual-shape GPU check passed 75 updates (25 warmup) plus
a length-36 forward check. Measured update time is about 17 ms.

Frozen 46-file execution source SHA:
`657d045a831613fa3accedc326dfc87781c477734cf988d27335a522108a2ad2`.
See lineage `checks/source-freeze.json`. Both historical 43-file and reference
46-file source identities remain unchanged.

Completed training: <https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/whppi5e6>.
At 10k, short dev token accuracy is 50.9282%, with whole-word exactness zero;
the OOD E≥50% horizon is 6. Original SEQ at 10k has 39.7189% short dev token
accuracy and horizon 5; the full released architecture has 70.9291% and horizon
8. These are intermediate diagnostics; the primary comparison remains 50k.

## Final result and review point

The [50k report and plots](reports/rt-a5/rope-only-50k/report.md) and
[W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/eiwzzng3)
use exactly matched budgets and evaluation words.

| Backbone, all at 50k | L12 dev token accuracy | L12 dev whole-word exactness | Last OOD prefix with E≥50% | OOD E(13) |
| --- | ---: | ---: | ---: | ---: |
| Our Transformer, ALiBi | 56.0525% | 0.0352% | 6 | 0.0029% |
| Our Transformer, RoPE | 75.2426% | 0.8223% | 9 | 0.0625% |
| Authors' GPT architecture, context | 90.6925% | 5.2578% | 11 | 0.1318% |
| Our RT, context | 99.9793% | 99.9287% | 13 | 81.7979% |

RoPE alone adds **19.1901 percentage points** of short dev token accuracy.
On that particular metric it closes approximately **55.4%** of the gap to
the full released architecture, calculated as `(RoPE - ALiBi)/(GPT - ALiBi)`.
This is a descriptive fraction for this seed and budget, not an additive
decomposition of interacting architecture changes. The exact-prefix horizon
moves three positions, from 6 to 9, while the full released architecture
reaches 11. RoPE is a major contributor under this protocol, but leaves a
substantial gap, especially in whole-word accuracy at the training length.

The remaining architecture differences include RMSNorm, absent Q/K
normalization, SwiGLU and initialization. This run does not determine which
of those changes or their interactions account for the rest. RoPE in RT
remains untested; the current experiment changes only the nonrecurrent SEQ.

The complete primary model configuration differs only in the two PE flags.
The stored-state audit confirmed that the actual saved RoPE initialization
equals the retained original SEQ initialization, and all 19 final parameter
and Adam states are finite FP32 with counters exactly 50k. All four arms'
first 50k batch-order hashes and shared execution contracts agree. The 16
reporter tests also passed; full/zoom/E-only curves share the same data.

All four new checkpoints are retained in GCS with verified checksums.
Archive/readback receipts are stored in the lineage as `evidence-storage.json`,
`evidence-storage-upload.json`, `evidence-readback.json`, and
`evidence-readback-upload.json`. The earlier 100k comparison is independently
archived and closed. **Pause here for user review; do not extend RoPE beyond
50k or begin an RT positional-encoding change without further instruction.**
