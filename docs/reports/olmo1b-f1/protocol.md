# F1: bounded actual-checkpoint integration and early profiling

Prespecified 2026-09-22 under the user-approved functionality-first
[v4 plan](../../fbt-rt-nextlat-research-plan-v4.md). This protocol authorizes no
quality comparison or long learning run.

## Model and objective

Use original OLMo-1B step60000 (~252B tokens), native revision
81b71efbce6f4dada57c94860301af4298bcd351, checkpoint SHA256
ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c.
Preserve native dimensions, tied readout, RoPE, LayerNorm and absence of Q/K norm.
RT selects layer0 except the explicitly named (0,15) case.
Selected RT uses existing eager dyadic tiling and explicit returned parameter VJP.
Ordinary blocks use SDPA; profile actual dispatch rather than assume Flash.

Keep existing FBT fusion initialization/scale and NextLat LM predictor recipe,
unit CE/latent/KL weights, gamma1 and pass0 + mean(extra-pass losses).
K includes the ordinary bootstrap; RT applies to extra FBT passes.
Vocabulary projections use chunks of128 positions without reducing vocabulary.
Freeze registered unused fusion matrices in FBT-off cases before creating AdamW.
Keep fusion trainable throughout the beta0-to-positive transition case.
NextLat-off construction omits the predictor.

All GPU commands run inside the verified project container on the H10080GB.
Use BF16 autocast with FP32 parameters/gradients/moments, native mixed RT
attention, TF32 off, no compile/graphs/distribution. AdamW LR1e-5,
betas(.9,.95), epsilon1e-8, weight decay.1, clipping1, two-update linear warmup;
existing optimizer helper owns reductions and updates. No objective changes.

## Cases and data

The authoritative machine-readable matrix is
[configuration](../../../configs/olmo_f1_integration.json).

- All eight combinations: B2/T32, three changed-input updates.
- Four B2/T32 all-three variants: K3, fractional alpha.37/beta.35,
  RT layers(0,15), and alpha/beta sequence(0,0),(.37,.35),(1,1).
- Ordinary and all-three B2/T128: two updates each.
- Ordinary, RT, FBT and all-three B1/T512: one observed update, then
  three warmup, three timed and one separately profiled complete update.
- Ordinary and all-three T32: save after update2; rebuild/load and replay update3.

These are bounded real-text operational fixtures from the existing two prompt
strings, with deterministic token rotations between updates, explicit EOS,
right padding and independent CE/latent/KL masks. Each row is one document.
The second row has five fewer valid tokens. The middle diagnostic update
accumulates separate rows against whole-update denominators. This exercises
unequal counts; existing tiny FP32 tests establish equivalent accumulation.
No benchmark data, held-out quality estimate, corpus exposure or learning
efficiency claim is implied.

## Checks and stopping rules

Reuse the retained independent tiny mathematical tests. Add only focused
configuration/observer/accounting/reporting tests for the new harness.
For actual-checkpoint cases, verify:

- Finite scalar losses, expected gradient ownership, finite parameters/moments,
  nonzero updates to active tensors and exact invariance of inactive state.
- Native embedding/readout ownership and unique optimizer/parameter counts.
- Changing feature participation under scheduled alpha/beta.
- Full save/load state, counters, cursor, next fixture, next complete update
  and next CPU/CUDA RNG draw for two representative BF16 cases.
- Same-semantics online whole-versus-split cache behavior at B1/T8 on the
  updated all-three model, and rejection of an incompatible cache mode.

Attempt exact checkpoint replay first. If the loaded boundary is exact but
the future BF16 update differs, distinguish backend nondeterminism from a
checkpoint defect before choosing any new criterion. No silent tolerance
loosening or fresh broad precision campaign. Historical reference budgets
remain unchanged. A falling loss or architecture quality win is not required.
Stop on a failed case, retain the evidence and localize before continuing.
Record a new execution lineage after fixes that change runtime sources.

The two full resume checkpoints are disposable validation artifacts. Delete
them only after their respective exact replay passes. Retain their hashes and
comparison evidence; do not repeatedly archive synthetic diagnostic weights.
Source native weights remain at the previously verified GCS object.

## Early profile and reporting

Use no gradient-observer hooks, digests, W&B logging or checkpoint operations
inside timing. Include the canonical loss, backward, clipping, AdamW and
scheduler step with nonzero LR and changed inputs/weights. Fixture creation
is outside timing. Report wall and CUDA-event median, valid input tokens/s,
CE targets/s, target counts, pass count and allocated/reserved peak memory.
Optimizer state must exist before timing. Profile one additional complete step
outside timing and classify observed SDPA operators plus top exclusive CPU/device
operators. A cuDNN fused operator is not specifically Dao FlashAttention.

This is a small B1/T512 directional screen, not a capacity search or final
FLOP accounting. It does not compare equally optimized implementations.
Attribute costs cautiously: all-three changes passes, recurrence and auxiliary
training together. Main purpose is selecting the next engineering bottleneck.

Store config, source/protocol hashes, runtime, fixture/mode records, update
checks, checkpoint comparisons, parameter ledger and profile summaries in
per-case JSON plus an aggregate report. Record W&B online under taylorbollman.
Retain small source/report/test evidence at gs://fast-chunks with receipts.
Completed historical runtime/protocol files must remain unchanged.
