# Saved NFR state: packed sparse, prepared and captured backward bridge

2026-09-29. This closes an execution-path gap left by canonical sparse numerical
probes and short optimizer tests. It is not another precision sweep or training
extension. No existing model, runtime helper, loader or historical protocol changes.

Use exactly the completed BF16 trajectory's update4 from the four-update NFR
diagnostic. Require independent checkpoint/report SHA256 pins and the original
report's successful eight-update contract. Reconstruct its original isolated
NFR model, fresh campaign optimizer and token scheduler solely to invoke the
unchanged strict full-checkpoint loader. Match its complete saved boundary, then
release optimizer/moment storage before measurement. There are **zero optimizer
steps** and no resumed data cursor advancement.
Require matching PyTorch/CUDA, GPU type/memory/driver and deterministic controls;
only the number of identical visible devices may differ for this one-process test.

Only after import, explicitly change document policy from isolated-v1 to
continuous-stream-v1. Preserve all tensor values, parameter identities, tying,
trainability and modes. Keep RT layers0/15, alpha1 on all four FBT passes,
beta1, jitter0.02 and combined CE + latent + KL with original weights and
stop-gradients. No full-precision promotion, normalization change or kernel edit.

Use the exact packed fixture SHA256
`4932410f9fd370d9dae20a1075bf2a191c642e1563eb8557a6fe02fd7e83a975`:
two B1/T1024 records, 2,048 inputs, CE2,046/latent2,040/KL2,032. The frozen fixture
loader accepts NF only. Supply it an explicitly recorded NF **data-construction
view** of the otherwise identical recipe, retaining its strict original schema
and noise checks. The actual model and execution recipe remain NFR. Verify
the view changes only arm; keyed noise does not depend on the RT selection.
Do not weaken the old loader or silently run an NF measurement.

Exactly five measured aggregate configurations (two physical records each):

1. FP32/math/eager RT, canonical sparse combined backward.
2. FP32/math/eager RT, prepared eager backward.
3. Production BF16/Flash/native Triton, canonical sparse combined backward.
4. The same BF16 path, prepared eager backward.
5. The same BF16 prepared runner, after CUDA-graph capture, two refilled replays.

Reuse canonical/component backward and the existing CampaignObjective and
CampaignGraphTraining implementation. Prepared eager execution has one gradient
initialization backward per runner, explicitly discarded. Reuse the BF16 runner
for capture, retaining gradient storage: ten warmup backwards plus one capture
backward, all discarded before replay. Thus the expected total is **10 measured
physical backwards +12 initialization/warmup +1 capture =23**. Record actual
counters and do not count warmup as training or measured data exposure. No
additional repetition or precision condition is authorized.

Maintain forced SDPA scope through backward, checkpoint recomputation, capture
and replay. BF16 autocast is forward-only with cache disabled. Set deterministic
controls before CUDA and TF32 off; use FP32 master parameters throughout.
Before capture, save and verify persistent input/gradient/parameter pointers;
both packed records must refill the owned tensors with their own token/mask/noise
values and global denominators. Clear all setup gradients and keep state/RNG
unchanged. Run graph last, then release graph owners before clearing storage.

Report raw loss sums by term, normalized combined objective, target counts,
complete gradient hashes and per-group relative/absolute error/cosine. Primary
comparisons are FP32 sparse/prepared, BF16 sparse/prepared, and BF16 prepared
eager/captured. Also report BF16/FP32 within each eager layout from the already
measured references, with no extra backward. Preserve existing legacy comparison
budgets as explicitly labeled diagnostics; do not invent or relax thresholds.
Finite losses/gradients, full parameter participation, exact inputs/counts/state,
RNG and pointer controls are operational gates. FP32 sparse/prepared and BF16
prepared/captured must additionally satisfy the existing semantic comparison
budgets before declaring this execution bridge complete. BF16 sparse/prepared
remains an explicitly separate numerical qualification if its legacy budget
still fails; passing other controls cannot erase that difference.

Root alone launches one GPU process after source/test freeze, with an external
timeout of **at most3,600seconds**. The helper checks a3,300-second soft budget
between configurations and estimates capture cost from the measured BF16
prepared run before warming up. If remaining time or GPU memory is infeasible,
retain partial evidence and stop; do not reduce context, alter the fixture or
launch broad variants. Reports and source snapshots are persisted incrementally
and logged to W&B. No new checkpoint is written; root retains the small evidence
bundle in GCS. Expected sub-hour stages require no unrecoverable long training.

CPU tests cover exact tiny sparse/prepared gradients and loss semantics, setup
accounting/refill, policy-only transition and data view, immutable endpoint
authority, loader clocks and failed pre-mutation guards. CPU tests cannot
substitute for actual CUDA capture. This one-GPU bridge does not clear DDP/NCCL,
large physical batches, H200, arbitrary documents, long-run BF16 optimization
or model quality. A packed result is not a retrospective rerun of earlier cold
3.40224%/1.6953% sparse/prepared observations; state, objective and fixture scope
are retained explicitly.
