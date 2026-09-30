# FBT-only optimization and feedback stability

Authorized 2026-09-30. The user approves approximately 6.5 hours of autonomous
work beginning around 07:07 UTC, with updates during the initial Figure 3-style
measurements. Continue to a useful evidence-driven milestone without waiting
for review. This protocol covers the FBT-only study; later component training
must have its own explicit recorded declaration before launch.

## Question and fixed treatment

Determine whether the current four-pass feedback recipe learns and stabilizes
without NextLat or temporal RT. Distinguish finite optimizer health, empirical
stability under repeated feedback, and preservation of useful language
prediction. No quality win or global contraction proof is required. A stable
poor predictor is not a successful recovery. Frequent clipping alone is not
an execution failure.

Use original OLMo-1B step60000/tokens252B and the same retained fusion128 weights
used at the NF/NFR cohort origin. Import weights only, then create fresh Adam
for all active backbone/fusion parameters. Do not fork trained NF64 weights or
import the warmup optimizer. Authenticate original NF startup authority and
explicitly construct the F-only wrapper without predictor ownership. Record
all prior fusion exposure separately from this new run.

Keep K4 (one ordinary pass plus three feedback passes), beta1, training jitter
0.02, CE coefficients (1/2,1/6,1/6,1/6), fully differentiated training feedback,
native RoPE/normalization, no sampled prefix mixin, no new gate ramp. No RT,
NextLat regression, KL, predictor rollout or Q/K changes. Use the accepted BF16
mixed path, FP32 masters/Adam, captured DDP, activation checkpointing and fused
AdamW. Numerical qualifications from earlier work remain.

T1024, two H100s, physical B12/rank and 524,288 real inputs/update. Use the same
ordered packed Dolma mixture, CE/document policy, keyed jitter and first128
logical update memberships as the completed cohort. Peak LR2e-4, unchanged
100-update 10%-to-100% warmup and constant plateau; preserve actual token-based
schedule. Declare192 updates as a ceiling, initially stop128. Confirm first128
LR/member prefix equality; extending the finite declaration is not resetting
the optimizer or warmup. Additional128→192 training is conditional on remaining
time and a useful adaptation question, not an automatic outcome.

## Observation and persistence

- Every update: raw CE, preclip norm/coefficient, LR, token and update counters,
  finite update health, memory and separately scoped time/throughput.
- Every8 updates: fixed small dev-main panel, noiseless common-FP32 K1..K8.
- At0/32/64/96/100/128, and192 if reached: same small panel K1..K32.
- Every16 and100: original64-row trained-pass development evaluation and
  complete resumable checkpoint. Also preserve initial/terminal states and
  the accepted600-second recovery trigger at completed boundaries.

Each pass curve reports CE in nats/target, hidden-state change RMS and
normalized change, cosine, hidden scale, prediction entropy, and position/tail
breakdowns. Track positions beyond the causally settled prefix separately.
Pass1 is the ordinary pass; pass number and feedback count differ by one.
Use fixed content-independent panel membership, masks and denominators.
Common FP32/no-jitter evaluation is distinct from BF16/jitter training.

Stream evaluation states and chunked readouts to bound memory. Preserve live
model/optimizer, existing gradients, RNG, data cursor, training flags and CUDA
graph state. Reuse accepted preservation machinery; verify streaming outputs
against the ordinary finite-pass model before native use. Scope small-model
exact continuation separately from native operational observations.

Save immutable states to owned SSD directories; one CPU worker uploads and
verifies them asynchronously. Keep two verified local states, retain published
cloud states in gs://fast-chunks. Named checkpoints are additional to wall-time
saves: wall-time saves alone do not promise a particular optimizer boundary.
Cloud rollback may reach the preceding publication if VM/SSD disappear while
upload is pending. Save code/reports and retain evidence every20–30minutes.

## Follow-up sequence and decision points

The ordinary B control can resume its exact shared update32 state through128
using the unchanged original128 declaration/runtime while new F diagnostics
are implemented. This overlaps GPU work with CPU preparation and provides
matched input/LR milestones. It is not a new B initialization or precision test.

After the initial F study, summarize findings and proceed without a user-review
pause to the next useful scoped task. Candidates: same-recipe extension toward
192; bounded exact-sequential versus finite-pass hidden/logit/CE comparison;
identical curves on saved NF/NFR checkpoints; or the earlier proposed paired
NFR KL1 versus0.1 continuation. Choose using observed behavior and remaining
time, record the choice before execution, and do not silently change the F
training recipe to fill the time window.

Exact-online currently supports isolated-document rows. For this check use a
separately declared one-document panel and compare finite and sequential
execution on the identical prefixes. Do not imply those losses are comparable
to unmatched packed dev-main rows or label teacher forcing free generation.

If F itself remains poor, inspect first/later CE and their gradients before
changing pass count, strength, prefix sampling, loss weights or optimizer. No
automatic gate ramp or architecture sweep is authorized by this protocol.
Stop a failed execution safely on nonfinite values, source/state mismatch,
publication failure or user request; retain its evidence.

## Source and reporting contract

Historical200-source runtime/model files remain unchanged. New versioned
executor, importer, controller, diagnostics and auditor carry explicit source
pins and fresh output paths. Use focused CPU tests and bounded two-GPU
acceptance for new seams; reuse prior kernel/restart evidence where unchanged.
Log online to taylorbollman/pretrained-fbt-rt-nextlat. Reports distinguish
measured checkpoint curves from unobserved transitions, and empirical shrinking
state changes from a proof of contraction or universal numerical clearance.

Reference: https://arxiv.org/html/2608.08888#S3.SS3. Figure3 compares trained
recipes, not successive training checkpoints showing a binary transition.
