# Conditional NFR KL continuation

Prepared 2026-09-30 as a possible next milestone after assessing the clean
FBT-only study. Preparation is not a GPU launch. The root agent may activate
this predeclared comparison if the evidence and remaining authorized overnight
window support it; no automatic dependency launches training.

Both branches restore the exact same original NFR update32 checkpoint from the
first adaptation cohort. Preserve all model tensors, populated Adam moments,
scheduler, counters, ordered-data cursors and rank RNG. Native RT remains at
layers0/15; FBT remains four total passes, full fusion beta1 and jitter0.02;
NextLat latent weight remains1. Only KL weight differs: control1, reduced0.1.
Both stop at64, with the original128-update token/LR plan unchanged. Context is
1024, two H100 ranks, physical batch12/rank, effective524288 inputtokens/update.
The added exposure is16,777,216 inputtokens per branch. All previous mixed-
precision qualifications remain open.

The new `olmo_nfr_kl_execute.py` calls the existing
`olmo_kl_continuation.run_stage` and its engine directly. No engine callbacks,
model mathematics, kernels, optimizer or evaluator are copied or modified.
The old entrypoint restricted native scope to NF; the new narrow authorizer
admits only NFR32→64 and authenticates an independently pinned scope declaration.
Its exact bytes join the child checkpoint source identity. Reports retain the
shared engine schema and add a separately versioned `nfr_scope` field.

The original200/210 and the FBT-only208 pinned sources are immutable. Each arm
has a distinct output/SSD/GCS namespace. Parent files must remain available even
for child resume, as in the accepted NF comparison. Checkpoint time cadence,
immutable SSD save, background full cloud verification and terminal drain are
reused unchanged. No quality improvement, harmless divergence or BF16 clearance
is implied by finite updates alone.

Native prior NFR took approximately114minutes for its first32updates including
startup/checkpoints. Budget roughly3–4hours for the pair. Review raw per-pass CE,
raw auxiliary losses and gradient/clipping scales separately; objective values
with different KL coefficients are not directly comparable. A lower gradient
norm or closer later-pass losses alone does not establish useful refinement.
