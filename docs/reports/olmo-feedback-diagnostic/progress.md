# Saved-state feedback diagnostic progress

2026-09-30: user approved the bounded next-step proposal from PR52. Branch
`feat/olmo-feedback-diagnostic`, base `19af4d0`. No learning extension is authorized
by this milestone. Existing 200 runtime source pins and checkpoint bytes stay
unchanged. New helpers observe the existing objectives at FP32/no jitter.

Fixture prepared: `.runtime/olmo-feedback-diagnostic/fixture-01/report.json`, SHA
`b83df92927f56c0528bef44ff17a5928e1579d3834cb7e98d5c09f33afb99c7f`.
Eight dev rows and two separately declared two-row unused training batches;
all T1024. No training cursor advances.

New gradient helper's six CPU tests and fixture's thirteen CPU tests pass.
Forward observer, weights-only diagnostic importer and focused checks are being
completed. Both H100s were available at preflight; no GPU diagnostic launched yet.

03:21 UTC update: the above initial preflight is superseded. NF origin forward02,
NF endpoint forward01, and NF endpoint primary-gradient01 are complete with exact
state preservation. NF conditional-gradient01 and NFR endpoint-forward01 are
running, one GPU each. NFR primary-gradient confirmation is next; no training.
Code commits `710d5d2` / `b623419` are pushed. First GPU attempt stopped before
CUDA because newly added observational modules changed dynamic source discovery;
the retry explicitly authenticates the pinned accepted plan and all 200 unchanged
runtime files. Failed evidence is retained without overwriting it.

NF beta0 controls pass exactly; beta0.5 does not repair later-pass CE. Starting
later-pass deficit already exists. Endpoint KL falls while readout distributions
broaden; this is not evidence of useful refinement or a collapse diagnosis.
NF primary-gradient reconstruction relative L2 is 1.61e-6. Backbone first-CE
versus KL cosine is -0.369, versus latent -0.665; combined CE/auxiliary cosine is
only -0.058, and fusion CE/auxiliary alignment is +0.870. This selective conflict
triggered the predeclared second-batch and NFR confirmation, not a wider grid.

NF origin evidence and fixture metadata/payload are already GCS retained under
`olmo-two-gpu/20260930T031600Z/feedback-*`; receipts live outside active writers in
`.runtime/olmo-feedback-retention`. Remaining artifacts will be retained on completion.

Older SSD restore rejected the async checkpoint identity before downloading any
state. Its failed attempt is retained. The matching async restore path is being
used for NF origin0; NF32 and NFR32 are already on SSD and cloud-verified.

After interruption, inspect this directory and `.runtime/olmo-feedback-diagnostic`
before launching anything. Use the project container for GPU work. Save each case
and retain evidence every 20–30 minutes. Do not resume or modify PR52 training.
