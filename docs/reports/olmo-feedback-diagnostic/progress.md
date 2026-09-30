# Saved-state feedback diagnostic progress

**COMPLETE, 2026-09-30.** All six native probes pass, all seven W&B runs including
summary are synced, and both H100s are idle. No optimizer update or continuation
is active/queued. Read results.md and next-steps.md; the proposed paired KL-weight
continuation remains a review proposal only.

Summary SHA256 `6508fd8c5d7276482c0fca8a12b6fcb1dd1e53fc1c6d0c4f3be4f075bafb815a`;
W&B `p5xs1bod`. Six case wall times are64.75/79.52/181.34 seconds for
NF0/NF32/NFR32 forwards and90.41/88.53/181.01 seconds for NF primary/NF
conditional/NFR primary gradients; include load/hash/logging overhead, not
training throughput measurements. Model/checkpoint bytes remain unchanged.

Final55 CPU tests pass. Independent packaging audit confirms unchanged helper
function bodies after relocation to scripts, exact original137/200 discovered
source inventories, and successful fresh original training `load_spec` for
both NF/NFR with no CUDA, model, optimizer, process group or RNG changes.
Audit SHA256 `f53a583fffe0bcfd40e205c2b43a794580d98185942677b7c00bd4f5aec276da`;
its cloud receipt is `packaging-retained-01.json` under the retention directory.
Source snapshots in completed GPU reports preserve the original execution code;
no historical report was rewritten to claim execution of relocated paths.

All six probe archives, summary, fixture metadata/payload, failed preflight and
packaging audit are cloud verified. Final code/docs/test/receipt closeout uses
`feedback-closeout` under the same timestamp prefix; PR/merge identifiers are
recorded below.

[PR53](https://github.com/taylorbollman/cdrm-w-latent/pull/53) merged as
`fcc5a29adbdc0d11c5380d3c132d61225b874e47`; reviewed head
`c79282d05877d3a9b00dcc27b7999ff9dd9cd275`. No repository CI checks were configured;
the explicit CPU/GPU and independent compatibility evidence above supplies the
validation record. Working branch returned to main.
Closeout receipt `.runtime/olmo-feedback-retention/closeout-01.json` has SHA256
`63d5cb9b765972437df9923c462d3850de4ea8d7e7ed68baa31ec152edd56a05`.
The38-member code/docs/test/receipt archive is fully download verified. This
post-merge metadata and that receipt are separately retained under
`feedback-admin` at the same cloud timestamp prefix.

## Earlier progress (superseded by completion above)

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
