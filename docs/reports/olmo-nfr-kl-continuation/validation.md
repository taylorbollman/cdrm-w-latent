# Conditional NFR KL validation

This comparison remains **prepared, not launched**. It is conditional on the
FBT-only assessment. The new authorizer admits only the original NFR update-32
checkpoint, native RT at layers 0 and 15, K4, and continuation through update
64 under the unchanged 128-update plan. The only intervention is KL weight
1 versus 0.1; latent weight stays 1.

Independent review confirms that the launcher directly calls the accepted KL
stage and engine. Its scope declaration is included in the 215-source execution
identity. It does not change the model, optimizer, attention, evaluator or
training callbacks. The existing 200/210-source lineages and separate F-only
208-source runtime remain unchanged.

The separate JSON-only `scripts/olmo_nfr_kl_audit.py` retains the accepted
training, evaluation, exact-parent-state and paired-comparison checks. The
copied branch validator changes only the native arm restriction from NF to NFR;
an AST test verifies that narrow difference. Additional checks bind the actual
scope, original checkpoint/report pins, selected RT layers, unchanged pass count,
latent weight and 128-update plan to an independently pinned 215-source map.
There is no report relabeling or mutation of the historical auditor's globals.

**22 independent audit tests pass** in the CPU container. They reject changes
to source identity, parent checkpoint, declared scope, RT selection/strength,
pass count, latent weight, plan length and activation conditions. The separate
authorizer's 26 tests cover its argument/parent checks and direct engine reuse.

An independent CPU metadata audit of the actual saved NFR authority passed
**685 checks**. It verifies the published update-32 manifest, original model
identity, populated Adam state digests for every active parameter, scheduler
epoch 32, rank cursors, unchanged finite plan and the live bytes of all 215
sources. Evidence is in
`.runtime/olmo-nfr-kl-continuation/independent-preflight-audit-01/report.json`.
This audit does not load model or optimizer tensors and does not run CUDA.

If activated, the accepted strict checkpoint loader must verify complete model,
Adam, scheduler, RNG and cursor restoration before graph construction. The
paired report audit will then require both arms' full origin state to match the
saved NFR32 boundary, identical data and LR sequences, identical first-update
raw forward losses, and only the declared KL coefficient change. Both terminal
checkpoints must have verified publication receipts. Successful preparation is
not evidence of optimizer stability, useful refinement or BF16 equivalence.
