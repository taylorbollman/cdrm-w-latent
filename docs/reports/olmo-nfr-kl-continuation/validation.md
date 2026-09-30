# NFR KL paired-continuation validation

Both NFR branches completed updates **33–64**, published verified update-64
checkpoints and synchronized W&B. Their host launchers exited successfully.
The independent paired audit passed **16,483 checks with zero failures** after
both terminal reports were immutable. The comparison uses the original NFR
update-32 checkpoint, native RT at layers 0 and 15, K4, and the unchanged
128-update plan. The only intervention is KL weight 1 versus 0.1; latent weight
stays 1. It was activated after the separate FBT-only assessment.

## Preparation evidence

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

Before execution, an independent CPU metadata audit of the saved NFR authority passed
**685 checks**. It verifies the published update-32 manifest, original model
identity, populated Adam state digests for every active parameter, scheduler
epoch 32, rank cursors, unchanged finite plan and the live bytes of all 215
sources. Evidence is in
`.runtime/olmo-nfr-kl-continuation/independent-preflight-audit-01/report.json`.
This audit does not load model or optimizer tensors and does not run CUDA.

## Completed paired audit

The strict checkpoint loader verified complete model, Adam, scheduler, RNG and
cursor restoration before graph construction. The paired audit confirms that
both full origin states match the saved NFR32 boundary, data and LR sequences
are identical, first-update raw forward losses agree exactly, and KL weight is
the only declared objective change. It also checks the independently pinned
215-source scope, per-loss denominators, development-state preservation,
complete update histories and terminal checkpoint publication metadata.

The one-shot host CPU launcher required both closed update-64 reports, successful
host exits, synced W&B and verified cloud-64 receipts before invoking the
existing auditor. Neither the audit nor evidence retention loaded tensors or
performed GPU work. The evidence is in
`.runtime/olmo-nfr-kl-continuation/native-pair-audit-01/`; its exact command is
retained in `native-pair-audit-command-01.json`.

| Authority | SHA256 |
| --- | --- |
| Independent paired audit report | `2ee9b737c8269ef071fd63792f9d4ed2d87598b900347b571d7811e1aadcb211` |
| KL1 terminal training report | `4e47728173364a6a3a6ed14887517cea9df8bcd2247d1b5d8ca743eb2f3b27fe` |
| KL0.1 terminal training report | `9de466ea50ea837db1aaf6f0e88f23675c6a13af1f24a196fab992ea54a96e8c` |

The audit archive retains eighteen files (77.66 MiB uncompressed), including
exact input snapshots, audit sources, the command receipt and source inventory.
It is verified at
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/nfr-kl-pair-audit-01/`.
The compressed archive SHA256 is
`787b300aa944b7698b369d53d9266135ebae221938dd6e624d95d972b3968c94`.
Its archive, manifest and receipt passed server-size, MD5, SHA metadata and
downloaded-SHA verification. The local receipt is
`.runtime/olmo-nfr-kl-retention/pair-audit-01.json`. Branch archives and model
checkpoints are retained separately; this operation did not upload checkpoints
or delete local files.

Passing establishes recorded-state, scope and accounting consistency for this
paired continuation. It does not establish optimizer stability, useful
refinement, BF16 equivalence or a generalization advantage.
