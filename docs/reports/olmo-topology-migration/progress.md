# Topology readiness progress

2026-09-30: branch feat/olmo-topology-migration from f93e673. User authorized
remaining two-GPU-suited steps. Read plan.md and the detailed implementation map
in ../olmo-gpu-allocation/topology-migration-plan.md.

Two H10080GB verified idle inside the required project container. Boot has
about87GiB free; SSD about985GiB free. Native checkpoints/gradient artifacts
will use SSD; small evidence/code live in the persistent project. No native
learning extension is authorized or scheduled.

Parent owns scripts/olmo_topology_execute.py and GPU scheduling. Parallel work:
allocation_runner owns new contract/checkpoint wrappers and tests;
allocation_acceptance owns independent numeric/restart auditor and tests;
memory_evidence owns independent-job isolation helper and gap assessment.
Historical runtime and checkpoint source inventories stay unchanged.

## 2026-09-30 22:16 UTC — pre-GPU implementation review

- Native NFR127 authenticated in full (manifest and15.215GB state); all422
  historical source entries remain unchanged. Both H100s were idle at preflight.
- Explicit migration/import and independent numerical audit are implemented.
  Destination checkpoints receive a fresh allocation identity compatible with
  the existing asynchronous SSD-to-GCS publication path.
- Review added TF32-off controls, source/precision checks, historical counter
  validation and cancellation-safe retention cleanup before any GPU execution.
- Tiny execution helper CPU tests pass; independent-job isolation review found
  no ownership/cleanup blocker and is adding explicit post-failure displacement
  evidence. Native N/R/NR/FR smoke helper is being prepared independently.
- No new GPU run or scientific continuation has started. Next: freeze sources,
  tiny CUDA/NCCL migrations, native127-to128 replay/restart, isolation and gaps.

## 2026-09-30 22:21 UTC — tiny GPU acceptance and native import

- Four required tiny CUDA/NCCL audits passed:2→1 and1→2 FP32 raw-gradient
  relative L2 error4.835e-8; actual Adam-displacement relative error8.198e-7.
  Fresh-process imported-boundary and live-graph-boundary restarts were bitwise
  exact. A redundant cross-history comparison correctly rejected distinct
  intermediate checkpoint manifest identities despite exact tensor/RNG state;
  this is retained explicitly rather than weakening the strict auditor.
- Cooperative and deliberate abrupt-exit independent-job isolation both passed.
  Peer graphs survived, four additional real updates changed parameters, and
  peer checkpoint bytes and fresh CPU model/Adam restore were verified.
- Native two-rank127→128 control is running from authenticated original127,
  preserving the actual finite schedule and populated optimizer. No update129
  is authorized or supported by this fixture. Later native migration and strict
  restart remain pending. Retention of the completed tiny evidence is underway.

## 2026-09-30 22:27 UTC — native publication integration issue

Native control attempt01 authenticated/imported/saved127 but its asynchronous
CPU worker rejected the identity schema: SSD staging uses campaign identity,
whereas the reused pilot worker hardcodes pilot identity. This was an omitted
end-to-end storage compatibility case, not a numerical failure. Parent stopped
only its authenticated owned container during graph preparation; original127
and128 remain untouched, imported127 remains local, and no publication is
claimed. Logs plus operator-stop receipt are preserved. The fix uses the
existing retention manager's explicit worker hook with a new campaign-identity
CPU worker; historical validators, files and checkpoint formats stay frozen.
Before retrying native, exercise actual tiny asynchronous cloud publication.
