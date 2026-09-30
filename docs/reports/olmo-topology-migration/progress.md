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
