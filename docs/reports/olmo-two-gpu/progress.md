# Two-H100 completed milestone and interruption handoff

Updated 2026-09-25. All planned stages completed; no GPU job or quality-training
campaign is queued. Work is on `feat/olmo-two-gpu`, base `25bc29d` (PR29).
The measured runtime's latest code change is `bfa3649`; later commits document
results. Read [results](results.md), [summary](summary.md), [usage](usage.md),
[test ledger](test-ledger.md) and [storage receipt](storage-receipt.md).

## Verified position

- Two H10080GB, NV18 peer link; real NCCL/DDP. PyTorch
  2.13.0a0+8145d630e8.nv26.06, CUDA13.3, NCCL2.30.5.
- Tiny eight-mode eager, actual ordinary/RT eager, fixed-state combined updates,
  actual RT/combined graphs and selected recovery checks pass. CUDA graphs
  capture real DDP backward/NCCL reductions after11 warmup backwards.
- Actual RT eager recovery passes24checks; combined graph reconstruction
  passes42. ZeRO-1 full-Adam comparison and recovery pass13checks/rank.
  These rebuild model/optimizer/DDP within the same process group, not a fresh
  process launch. Both graph recovery branches recapture.
- ZeRO-1 keeps FP32 weights/gradients/moments. Scoped loader fixes native fused
  step placement/duplicated outer state; buffer-view consolidation preserves
  native bytes and mapping. Actual corrected checkpoint save83.4seconds, exact
  next-update recovery; cloud upload excluded.
- All31 completed attempts retained:26passed,5failed,0OOM,0running. All4,492
  report-pinned source pairs rehash;31verified stage receipts,8checkpoint stages.
  Two setup-failure stubs lack full source pins and remain explicit in audit.
- Matched globalB128: RT28.0k/s single versus47.4k/s twoGPUs B64/rank(1.69x);
  combined12.36k versus23.41k(1.89x). Five timed complete updates per row.

Actual configuration: original OLMo-1B step60000(~252Btokens),16layers/D2048,
RT at0/15. Combined=K2 FBT ordinary bootstrap then feedback RT pass+NextLat.
BF16mixed/FP32persistentstate, native ordinary RoPE, rounded compiled SwiGLU,
fusedAdam, ordinary checkpointing, FlashSDPA; native RT Triton/recompute backend.
No Q/K-normalization change, FA4 adoption, bucket-view adoption or all-layer RT.

## Practical operating points

| Purpose | Mode/backend | Local/global batch, T512 | Aggregate tokens/s | Minimum sampled free GiB/GPU |
| --- | --- | ---: | ---: | ---: |
| Room for changes | RT DDP |128/256|55,768|24.8|
| Room for changes | Combined DDP |64/128|23,406|30.5|
| Fixed larger batch | RT ZeRO-1 |192/384|58,187|14.5|
| Fixed larger batch | Combined ZeRO-1 |128/256|24,398|7.5|

DDPRT192 also passes59,035/s with10.5GiBfree. DDPcombined128 passes24,657/s
but only3.4GiBfree; avoid that as default for additions. ZeRO-1 adds headroom
at roughly1–1.5% measured throughput cost. No runtime default automatically
changed. GlobalB512/localB256 was not established; accumulation does not enlarge
physical RT matrices. ZeRO-2 and boundary-maximization remain conditional.

## Preserve qualifications

`actual-eager-01` remains failed: combined independent update2 passes loss,
raw-gradient budgets(globalrelativeL2=1.3509761003e-5) and exact rank replicas,
but20parameter/7moment tensors miss strict elementwise complete-update budgets.
Maximum parameter absolute difference1.383945346e-6. Fixed-state anchored
comparisons pass unchanged budgets; independent BF16 trajectory equivalence
and older native/author BF16 compatibility remain unresolved.

Four other retained failures are corrected setup/restoration contracts:
empty DDP forward args, fixed-buffer version changes, missing recorder argument,
and missing NCCL async-error setting. The buffer-restoration failure executed
four optimizer updates/rank; others zero. No numerical tolerance was relaxed.

## Persistence and resumption

Evidence prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T172000Z/`.
Local stage receipts: `.runtime/olmo-two-gpu/retention/`. Full checkpoints are
13.2–14.2GiB; generated local duplicates were removed only after remote
verification and local rehash. Keep manifests/cleanup receipts. Original
pretrained weights use O1 retention. Persistent disk has approximately19GiB
free; keep at most one newly generated full checkpoint before upload/verification.
Local SSD is disposable. GCS uses `env -u GOOGLE_APPLICATION_CREDENTIALS` in
CPU container to select mounted ADC; inherited custom ADC path is stale.

GPU commands must use project Docker and verify `/.dockerenv`, working directory
and `nvidia-smi`. Graph launchers require both NCCL async-error variables set0
and an external timeout. See usage for commands. CPU checks explicitly disable
GPU passthrough. Never print environment-file values or credentials.

Audit command: `python3 .runtime/olmo-two-gpu/audit/inventory.py`. It writes only
derived audit files and reads retained receipts; it does not repeat remote
verification or load checkpoints. The closeout archive preserves the audit and
CPU logs. Final CPU scopes77 and54 pass; they overlap earlier91/129 scopes.

This is the planned review point. Next select a workload/physical batch, then
rehearse an actual fresh-process checkpoint restart with its data cursor before
a long campaign. Dynamic graph layouts, other layer selections/context lengths
and quality experiments need their own scoped checks. Do not restart completed
probes or broaden precision/kernel work without a concrete reason.
