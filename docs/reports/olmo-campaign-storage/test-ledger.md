# Test ledger

## CPU

215 distinct tests passed; these are the union of focused scopes, not the sum
of repeated invocations. All torch-related testing used the GPU-disabled project
container. No host CUDA or silent CPU fallback was used.

| Scope | Distinct tests |
| --- | ---: |
| SSD storage ownership/publication/pruning | 34 |
| New executor identity/path/retention contract | 15 |
| SSD streaming restore and failure ordering | 8 |
| Independent SSD transition/resume/storage auditor | 45 |
| Existing streaming restore regression | 41 |
| Existing execution/evaluation auditors | 72 |
| Total | 215 |

Final storage/executor/restore scope:57passed in5.40s. Restore/executor plus
historical restore scope:64passed in5.34s. Final new/prior auditor scope:117passed
in1.06s. Two external Google package future warnings appeared in the restore
scope. They did not change results. An early combined invocation selected two
agent test files before they existed and collected no tests; its log is retained
separately and it contributes zero to the pass count.

Copied logs and counts are under
`.runtime/olmo-campaign-storage/cpu-acceptance-01/`. Runtime source authority is
`54ae688` (172pins). The separately reviewed auditor source SHA256 is
`f78841bc71551feeafa8bd55c9611e1ca77be88f79f4004fa3df84d976e3ca3b`;
its test source SHA256 is
`e20f7e053e218b06310e1add921276cdf445b36a74338d8cd7449b0e9a7ae60c`.

## GPU and recovery

| Stage | Wall seconds | Audit |
| --- | ---: | --- |
| tiny-reference-01 | 20.050 | Transition:2,121checks;336source snapshots |
| tiny-stop-01 | 18.223 | Verified update2 cloud publication |
| tiny-restored-01 | 1.935 | 20,132,121bytes; CPU streaming recovery |
| tiny-resume-01 | 14.466 | Resume:1,649checks;344source snapshots |
| tiny-terminal-01 | 3.768 | Terminal:1,425checks;344source snapshots |
| native-assets-01 | 70.631 | 15,215,060,305bytes; CPU streaming recovery only |

The JSON auditor separately verifies source/schema/storage differences allowed
for the PR47→SSD comparison, exact complete identity for same-lineage resumes,
exact resume-source publication pin, per-pass evaluation, target accounting,
full committed state and the keep-two journal/receipt/prune sequence. Audit scope
is explicitly tiny acceptance. It does not pretend that metadata alone verifies
a fresh cloud readback or physical deletion; the producer/restore checks and
actual local file inventory supply those separate observations.

All four GPU runs were inside the required container, with two H10080GBs and
bounded600-second external timeouts. Recovery used the CPU-only container and
bounded1,200-second timeouts. Final container `nvidia-smi` shows both GPUs at
0MiB and0% utilization. No native training update was run in this milestone.
