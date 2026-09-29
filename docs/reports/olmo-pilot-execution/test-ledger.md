# Test and acceptance ledger

The frozen runtime at `be74dde` passed **149 distinct new CPU tests in58.75s**
under the GPU-disabled project container. The complete log is
`.runtime/olmo-pilot-execution/cpu-final-01.log`; CPU evidence and the192source
snapshot are retained in `pilot-execution-cpu`. Earlier scoped counts overlap
this collection and must not be added to149.

| Scope | Distinct tests | Checks |
| --- | ---: | --- |
| Execution contract | 31 | Ordered data/source/recipe authority; finite plans; startup exposure; exact same-lineage resume metadata. |
| Named evaluation | 36 | Dev-only named membership; independent physical batches; per-pass reductions; complete CPU state preservation and failure paths. |
| Executor | 25 | All-eight component ownership; pre-CUDA rejection; ordered identity; shared native constructor; no precision/topology fallback. |
| Synthetic ordered fixture | 5 | Actual pinned native tokenizer and packed indexes; simulated upstream metadata labeled; document boundaries and uneven rank slots. |
| Original independent auditor | 29 | New schemas, exact training comparison, evaluation reductions, published checkpoint and storage evidence; mutation rejection. |
| Restore adapter | 17 | New identity accepted/old rejected; exact generations and byte bounds; manifest-last publication; CPU-only import. |
| Storage adapter | 6 | Inherited lifecycle methods; exact owned publication/pruning; corrupted/unretained/foreign/old-identity rejection. |

Early records remain unmodified. The first executor scope had9failures/16passes
because its test-only cloud prefix was outside the unchanged retainer's allowed
namespace; correcting that fixture produced25passes. One contract test incorrectly
indexed the serialized default document policy; parsing through CampaignRecipe
fixed its assertion (final31pass). The first synthetic fixture had insufficient
dev reserves; the second hit a report-only rank-batch length bug. The accepted
third fixture and final tests retain the data-capacity rejection rules.

## Actual GPU acceptance

Tiny reference and evaluation insertion completed on two H10080GB GPUs with
CUDA graphs. The original independent insertion audit passes2063checks, including
192sourcepairs, exact rawgradient/update/final-state comparisons on bothranks and
separate four-pass results for dev-main/books at update2. Cloud checkpoint2
recovery downloaded20,150,587bytes in1.92s and the fresh process completedupdate3.
A terminal restored checkpoint2 evaluates without a newupdate or graphcapture.

The original resume auditor compared `preservation.elapsed_seconds` exactly;
this produced a retained false failure even though meaningful preservation
fields, update3 gradients/metrics and finalstate are identical. The zero-update
terminal path also required an empty-updates case in the comparison auditor.
A separate v2 auditor was added at `2d9fb70`, with27additional distinct CPU
tests passing in0.65s. It excludes only preservation wall time from cross-process
equality (still validates that it is finite/nonnegative), and handles a validated
zero-update terminal segment explicitly. All meaningful preservation fields
remain exact. The192runtimepins and GPU evidence are unchanged. Both v1 failures
remain retained alongside the final audits.

The v2 actual resume audit passed2,136checks; terminal evaluation passed1,914.
Update3 inputs, gradients, metrics and full state match uninterrupted execution.
Terminal evaluation repeats at checkpoint2 with no optimizer update, graph
preparation or new checkpoint publication.

Native B32, B64 and NFR12 each completed eight updates, final named FP32
evaluation, preparation/evaluation preservation checks and verified checkpoints
at updates 0, 4 and 8. The final independent summary revalidated all runtime
source pins, declarations, allocations, counts and ownership for the three runs.
All 24 updates were finite. NFR still clipped heavily and its later dev passes
were worse than pass one; execution acceptance is not learning success.

Timing uses only updates 4–8 of each native run. NFR8 fallback was unnecessary.
These are operational capacity fixtures with different logical batches and
startup histories, not a matched cohort or cross-precision trajectory clearance.
Native restart was not repeated under these new physical layouts; exact ordered
integration restart is scoped to the tiny fixture above. Prior native recovery
evidence remains separately scoped to its original layouts.
