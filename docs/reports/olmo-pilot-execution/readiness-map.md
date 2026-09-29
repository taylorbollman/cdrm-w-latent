# Readiness map for the first learning comparison

Use this map with the current milestone's results and
[next-steps.md](next-steps.md). Readiness means the selected implementation can
execute and recover under its tested conditions; it does not establish a useful
architecture improvement. Native capacity numbers belong in the results report
after each run finishes, not in this checklist as forecasts.

| Area | Current evidence or status | Minimum remaining for a short learning pilot |
| --- | --- | --- |
| Model and objective | Pinned original OLMo-1B; eight component configurations; native RT; K4 FBT and accepted NextLat regression/KL semantics | Choose the initial arms and startup explicitly; keep the accepted math |
| Precision | Qualified BF16 functionality; prior FP32 comparisons and optimizer-history diagnostics; trajectory divergence remains documented | Monitor actual health; no general numerical requalification unless a concrete new failure appears |
| Data | PR49's pinned Dolma selection, 134.2M-input ordered train capacity, disjoint dev/confirmation identities, source/coverage audit and cloud recovery | Freeze a smaller pilot exposure and the development prefix; retain limited books coverage and unknown original pretraining exposure as qualifications |
| Distributed execution | Tiny two-H100 ordered graph integration; evaluation insertion 2,063 audit checks, cloud resume 2,136, terminal evaluation-only 1,914; exact accepted comparisons | Resolve the actual logical batch; exercise its first updates as the pilot begins |
| Native T1024 capacity | Completed B32, B64 and NFR12 T1024 measurements, including final FP32 evaluation and cloud checkpoints; comfortable tested headroom | Pick comfortable physical batches from successful runs, independently of a common effective global batch |
| Evaluation | Named dev prefixes, explicit per-arm FP32 batch, all trained passes, separate loss denominators, preserved training state | Choose useful monitoring membership and cadence from coverage and measured total cost; confirmation remains separate |
| Checkpoints | New ordered identity, unchanged shared SSD engine, narrow storage metadata adapter, manifest-last exact-generation recovery, verified retention before pruning | Freeze cadence/storage budget; preserve full Adam/schedule/RNG/cursor on same-lineage resume |
| Throughput and cost | Update regions, setup, evaluation and checkpoint/transfer costs are separately observable; parameter ownership is retained | Price the proposed segment from actual measurements; do not confuse compute-region throughput with all-in training throughput |
| Scientific comparison | Common-recipe/token-matched plan is available; no matched learning cohort has been launched by this milestone | Choose original/fresh-Adam versus explicitly adapted ancestry, finite ceiling, first stop point, arms and review criteria |
| Hardware portability | Two H100 80GB GPUs under the recorded software/runtime | For H200 or a changed rank count, do a bounded local capacity and restart check; do not assume exact cross-topology resume |
| Later outcomes | Longer pretraining, cooldown, SFT, generated tasks and compute-matched extensions are proposed research stages | Implement only the next selected stage after short-pilot review |

The native B/NFR capacity runs are operational fixtures. Their effective batches
and startup routes can differ, so their learning values are not a matched
architecture comparison. None of the audit check counts above measures model
quality or grants general BF16/H200 clearance.

The frozen capacity sequence completed B32/rank, B64 and NFR B12/rank; B8
fallback was unnecessary. Eight-update fixtures measure updates 4–8. Their 600-second,
every-four-update checkpoint policy is a test setting; future cadence and the
provisional learning budget still require the review described in next-steps.

## Terms and comparison boundaries

**B** is ordinary OLMo continuation. **N** adds NextLat training losses and its
predictor, without learned latent-rollout inference. **F** adds the feedback
fusion pathway and repeated shared-weight FBT passes. **R** enables native
temporal RT in layers 0 and 15. Combined names mean the corresponding elements
are active together.

**Fusion** is the learned transformation that combines causally shifted upper
hidden states with token embeddings for a later FBT pass. **K** counts total
passes, including the initial pass: this campaign uses K4 for F-containing arms
and K1 otherwise. Historical K2/K3 experiments are different compute and training
settings, not interchangeable throughput references.

**Physical batch** is the number of rows processed together on each GPU.
**Effective batch** is the global valid-input exposure accumulated before one
optimizer update. Loss denominators are their own global eligible-target counts.
Four passes do not multiply data-token exposure by four, but their forward,
backward and recomputation work must be counted in cost.

**Common original startup** and **fusion128 startup** are separate ancestries.
The latter adds only the selected fusion weights and starts new all-active Adam;
its historical adaptation cost must remain visible. A recovered checkpoint from
either accepted lineage restores complete state, including Adam, rather than
starting a fresh optimizer again.

**Development monitoring** guides choices. **Confirmation** evaluates a frozen
choice later. Main and source panels share some chunks, so their results remain
separate views of data rather than independent replications. Frequent small
prefixes can monitor direction without supporting broad quality claims.

**RT screening** asks whether recurrence earns its incremental cost in the
tested recipe. The primary contrast is NFR versus NF, with R/B, NR/N and FR/F
retained as distinct questions. A failed early warmup pilot or one negative
contrast does not establish that RT is generally ineffective.
