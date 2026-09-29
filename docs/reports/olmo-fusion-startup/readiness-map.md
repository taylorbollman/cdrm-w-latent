# What the overnight work does and does not make ready

2026-09-29, active work. This separates operational readiness from the open
precision decision. A successful small operational test does not establish
model quality, production throughput or numerical equivalence.

| Area | Current evidence | Remaining scope |
|---|---|---|
| Native model and component ownership | Existing pretrained checkpoint fidelity; explicit backbone/fusion/predictor counts for all eight combinations | No architecture or Q/K-normalization change in this work |
| Prepared corpus recovery | All 86 prepared files retained; earlier complete cloud restore and current checksum audit | This is a readiness corpus, not the final training mixture |
| Packed semantics | Saved full-NFR T1024 FP32 semantics pass; BF16 prepared and captured losses/gradients are exact | Small BF16 sparse/prepared elementwise compatibility miss remains documented |
| Fusion startup | FP32 fusion-only warmup sharply reduces several matched-state BF16 gradient discrepancies | Does not remove temporal-RT sensitivity or establish a universal warmup duration |
| Short full-model optimization | Four paired updates plus 16 conditional updates per precision complete; endpoint held-out losses closely agree | Backbone cumulative updates still differ 16.27%; shared BF16 origin, four dev documents and short isolated T128 scope |
| Resource accounting | Parameters and analytical FLOPs recorded for every component combination | Diagnostic copying, hashing and checkpoint I/O are not throughput measurements |
| Common lifecycle loop | Tiny two-GPU captured-DDP stop, retention, fresh resume and coordinated logging-failure recovery pass; pretrained ordinary reference and cloud restore pass | Fresh ordinary continuation updates match exactly; final retention/exit audit still pending |
| Abrupt worker loss | Deliberate rank exit after retained update1; fresh process exactly reproduces reference updates2/3 | No rollback of an in-progress Adam step, actual VM power loss or every NCCL failure is claimed |
| Evaluation inside a live graph-training process | Actual tiny two-GPU reference/insertion pair reproduces all following updates exactly | Tiny-model final-pass CE scope, not a production evaluation protocol |
| Configuration to execution | All-eight CPU manifest resolution and 51-check independent audit pass | A narrow B-only manifest execution adapter is in development; all-arm/adapted startup remains later work |
| Multi-GPU and hardware | Existing two-H100 packed/checkpoint evidence plus new tiny lifecycle checks | H200 capacity, topology and actual production batch still need hardware-specific qualification |

The next decision is whether a modest BF16 pilot is justified after bounded
full-model optimization and packed execution checks. It is not whether every
gradient must match FP32 exactly. We need to retain meaningful discrepancies,
examine actual optimizer effects and losses, and ensure execution machinery is
not changing the intended objective. A qualification that remains unresolved
should be carried into the next protocol rather than silently converted to a pass.

Production campaign setup still needs a selected training mixture, fixed token
and evaluation budgets, final batch/accumulation choices on the target hardware,
and a recoverable launch package. Those choices should follow the numerical and
resource evidence; the readiness fixture and tiny diagnostic update sizes are
not defaults for that campaign.
