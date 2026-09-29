# CPU campaign manifest resolution draft

2026-09-29. This is a bounded preparation step for a later common training
entrypoint. It adds a strict JSON manifest resolver, focused CPU tests and a
readiness example. It constructs no model, optimizer, scheduler or model input
tensor, and performs no forward/backward, CUDA execution, training, download,
cloud publication or W&B run. It does hash local pinned model and corpus bytes;
the original artifact validator also checks tokenizer fixtures offline.

Successful status is **`cpu_plan_validated_not_authorized`**, with
`launch_authorized=false` and `numerical_clearance=false`. Structural coherence
of a BF16 NFR declaration does not clear the retained gradient/trajectory
qualifications. A training launcher is not part of this implementation.

The manifest explicitly selects one or more of B/N/F/R/NF/NR/FR/NFR and includes
all CampaignRecipe fields, the original OLMo-1B model/manifest/checkpoint pins,
corpus and packed-index manifests, exact source inventory, finite update count,
common valid-token target, world size and physical batch for each selected arm.
The original T1024, RT0/15 and continuous-stream policies are retained. Startup
is restricted to original pretrained weights with fresh Adam and all active
parameters trainable. Adapted startup, inherited optimizer state, hidden warmup,
cooldown and SFT are rejected rather than silently converted to original startup.

Declared execution paths are the existing FP32 math/eager reference with eager
prepared execution, or production BF16 Flash/native-Triton with prepared eager
or captured execution. Checkpointing, deterministic controls, FP32 masters,
autocast-cache policy, TF32, optimizer and pointwise/RoPE choices are explicit.
The historical recipe's nominal precision label is retained; the outer execution
declaration is authoritative. A declaration is not hardware/backend acceptance.

Use PackedCampaignData's immutable metadata reader, canonical `peek_update`
and `partition`. Never call token materialization or `commit`. Record exact
chunk-membership digests, start/end cursors, all target/boundary counts, valid
token schedule prefix, physical slots, dummy rows and padding by arm/rank.
Whole-chunk overshoot is explicit. Reject exhaustion or a shortened final update;
never cycle, shuffle, drop the tail or alter the pinned stream. The planner is
bounded to 4,096 updates and 1,048,576 chunk presentations; this is a CPU resolver
limit, not a recommended training budget. Per-update unique-document counts
must not be summed and presented as corpus-unique counts.

Retention declarations require an explicit fast-chunks prefix, verified
generation-before-prune policy, unchanged-topology resume and checkpoint cadence
at most 600 seconds checked at completed boundaries. Evaluation is either
explicitly deferred with a reason, or a pinned finite-pass, all-trained-pass,
no-jitter FP32 declaration on the existing disjoint dev split. The resolver can
describe its fixed metadata footprint but executes no evaluation. Generation is
unsupported. Tracking metadata names the authorized taylorbollman account; no
credentials or SDK calls are needed.

Parameter cards reuse the completed actual CPU ownership ledger, SHA256
`f7d111fdbbac0cc9147766a6b74cd2b09d529e9de856b0608fe5c9bca5f5fea9`,
cross-checking the native architecture formula and inherited source hashes.
Tied parameters count once; dormant fusion remains resident; NextLat is
training-only. Reuse the ledger's matrix-work formulas at each arm's aggregate
physical footprint. Distinguish selected CE/latent/KL/predictor work from dense
prepared capacity, including dummy/padded positions. The predictor-source union
equals latent pairs for these fully supervised packed documents. Estimates
exclude pointwise, communication, optimizer, graph setup and hardware padding;
they are not measured FLOPs, GPU memory estimates, MFU or throughput.

Focused tests cover all-eight schema and ownership cards, real packed metadata
versus independently materialized token masks, heterogeneous rank allocations,
exact schedule-prefix arithmetic, incomplete corpus, unsupported policies,
source/tokenizer/authority mismatch, explicit evaluation declarations, and
no-model/no-CUDA behavior. No old frozen file is edited. Root reviews before
one actual CPU resolution of the existing readiness corpus and pinned artifacts.
Retain manifest, resolved JSON, readable card and source snapshot locally.

The example uses three 16,384-token updates, two declared ranks and B8 for all
eight arms solely to demonstrate a shared plan. **It is not a production
configuration or an all-arm capacity recommendation.** Evaluation is explicitly
deferred. Future review must select data mixture/order, startup, precision,
hardware/partition, training/evaluation budgets and generation contract before
implementing and accepting the actual campaign launcher.
