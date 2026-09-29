# Execution startup and recovery contract

This new versioned module wraps the frozen CPU manifest resolver without changing
its declarations, source guards or original-weight semantics. It performs local
metadata resolution and byte hashing only: no model construction, tensor loading,
GPU execution, download, upload or training. Successful resolution is not launch
authorization, numerical clearance or a choice of production experiment.

## Declaration and supported startup

`olmo-campaign-execution-declaration-v1` contains exactly `schema`,
`planning_manifest`, `startup` and `implementation_sources`. The nested planning
manifest is an unchanged v1 declaration for original-backbone construction,
data membership, finite token budget, parameter ownership and physical partition.
The outer **actual startup** is authoritative for the execution origin. Both
documents are retained; adapted weights are never silently relabeled as fresh
original weights or normalized into the old startup schema.

The wrapper supports two explicitly separate cases:

1. `original-pretrained-fresh-optimizer-v1`: all eight B/N/F/R/NF/NR/FR/NFR arms,
   original pinned native weights, declared new branch seeds and fresh Adam for
   all active parameters.
2. `fusion128-fresh-all-adam-v1`: NF/NFR only. Import exactly the complete fusion
   state from the retained FP32 fusion-only update-128 checkpoint. First construct
   the historical isolated NF model and call the unchanged strict weights-only
   loader, then verify its receipt. Only afterward perform a separately checked,
   weights-preserving policy/arm transition into the declared packed NF/NFR mode.
   The root runner owns the actual model construction/transition and must validate
   state bytes, parameter identities/trainability, modes, ties and target configs.

The adapted declaration pins checkpoint SHA256
`892ff2fdcdeec89e3008a16a12e91158250ebe05adfe0e9efce8f153409b8cfc`
(103,240,258 bytes) and historical train-02 report SHA256
`79a148ff18af694fca542b7f02eaca5d3005ab8fd4c140a369071099eb8b3edc`.
It explicitly requests complete fusion weights only, **fresh all-active Adam**,
reset optimizer/scheduler/RNG/cursor lineage, and the new packed source prefix.
The old fusion Adam, scheduler and RNG are not imported. The selected prior
exposure remains separately recorded: 128 updates, 1,073,565 valid input tokens,
1,048,576 CE targets, zero latent/KL targets, 9,235 document presentations and
1,219 physical microbatches. The report also records the old data authority and
frozen backbone/predictor/scale pins. This is not exposure matched to untouched
original weights. Documents here are presentations, not unique documents.

The checkpoint's historical source inventory must still match current pinned
files. Fusion/predictor seeds and feedback jitter must match the selected origin.
New training hyperparameters and physical allocation are explicit in the new
manifest. The old import is not broadened to arbitrary predictor-free arms,
other adapted weights, inherited Adam, schedule migration or runtime conversion.
Such unsupported declarations fail before model construction.

## Public API and division of responsibility

- `validate_declaration(declaration, sources=None)` checks the complete versioned
  schema and calls the old validator directly.
- `resolve(declaration)` independently resolves the old plan, authenticates any
  selected fusion origin and returns an inspectable contract with a canonical
  SHA256. The old planning status is retained alongside actual startup authority.
- `load_declaration(path, sha256, resolved_path=None, resolved_sha256=None)` reads
  exact pinned JSON, re-resolves before CUDA and optionally requires equality
  with an independently pinned resolved artifact. Both optional pins are required
  together. No stale resolved file is trusted instead of re-resolution.
- `startup_plan(resolved, arm)` exposes the exact historical constructor/import
  recipe, expected import receipt, target recipe/mode/config and permitted
  weights-preserving transition. Original startup has no adapted import.
- `validate_import_receipt(receipt, plan)` checks the unchanged historical
  loader's configuration/source digests, all import guards, saved fusion pins,
  old counters/cursor and the explicit weights-only restore scope.
- `execution_identity(resolved, arm, *, runtime, determinism, model_contract,
  extra_sources=None)` binds actual model ownership and runtime observations to
  the plan and startup. Extra runner/observer sources are explicitly pinned and
  checked against local bytes; conflicts with historical sources are rejected.
- `expected_counters(identity, completed)` calculates exact exposure from the
  immutable logical plan and arm-specific physical allocation. It includes only
  enabled auxiliary targets and uses packed rows for the existing document clock.
- `validate_resume_metadata(checkpoint_manifest, identity)` checks same-lineage
  committed metadata, topology, complete counters and every rank cursor. It
  returns whether the entire finite plan is already complete; callers must exit
  before indexing another update or preparing a graph in that case.

The root entrypoint must reject execution choices it has not implemented even
when metadata resolution supports them. In particular, this module can resolve
FP32/eager and evaluation declarations; their existence does not implement those
paths in a BF16/captured/deferred-evaluation runner. Tiny-model acceptance has a
separate explicitly labeled fixture adapter, not an exception to native weight
or manifest authority.

## Immutable identity and generic recovery

The per-arm identity contains the resolved-contract digest, actual startup,
recipe/model ownership, complete finite data/count/allocation plan, schedule,
backend and evaluation declarations, runtime/determinism, and all execution
source pins. Rank cursors use `olmo-campaign-execution-cursor-v1`. It excludes
segment output directories, a requested early stop and observation verbosity;
lean and acceptance observations therefore need not fork the training lineage.
The declarations themselves, including their retention/tracking choices, remain
bound through the resolved digest.

The new distributed checkpoint configuration stores the complete identity at
`configuration.execution_identity`, and its source fingerprint stores the
canonical SHA at `execution_identity_sha256`. Generic same-lineage recovery
requires the independently pinned committed manifest and its exact state bytes,
then the unchanged distributed loader's full tensor, optimizer, scheduler,
ownership and per-rank RNG validation. **The metadata helper alone is not a
checkpoint loader or proof of byte integrity.** Save a validated new distributed
origin for adapted forks; historical compact checkpoints are not generic
two-rank checkpoints.

A stopped, failed or absent run report does not invalidate an otherwise valid
committed boundary. A completed uninterrupted reference is unnecessary. Optional
acceptance observers can still compare exact reference updates separately.
Changed arm, startup, source, mode, policy, schedule, physical partition or
runtime identity requires a new reviewed lineage; it cannot masquerade as a
same-lineage resume. Checkpoint cadence remains a completed-boundary target,
not a guarantee against arbitrary shutdown or slow transfer.

CPU tests cover direct all-eight resolver reuse, malformed declarations,
selected-origin pins and receipt guards, finite-plan counters, separate startup
and exposure provenance, exact identity changes, and metadata resume at initial,
stopped and completed boundaries with no reference report. Independent runner
tests must additionally exercise actual historical import/transition and real
generic tensor/optimizer/RNG recovery before GPU acceptance.
