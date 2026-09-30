# Independent validation

This note records independent review for the NFR64 pass curves and KL0.1
continuation to update 128. Both endpoint probes and their CPU summary have
completed and passed closure checks. Continuation execution is separate;
its terminal validation remains pending.

## Existing endpoint authority

Read-only inspection on 2026-09-30 confirmed that the original NFR32 and both
NFR64 checkpoint directories still contain their manifests and state files.
Manifest hashes match their successful training reports; state sizes match the
published receipts. This inspection did not rehash the three approximately
15.2 GB state files. Their existing publication receipts record download-SHA,
server size/MD5 and SHA metadata verification. The diagnostic and strict resume
loaders must independently authenticate the actual state they load.

| State | Report SHA256 | Manifest SHA256 |
| --- | --- | --- |
| Original NFR32 | `01bceb2a1191adb513bea974d8dcb0a5b52e8b5c3b384dbd0e69d21cfea665f7` | `1c83b37812005b9d87a3ce06a1156b18789498e0d4a1ce7964cb16c5338bc82c` |
| NFR64 KL1 | `4e47728173364a6a3a6ed14887517cea9df8bcd2247d1b5d8ca743eb2f3b27fe` | `f46420d563943d4e28f29b67c3f182c0c1e61f96406827061cb4e7beca4a14a3` |
| NFR64 KL0.1 | `9de466ea50ea837db1aaf6f0e88f23675c6a13af1f24a196fab992ea54a96e8c` | `4f24a233c89e2eb20a0c355d1a3dda530fb906981402297278b2decffde611ea` |

All 215 files in the existing NFR execution inventory still match their frozen
hashes. The passing paired audit remains
`2ee9b737c8269ef071fd63792f9d4ed2d87598b900347b571d7811e1aadcb211`.
The reduced endpoint has populated Adam state for 71 owned parameter tensors,
64 optimizer updates, scheduler epoch 64 and 33,554,432 input tokens.

## Review criteria

The saved-state probes must bind the exact audited endpoints and the same eight
packed development rows, use common FP32 with no jitter, retain the RT layers,
and leave weights, RNG, module modes and gradient buffers unchanged. The
NextLat predictor remains present but must not execute during these curves.

Settling is assessed using finite hidden/input/pre-normalization scales and
successive-pass changes on the tail and unsettled suffix. K4/K8 residuals against
K32 describe a finite reference, not exact online inference. The first causal
positions can settle by construction; aggregate changes alone are insufficient.
Reaching the numerical floor and improving CE are separate questions. Report
absolute CE at passes 1, 4, 8 and 32, not just a shrinking later-minus-first gap.
NextLat predicts the next token-position representation within each pass, not
the next FBT pass.

Continuation must restore the exact reduced64 model, populated Adam, scheduler,
rank RNG and data cursors, then consume updates 65–128 of the original plan.
The original schedule hash is
`993d922340322268d7d0092b3303e8650c9e96a6b33e3c14055353e91041ef1c`;
warmup must not restart. Reproduce the restored development measurement and
retain/evaluate updates 96, 100 and 128. Finite large clipped gradients or worse
later-pass CE remain important observations, but alone are not execution faults.

Two implementation hazards were identified before coding: the old branch
identity includes `review_stop=64`, and its evaluation schedule does not include
100. New continuation authority must handle those changes explicitly while
preserving inherited training state and the old frozen source/configuration
lineage. Merely changing the old CLI stop or relabeling its report is inadequate.

## New implementation review

Read-only source review found no unresolved correctness issue in the new
endpoint producer, summary or continuation adapter. The endpoint and
continuation implementations are frozen. This is not a native execution
result.

The endpoint producer binds the passing pair audit, original 215-source
authority, unique published checkpoint, configuration, cursor and eight-row
panel. A temporary hook observes one canonical K32 forward; it does not add a
second stack pass or execute the predictor. Its residual pools squared
coordinate-mean differences and reference norms before taking their ratio.
For K-versus-K32 it excludes positions below K; this correctly differs from
excluding positions below K−1 for consecutive-pass differences. Focused tests
cover swapped endpoints, malformed authority, exact preservation of existing
curve metrics, one forward, forbidden predictor execution, hook removal after
failure, analytical pooled norms and empty suffixes.

The summary's metric names and region selections match the producer. Requested
guards now check preservation flags, identical materialized batch digests,
8,192 input tokens / 8,184 CE targets and finite plotted measurements. Finite
K32 residuals remain explicitly labeled as such.

The continuation reconstructs the exact original32 and reduced64 configuration
and fingerprints, strictly loads the latter, and checks the complete inherited
boundary against the independently saved boundary. A metadata-only transition
records the new execution scope, added update100 evaluation and named save
boundaries. It preserves the old objective branch, model configuration and
original schedule. The new activation receipt binds both preserved endpoint
reports to the same panel and scope and the selected reduced64 checkpoint;
the explicit health judgment remains a separate recorded decision.

An independent AST comparison found the following ten callbacks unchanged from
the frozen KL engine: `model_contract`, `current_boundary`, `save`,
`create_manager`, `submit`, `poll`, `accept`, `prepare`, `update` and `log`.
This confirms unchanged mathematical/update and storage callback code; it does
not substitute for the strict live restoration checks or native measurements.

The continuation owner reported 36 focused CPU tests passing in 2.07 seconds, including
rejection of altered evaluation panels, additional undeclared evaluations,
objective/schedule changes and mismatched endpoint activation receipts. An
independent read of the completed metadata resolution confirmed all 222 source
pins currently match, the original schedule is intact and the only added
evaluation boundary is 100. Resolution SHA256:
`df798d05481d26113032aee34a074373421bd49e50b84826b8b5ef8eb58459a5`.
Scope SHA256:
`563c6bbb3b5491bffae75879fbfc9299b94021a25c8c391a91d451a1622d4523`.
These are preparation receipts, not an activation decision or a tensor-resume
result. The five new implementation/test files are committed as `22a05c5`;
the frozen resolution contains 222 source pins. Further runtime source changes
require new authority rather than silently updating this resolution.

The endpoint owner reported 44 CPU tests passing in 3.02 seconds across the
new endpoint and existing component suites. Both completed CPU preflights use
the same eight materialized row digests and 230-source authority. Scope SHA256:
`e8d52df40aa5456639c919ff32d58cde12a65a3fcf49df07d91778f218d48880`.
Control/reduced preflight report SHA256 values are respectively
`f74d3fd7cd3607f305eb3fc4cc53ca3717673553bc273c89558ddee2366fe997`
and `188b0b08ad8ce219e817e355a4f8fef3a4b5cd6b3e4432e611077866b397d0de`.
Preflight explicitly does not verify or load checkpoint state tensors. The
immutable preparation evidence was independently copied and retained; see
[storage-receipt.md](storage-receipt.md).

## Completed endpoint closure

Both endpoint launchers exited 0, each probe completed all eight rows and 32
passes, and both W&B runs synced. Report SHA256 values:

- KL1: `6a80643b3855829e4ad444a0a42dc188181348e7e9899879c40fe1d4ae276ac9`.
- KL0.1: `97fe4e9029f809c7bc46956b8fee989e590a6f9839959e8bef64f72cd4378a0b`.

Independent CPU closure confirmed identical panel, policy and batch tensor
digests; zero optimizer updates; all runtime/mode/RNG preservation checks;
unchanged weight hashes; absent gradient buffers; and all 230 live source
pins for each probe. Reaggregating every raw per-row/pass/region sufficient
statistic exactly reproduced the reported pooled sums and CE. Reaggregating
the direct K4/K8-versus-K32 residuals exactly reproduced the reported squared
sums and relative norms. Every derived pass metric was finite.

The completed, synced summary binds those exact report hashes:
`4396a81af23dbbb9f595dec1e395b9b9bcb8a124d0b1bb4e4d63ee144f768525`.
The explicit continuation activation binds the same pair and reduced64 source:
`9a85cd4f6b2d48e95a0e301c92f6536bee302acb3b5af0496da15c1247c95f71`.
These checks support the integrity of the observations, not a conclusion that
later passes improve CE. K32 remains a finite reference, and the existing BF16
optimization-equivalence qualification is unchanged.

The CPU-only closure receipt is
`.runtime/olmo-nfr-stability-128/endpoint-terminal-retention-input-01/closure.json`,
SHA256 `01e2a5087f5ad5e0e61539d128989ce2ad285aedcec957d88640e2c3ee9f1d32`.
No GPU call, model load or checkpoint-state rehash was performed by this
independent closure. The closed outputs and activation were retained separately
from the active continuation; see the storage receipt.

## Prepared final observer and reporting review

Read-only review of `olmo_nfr_final_curves.py` found no unresolved correctness
issue. It admits only the completed, synced and cloud-published reduced-KL
NFR128 continuation with exact resolution, configuration, source inventory and
ordered cursor. It reuses the frozen update64 observer's mathematical functions
and compares materialized batch hashes against the exact reduced64 probe. The
owner reported 30 focused CPU tests passing. Actual terminal128 scope binding,
state authentication and preflight remain deferred until that checkpoint exists.
Preparation SHA256:
`e00655584ae906401bc12bab826ae24c68edf830a75819bae7327c8e13dddde8`.
All 244 preparation pins matched at independent retention time.

The continuation summary's time aggregation is appropriate: sum the selected
regions within each rank, take the slower rank for each update, then divide
total real input tokens by the sum of update durations. It does not sum rank
times, average rates or multiply inputs by four FBT passes. Memory is the
maximum sampled per-GPU value. Development64 is required to match exactly
across the parent and restored run and is included only once. Raw loss target
counts are retained separately; training CE keeps its pass weights and the
auxiliary terms are pass-averaged before objective coefficients. Executor wall
time is explicitly distinguished from final W&B/host closeout.

The optional final-curve summary binds the new final128 schema to the exact
reduced64 report SHA and requires identical data tensors, panel policy, pass
coverage and preserved state. Its overlay selects reduced64 and reduced128,
with distinct labels and a separate summary schema. It does not overwrite or
relabel the retained paired64 comparison.

The new terminal audit was also reviewed. Its reconstructed metadata,
inherited boundary, unchanged data/LR, required update100 evaluation/save and
accepted validator reuse are appropriate. Its explicit scope is an
uninterrupted 64→128 segment; a restart requiring multiple report segments
would need an independently reviewed history join. Imported audit helper
sources must be included in the audit's own retained source inventory.
