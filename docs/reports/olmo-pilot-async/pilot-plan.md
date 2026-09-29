# Bounded adaptation pilot after asynchronous checkpoint acceptance

Asynchronous checkpoint acceptance and the separate four-update native
capacity/overlap check are complete; see [results](results.md). The measured
fixture uses the proposed effective batch, but its endpoint is not the start
of this comparison. The CPU-resolved 32-/128-update learning cohort below
remains unlaunched.

This proposal keeps the user's concern about heavy clipping and poor later-pass
CE explicit. Those are adaptation questions for the bounded pilot, not reasons
to repeat the broad BF16-versus-FP32 investigation before observing training.

## Concrete cohort and startup

Use the pinned original OLMo-1B `step60000-tokens252B`, revision
`81b71efbce6f4dada57c94860301af4298bcd351`, at T1024. Keep the accepted model,
native Q/K behavior, attention kernels, CUDA graphs, activation checkpointing,
BF16 mixed training, NextLat regression/KL weights and feedback jitter unchanged.

| Arm | Meaning | Initialization | Physical batch/GPU | Accumulation slots/GPU |
| --- | --- | --- | ---: | ---: |
| B | Ordinary continuation control | Original weights, fresh active-parameter Adam | 32 | 8 |
| NF | NextLat plus K4 feedback | Original backbone, paired fresh predictor initialization, imported fusion128 weights, fresh all-active Adam | 12 initially | 22 |
| NFR | Same as NF plus native temporal RT at layers 0 and 15 on every pass | Exactly the same fusion128 import and remaining initialization as NF | 12 | 22 |

The immediate treatment comparison is **NFR minus NF**. Both receive identical
ordered examples, logical batches, initialization rules, imported fusion state,
new optimizer recipe and token schedule. B is the moving original-model control;
its prior adaptation exposure is different and must remain labeled.

Using the shared fusion128 import is preferable for this first practical pilot
because it is the startup route supported by the existing numerical evidence.
Fusion-only adaptation reduced matched BF16/FP32 gradient discrepancies without
changing the pretrained backbone. It does not eliminate the known trajectory
qualification or establish useful feedback refinement. That import previously
consumed **1,073,565 valid inputs / 1,048,576 CE targets**; charge its exposure
and compute separately. Import neither the short capacity checkpoint nor the
later complete-model adaptation checkpoint as if either were only fusion warmup.
See [startup evidence](../olmo-fusion-startup/results.md).

NF physical batch 12 is a conservative initial allocation, not a measured NF
capacity result. Observe its first actual updates and memory before committing
to a longer segment. NFR12 now has native acceptance at all 22 accumulation
slots. B32 has prior native one-slot acceptance; observe its first accumulated
updates as well.

The existing declaration contract accepts either original startup or the
fusion128 route, and restricts that adapted route to NF/NFR. Consequently use
**two declarations**: B-original and paired NF/NFR-adapted. Do not put B into an
adapted declaration, relabel old checkpoint ancestry, or silently alter the
contract to make all three appear to have identical prehistory.

## Logical batch, optimizer and finite schedule

Every arm uses **524,288 valid input tokens/update**, or 512 full T1024 rows
globally on two GPUs. B32 requires eight slots. NF/NFR12 require 22 slots,
including 16 dummy rows globally in the final slot. The actual resolver assigns
260 real rows to rank 0 and 252 to rank 1: four and twelve dummy rows respectively,
so rank 1's final slot is fully dummy. Losses
normalize by actual global CE targets, latent pairs and KL triples; dummy rows
contribute to neither exposure nor the normalization denominators. Additional
FBT passes increase work, not input-token exposure.

Keep fused AdamW, peak LR `2e-4`, betas `(0.9, 0.95)`, epsilon `1e-5`, existing
decay exclusions, gradient clipping at 1.0 and token warmup from 10% to 100% LR
over **52,428,800 inputs**. That is 100 updates at this batch. Do not infer an LR
change from small one-slot capacity norms or start an LR grid now.

Declare a **128-update ceiling / 67,108,864 inputs** before launch. Stop the
initial learning segment at **update 32 / 16,777,216 inputs** for assessment.
This preserves a compatible finite schedule if continuation is warranted.
Stopping at 32 is only 32% through warmup; it can expose nonfunctioning
adaptation but is too early to adjudicate the scientific value of RT. Do not
silently expand the finite ceiling or convert a separate capacity fixture into
the cohort's history.

Current materialization stages all CPU feedback-noise tensors for a logical
update before replay. At 22 B12 slots this is approximately **6.18 GiB/rank**
for the three FP32 `(12, 1023, 2048)` noise tensors per slot, plus temporary
construction buffers and data metadata. Noise moves through fixed physical
graph buffers on the GPU; accumulation does not imply keeping all microbatch
activations in VRAM. This is another reason to measure the actual accumulated
path rather than multiply one-slot performance mechanically.

## Fixed development monitoring

Choose the **65,536-input prefix of `dev-main`**, common FP32, jitter disabled,
all trained passes, physical evaluation batch 1/GPU, **every 16 updates**.
The first segment therefore has development observations at 16 and 32, and
the full ceiling at 16, 32, ..., 128. Freeze membership and cadence before
learning, using the already completed metadata inspection rather than outcomes.

This prefix touches 193 unique documents and 29 source objects across seven
of nine strata. It omits books and Wikipedia. It offers substantially better
coverage than the five-row capacity prefix while retaining a modest bounded
evaluation cost. These are touched documents, not 193 complete documents or
independent source families. See [coverage](../olmo-pilot-execution/dev-prefix-coverage.md).
Keep omitted-source diagnostic panels and unopened confirmation data separate;
do not imply full source coverage or independent replications from overlapping
main/source panels.

The accumulated diagnostic measured **225.45 seconds / 3.76 minutes** for the
65,536-input evaluation at this physical batch. Two such evaluations would add
about 7.5 minutes to its first 32 updates if that timing holds. Allow variation
when budgeting. The ordinary model should be much cheaper in
absolute time, although evaluating equally often can consume a greater
fraction of its much faster training time.

The roughly 10% evaluation-overhead target is a guide, not a reason to shrink
the chosen panel after seeing outcomes. The proposed cadence may cost somewhat
more than 10% for these short segments. Measure it once and report the actual
cost. Any later cadence change should be declared at a new compatible boundary.

The accepted controller does not schedule update-zero evaluation. Comparisons
begin at the first declared evaluation, so this pilot alone cannot measure
immediate pristine-start retrofit damage. Do not substitute a later capacity
checkpoint for an origin measurement. Adding an explicit preserved origin
evaluation is optional future scope rather than a prerequisite for this
bounded functionality/adaptation test.

## Keep the refinement concern visible

PR50's eight small NFR updates were finite, but raw gradient norm remained
large, falling from 222.53 to 24.14 under clipping at 1.0. Its small dev-main
prefix had per-pass CE approximately **2.993, 7.654, 7.652, 7.631**. The capacity
batch/startup/exposure differ from the new cohort; these are reasons to monitor,
not an architectural verdict or proof of a precision bug.

For each update, preserve CE, latent regression and KL separately, their
denominators, LR, input exposure, raw gradient norm and whether clipping was
active. Derive the clipping coefficient from the recorded norm and configured
threshold using the implemented clipping rule if that coefficient is not already
logged. Summarize clipping frequency and norm ranges over fixed intervals.

For each fixed development observation, report **all four pass CEs**, plus
`CE(pass k) - CE(pass 1)` for k=2,3,4, and the existing latent/KL terms. A falling
weighted objective is not sufficient if auxiliary losses shrink while CE
regresses. NF/NFR should be compared at equal new input exposure; B supplies
the ordinary continuation reference with its different prior exposure stated.

Stop immediately for an actual nonfinite value, incoherent state/counter,
failed checkpoint integrity, or a concrete execution failure. Persistent
clipping alone is not an automatic stop: freshly activated objectives can
produce it. At update 32, require an explicit written assessment of whether
later-pass CE gaps are closing, whether the first pass is degrading, and whether
the gradient/auxiliary scales are settling. If none improves, pause before
spending the remaining ceiling and inspect loss weighting/fusion adaptation
with a targeted diagnostic. If auxiliary loss improves while CE worsens, retain
that as a flagged issue even when everything stays finite.

Do not introduce an arbitrary universal gradient-norm threshold, remove RT,
change Q/K normalization or reopen the broad precision study merely because
these trajectories differ. A new, localized failure would justify such work.

## Cost and checkpoint policy

The recorded compute-plus-materialization rates are **67,279 inputs/s for B32**
(the prior one-slot fixture) and **3,568 inputs/s for NFR12** (the new four-update,
22-slot fixture). They omit health/logging/coordination gaps, evaluation, setup
and checkpointing. The NFR measurement covers the actual proposed logical batch;
B's accumulated timing is still an extrapolation. The following longer-run
figures are exposure/rate estimates, not measured pilot throughput or guarantees:

| Extent | New valid inputs | B32 selected-region estimate | NFR12 selected-region estimate |
| --- | ---: | ---: | ---: |
| 4-update accumulated diagnostic | 2,097,152 | 0.52 minutes | 9.80 minutes measured |
| 32-update first learning segment | 16,777,216 | 4.16 minutes | 78.38 minutes |
| Full 128-update ceiling | 67,108,864 | 16.62 minutes | 313.50 minutes |

NFR graph preparation took 7.48 minutes. At 32 updates, adding that setup and
two measured-size development evaluations gives about **93.37 minutes before
checkpoint stalls and unmeasured host work**. Populated local-save regions cost
about 76 seconds each, while the middle checkpoint's background retention took
346 seconds and overlapped following update callbacks. The final drain still
waits. Budget roughly **two hours for the first NFR segment**, with additional
margin for variation. Full NFR128 is a multi-hour run even
with no checkpoint overhead. NF has no native accumulated throughput measure;
do not manufacture a precise cohort estimate from layer/pass counts. Measure
its first updates and then update the budget. All-three-cohort wall time is
therefore not yet established.

Continue to save an immutable completed checkpoint on local SSD before training
resumes. The CPU worker may upload and verify that immutable file while GPUs
advance, and publish it as the latest cloud recovery point only after successful
verification. Keep the previous verified cloud checkpoint authoritative until
then. If the VM and SSD disappear while a new upload is pending, recovery uses
that previous verified generation and loses the newer work; it need not lose
the whole run. A process restart with SSD intact may also use the separately
identified completed local state. Local completion and cloud durability must
be distinct in reports and checkpoint selection.

For the future learning cohort, retain the **600-second trigger** and terminal/
review boundaries, with **every 32 updates** as the explicit update milestone.
This is a save trigger checked at completed updates, not a guarantee of a
cloud checkpoint or at most ten minutes of rollback. Update duration, the
local save, transfer and any drain can extend that interval.
One in-flight publication plus a bounded pending policy is sufficient initially;
avoid an unbounded backlog and never prune a needed in-flight or last durable
checkpoint. Emit local-save stalls, worker transfer/verification time, durable
lag in updates/tokens/seconds, GPU useful progress during retention, and final
drain time separately. The exact async clock/backpressure semantics belong in
the new runtime policy and its acceptance evidence, not an implicit alteration
to an old checkpoint declaration.

Moving retention off the critical path does not remove every pause: existing
local boundary checks and serialization take about 76 seconds at native size,
and the worker shares CPU/SSD/network resources with data materialization. The
bounded native check demonstrates overlap; it is not a matched native blocking
versus async speedup experiment. Keeping full cloud readback isolates concurrency as
the change; lighter upload verification can be a separate optimization if it
remains worthwhile. The user's willingness to accept some extra redo after an
interruption is compatible with this explicit distinction between completed
local and verified cloud boundaries.

## This milestone's scope and concrete artifacts

1. Accept asynchronous publication with focused CPU failure/queue tests and
   tiny two-GPU unchanged-update/evaluation/restart checks. Preserve generation
   identity, checkpoint ownership and old recovery authority while transfer is
   pending or fails.
2. Resolve the two pilot declarations against the retained real corpus and
   ordered dev suite. Verify all 128 per-update memberships/counts match between
   B and the NF/NFR pair; only physical allocation and startup differ.
3. After runtime sources are fixed, bind a new async executor identity rather
   than migrating or relabeling prior PR50 checkpoints. A CPU planning result
   alone does not authenticate new worker code or promise exact cross-version
   resume.
4. Use a separate four-update NFR fixture at the real 524,288-input logical
   batch for accumulated memory/throughput, evaluation and publication overlap.
   If desired, schedule its evaluation at update 4 and checkpoints at 0,2,4;
   those are diagnostic cadences, not the longer cohort's policy. Report its
   extra exposure separately and retain it for inspection, not as the cohort's
   initialization.
5. Close this readiness milestone with the measured overlap/cost and proposed
   B/NF/NFR budget. Then begin the declared cohort in resumable segments when
   its cost and remaining adaptation questions are understood.

The one-off helper is
`.runtime/olmo-pilot-async/prepare_pilot_declarations.py`. It invokes the accepted
PR50 CPU resolver in the GPU-disabled project container, checks the frozen
192-source inventory, snapshots its own source and the inherited sources,
authenticates retained template declarations and writes fresh plans/evidence.
It neither launches GPUs nor binds a future async executor. Concrete output and
hashes are recorded below.

CPU resolution completed in
`.runtime/olmo-pilot-async/pilot-declarations-02/report.json`, SHA256
`e7b423773784d23585e23f0da92e1886d2a8df2eb2c3b76e496c6457cf734b7e`.
Both 128-update plans have exactly the same ordered logical memberships and
counts at every update, no token overshoot, and the intended physical
allocations. Each full plan contains 67,108,864 inputs and 67,043,328 CE targets;
the shared data also supplies 66,924,093 eligible latent pairs and 66,739,567 KL
triples for enabled objectives. B's unused auxiliary counts describe eligible
data, not enabled training losses. The fixed dev prefix contains 65,472 CE
targets, 65,338 eligible latent pairs and 65,140 KL triples.

| Declaration | Declaration SHA256 | Resolved SHA256 |
| --- | --- | --- |
| B-original | `7a00d97a6cede064571ffbcd3d7edabe77c9715981db79678acdfcd9116092f1` | `268ae31442fd321672cd9f3cc4481f7135a7a16394f9ef529b67b2c9e1dd2363` |
| NF/NFR fusion128 pair | `259fc4b84f0b5356bd3a038cee9142ca73fe65efed1cdb7a56ed63a325222109` | `44d0e8ca4bf8b4d7dbd9a767980faca7c1951375f701e5f6dc0d4ff4ff238a8d` |

Attempt `pilot-declarations-01` is retained as failed helper evidence: it
incorrectly assumed the 16 dummy rows would split 8/8 between ranks. The
accepted allocator correctly produces 4/12 under contiguous slot ownership.
The helper assertion was corrected and rerun into fresh `-02` output; no
allocator, training code, corpus or historical evidence was changed. The two
successful resolutions took about 21.88 and 29.43 seconds respectively, with
no GPU, model construction, training or network use. These are CPU planning
artifacts. The accepted asynchronous runtime was separately bound for the
four-update diagnostic; the future cohort still needs its own declared
execution identities and fresh segment directories.
