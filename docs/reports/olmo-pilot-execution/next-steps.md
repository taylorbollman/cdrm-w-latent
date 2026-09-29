# From execution readiness to a short learning pilot

The next review milestone is a concrete, costed pilot declaration. The ordered
stream now works with the shared two-GPU update engine, named development
evaluation and interruption recovery. Finish the bounded native capacity results
in this milestone, then choose the learning configuration. Do not repeat the
general numerical investigation or launch the eight-arm campaign automatically.

This updates the implementation status in the
[broader experiment plan](../../olmo-nextlat-fbt-rt-experiment-plan-v1.md).
Its later research questions remain useful, but its older data, checkpoint
storage and implementation-status sections are superseded by the completed
ordered-stream work. [readiness-map.md](readiness-map.md) separates accepted
functionality from the remaining scientific and hardware decisions.

## 1. Choose the actual startup, not a capacity checkpoint

The base remains the pinned original OLMo-1B `step60000-tokens252B`, revision
`81b71efbce6f4dada57c94860301af4298bcd351`: 16 layers, width 2048, 16 heads,
SwiGLU, tied embedding/readout, RoPE and non-affine LayerNorm. Train at T1024,
without newly added Q/K normalization. Native RT is in layers 0 and 15 when
enabled. FBT uses four trained passes; RT remains active in all four when both
are enabled. NextLat includes its existing latent regression and KL objectives.

Two supported starting routes answer different questions:

| Route | Weights and optimizer | Interpretation |
| --- | --- | --- |
| Common original start | Original backbone, paired fresh fusion/predictor initialization where applicable, fresh all-active Adam | Cleanest initial common-recipe comparison across all eight arms; adaptation of new modules is part of each treatment |
| Shared fusion128 start for NF/NFR | Original backbone and predictor initialization plus the same imported 128-step fusion-only weights; fresh all-active Adam | Uses the bounded adaptation route already exercised, with an explicit extra exposure/compute history |

For a common-recipe cohort, prefer the first route unless the current functional
evidence motivates choosing the second explicitly. If the immediate question is
NF versus NFR using fusion128, give both arms exactly the same fusion import.
A contemporaneous B run is still useful, but do not describe its ancestry as
identical to those adapted arms. The existing adapted import supports NF/NFR;
applying it to F/FR is a separate startup design decision.

The fusion128 adaptation saw **1,073,565 valid input tokens and 1,048,576 CE
targets**. Account for these separately from new continuation exposure and
include their compute in any total-cost comparison. Do not import the later
full-NFR checkpoint as if it were only fusion warmup: that checkpoint changes
the backbone, predictor and optimizer history as well. The short native capacity
runs have their own batch/exposure declarations and are not the start of a
matched learning cohort.

Original pretraining optimizer-state retrieval remains unverified. Fresh Adam
is an explicit experimental choice and does not require resolving that retrieval
first. Same-lineage interruption recovery must restore all Adam state, schedule,
RNG and cursors; never assign historical optimizer steps to zero moments.

## 2. Keep physical capacity separate from the learning batch

Use the measured comfortable physical batch for each arm, leaving the observed
evaluation/capture/checkpoint memory margin. Preserve a common global effective
batch for the comparison. The existing proposed starting point is **524,288
valid input tokens per update**, or 512 full T1024 rows. It is a recipe choice,
not a number implied by whatever fits on one GPU.

The current capacity protocol starts ordinary B32/rank, conditionally B64, and
NFR B12/rank with B8 fallback. Each runs eight updates and measures updates 4–8
after excluding the first three. These candidates do not select the future
learning allocation until the complete memory/evaluation results are reviewed.

For example, on two GPUs this requires eight accumulation slots at B32/rank,
four at B64/rank, or 32 at B8/rank. B12/rank needs 22 slots, with 16 dummy rows
in the final global slot. The ordered planner records those allocations and
normalizes over actual CE targets, latent pairs and KL triples. Extra FBT passes
increase work, not counted input exposure. Larger physical batches can improve
RT utilization; accumulation alone does not have that effect.

Retain the current provisional common optimizer settings for the first pilot:
fused AdamW, LR plateau 2e-4, betas (0.9, 0.95), epsilon 1e-5, existing decay
exclusions, clipping at 1.0, and token-based warmup from 10% to 100% LR over
52,428,800 valid inputs. Do not open a learning-rate grid without an observed
learning or stability concern. Any later tuned effective batch or optimizer
recipe becomes a separately labeled comparison.

Physical capacity fixtures often use one global physical slot per update. Their
tokens/s estimate does not remove the need to resolve the proposed accumulated
logical batch. Check its CPU membership/count/allocation plan, then observe the
first two real updates as the beginning of the pilot. Reuse the accepted graph,
accumulation and restart behavior; add another small integration check only if
the chosen topology, startup or execution contract is materially different.

## 3. Freeze development membership and evaluation cost

Capacity evaluation uses a 5,120-input prefix of `dev-main` and a 2,048-input
prefix of `dev-source/books`, both common FP32 with physical B1/rank. These tiny
prefixes establish operation and cost. They are not yet a representative quality
panel. Before the learning declaration, use metadata to report the chosen
prefix's actual source and document coverage and choose its size without looking
at model outcomes. The full `dev-main` contains 1,048,576 inputs; evaluating it
every few updates is not required.

The completed [prefix coverage check](dev-prefix-coverage.md) finds 14, 52 and
193 unique documents at 5,120, 16,384 and 65,536 inputs, covering four, six and
seven of the nine strata respectively. All three lack books and Wikipedia;
the smallest also lacks C4, peS2o and Reddit. These nested-prefix counts inform
the pending cost/coverage decision without selecting membership or reading
model outcomes.

Prefer one fixed `dev-main` prefix for routine monitoring. Select a larger
prefix if the cost estimate supports useful coverage; use occasional source
panels only for a stated diagnostic. Books and Wikipedia have little weight in
the main mixture, and the book pool contains very few documents. Main/source
panels overlap, so retain separate results and never pool them as independent
replications. Keep confirmation outcomes unopened during selection and tuning.

Choose cadence from measured total evaluation time, including preservation and
data verification. A useful initial target is evaluation time below roughly 10%
of the intervening training time. Keep at least an intermediate and final
measurement in the first pilot segment; ordinary per-update health metrics
continue between evaluations. Freeze membership, physical evaluation batches
and cadence in the declaration before learning begins.

Smaller evaluation batches reduce FP32 forward memory. They do not reduce all
fixed observer costs: graph-owned input/noise/layout hashes, gradient-zero
checks and opening an authenticated corpus remain. Lean evaluation avoids full
model/Adam byte hashing, while tiny acceptance deliberately includes it. Report
these costs separately from useful update throughput before changing any check.

Current scheduling starts at positive update multiples. If measuring immediate
retrofit damage at update zero is needed, add a narrowly scoped evaluation-only
origin invocation and check its state preservation. Otherwise start comparisons
at the first declared evaluation and omit claims about unmeasured initial
damage; a later capacity checkpoint is not a pristine reference.

## 4. Declare a small resumable pilot and an explicit review point

The 134,217,728-input training panel is available capacity, not an agreed training
budget. A practical proposal at the common batch is a finite **128-update
ceiling, 67,108,864 inputs**, with the **first segment stopping at update 32,
16,777,216 inputs**, for review. This leaves room to observe the end of the
100-update warmup and 28 plateau updates if continuation is later chosen.
Estimate wall time and cloud/checkpoint storage from the measured arms before
adopting these numbers. Neither the ceiling nor this document authorizes a run.

Declare that finite ceiling before starting and use `stop_after=32` for the
first segment. The accepted execution identity includes the complete finite
schedule. Silently extending a declaration that originally ended at update 32
would be a different contract and is not a supported exact resume. A planned
stop inside a longer, already frozen ceiling preserves ordinary continuation.

For the first small cohort, B supplies the moving continuation control and
NF/NFR directly address the principal RT question. A first 32-update segment is
mainly an adaptation/health assessment: with the retained warmup it is too early
to declare that recurrence helps or fails. Inspect training loss, per-pass dev
CE, auxiliary terms, clipping/gradient behavior and whether the live execution
stays finite and recoverable. Extend only within the declared ceiling after the
review; change the recipe only for a concrete concern and label new lineages.

The native capacity fixtures use a 600-second policy and every-four-update
checkpoints. That is a test setting, not a future learning-cadence decision.
For a learning pilot, retain **600 seconds plus terminal and meaningful update
milestones**, rather than automatically checkpointing every four updates.
Keep verified GCS publication before pruning and two retained local boundaries.

The completed B32 fixture makes the cost material: selected checkpoint regions
at updates 4 and 8 took **367.55 and 383.71 seconds**, excluding later untimed
publication/validation. Its eight-update stage lasted **17.49 minutes**, mostly
checkpoint work. The loop resets its checkpoint clock **after verified
publication**. Consequently, 600 seconds of advancing work followed by another
approximately 6.3-minute checkpoint implies roughly 16–17 minutes or more
between durable publications, plus boundary overshoot and unmeasured publication
cost. This is an estimate, not a timing guarantee.

There is no need to raise the current 600-second limit to target the user's
20–30 minute interruption window. Provisionally budget roughly **40% or more**
of elapsed time for checkpoint work under such a cadence, then revise from the
actual NFR costs; this is not a measured steady-state overhead fraction. The
short fixture's every-four-update policy is much more expensive proportionally.
See [checkpoint-cost.md](checkpoint-cost.md) for measured regions, clock semantics
and the scope of a later focused I/O improvement.
Report any indivisible setup, evaluation or transfer operation likely to exceed
that window. Continue online W&B and immutable local/cloud evidence. A successful
short pilot may form the beginning of the eventual learning run only when its
recipe, ancestry, data order and finite schedule remain compatible.

## 5. Expand the research comparisons in phases

The full factorial remains B, N, F, R, NF, NR, FR and NFR. The four conditional
RT contrasts are **R−B, NR−N, FR−F and NFR−NF**. Prioritize the last one, retain B
as a control, then add N/NR, F/FR and R as resources permit. Do not infer the
missing contrasts from NFR alone; every newly used configuration gets its own
short health and memory observation without repeating a broad numerical study.

The proposed approximately 500M warm-token RT screen, cooldown, common SFT and
selected compute-matched extensions remain later decisions. The current stream
does not supply that budget without a declared data extension, and the current
finite warmup/plateau runner does not implement arbitrary schedule extension,
cooldown branching, SFT or generated-task evaluation. Add these only as the
chosen next comparison requires them. Keep token-matched findings distinct from
actual compute-matched runs; a timing column alone does not make them compute
matched.

BF16 and FP32 can follow different optimization paths, particularly after
restarting Adam while adding modules. Prior matched-state and optimizer-history
work supports proceeding with a qualified functional pilot, not bitwise
precision equivalence or a universal harmless-error threshold. Keep the
qualification visible and monitor actual optimization. Reopen local numerical
analysis for a new concrete failure, not simply because trajectories differ.

Two-H100 acceptance applies to the tested native implementation and declared
layouts. Moving to H200 or changing world size needs a short launch, memory,
evaluation and completed-boundary restart check on that topology. It does not
require repeating the full mathematical investigation. Cross-topology exact
checkpoint migration is not currently accepted; decide that transition before
assuming an H100 pilot can resume unchanged on a different device count.
