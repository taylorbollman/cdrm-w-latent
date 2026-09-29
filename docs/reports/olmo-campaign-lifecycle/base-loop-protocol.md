# Ordinary pretrained OLMo: common host-loop integration

2026-09-29. This bounded acceptance connects the already-tested packed producer,
pretrained ordinary model, captured DDP and distributed checkpoint implementation
to the common lifecycle loop. No frozen source, core model, objective, optimizer,
kernel, data or backend policy is changed. Root alone launches GPU jobs after
source/test freeze and review. No additional quality campaign is authorized.

Use the original OLMo-1B step 60000 authority and **ordinary arm B**: no temporal
RT, no FBT feedback and no NextLat predictor or auxiliary losses. Keep the
existing dormant fusion wrapper frozen and account for its resident parameters
separately. Preserve native tied readout ownership, FP32 master parameters,
production BF16 mixed precision, deterministic ordinary Flash SDPA, ordinary
activation checkpointing, native RoPE and fused campaign AdamW. Force the same
Flash scope through preparation, backward, checkpoint recomputation, capture
and replay; autocast remains forward-only with cache disabled. Set determinism
before CUDA and disable TF32. Do not substitute a tiny initialization.

Use the original verified continuous-stream train index at T1024 and physical
B8 **per GPU**, two H100 80GB GPUs. Exactly three planned logical updates use
M1/M2/M3 accumulation: 16,384/32,768/49,152 valid input tokens. The uninterrupted
reference therefore presents 98,304 distinct stream tokens in 96 non-overlapping
chunks, with 98,208 CE targets. Latent/KL counts are zero because those objectives
are disabled. Report actual packed document boundaries and row presentations;
do not infer unique-document counts from the legacy `documents` counter.

These explicit varying diagnostic targets override the ordinary campaign's
nominal 524,288-token update size for this smoke only. Retain the original
52,428,800-token LR warmup, plateau and optimizer settings. Schedule exposure
uses the actual 98,304 tokens, with no reset or accelerated warmup. This is not
a production batch choice, throughput measurement or learning-quality result.

Two fresh processes/stages are authorized:

1. Reference: construct the exact pretrained model, prepare/capture the existing
   local and synchronized DDP backward graphs with 11 warmup iterations, and run
   updates 1/2/3. Save and retain checkpoint 1 while the graphs remain live, then
   continue updates 2/3 and save/retain checkpoint 3. Do not write checkpoint 0;
   the original pretrained weights and explicit fresh-state recipe/seeds are
   already retained by authority.
2. Resume: root first restores checkpoint 1 from its exact retained GCS
   generations. New torchrun processes strictly load it before DDP/capture,
   verify the saved complete boundary, reconstruct graphs with Adam resident,
   and run only updates 2/3. Require bitwise equality of inputs, raw gradients,
   losses/metrics, model/Adam/scheduler/counters, committed cursor and rank-local
   RNG against the uninterrupted reference. Save only the final checkpoint3
   during an uninterrupted resume; do not overwrite the imported checkpoint.

The reference intentionally prepares DDP/graphs with initially empty Adam
state; the first actual update creates its moments. The resume loads populated
Adam before preparation. Record this difference explicitly and keep exact
next-update equality as the gate, rather than manufacture optimizer history.
This follows the successful frozen packed pair: its [write ledger, lines
362–364](../olmo-packed-campaign/test-ledger.md#corrected-deterministic-write-and-continuation)
records absent initial Adam, and its [fresh-resume ledger, lines
378–393](../olmo-packed-campaign/test-ledger.md#corrected-cloud-restored-fresh-process-continuation)
records loaded moments and exact continuation. The authoritative reports are
`.runtime/olmo-packed-campaign/pretrained-write-02/report.json` (SHA256
`a3c1b48913ecc95f087cc1a79ddc3c70e6ac0642e0f0e5ee0f3883b42b230754`)
and `pretrained-resume-02/report.json` (SHA256
`18666d8cfed2ef79bb5a20f0f569c81c25dcf8b08b6a20842a5983f42e821603`).
That prior success does not replace this new ordinary-B acceptance.

Use the existing common host loop for coordinated logging, stop/signal handling,
checkpoint cadence at most 600 seconds, retention and publication. Preserve local
RNG around cloud SDK work. A graceful stop or elapsed cadence may publish an
intermediate completed boundary; label a short segment as stopped, not passed
three-update acceptance. Do not add failure injection or evaluation variants.
Unknown CUDA/NCCL/update failure retains external teardown and previous complete
checkpoints; no emergency checkpoint of an uncertain optimizer state is claimed.

Pin runtime/topology, corpus/index, source checkpoint, recipe, exact finite data
plan, optimizer ownership and source inventory in both checkpoint and report.
Check all data membership/counts and committed clocks, dormant state integrity,
finite active parameters/moments/gradients, replica equality, preparation
invariance, graph/persistent-buffer ownership and fresh-resume equality. Persist
candidate observations before reference equality gates. Report allocated and
reserved memory and stage/update times with their diagnostic/I/O scope; do not
present these few instrumented updates as optimized throughput.

Root reserves roughly 15–25 minutes including checkpoints and cloud verification
for both stages, with an external bound of at most 1200 seconds per stage. Reports,
source snapshots and W&B logging remain incremental; full checkpoints are saved
to persistent storage and retained immutably. The complete reference plus resume
executes five physical optimizer updates but only three unique logical updates.
CPU tests cover real packed plans/cursors, B-only ownership, literal CE/native
gradient equivalence, strict source/checkpoint reference guards and checkpoint policy.
The new test is scale/lifecycle integration, not additional NFR BF16 clearance,
H200 qualification, production data selection or a new model-quality experiment.
