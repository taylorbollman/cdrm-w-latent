# Native RT: single-H100 closeout and two-GPU readiness

Updated 2026-09-25 after interruption. The user approved this execution
order after reviewing the updated plan. It refines the
[large-batch plan](native-rt-large-batch-plan.md) and V4 section 8. It does not
change the frozen prospective benchmark protocol or authorize quality training.

## Verified current position

The final large-batch queue is complete:22reports,16passed,three numerical
failures and three OOMs, totaling153physicaloptimizerupdates. Reverse repeats
and two untimed profiles are complete; see the [final results](reports/olmo-rt-large-batch/results.md).
FA4 is not adopted: it does not move the RT B256 capture boundary, and its
combined integration fails output/gradient budgets for a modest1GiB current
reserved-memory saving. A1/A2 below are completed measurement scopes with verified GCS retention.
A3 is also complete: [one-GPU preparation results](reports/olmo-single-gpu-readiness/results.md)
record70passing GPU gates/20updates for RT/combined accumulation, complete
optimizer comparisons and checkpoint/graph reconstruction. Keep the original
zero-update eager mask-dispatch failure and its checked opt-in fix. Both recovery
branches rebuild graphs; accumulation uses same-shaped eager microbatches.
No real DDP/NCCL, distributed graphs/recovery, sharding or scaling is established.
The current action is C on two actual GPUs; optional B does not block it.
Only one H100 is exposed now, idle, with no further GPU job queued.

The actual step-60000 OLMo-1B has 16 layers, width2048, 16 attention heads and
SwiGLU8192 per branch. These measurements select native RT at indices0/15.
Combined means K2 FBT (ordinary bootstrap, then a feedback pass with RT) plus
NextLat. No changes to Q/K normalization. All rows use T512, full valid CE,
BF16 mixed with FP32 model/gradient/Adam state, ordinary activation
checkpointing and CUDA-graph forward/loss/backward. Rates include the complete
optimizer update, and count original input tokens once.

| Candidate and physical batch | Input tokens/s | Seconds/update | Setup peak reserved GiB | Sampled free after capture GiB |
| --- | ---: | ---: | ---: | ---: |
| RT B128 | 28,008.6 | 2.340 | 48.377 | 29.684 |
| RT B192 | 29,743.7 | 3.305 | 62.926 | 17.145 |
| Combined B64 | 11,707.6 | 2.799 | 41.713 | 36.809 |
| Combined B128 | 12,410.9 | 5.281 | 65.971 | 12.877 |

The table retains first-run values. Reverse fresh-process repeats measured
RT B12828,022.56/s, RT B19229,748.39/s, combined B6411,698.26/s and combined
B12812,415.46/s, with identical setup/current reserved memory at each shape.
Minimum free memory sampled across phase boundaries was15.565GiB for RT B192
and12.520GiB for combined B128; sampling is not continuous peak monitoring.
Both recommended candidate points pass all five large-shape gates and eight
updates. Their B8 cross-configuration and own graph/full-Adam checks also pass.

The candidate uses native ordinary RoPE, rounded compiled ordinary SwiGLU,
fused AdamW and deterministic Flash SDPA. Native RT uses its validated Triton
tiles, recompute backward, reused casts/RoPE tables and K/V-only writes. The
Dao ordinary RoPE integration retained a loss-only screen failure in RT, so it
is not the default candidate for this scope. Historical native/author BF16
qualifications remain open; no new broad numerical campaign is implied.

RT B256 completes three eager updates but fails during graph capture in native
RT's reconstructed full-sequence parameter VJP. The earlier B192 OOM instead
occurred during eager validation beside a live graph. Moving eager references
before capture and releasing the graph before final eager validation fixed that
measurement overlap, without dropping any exact checks. These two failures must
remain distinguished. B512 has not been executed or established feasible.

## A. Finish on one H100

1. **Close the current capacity milestone.** Repeat RT B192/B128 and combined
   B128/B64 in reverse order, with one useful-batch profile of each mode. Retain
   full-update and graph timing, setup/current memory, parameters, matrix-FLOP
   ledgers and failed attempts. Keep profiles outside timed samples; generic
   GEMMs must not be assigned exclusively to RT. Finalize report/plots, test
   ledger, W&B links and verified GCS retention.
2. **Finish the already-triggered FA4 question.** Small B8 integration for
   `fa4-native` versus `compiled-native`, then ordinary FA4 versus Flash SDPA
   at RT B192 and the failed B256 boundary, and combined B128. A single combined
   B160 boundary probe is optional if the matched memory measurements leave
   the practical recommendation uncertain. Native RT attention remains Triton.
   Preserve loss-screen failures and operational gates; a finite numerical-only
   failure can support qualified timing, not automatic adoption. Stop if FA4
   offers no useful memory or full-step advantage. Do not expand to a full grid.
3. **Prepare recoverable distributed training on the existing machine.** Add a
   forward adapter that routes the canonical objective through DDP's future
   wrapper, preserving native semantics in single-process mode. Build separate
   CE/latent/KL global-count normalization tests, including unequal/zero local
   counts. Establish a one-GPU accumulated-gradient/full-update reference and
   a bounded save/reload/recapture continuation for RT and combined with the
   selected execution/optimizer flags, RNG and data cursor. Reuse prior recovery
   machinery; this is an integration check, not another extensive precision
   investigation. Tiny CPU tests and one-process GPU checks prepare the code;
   they do not validate distributed collectives.

Section A is complete. Its serial measurements and CPU implementation did not
require a second GPU. No additional single-GPU optimization is required before
beginning the actual two-GPU checks below.

## B. Optional single-GPU optimization, not a two-GPU prerequisite

The measured memory failure makes a bounded native RT backward-lifetime audit
more relevant than another broad fused-activation search. Consider freeing
unneeded projection/adjoint references, then chunking the batched parameter VJP
to bound full-sequence MLP intermediates. Preserve causal forward recurrence and
batched weight-gradient efficiency; chunked reductions may change BF16 rounding
and require a small existing-budget comparison plus own full-update checks.

For speed, use the profile to decide whether to compile a contiguous RT
finish/writer region. CUDA graphs already remove much host launch overhead;
measure added device-time benefit rather than assuming it. Treat memory and
leaf compilation as separate interventions. Neither achieving physical B512,
implementing both optimizations, nor resolving every historical numerical
qualification is a gate for starting two-GPU work.

Additional layers/longer contexts and bounded Q/K health probes remain useful
on one GPU when tied to the next intended configuration. Keep native Q/K math
unless a new concrete activation/gradient concern warrants changing it.

## C. When to move to two GPUs

Move after A establishes a frozen working configuration, repeatable resource
points, the global-loss reference and recoverable state. We already have enough
single-GPU capacity to justify this move. If the second GPU becomes available
earlier, eager distributed correctness can start once the forward adapter and
loss reference are ready; optional kernel work need not delay it.

Prefer two H10080GB devices in one machine with a fast direct interconnect.
Inventory actual topology, peer access, driver/CUDA/PyTorch/NCCL versions and
measure collectives before interpreting scaling. Do not infer topology merely
from an H100 instance label. CPU collectives or two processes sharing one GPU
do not establish two-GPU readiness.

1. **Eager DDP correctness first.** Tiny eight-mode RT/FBT/NextLat coverage,
   followed by actual-model ordinary, RT and combined updates. Match identical
   global examples against one-GPU full-batch/accumulation references; compare
   numerically rather than requiring bitwise equality across different reduction
   groupings. Test unequal objective counts, zero targets on one rank, tied
   readout/shared FBT parameters, inactive branches and coordinated nonfinite
   handling. For default DDP averaging, each local objective sum is scaled by
   `world_size / global_valid_count`, separately per objective; handle globally
   empty terms consistently. Clip once after the completed gradient reduction.
   Explicitly test a predictor used in an earlier `no_sync` microbatch but
   unused in the final synchronized microbatch, with different local usage on
   the other rank. Simulated CPU rank averaging cannot validate the reducer's
   unused-parameter bookkeeping for that pattern.
2. **Recovery and CUDA graphs.** Check a short same-world-size save/resume,
   per-rank RNG/data cursors, global metrics and rank-zero artifact/W&B ownership.
   Then add graph execution and validate persistent gradient addresses, reducer
   hooks, repeated shared-backbone use and identical collective ordering. The
   current plan calls the backbone directly and owns gradient buffers: wrapping
   the object alone is insufficient. Keep an eager distributed diagnostic
   reference. If full capture conflicts with reduction, evaluate a separately
   validated graph-compute/explicit-reduction boundary before broader redesign.
   Use the installed PyTorch/NCCL capture contract and sufficient DDP-enabled
   warmup (current upstream guidance requires at least11 eager DDP iterations).
   Audit any gradient-as-bucket-view option before fixing capture addresses.
   Native RT's local `autograd.grad` calls operate inside its custom backward,
   which returns parameter gradients to outer autograd; verify reducer hooks
   at that outer boundary rather than assuming local VJPs prove compatibility.
3. **Measure two different scaling questions.** First equal global batch:
   e.g. one GPU B128 versus two GPUs B64 each on identical examples. Then the
   largest comfortable per-rank batches, starting conservatively and allowing
   for communication buffers. Report tokens/sec total and per GPU, GPU-seconds
   per token, communication time, per-rank setup/current memory and parameter/
   FLOP ledgers. RT B192 per rank (global384) and combined B128 per rank
   (global256) are candidate targets, not proven distributed capacities.
4. **ZeRO-1, then conditional ZeRO-2.** After DDP works, test optimizer-state
   sharding; add gradient sharding only if the extra memory supports a useful
   local batch or another required setting. Keep parameters replicated and
   avoid CPU/NVMe offload initially. Preserve FP32 parameter/gradient/moment and
   loss policies; do not silently combine optimizer/precision/sharding changes.
   Current gradient-buffer ownership makes ZeRO-2 a larger integration change.
   Stage3/FSDP/tensor/pipeline parallelism remain later unless a concrete limit
   makes them necessary.

DDP replicates model state and does not pool the two GPUs' memory. Two-way
optimizer-moment sharding alone would theoretically save about4.4GiB per rank
for RT or4.7GiB for combined; additionally sharding current FP32 gradients could
save about2.2/2.4GiB. These estimates use the current inventory and exclude new
communication/capture buffers, redundant master copies and allocator overhead.
They are not promises of net savings or B512 feasibility. Verify the actual
framework's precision/storage policy before using these estimates.

In particular, global B512 on two GPUs is physical B256 per GPU, which currently
fails RT graph capture on one GPU. DDP alone does not cure that. Global B512 via
accumulation also does not give a physical B512 RT matrix on either GPU. The
paper's physical-batch utilization argument remains distinct from global
optimizer-batch size.

## Review point and sources

Two-GPU readiness is complete when representative eager/graphed updates,
same-world-size recovery, correct aggregated metrics and measured resource
cards pass in the declared scope. Then choose short learning pilots or a longer
ordinary baseline using measured costs. Native RT is the selected backend;
FBT/NextLat remain independently configurable. No quality contest is a
prerequisite for this engineering milestone.

Primary implementation references, checked for this plan:
[PyTorch DDP](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html),
[CUDA graph/DDP constraints](https://docs.pytorch.org/docs/main/notes/cuda.html),
[DeepSpeed ZeRO stages](https://www.deepspeed.ai/tutorials/zero/).
These describe framework contracts, not proof that our installed versions and
custom backward are already compatible. Existing local evidence:
[capacity audit](reports/olmo-rt-large-batch/capacity-audit.md),
[live milestone record](reports/olmo-rt-large-batch/progress.md), and raw reports
under `.runtime/olmo-rt-large-batch/`.
