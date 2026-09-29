# Paired fusion-only continuation through update144

**Both precision paths completed successfully, with closely matched held-out
losses. Their learned updates diverged measurably, so this is evidence for
bounded workability, not trajectory equivalence.** A fresh BF16 process also
reproduced updates137–144 bitwise from checkpoint136, including gradients,
Adam, counters, RNG and the common FP32 evaluations.

The [continuation protocol](continuation-protocol.md) started both runs from the
same complete FP32 startup128 boundary. Each used the next sixteen original
training selections,128..143: another **131,072 CE targets** and133,978 valid
input tokens, physical B8/T128. The model retained K4 feedback, beta1 and
jitter0.02, isolated documents and CE-only training. Only the two fusion matrices
were trainable; the pretrained backbone, tied readout, seeded predictor and
saved output scale stayed frozen. Autograd still traversed the later backbone
passes. Adam moments and the completed scheduler were inherited, with fixed
LR1e-4, beta(.9,.95), epsilon1e-8, decay0.1 and clipping1.

One path used full FP32/math attention; the other used ordinary Flash attention
with BF16 forward autocast and FP32 masters. Both were deterministic with TF32
off. Checkpoints at128,136,144 were saved and retained. The original startup
loader and model sources were unchanged; continuation checkpoints have their
own strict source/configuration identity.

Both trajectories were evaluated on the same four held-out T128 documents
under a **common FP32/math path**, without gradients or changing RNG, modes,
weights, flags, optimizer or cursor:

| Completed update | FP32 trajectory CE | BF16 trajectory CE | BF16 minus FP32 |
|---|---:|---:|---:|
| 128 | 5.253002737 | 5.253002737 | 0 |
| 136 | 5.170248437 | 5.170717495 | 0.000469058 |
| 144 | 5.137168103 | 5.137339194 | **0.000171091** |

Values are nats per CE target. These are small fixed-fixture readiness
observations, not a model-quality comparison. Across matched training selections,
the signed BF16-minus-FP32 CE differences ranged from −0.0004996 to+0.0007418.
All losses, gradient norms and update norms remained finite; no operation or
state-preservation gate failed.

Compact checkpoint comparisons also show why similar losses should not be
called identical optimization. At144, the norm of the BF16-minus-FP32 fusion
weights is **0.233800**. This is **0.36445%** of the full FP32 fusion-weight norm,
but **18.4122%** of the cumulative FP32 change since128. The two cumulative
change vectors have cosine **0.983001**. At136, their relative difference was
14.4551%. Adam's first and second moments at144 differ by **9.9456%** and
**2.3627%**, respectively.

After the first new update, these comparisons include feedback from different
weights and Adam histories; they are not same-state local rounding tests.
Clipping and shared pre-128 optimizer history also affect the actual updates.
The measured divergence should stay visible, but it is not by itself evidence
of training failure: both short trajectories remained finite and their common
held-out losses stayed close. Conversely, the small held-out loss gap cannot
establish that later or broader training would be equivalent.

There were **40 physical optimizer calls**:16 FP32,16 BF16 and8 replayed BF16,
covering16 unique new training selections. The comparison helper independently
checked55 report/checkpoint conditions, including identical original boundaries,
all paired inputs/noise/LRs/counts, preserved scale, inherited clocks and every
replayed update/evaluation/final boundary. No new training was performed by this
CPU analysis.

The next practical step is the separately bounded full-parameter NFR combined
update test. This fusion-only result does not clear backbone unfreezing,
NextLat losses, RT, packing, prepared loss layouts, CUDA-graph training or DDP.
Those are distinct execution and optimization scopes.

Evidence paths under `.runtime/olmo-fusion-startup/`:

- `continue-fp32-01/report.json`, SHA256 `538639fc63028f022c549cf7af9e7dbda809a77c956c24a64692a5f6186a333d`; [W&B17ty29w3](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/17ty29w3).
- `continue-bf16-01/report.json`, SHA256 `a01e9d102fc4e47ec345bf29f3625cc0f48fa671697f4ead5d91b65ae66dc4c3`; [W&B2l8pyh82](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/2l8pyh82).
- `continue-bf16-replay-01/report.json`, SHA256 `de4cccaf6cc7e3868c382672111efdc7e64626ac5997f24470ccad4b1ffa1dad`.
- `continuation-comparison-02/report.json`, SHA256 `b7b0667fc9dd919f25767a7e261b04daefe08dd9c013f617a2be0e48c7d635a2`, created by `scripts/olmo_fusion_startup_continuation_compare.py`. The earlier `comparison-01` excluded the then-pending replay and remains separate evidence.

The continuation's73 affected CPU tests passed before GPU launch, including
exact gradient oracles, fresh-model checkpoint recovery, malformed import
rejection and common-FP32 evaluation restoration. The CPU comparison report is
a post-run geometry/consistency analysis and makes no additional GPU claim.
