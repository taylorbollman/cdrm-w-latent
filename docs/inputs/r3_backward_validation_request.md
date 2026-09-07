# Request: resolve the FP32 recurrent-backward uncertainty

Please perform a bounded numerical investigation before further substantial recurrent research training. Keep the model in the existing OLMo-based project. This task is validation and, if necessary, a focused correctness fix; it is not a new architecture or training sweep.

## Context and question

Stage B compared SEQ with R3 at `rho=1`: 12 blocks, width 256, four heads, sequence length 128, physical batch 64, FP32 parameters and computation, and no gradient accumulation. Only block index 3 was recurrent. There was no CDRM or block-8-to-block-3 pathway.

The report says the selected tiled path passed its declared unit-direction fixtures, while raw unnormalized-cotangent failures remain. Determine whether these reflect scale-sensitive acceptance criteria, ordinary floating-point error, or an incorrect backward calculation. Do not assume that they invalidate Stage B, or dismiss them solely because normalized fixtures passed.

## Requested work

1. **Inspect and reproduce the existing failure.** Reuse the recorded fixture, checkpoint/configuration, compiler profile and tolerances where available. Explain what “unit-direction” meant and exactly which gradients failed. Separate BF16 and accumulation failures from the FP32, no-accumulation path actually used in Stage B. Reuse adequate existing evidence instead of rerunning it unnecessarily.

2. **Compare naïve autograd and tiled backward under the actual task loss.** Use identical weights, examples, target alignment, answer masks and loss reduction. Start with tiny fixtures, then a small-microbatch end-to-end model at the Stage B width/depth and T=128. Include initialization and one retained trained checkpoint. Prioritize the task/fixture most relevant to the recorded failure; expand only if a concrete uncertainty remains. Do not silently substitute a mean-logit surrogate for masked cross-entropy, change normalization, detach recurrent history, or disable the tiled code path being assessed.

   Compare logits, masked CE, gradients with respect to the recurrent block's input, and **every intended trainable parameter gradient**, including missing-versus-zero gradients. Map parameters explicitly if registration differs. Show that nontrivial gradients traverse earlier persistent writes. Record effective precision, autocast, TF32 and compiler settings; use strict FP32 settings for the reference comparison and identify any difference from Stage B's execution settings.

3. **Check backward scaling independently of optimizer behavior.** At a fixed forward evaluation, let G(v) be the input/parameter gradients produced by output cotangent v. Use the original failing cotangent and a small set of representative scales, for example alpha in {1/32, 1, 32}. Check both naïve-versus-tiled agreement and G(alpha*v) approximately equals alpha*G(v). Include the actual CE-induced gradient scale. Apply this check before gradient clipping or optimizer transforms; those need not scale linearly.

   Report maximum absolute errors, relative L2 errors with a documented denominator floor, gradient/reference norms, and the worst offending tensors. Near-zero reference entries need separate interpretation. Do not loosen thresholds merely until tests pass: justify scale-aware tolerances and classify each remaining failure. A small FP64 reference or targeted finite-difference/directional check is a fallback only if needed to resolve a discrepancy.

4. **Compare an actual optimizer update.** Start the two implementations from identical model and Adam/AdamW state, including matching step counters. Use the configured Stage B hyperparameters and clipping policy. Compare gradients before clipping, clipping norms/coefficients if applicable, parameter updates, and optimizer moments. Report update differences as well as post-step weight differences, since unchanged weight magnitude can hide an erroneous small update. Do not demand bitwise identity between different numerical evaluation orders.

5. **Conclude and stop.** Use one GPU and bounded fixtures, without a research-training sweep. Do not make BF16, accumulation, T=512, distributed training, or CDRM support prerequisites for clearing the existing FP32/no-accumulation path. Those remain unsupported for new research runs until separately validated.

## Deliverables and decision

- A concise report with exact commands, revisions, fixture/checkpoint identifiers, precision/compiler settings, tolerances, numerical results and any patch.
- An explicit conclusion: **FP32/no-accumulation path cleared for the tested regime**, **backward defect found and fixed**, or **unresolved**. State the supported shape/configuration scope.
- If a defect is found, assess whether it affected the actual Stage B loss/backward path. Preserve the original artifacts and annotate their status; use a new lineage for any subsequent rerun. A failure confined to unused BF16 or accumulation must not be presented as proof that Stage B FP32 results are invalid.

## Architecture reminder

The interpolation parameter rho is our proposed warm-start/ablation control. The intended Oncescu-style output-derived write corresponds to rho=1. It is not the switch for cross-depth input.

For **R3**, let x_t be the input to block 3 (the output of block 2). Form the query and temporary K/V from x_t. Read earlier permanent records plus the current temporary pair, producing a_t. Then

\[
z_t=\operatorname{Post3}(x_t,a_t),\qquad
w_t=(1-\rho)x_t+\rho z_t,\qquad
(k_t,v_t)=\operatorname{KV3}(w_t).
\]

Pass z_t upward to block 4 regardless of rho. The permanent record is written only after the current read. At rho=1 it stores projected processed output; at rho=0 it stores projected input. Rho=0 matches SEQ only under the validated compatible preprocessing/normalization contract, not for arbitrary upstream configurations.

For **CDRM**, run ordinary blocks 0–8 first and save their post-block outputs p3_t and p8_t. Form the query and current temporary pair from p3_t; read the side memory to obtain a_t. Then

\[
c_t=p3_t+\epsilon A_d\!\left(N_d(p8_t-p3_t)\right),
\]
\[
\hat m_t=\operatorname{Post3}(c_t,a_t),\qquad
m_t=(1-\rho)p3_t+\rho\hat m_t,\qquad
M_t=\operatorname{KV3}(m_t),
\]
\[
v8_t=p8_t+\lambda A_b\!\left(N_b(\hat m_t-p3_t)\right).
\]

M_t is the new permanent K/V pair, unavailable to the current read. Feed v8 to ordinary blocks 9–11. A_d and A_b are adapters; N_d and N_b are normalizers; Post3 includes the selected residual/MLP computation and KV3 includes the validated projection/normalization policy.

- **epsilon:** explicit deep-source contribution to the writer.
- **rho:** output-derived persistent-storage strength.
- **lambda:** bridge strength into the post-block-8 representation.

The bridge deliberately uses hat_m, not interpolated m. Thus rho=0 removes recursive read-conditioned writes while leaving the extra read/writer/bridge computation active. Lambda=0 restores SEQ logits when the backbone is otherwise ordinary and execution is deterministic. CDRM executes its side scan after the preview; it does not retroactively change that preview or replay blocks 4–8.
