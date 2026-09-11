# Five-block tiled CDRM BF16: implementation and preliminary validation plan

Status: implementation and bounded validation completed on 2026-09-07 at a
PR review point. See [results and qualified disposition](reports/cdrm-tiled-bf16/results.md);
the frozen numerical failures remain visible. The implementation is opt-in.
Execution lineage: `.runtime/cdrm-tiled-bf16/20260907T191403Z/`.
It follows the completed [tiled R3 investigation](reports/r3-bf16-tiled-resolution/results.md).

## Architecture and meaning

CDRM means **Cross-Depth Read-Conditioned Recurrent Memory**. An ordinary causal
Transformer produces an early and a deeper preview. A side-memory scan reads
earlier records, combines that read with the current deep-conditioned candidate,
writes a record for subsequent tokens, and supplies a correction to the deeper
preview. Its stored records therefore depend on earlier reads as well as current
deep information.

Use five ordinary blocks with **zero-based indices 0–4**. One fabric connects
early site 1 and late site 3. Blocks 0–3 execute once; the side scan shares block
1's canonical parameters; the bridged state enters ordinary block 4. The tiling
belongs to the side scan. Both attachment sites remain ordinary blocks.

Let p_t and d_t denote the ordinary outputs after blocks 1 and 3. For each token:

\[
c_t=p_t+\epsilon A_d(N_d(d_t-p_t)),
\qquad(q_t,k_t^{tmp},v_t^{tmp})=\operatorname{Pre}_1(p_t),
\]
\[
a_t=\operatorname{Read}_1\left(q_t,
\{M_i:i<t\}\cup\{(k_t^{tmp},v_t^{tmp})\}\right),
\]
\[
\hat m_t=\operatorname{Post}_1(c_t,a_t),\qquad
m_t=(1-\rho)p_t+\rho\hat m_t,\qquad M_t=\operatorname{KV}_1(m_t),
\]
\[
z_t=d_t+\lambda A_b(N_b(\hat m_t-p_t)).
\]

Block 4 receives z. Pre includes the owner's learned pre-normalization, QKV
projection and Q/K normalization; Read includes scaled softmax attention,
absolute-position ALiBi and the attention-output projection; Post is the
owner's residual/MLP path; KV applies the persistent-record normalization and
projection path. Each operation executes once in its stated role. N_d and N_b
are the existing stateless RMS normalizers with epsilon 1e-6. The two bias-free
D-to-D adapters are the only additional parameter owners and start nonzero.

Queries and temporary records use p, while the residual candidate uses c. The
current read can see its temporary pair; its permanent write becomes visible
only to later tokens. The bridge uses proposed state hat_m and subtracts p.
Memory resets for each independent example/forward. Lambda zero gives the SEQ
bypass; rho zero generally leaves an active side computation.

Use epsilon=0.1, rho=1 and lambda=0.01. Initially support **rho=1** in tiled
production; the generalized equations describe the existing architecture and
reference controls, without requiring all controls in the optimized backend.

## Implementation boundary

1. Add explicit tiled CDRM and BF16-with-FP32-state options. Preserve the strict
   naive FP32 oracle, canonical checkpoint names, unique optimizer ownership,
   and the supported SEQ/R3 paths. Keep preview replacements empty. Reject
   unsupported optimized settings explicitly.
2. Keep candidate construction, Q/temporary-KV preprocessing and the bridge in
   ordinary autograd. Implement a CDRM scan returning hat_m, using R3's tiled
   attention schedule and tested primitives. Time-dependent reads/writes still
   require ordered processing; tiling batches their attention contributions.
3. Give the scan its own backward, accepting explicit tensor inputs and unique
   canonical owner parameters. Return their gradients so outer autograd sums
   preview, preprocessing and side contributions exactly once. Fused QKV slices
   remain views of one parameter. Avoid copying the R3 function's nested
   parameter-backward side effects.
4. At rho=1, reverse-time processing must combine direct bridge credit with
   future-reader credit before differentiating Post:

   \[
   \bar{\hat m}_t=\bar{\hat m}^{direct}_t+
   J_{\operatorname{KV}_1}(\hat m_t)^T\bar M_t^{future}.
   \]

   Propagate that total through the candidate and attention read, then into
   earlier permanent records. Batch dense parameter-gradient contractions where
   supported by the dependencies; preserve the forward operands and cast
   boundaries during replay.
5. BF16 handles dense operations and projected storage. Keep master parameters,
   differences d-p and hat_m-p, adapter/learned normalization arithmetic,
   softmax and recurrent accumulators, temporal adjoints, residuals, gate
   arithmetic and optimizer state FP32. Promote adapter outputs before gain
   multiplication/residual addition. Specify casts after Q/K normalization;
   merely removing today's FP32 guards would not define a correct policy.

## Bounded validation sequence

**First establish tiled FP32 correctness.** Reuse the independent tiny FP64
oracle and the ordinary-autograd CDRM reference. Start B1–2, D16–32, with T1,5,16,17
to cover terminal writes and irregular tile boundaries. Compare proposed states,
records, bridge outputs, both independent preview-input gradients, every shared
parameter and both adapters. Include causal perturbations, fresh memory per
example, read-conditioned writes, earlier-deep-preview temporal credit, lambda
zero, shared-owner gradient summation, and optimizer/checkpoint ownership.
Check ordinary-autograd differentiation interfaces as well as training backward.
An unused terminal permanent write is allowed to have no consumer gradient.

**Then validate BF16 locally.** Hold side inputs and incoming output gradients
fixed. Examine hat_m directly and isolated side gradients before lambda scaling,
so lambda=0.01 cannot hide a defective branch. Check exact intended dependency
paths, replay consistency, fixed-forward power-of-two gradient scaling, and
bounded dense contractions against matched-operand FP32/FP64 diagnostics.
BF16 acceptance targets tiled FP32; naive BF16 is optional diagnostic evidence.

**Finally check the actual small model.** Preserve the established D128/H16,
MLP512/GELU, full-MHA learned normalization/QK normalization, ALiBi and no-dropout
profile, changing depth to five. Use selective copying V16/T256 with 96 copied
tokens and the existing generator/answer-mask loss semantics. Run B2 diagnosis,
then physical B64 on the H100 if it fits. Use all three primary arms:

- naive CDRM FP32 versus tiled CDRM FP32: implementation check;
- tiled CDRM BF16 versus tiled CDRM FP32: precision suitability;
- tiny independent FP64 oracle: composition check and discrepancy localization.

Freeze configuration, cast/reduction/compiler policy, initial weights, data and
criteria before fresh confirmation. Reuse the previous FP32-centered BF16
engineering screens as initial proposed B64 guards: approximately 1.56% global
gradient error, 3.13% per tensor, 6.25% maximum/reference-maximum, with the same
explicit absolute floors, plus 0.01-nat same-state CE difference. Retain tails,
near-zero diagnostics, and adapter/side-only summaries. Apply strict semantic
checks independently of aggregate budgets. Investigate failed screens; do not
silently move thresholds or require exact BF16/FP32 equality.

Confirm at initialization and a checkpoint from the new five-block run, using
identical starting weights/moments for each comparison and separately frozen
fresh minibatches. Existing six-block checkpoints are not interchangeable with
this model. Record clipping and Adam deltas; retain the previous initial-update
cosine and trained-moment relative-update guards as prospective screens.

## Preliminary operation and PR pause

After the initial numerical checks are satisfactorily resolved, run a bounded paired
100-update tiled FP32/BF16 smoke test from identical initialization and data
order. Evaluate loss, answer-token accuracy and whole-sequence exact match on
development data; monitor adapter/shared-weight gradients and correction ratios.
A separate tiny repeated-batch check should demonstrate loss reduction and
active learning paths, without becoming a long synthetic research comparison.
Use a midpoint checkpoint for the same-state numerical comparison and a BF16
resume check with the retained shared Inductor cache. Keep the existing optimizer
and schedule semantics explicit. No long-run learning-equivalence claim follows.

Run a short warmed-up benchmark of complete updates, measuring both time and
peak memory; BF16 need not be faster. Final confirmation uses the compiled
execution without diagnostic hooks. Test changed config/model integration and
existing CDRM/SEQ/R3 behavior where affected. All GPU execution uses the project
container with a successful in-container GPU check. Log graphable runs online
under taylorbollman in a new CDRM tiled-validation W&B project, and retain source,
data identities, checkpoints, reports and required compiler cache under
gs://fast-chunks/cdrm-w-latent/ in a new lineage.

The review point is a working five-block tiled CDRM, demonstrated local and
end-to-end gradient behavior, a bounded optimizer/training/recovery check, and
measured cost with explicit limitations. General rho, multiple fabrics, packing,
cached generation, gradient accumulation, distributed execution and a renewed
SEQ-versus-CDRM research pilot remain separate scope decisions.
