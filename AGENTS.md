# Workspace

Use `/home/taylorbollman/cdrm-w-latent` as the active project. The old projects
are unrelated archives and must not be searched unless explicitly requested.
Do not print credentials or environment-file values.

# Container execution

Never run CUDA, training, GPU evaluation or profiling in the host shell and
never silently fall back to CPU. Bootstrap GPU instances with
`bash /home/taylorbollman/start.sh`; CPU instances with Local SSD use
`bash /home/taylorbollman/start_cpu.sh`.

After bootstrap, use `bash /home/taylorbollman/cdrm-w-latent/scripts/docker_shell.sh`.
For a single command, append `bash -lc '<command>'`. The container working
directory must be `/workspace/cdrm-w-latent`. Before GPU work, verify that the
command is inside the container and that `nvidia-smi` succeeds there.
Use `CDRM_DOCKER_GPUS=none` explicitly for CPU container work.

Add dependencies to `docker/requirements-docker.txt` and rebuild with
`scripts/docker_build.sh`. Keep `.env` and `.docker-home` out of Git.

# Experiment tracking

For future training and evaluation runs with graphable metrics, log online to
Weights & Biases under the `taylorbollman` entity. The user authorizes creating
appropriately named projects and runs there. Use the credentials in `.env`
without printing them, and include the project or run URLs in progress updates
and results. Keep the existing local records and GCS artifact retention alongside
W&B tracking.

# Pretrained model handoff

For pretrained OLMo / RT / FBT / NextLat work, first read
`docs/fbt-rt-nextlat-handoff.md`, then
`docs/fbt-rt-nextlat-research-plan-v4.md`. Original OLMo-1B at step 60,000
(approximately 252B tokens) is the selected primary model; its native checkpoint,
source candidate and tokenizer pins are in the handoff and selection audit.
O1 native ordinary fidelity/sequential RT and O2 native tiled execution/backward
are complete, with bounded GPU evidence and a documented raw-input roundoff
qualification. O3 language-model NextLat, optimizer/save-resume and bounded
single-H100 profiling are complete. O4
matched Python continuation is complete; read its results and assessment. The
user asked to assess and continue: O5a bounded FBT correctness now passes;
O5b matched ordinary-versus-FBT-only learning and its endpoint diagnostic are
complete and assessed. O5c fusion-only code versus code/general-text adaptation
is also complete: mixed training repairs the measured retention deficit while
the native backbone remains unchanged. O5d is complete:
fixed-weight finite K2/K3/K4 versus exact sequential feedback confirms that the
repair survives teacher-forced online execution through512-token contexts.
Read the current handoff and O5d assessment for retained evidence and the
ordinary additional-training control. O5e is complete: shared-source ordinary
continuation on the exact O5c mixed plan beats both fusion-only endpoints in
code/WikiText NLL, with a140-fold trainable-capacity qualification. Read its
assessment/results and current handoff. All four full checkpoints are retained;
no GPU job or further learning is queued. The user has reset the next priority
to functionality, bounded numerical health, integration, parameter/throughput/
FLOP accounting, Q/K-normalization assessment, native tiled-RT/Flash efficiency
and multi-GPU execution before quality comparisons. The user approved V4,
including early profiling. F1 is complete:18actual-checkpoint cases, two exact
BF16 recovery checks, short online cache parity and244scoped CPU tests pass.
Read its assessment/results and handoff. Early B1/T512 profiling identifies eager
RT scheduling/replay/launch overhead as a leading bottleneck; large clipped
startup gradients motivated F2, now complete and assessed. Ordinary-only
checkpointing passes exact BF16 complete-update parity and allows B128/T512:
RT20.2kinputtokens/s at41.4GiB, combined10.5k/s at51.9GiB. Native Q/K math stays.
Deterministic Flash SDPA gives exact B8/T512 native-stack graph checks; default
cuDNN has separately measured eager-repeat variability. F3's canonical combined
CUDA-graph integration is now complete: seven actual-checkpoint cases have exact
loss/gradient/full-Adam parity; six paired B32/64/128 capacity checks and 267 scoped
CPU tests pass. Read F3 assessment/results/usage and the current handoff.
Graphs capture forward/loss/backward with ordinary checkpointing; clipping,
AdamW and scheduler remain outside. At T512, graph RT B128 reaches24.7kinputtokens/s
at42.1GiB; combinedK2+NextLat B64 reaches10.4k/s at40.8GiB, B12810.8k/s at58.2GiB.
Use B64 for common development checks, preserving memory headroom with96.4% of
combined B128 throughput. Peak reserved setup and current postcapture memory
are separate. Deterministic ordinary Flash/no-autocast-cache settings differ
from F2 timings. Only RT layer0 is selected.
F3b's bounded forward prototype is now complete: per-invocation weight-cast
reuse plus a Triton historical attention tile, both opt-in. Read F3b assessment,
results, usage, FA4 environment note and resource accounting. Twelve GPU reports
pass, including 48 frozen tiles, 12 tiny blocks, four native correctness cases
and four capacity cases; 307 scoped CPU tests pass. Fused actual B8/T512 gradient
relative L2 versus original BF16 is 0.00354 for RT and 0.01007 for combined;
same-candidate graph and complete AdamW update comparisons are exact. T512 graph
RT B128 now reaches26.1kinputtokens/s at42.1GiB and combined B64 10.9k/s at40.8GiB,
about5.7%/4.4% above reference. Only layer0 is RT. The installed FA4/CuTE wheel
works with the explicit installed-source launcher selector; RT uses Triton, not
FA4. Analytic parameter/FLOP cards cover all eight combinations, with broader
runtime coverage still pending. F3c historical backward fusion is also complete:
independent opt-in backward_tile_backend, seven GPU reports/78gates and432 scoped
CPU tests pass. RT B8 initial losses/gradients equal F3b control bitwise; combined
initial losses equal bitwise and global gradient relative L2 is0.00139246. Both
have exact same-candidate graph/full-Adam parity. T512 RT B128 reaches26.48k
inputtokens/s; combined B64 10.97k/s, about1.45%/1.06% additional gains with
unchanged allocated peaks. Read F3c assessment/results/usage and current handoff.
F3d bounded backward workspace is complete: opt-in backward_memory="recompute"
retains row statistics and recomputes attention/adjoint tiles, preserving BF16
whole-product rounding, temporary self and query/prefix gradients. Materialized
stays default/reference. Ten final reports pass 90 gates;428 scoped CPU tests pass.
RT B8/T512 and combined B8/T512/B2T1024 initial gradient global L2 versus F3c is
.00089227/.00237615/.00224863; same-candidate graph/full-Adam parity is exact.
Fresh capacity saves 1.75–3.50 GiB with 0.39–0.74% lower throughput: RT B128/T512
26.26k/s at 38.60 GiB; combined B64/T512 10.93k/s at 39.09 GiB; B16/T1024 8.58k/s
at 31.29 GiB. Isolated reconstruction memory is approximately linear throughT2048;
full-model memory is not. Read F3d assessment/results/usage and handoff. One
failed capture attempt (fixed scalar indexing) and an earlier probe are retained
separately from the final10-run selection. Native Q/K math remains unchanged.
No GPU or quality run is queued. Next is broader optimized RT layer/context
integration and F4 runtime cards. Only RT layer 0 is selected in actual F3d
full-model checks. Forward rectangles above 256 still use eager fallback;
recompute backward is fused through 2048. Native T2048 complete updates, more/all
RT layers, padded graphs, graph recovery/accumulation and genuine multi-GPU remain
untested. Direct Triton CPU observer attribution undercounts kernels; use device
traces/full-step timings. Two-GPU checks need a second GPU; one H100 is exposed.
Read the handoff
for current authorization and evidence. Do not infer long-run
authorization from platform work. Completed OpenELM code/results are historical
reference evidence; do not resume its superseded next milestone by default.

# RT numerical handoff

For future changes to the base Recurrent Transformer or its numerical tests,
read `docs/rt-numerical-handoff.md` first. It records the completed precision
baseline, reusable validation methods, harness adaptation constraints and
retained artifacts. The historical precision lineage is closed; future
experiments should have new output directories and an explicit scope.

# A5 experiment handoff

For the two-layer A5 experiment, read `docs/rt-a5-handoff.md` and
`docs/rt-a5-usage.md`. The first paired 10,000-update development pilot is
complete. The user chose full FP32 and minimal bounded correctness checks;
do not automatically restart mixed-precision or compiler studies. Final
confirmation remains unevaluated.
