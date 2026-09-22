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
`docs/fbt-rt-nextlat-research-plan-v3.md`. Original OLMo-1B at step 60,000
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
no GPU job or further learning is queued. Do not resume completed diagnostics
or infer authorization for the proposed full-backbone FBT comparison.
Two-GPU correctness remains untested. Read the handoff
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
