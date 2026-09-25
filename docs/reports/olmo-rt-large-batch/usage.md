# Native RT large-batch diagnostic usage

These commands run bounded execution probes, not quality training. Enter the
project container and verify its working directory/GPU before GPU commands.
Each run requires a new output directory and online W&B. Sources, external
dependencies and the prospective protocol are frozen into the output.

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_rt_large_batch.py --stage correctness --case rt \
   --arm compiled-native --batch-size 8 \
   --output-dir .runtime/olmo-rt-large-batch/NEW-CHECK'
```

Use `--case combined` for K2 FBT+RT+NextLat. Both modes select RT at indices0/15
inside the original 16-layer pretrained OLMo. K2 bootstrap stays ordinary.
`compiled-native` retains native RoPE, uses rounded compiled ordinary SwiGLU and
fused AdamW, and preserves the optimized native RT tiles/recompute path.
`control` retains eager ordinary SwiGLU/scalar Adam. `optimized` additionally
uses Dao ordinary RoPE and retains a failed RT loss-only integration screen;
do not treat that arm as the primary cleared candidate.

For capacity, replace `--stage correctness` with `--stage capacity` and select
a supported physical batch, initially64 or128, at T512. One batch is one
optimizer update; no gradient accumulation is used. `--profile` adds separate
untimed backward/full-step traces and a ninth physical optimizer update.
Without profiles, each successful capacity run performs three preparation and
five timed updates. Correctness performs three eager and three graph updates.

`--release-transient-cache` is an explicit graph-preparation option; consult the
frozen runtime/protocol for its exact phase placement. It releases unused cache,
not live model/optimizer/gradient storage. Do not mix its capacity points with
the default setup without labeling the difference. Memory reports distinguish
phase peaks, graph setup, validation and steady timing. A validation OOM must
not automatically be called an intrinsic graph-training capacity limit.

Use `--validation-order before-capture --release-transient-cache` for the
recommended capacity setup. This saves exact eager references on CPU before
capture, checks every graph replay against them, and releases the graph before
the terminal changed-weight eager check. The default `live-graph` validation
keeps the graph pool alongside new eager workspace and produced an avoidable
B192 validation OOM. Both orders retain all loss/gradient gates. Reordered B128
has identical update records and throughput to the old B128, with less unused
allocator reservation and no live eager/graph validation overlap. This is a
preparation change, not a reduction in the model's live arithmetic workspace.

Conditional FA4 check: use `--arm fa4-native --reference-arm compiled-native`
for small-batch correctness, then `--arm fa4-native` for matched capacity. This
changes ordinary attention only; native RT historical attention remains Triton.
Finite numerical-only failures can complete operational diagnosis with explicit
`--continue-after-compatibility-miss`, but still exit unsuccessfully and remain
failed. That flag never permits structural, nonfinite or operational failures.

FA4 currently fails both B8 cross-configuration screens: RT has a loss-only
failure, and combined additionally fails output/global-gradient/tensor budgets.
Passing capacity rows validate their own execution only. FA4 capacity results
remain qualified throughput/memory evidence; keep Flash SDPA as the working
ordinary backend. The summary attaches selected same-case/arm integration
evidence to capacity rows, and plots label uncleared integration explicitly.

The report helper requires an explicit completed-run selection and matching
per-run Git revisions. It validates frozen sources, protocol, dependencies,
fixtures, counts, gates and timing scopes before creating summaries/plots:

```bash
python scripts/summarize_olmo_rt_large_batch.py \
  --runtime-commit COMMIT --runs RUN1 RUN2 \
  --run-commit RUN1=OLDER-COMMIT --plot
```

Use the explicit CPU container for summarization/tests. Its `--retain` mode
verifies the selected evidence and existing native checkpoint reference before
uploading a create-only archive to `gs://fast-chunks`; it never uploads credentials.
Retained numerical failures and OOM attempts are part of the evidence selection.
