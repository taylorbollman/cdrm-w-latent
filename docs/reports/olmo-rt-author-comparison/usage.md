# Reproducing the isolated author-backend comparison

Run from the project root through `bash scripts/docker_shell.sh`; CUDA commands
must execute inside the container with a working H100. No CPU fallback is
permitted. Read `protocol.md` and `author-port-audit.md` first. Use the native
step60000 checkpoint already retained by reference, and a fresh output directory
for each invocation. Small evidence belongs under the matching GCS prefix.

The new backend is an isolated experimental block function, not a model-wide
default or a supported replacement for the native FBT/NextLat stack. It requires
FP32 residuals/parameters, optionally BF16 autocast, full recurrence, equal Q/K/V
head counts, and no padding or cached prefix. It preserves packed native weights
and returns parameter gradients to outer autograd. Helpers compile lazily.

```bash
# Strict reference screen, followed by author graph/full-Adam checks.
bash scripts/docker_shell.sh python scripts/olmo_rt_author_compare.py \
  --stage verify --backend author --layers 1 --batch-size 1 --length 32 \
  --precision fp32 \
  --output-dir .runtime/olmo-rt-author-comparison/verify-author-fp32-b1-t32

# Repeat with BF16 mixed. Cross-backend precision misses remain failed even
# when each backend independently passes its own graph/update checks.
bash scripts/docker_shell.sh python scripts/olmo_rt_author_compare.py \
  --stage verify --backend author --layers 1 --batch-size 8 --length 512 \
  --output-dir .runtime/olmo-rt-author-comparison/verify-author-b8-t512

# Matched timing only after the relevant operational checks pass.
bash scripts/docker_shell.sh python scripts/olmo_rt_author_compare.py \
  --stage capacity --backend author --layers 1 --batch-size 32 \
  --output-dir .runtime/olmo-rt-author-comparison/capacity-author-l1-b32
bash scripts/docker_shell.sh python scripts/olmo_rt_author_compare.py \
  --stage capacity --backend native --layers 1 --batch-size 32 \
  --native-arm both --native-backward recompute \
  --output-dir .runtime/olmo-rt-author-comparison/capacity-native-l1-b32
```

Each run logs to `taylorbollman/pretrained-fbt-rt-nextlat`, group
`olmo-rt-author-comparison`, and retains its exact source/protocol snapshots.
The fixture uses the first one, two or six checkpoint blocks, fixed Gaussian
inputs/targets and FP32 MSE. Its throughput is block-stack input tokens per
second, **not full language-model throughput**. No quality result follows.
Raw gradients use a separate fixed approximately unit-L2 cotangent.

The author primary policy is `author_legacy`, with materialized backward
attention and four local MLP chunks. `fp32_state` is a separately labeled
diagnostic. Native primary uses the Stage A `both` switches, Triton historical
kernels, explicit cast reuse and bounded probability recomputation. CUDA graphs
capture gradient reset, forward, objective and backward; copies, clipping,
AdamW and scheduling are outside the graph and included in complete-update
timing. Compilation and capture are outside steady timings.

Source revisions and final selected runs belong in `summary.json`; consult the
final results for any failed precision or operational checks before interpreting
timings. Rebuild graphs after changing shape, metadata, precision, parameter
storage, backend or model settings. This harness does not exercise save/resume,
distributed training, generation caches or the integrated FBT/NextLat path.
