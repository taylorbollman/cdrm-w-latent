# Ordinary OLMo throughput: results and assessment

Completed 2026-09-23. Seven bounded runs pass, with 42 physical optimizer
updates and nine focused CPU tests. The user's concern exposed substantial
avoidable loss-path overhead. The historical 31.1k rate measured a conservative
16-layer validation harness; it is not an optimized ordinary-training baseline.
No production model, attention kernel, default loss chunk or precision policy
was changed. Runtime/protocol are frozen at `71fbccd`.

## Measured six-layer results

View the [throughput plot](throughput.pdf).

One H100 80GB HBM3 (79.65 GiB visible), random native Mitchell initialization,
D2048, native SwiGLU8192 per branch, tied50304 vocabulary, RoPE, no Q/K norm.
Active parameters505,675,776; the common harness additionally retains8,388,608
frozen unused fusion parameters. RT, FBT and NextLat are all disabled.
BF16 autocast/FP32 parameters+Adam, deterministic PyTorch Flash, TF32off,
no global autocast weight cache, CUDA graphs for forward/loss/backward.

All rates are three-update medians after three eager preparation updates and
ten backward warmups. Complete updates include token copy/validation, clipping,
Adam and scheduling. Separate backward-only timings include the forward and
CE computation plus persistent-gradient zeroing. No data reading, W&B logging,
profiling or capture setup is included in steady-state timing.

| Physical batch | Heads | CE targets/row | Position chunk | Ordinary checkpointing | Full-update input tokens/s | Forward/loss/backward tokens/s | Peak allocated / reserved GiB |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 64 | 16 | 256 | 128 | on | 57,380 | 59,709 | 14.24 / 21.78 |
| 64 | 16 | 256 | 2048 | on | 92,813 | 98,737 | 14.24 / 21.78 |
| 512 | 32 | 256 | 128 | on | 43,856 | 44,218 | 56.46 / 72.78 |
| 512 | 32 | 256 | 2048 | on | 93,096 | 94,279 | 56.46 / 72.78 |
| 512 | 32 | 511 | 2048 | on | 73,517 | 74,449 | 56.47 / 76.78 |
| 32 | 32 | 511 | 2048 | on | 75,002 | 82,359 | 11.19 / 16.40 |
| 32 | 32 | 511 | 2048 | off | 88,861 | 99,246 | 21.17 / 36.00 |

The B64/H16 bridge retains F4's heads and half-target loss policy, while reducing
16 layers to six and using random weights/tokens. B512 uses32heads to approach
the paper geometry; this is not a controlled head-count/batch-size attribution.
The chunk pairs at each shape are controlled: only the CE grouping changes.
Half-target and full-target rows are different workloads and must not be mixed.
All physicalB512 attempts succeeded, with no OOM. Its setup reservations
72.78GiB (half CE) and76.78GiB (full CE) leave limited headroom; fitting does
not make B512 the recommended ordinary operating point. Each row uses one
physical batch per optimizer update, not accumulated globalB512.

## What explains the gap

1. **The original size comparison was mismatched.** F4 used16nativeSwiGLUlayers;
   the paper quoted six GELUlayers. Their dense block parameter counts differ
   by3.56x. Even native six-layer SwiGLU has33.3% more dense block weights than
   the paper's six-layer GELU. Native vocabulary is also larger. Read the
   [pinned recipe audit](paper-comparison.md) for exact counts and source links.
2. **Small CE chunks were a measured bottleneck.** At B64, increasing position
   chunks128→2048 improves57,380→92,813tokens/s (+61.75%). AtB512 it improves
   43,856→93,096 (+112.27%). This retains all vocabulary rows and supervised
   positions. It reduces repeated large readout operations and intermediate
   traffic. No loss terms or target positions were dropped.
3. **The paired profile identifies the work removed.** Copy/conversion,
   addition and fill families account for88.83% of the reduced device duration.
   Flash time is essentially unchanged11.859→11.864ms. Initially Flash was
   only2.16% of device duration; after CE regrouping it is3.57%. The evidence
   points to loss/other memory traffic ahead of changing FA versions for this
   specific ordinary configuration. See [profile audit](profile-audit.md).
4. **Checkpointing contributes a separate cost.** At matchedB32/H32/fullCE,
   turning off ordinary block checkpointing improves75,002→88,861tokens/s
   (+18.48%), while peak allocation grows11.19→21.17GiB. The published ordinary
   recipe has checkpointing off. Checkpointed CE chunks remain in both arms.
5. **Huge physical batch is not needed for this ordinary workload.** FullCE,
   checkpointedB32 andB512 give75.0k and73.5k respectively. Small-batch optimizer
   overhead matters: atB32 without ordinary checkpointing, captured forward/
   loss/backward gives99.2k versus88.9k for complete updates. The authors use
   global512 from16microbatches of32, so their optimizer is less frequent per
   input token. That accumulation route was not benchmarked here.

## What remains different from the paper

The153kpaperordinaryrate is not reproduced. Paper recipe: sixD2048/H32 layers,
GELU8192, ALiBi, affine normalization including Q/K normalization, untied32128
vocabulary, physicalmicrobatch32/global512, ordinary checkpointingoff and
`torch.compile`default. Native diagnostic preserves OLMo's SwiGLU/RoPE/noQKnorm,
full50304tiedreadout, explicit conservative precision/determinism and graphs
without pointwise compilation. All paper experiments are described as single
H100; exactH100variant and the resolvedrun/timerbehind153k are unspecified.
Do not apply the paper's separate layer-only latency exclusions to this number.

These differences can plausibly account for a substantial remaining gap, but
this benchmark does not allocate the entire gap quantitatively or prove that
native ordinary performance is optimized. Random initialization is appropriate
for a dense throughput check, not evidence about pretrained learning behavior.
We did not replace OLMo's MLP, change positional encoding, switch Flash backend,
disable determinism or benchmark all-layer RT.

## Validation and next recommendation

Each run confirms actual ordinaryFlash dispatch, finite losses/gradients/model/
Adamstate, expected participating gradients and changed trainable weights.
A same-weight2048-position native-width CE forward check agrees exactly between
chunk128/2048 on these fixtures. The9CPUtests include independent FP32 dense
versus chunkedCE loss and both input/readout gradients. These do not establish
native BF16 gradient/update equivalence across chunk sizes: grouping changes
GEMM/reduction rounding. Existing RT+FBT numerical qualifications remain open.

Use a bounded integration check of larger CE chunks before making them the
shared training default. Then measure the original16-layer ordinary model
with that policy and consider ordinary microbatching/checkpointing/pointwise
compilation based on profiling. Only after establishing that baseline should
we interpret RT slowdowns or prioritize FA changes. No broad optimization or
quality run is queued. The earlier graph-readiness plan remains documented;
this user-requested ordinary-throughput investigation precedes it.

## Evidence

[Protocol](protocol.md), [summary](summary.json), [CPU test log](test-results.txt),
[paper comparison](paper-comparison.md), [profile audit](profile-audit.md).
Exact per-run sources, protocol, raw reports and both compressed traces are
retained locally and selected for GCS retention; the final storage receipt
records verified objects. Random disposable weights need not be retained;
initialization/token seeds and source snapshots are recorded. No checkpoint
from the pretrained research line was changed. Final GPU is idle.

| Run | W&B |
| --- | --- |
| `l6-h16-b64-half-c128` | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/9t8dw046) |
| `l6-h16-b64-half-c2048` | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/5ed164zu) |
| `l6-h32-b512-half-c128` | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/0e8wttqg) |
| `l6-h32-b512-half-c2048` | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/gkial8t3) |
| `l6-h32-b512-full-c2048` | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/7m7wh3pw) |
| `l6-h32-b32-full-c2048-checkpoint` | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/xcb784vb) |
| `l6-h32-b32-full-c2048-no-checkpoint` | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8cw7ojkm) |
