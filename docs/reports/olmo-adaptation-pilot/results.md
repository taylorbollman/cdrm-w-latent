# First 32-update adaptation pilot

**In progress.** Ordinary continuation B completed all 32 updates; NF is running,
followed by queued NFR. No arm is authorized beyond update 32
in this milestone. See [protocol](protocol.md) for the frozen comparison and
[progress](progress.md) for execution/recovery state.

All arms use the same 16,777,216 new input tokens at T1024. B starts the original
OLMo weights with fresh Adam. NF/NFR share the fusion128 import and fresh paired
predictor/optimizer initialization; that prior exposure is separately charged.
The main treatment comparison is NFR minus NF. This remains an early-warmup
functionality/adaptation pilot, not a definitive test of RT's scientific value.

| Completed arm | Dev CE at 16 | Dev CE at 32 | Clipped updates | Training + materialization inputs/s | Sampled peak reservation/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| B: ordinary continuation | 2.63112 | 2.63179 | 0 / 32 | 71,006 | 42.50 GiB |

B gradient norms range from 0.3797 to 0.4532, below the threshold of 1.0.
Development CE is effectively flat (+0.000676 nats/target between the two
observations). This provides a stable ordinary-model reference for the other
arms; there is no update-zero evaluation or claim of pristine-start improvement.

B's selected compute-plus-materialization regions total 236.28 seconds. Full
executor elapsed time is 757.34 seconds, including setup, local checkpoints and
final cloud retention; these are different timing scopes. Graph preparation
took 32.58 seconds, both evaluations together 6.19 seconds, and selected local
save regions 95.93 seconds. Background checkpoint time overlaps other work and
must not be added as sequential time. The sampled minimum free GPU memory is
35.11 GiB; it is not a continuous free-memory minimum.

The B endpoint is cloud-verified at update 32, W&B is synced, and the independent
JSON summary validates runtime/declaration pins, counts, allocation, schedule,
ownership, evaluations and storage metadata. Lean evaluation preservation checks
cover cursor/graph metadata, tensor ownership/version, gradients, runtime/modes
and RNG; this is not a new full-byte model/Adam equivalence experiment.

B run: [nxm2prv9](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/nxm2prv9).
Report SHA256: `203b8fd63448cd4da7a90f424d374d37417758a73c85ca9825d26129bb5035bb`.
Subset summary SHA256: `4712d55fe8963b934904705be9ec549cce4fe8188cbdc36c16be3d81fa04d504`.

The heavy-clipping/later-pass CE question remains open until NF/NFR have been
assessed at the same development boundaries. Falling auxiliary losses alone
will not establish useful refinement. No numerical-clearance or architecture
change is made during this cohort.
