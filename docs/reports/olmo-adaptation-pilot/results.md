# First 32-update adaptation pilot

**In progress.** Ordinary continuation B and FBT + NextLat NF completed all 32 updates;
NFR is running under recovered queue-02 after a host scheduler interruption. No arm is authorized beyond update 32
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
| NF: FBT + NextLat (pass 1 CE shown) | 3.25361 | 2.96039 | 32 / 32 | 7,783 | 58.70 GiB |

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

NF completed without retraining through the server interruption; its GPU
container and retention worker survived. Final cloud32 and W&B sync verified.
Original NF launcher exit code is unknown because its host parent was lost;
completion is established from the final runtime/storage/evaluation authorities.
Its per-pass dev CE16 is3.25361/7.77496/7.79839/7.81583, versus
2.96039/7.39349/7.42586/7.43514 at32. Absolute CE improves for every pass,
but later-pass gaps only narrow from4.52–4.56 to4.43–4.47nats/target.
Useful refinement is not established. The final raw norm is6.60185, down from
424.732 at the first update, with intermediate spikes. Every update clips.

NF selected compute+materialization is7,783inputs/s; full update callbacks
including two development insertions are7,674inputs/s. Executor elapsed is
3112.31s (51.87min), including setup/saves/terminal retention. Memory minimum
sampled free18.76GiB; cumulative sampled reserved highwater58.70GiB.
NF run: [uf1ojrgl](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/uf1ojrgl).
Final reportSHA8e6fe4d8aa933650943320c4f277fd92f4d59e530e08ef456d23634e79cbd5a0;
B+NF summarySHA45de106de400358cf9cb318338da6d567a0712b80236766b467f34943d4dbda3.
See [assessment guide](assessment-guide.md) for exact objective semantics and
limitations. Await paired NFR before deciding next work.
