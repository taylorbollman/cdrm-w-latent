# Add a bounded three-arm CDRM learning pilot with exact continuation

The earlier 100-update smoke test did not establish a meaningful learning
comparison with SEQ. Add a retained pilot for ordinary SEQ-5 FP32, tiled CDRM-5
FP32 and tiled CDRM-5 BF16 on native MAD selective copying V16/T256/K96, physical
B64. All arms share the corresponding initial ordinary weights, data order and
original epoch-based LR schedule. Existing CDRM update-100 checkpoints continue
with exact optimizer, scheduler, RNG and data-position state; SEQ starts from
the original common backbone. Model arithmetic and precision policy stay
unchanged.

The child runner records immutable ancestry/source/configuration identities,
supports stopping-target extensions up to the prospectively capped 2500
updates, and saves checkpoints compatible with the existing NUM validator.
Development evaluation and lambda-zero branch controls preserve training state.
A preparation tool freezes fresh numerical slots by seed, endpoint and source
trajectory; a wrapper enforces same-state checkpoint/fixture roles around the
unchanged validator. A CPU reporting tool audits pairing, scheduling, recovery
and raw failed flags, and publishes combined curves to W&B with local artifacts.

At the completed seed-7500, update-1000 endpoint, CDRM FP32 reaches development
CE 1.58755 and token accuracy 32.959%, versus SEQ’s 1.77906 and 29.704%. Both
have zero exact sequences among 256 development examples. CDRM BF16 reaches
CE 1.82448 and 28.031% accuracy, also with zero exact sequences. CDRM uses much more
recorded update time, so this single development comparison does not establish
a general architecture or efficiency advantage. The fixed-weight branch
ablation is reported separately from the independently trained SEQ baseline.

Same-state validation at both update-1000 checkpoints finds meaningful
full-model BF16 gradient/per-tensor/maximum and Adam failures. Gradient relative
L2 errors are 4.4423% and 2.5737%; Adam-delta errors are 4.6329% and 2.2058%,
both above the 1.5625% Adam allowance. The independent memory checks
pass, while lambda-zero and FP32-head controls retain the broader discrepancy.
BF16 continuation is explicitly diagnostic and capped. A subsequent
two-development-point degradation trigger at update 600 was investigated: most
of the gap remains when both trained checkpoints are evaluated in FP32, so it
reflects their learned trajectories. The final BF16 development gap is +0.23693
nats, and its last-200-training-loss gap is +0.07487 nats, also triggering the
prospective guard. Stop at the first 1000-update cohort review; do not extend to
2500 or train a second seed. The frozen criteria and failed results remain
unchanged, and this PR keeps BF16 experimental without promoting it to the
default precision policy.

Validation: 3 preparation, 8 runner and 4 numerical-wrapper CPU tests; 27
aggregate checks including real trajectory audits and error injection; actual
retained initialization/u100 state checks; and H100 BF16 recovery from 190 to 210
with bitwise matching model, Adam, scheduler, RNG, data position and non-timing
metrics across the epoch boundary. Final results, numerical qualifications,
curves and retention evidence are summarized in [results.md](results.md) and
the [prospective protocol](protocol.md).
