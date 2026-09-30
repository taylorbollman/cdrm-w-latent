# Validation scope

125 distinct focused CPU tests pass: branch helpers6, integration9, evaluator25,
original audit51, corrected audit18, and posthoc summary16. The40 initial tests
and all91 prelaunch tests passed together; later audit/summary additions passed
in their focused CPU-only containers. No old runtime source was modified.

The210 execution pins comprise the original200 files plus10 explicit branch
implementation/test/protocol sources. New posthoc v2 audit and summary helpers
are separately pinned and do not alter training source discovery. Original
engine prepare/update/save/retention callbacks remain AST-identical to accepted
callbacks. Numerical kernels and mathematical production callbacks are reused.

Tiny NFR1→3 control exactly matches the accepted update2/3 suffix, including
raw gradients, Adam/model/scheduler/counters, rank RNG/cursors, data/noise and
raw evaluation. Its paired reduced-KL run starts from the identical complete
parent boundary; raw first-forward loss sums match. Independent pair audit
passes2,901 checks, including source snapshot bytes.

The reduced-KL update2 checkpoint was downloaded from exact cloud generations
and byte-verified, then loaded in a fresh process. Update3 exactly matches the
uninterrupted reduced-KL reference. Independent restart audit passes2,289 checks.
Each branch records truthful inherited optimizer/objective metadata. Child resume
loads its own strict identity and does not reapply the parent transition; it
still requires original parent files/report as documented in the protocol.

The first tiny launch failed before parent loading or updates because its new
GCS prefix was outside the accepted storage root. The host launcher was fixed;
no runtime source changed. A post-freeze audit v1 assumption required native
payload.schedule in the tiny report. Separate v2 correctly checks the actual
tiny configuration.schedule and every used/next LR, preserving frozen v1.
Neither issue required changing model math, tolerances or stored checkpoints.

Native audit and endpoint results remain pending while the paired continuation
runs. Tiny exact restart evidence is not native exact restart evidence. Native
lean observations bind the ordered plan, materialization/masks/jitter sources
and row/logical-update identities; they do not retain every tensor input/noise
byte hash. The two-GPU parent restore/transition compares complete saved-state
hashes independently against the parent report before preparing graphs.

This milestone does not clear earlier BF16 compatibility qualifications, claim
RT benefit, or estimate generalization across seeds/data. The native comparison
contains no active RT layer. Both branches remain in the original100-update
warmup at the stop64 boundary; effective inputs/update and token budget are fixed.
