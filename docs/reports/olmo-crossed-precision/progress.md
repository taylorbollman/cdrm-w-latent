# Crossed-state precision progress

2026-09-29. **Diagnostic complete**, branch `feat/olmo-crossed-precision`,
from318948c/PR43. Read results.md and next-steps.md for assessment and proposal.

Protocol/position baseline pushed at6dd5a16; validated runtime frozen and pushed
at3364376 before execution. Prior helpers/importer/tests/protocols unchanged.
Final focused CPU scope:58passed in3.63s, one unchanged test-only warning;
`cpu-final-02.log` includes final position/W&B summaries. Earlier58-test run
is overlapping and not an extra independent scope.

`crossed-01` completed four aggregate cases/eight physical backwards in177.394s,
no optimizer updates. Root used GPU0 inside the verified project container;
GPU1 unused. All state, first-pass, health and final integrity controls pass.
W&B `cwxjdnwe` synced. Report SHA256:
`ba7989f586e19937ff6c64d7f3c65306eae9f61440b4fedb3942bdd3ff18e298`.
Postflight shows both H10080GB GPUs idle:0MiB,0%.

CA backbone/fusion gradient relative L2:8.7328%/13.2529%; reverse
AC:53.8216%/54.4150%. CA substantially improves absolute/directional agreement;
AC absolute discrepancies worsen versus original CC. Residual CA cotangent
error16.57%, AC supported hidden drift9.323%, and old AA12.44% unlocalized
spike remain qualifications. No BF16 production clearance.

Stage evidence retained and independently read back:2objects,111inventory
members,109source/snapshot pairs; all four first-pass identities and464position
mask/support records checked. See storage-receipt.md for exact generations.
Retention namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T072000Z/`.

Recommended next step is a bounded FP32 fusion-only warmup from original
backbone/fresh fusion. This remains a proposal, not a queued or launched run.
Save/push every20–30minutes; retain checkpoints every10minutes during future
training. Both GPUs idle. Final documentation closeout and PR pending below.
